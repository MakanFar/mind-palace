"""Retrieval. RRF orders results; a separate evidence gate decides relevance."""

from __future__ import annotations

import re
import sqlite3
from collections.abc import Sequence
from dataclasses import asdict, dataclass

from mindpalace.config import Config
from mindpalace.embed import Embedder
from mindpalace.index import vectors

# LOCAL_KINDS includes "unparsed" so a file we could not parse is still
# reachable by search rather than silently absent from the vault.
LOCAL_KINDS = ("capture", "note", "entity", "unparsed")
RRF_K = 60
CANDIDATES = 40
TOKEN = re.compile(r"[A-Za-z0-9]+")

# Appended to whatever abstain message the vault's own config supplies. The
# instruction to say so rather than fabricate from the model's own weights is
# the safety-critical part of this message; it must survive even if an
# operator's custom template is short or forgets to say it explicitly.
ABSTAIN_INSTRUCTION = " Say so rather than answering from your own knowledge."
DEFAULT_ABSTAIN_MESSAGE = "Nothing in the vault is relevant to this query."


@dataclass(frozen=True)
class Hit:
    id: str
    kind: str
    title: str
    snippet: str
    bm25: float
    cosine: float
    score: float


def rrf(rank_lists: Sequence[Sequence[str]], k: int = RRF_K) -> list[tuple[str, float]]:
    scores: dict[str, float] = {}
    for ranked in rank_lists:
        for position, doc_id in enumerate(ranked):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + position + 1)
    return sorted(scores.items(), key=lambda item: (-item[1], item[0]))


def passes_evidence_gate(
    hits: Sequence[Hit], bm25_floor: float, cosine_floor: float
) -> bool:
    """Relevance is judged on raw signals, never on the fused ordering score."""
    return any(hit.bm25 >= bm25_floor or hit.cosine >= cosine_floor for hit in hits)


def _fts_query(text: str) -> str:
    terms = TOKEN.findall(text)
    return " OR ".join(f'"{term}"' for term in terms)


def _fts_candidates(
    conn: sqlite3.Connection, query: str, kinds: Sequence[str], limit: int
) -> dict[str, dict]:
    expression = _fts_query(query)
    if not expression:
        return {}
    placeholders = ",".join("?" for _ in kinds)
    rows = conn.execute(
        f"SELECT doc_id, kind, title, -bm25(docs) AS score, "
        f"snippet(docs, 3, '', '', '…', 12) AS snippet "
        f"FROM docs WHERE docs MATCH ? AND kind IN ({placeholders}) "
        f"ORDER BY bm25(docs) LIMIT ?",
        (expression, *kinds, limit),
    ).fetchall()
    return {
        row["doc_id"]: {
            "kind": row["kind"],
            "title": row["title"],
            "snippet": row["snippet"],
            "bm25": max(0.0, float(row["score"])),
        }
        for row in rows
    }


def _doc_meta(conn: sqlite3.Connection, doc_id: str) -> dict:
    row = conn.execute(
        "SELECT kind, title, substr(text, 1, 200) AS snippet FROM docs WHERE doc_id = ?",
        (doc_id,),
    ).fetchone()
    if row is None:
        return {"kind": "", "title": "", "snippet": ""}
    return {"kind": row["kind"], "title": row["title"], "snippet": row["snippet"]}


def _abstain_note(config: Config) -> str:
    message = config.templates.get("abstain", DEFAULT_ABSTAIN_MESSAGE)
    return f"{message}{ABSTAIN_INSTRUCTION}"


def local_search(
    conn: sqlite3.Connection,
    embedder: Embedder,
    query: str,
    config: Config,
    k: int = 8,
    expand_graph: bool = False,
) -> dict:
    query_vector = embedder.embed([query])[0]
    vector_hits = vectors.search(
        conn, query_vector, embedder.model_id, LOCAL_KINDS, CANDIDATES
    )
    cosines = {doc_id: score for doc_id, score in vector_hits}
    lexical = _fts_candidates(conn, query, LOCAL_KINDS, CANDIDATES)

    fused = rrf([[doc_id for doc_id, _ in vector_hits], list(lexical)])

    hits: list[Hit] = []
    for doc_id, score in fused[:k]:
        meta = lexical.get(doc_id) or _doc_meta(conn, doc_id)
        hits.append(
            Hit(
                id=doc_id,
                kind=meta["kind"],
                title=meta["title"],
                snippet=meta["snippet"],
                bm25=float(meta.get("bm25", 0.0)),
                cosine=float(cosines.get(doc_id, 0.0)),
                score=score,
            )
        )

    signals = {
        "max_bm25": max((h.bm25 for h in hits), default=0.0),
        "max_cosine": max((h.cosine for h in hits), default=0.0),
        "bm25_floor": config.thresholds.abstain_bm25_floor,
        "cosine_floor": config.thresholds.abstain_cosine_floor,
    }

    if not passes_evidence_gate(
        hits,
        config.thresholds.abstain_bm25_floor,
        config.thresholds.abstain_cosine_floor,
    ):
        return {
            "hits": [],
            "neighbours": [],
            "signals": signals,
            "note": _abstain_note(config),
        }

    neighbours = _expand(conn, hits) if expand_graph else []
    return {
        "hits": [asdict(hit) for hit in hits],
        "neighbours": neighbours,
        "signals": signals,
        # Present (as None) on the success path too, so callers can read
        # result["note"] unconditionally instead of branching on abstain vs.
        # not.
        "note": None,
    }


REPORT_KIND = ("report",)
# RRF relevance sits in [0, 2/(RRF_K+1)] ~= [0, 0.033), and the smallest gap
# between adjacent candidates in a 40-deep ranking is ~1/(RRF_K+40)^2 ~= 1e-4
# (see _fts_candidates/CANDIDATES). Impact rank is a 0-10 score; for it to act
# as a genuine *tiebreak* rather than a second vote that can outrank real
# relevance differences, its full 10-point spread must stay well under that
# smallest gap. 1e-6 keeps the spread at 1e-5, an order of magnitude under it.
RANK_TIEBREAK_WEIGHT = 0.000001


def global_search(
    conn: sqlite3.Connection,
    store,
    embedder: Embedder,
    query: str,
    config: Config,
) -> dict:
    """Rank community reports for the assistant to answer from.

    At personal-vault scale ten to twenty reports fit in one context window, so
    the paper's distributed map-reduce is unnecessary; the assistant answers
    directly. Communities lacking a usable report are still returned with their
    members so no region of the graph silently vanishes.
    """
    communities = conn.execute(
        "SELECT lineage_id, level, members FROM communities ORDER BY level, lineage_id"
    ).fetchall()
    entity_count = conn.execute("SELECT COUNT(*) FROM entities").fetchone()[0]
    threshold = config.thresholds.cluster_activation_entities

    if not communities:
        return {
            "available": False,
            "entity_count": entity_count,
            "threshold": threshold,
            "communities": [],
            "instructions": "",
            "note": (
                f"No communities exist yet ({entity_count} entities; clustering "
                f"activates at {threshold}). Use local_search for this query."
            ),
        }

    reports = {report.lineage_id: report for report in store.iter_reports()}

    query_vector = embedder.embed([query])[0]
    relevance = dict(
        vectors.search(conn, query_vector, embedder.model_id, REPORT_KIND, CANDIDATES)
    )
    lexical = _fts_candidates(conn, query, REPORT_KIND, CANDIDATES)
    ordering = {
        doc_id: score
        for doc_id, score in rrf([list(relevance), list(lexical)])
    }

    entries = []
    for row in communities:
        lineage_id = row["lineage_id"]
        report = reports.get(lineage_id)
        if report is None:
            state = "missing"
        elif report.stale:
            state = "stale"
        else:
            state = "present"
        entries.append(
            {
                "lineage_id": lineage_id,
                "level": row["level"],
                "members": row["members"].split(",") if row["members"] else [],
                "report": state,
                "title": report.title if report else None,
                "summary": report.summary if report else None,
                "rank": report.rank if report else 0.0,
                "findings": report.findings if report else [],
                "relevance": ordering.get(lineage_id, 0.0),
            }
        )

    entries.sort(
        key=lambda entry: (
            -(entry["relevance"] + RANK_TIEBREAK_WEIGHT * entry["rank"]),
            entry["lineage_id"],
        )
    )

    return {
        "available": True,
        "entity_count": entity_count,
        "threshold": threshold,
        "communities": entries,
        "instructions": config.templates.get("report_next", ""),
        # Present (as None) on the unavailable path too via the branch above,
        # so callers can read result["note"] unconditionally.
        "note": None,
    }


def _expand(conn: sqlite3.Connection, hits: Sequence[Hit]) -> list[dict]:
    slugs = [hit.id.removeprefix("e_") for hit in hits if hit.kind == "entity"]
    if not slugs:
        return []
    placeholders = ",".join("?" for _ in slugs)
    rows = conn.execute(
        f"SELECT source, target, type, weight FROM aggregates "
        f"WHERE traversable = 1 AND (source IN ({placeholders}) "
        f"OR target IN ({placeholders}))",
        (*slugs, *slugs),
    ).fetchall()

    seen: set[str] = set()
    neighbours: list[dict] = []
    for row in rows:
        for slug in (row["source"], row["target"]):
            if slug in slugs or slug in seen:
                continue
            seen.add(slug)
            neighbours.append(
                {"slug": slug, "via": row["type"], "weight": row["weight"]}
            )
    return neighbours
