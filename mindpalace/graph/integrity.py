"""Entity-type integrity checks over an already-folded source set.

`fold` raises on an unknown *edge* type because it cannot proceed without
one: the edge type drives `is_symmetric` and `cluster_weight`, so a name
absent from config leaves the fold with no defined behaviour. An entity
type is only a label. Nothing downstream is blocked by a wrong one, so
refusing the note would be a harsher remedy than the problem warrants --
and `fold` is the one place where refusing means the note leaves the graph
entirely. So these three checks observe rather than enforce: they run
after the fold, over exactly what the fold produced, and report.

That leaves the label free to be wrong in three ways today, none of which
produces any signal:

- a type not in `config.entity_types` (only `write_note` validates, and
  only for notes it writes -- a hand-edited note is never checked);
- two notes disagreeing about one entity, which `touch` settles
  last-write-wins by `(created, id)` and never mentions;
- a slug that only ever appears as an assertion or claim endpoint, which
  `touch(slug, note_id, None)` admits to the graph with the `unknown`
  sentinel type.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

from mindpalace.config import Config
from mindpalace.graph.fold import GraphTables
from mindpalace.ids import slugify
from mindpalace.models import Note


@dataclass(frozen=True)
class TypeFinding:
    """One reportable defect. `kind` is used verbatim as a `vault_issue` kind.

    `detail` names the notes itself rather than leaving that to the caller,
    because the right verb differs per kind: two of these list files that
    declared a type, and the third lists files that only mentioned a name
    they never extracted.
    """

    kind: str
    slug: str
    detail: str
    note_ids: tuple[str, ...]


def check_entity_types(
    notes: Iterable[Note], tables: GraphTables, config: Config
) -> list[TypeFinding]:
    """Findings for `notes`, which must be the exact set `tables` was folded
    from -- a note quarantined out of the fold must be quarantined out of
    here too, or the user is sent to a second finding that disappears the
    moment they fix the first.
    """
    declared: dict[str, dict[str | None, list[str]]] = {}
    for note in notes:
        for instance in note.entities:
            slug = slugify(instance.name)
            note_ids = declared.setdefault(slug, {}).setdefault(instance.type, [])
            if note.id not in note_ids:
                note_ids.append(note.id)

    findings: list[TypeFinding] = []
    for slug, by_type in declared.items():
        for entity_type, note_ids in by_type.items():
            # `None` is a deliberate "no configured type fits" with the
            # wording kept in `proposed_type` (docs/decisions/0001 §1); it
            # is surfaced through vocabulary proposals, not as a typo.
            if entity_type is None or entity_type in config.entity_types:
                continue
            findings.append(
                TypeFinding(
                    kind="unknown_entity_type",
                    slug=slug,
                    detail=(
                        f"{slug!r} is declared as {entity_type!r} in "
                        f"{', '.join(note_ids)}, which is not one of the "
                        f"configured entity_types "
                        f"({', '.join(sorted(config.entity_types))})"
                    ),
                    note_ids=tuple(note_ids),
                )
            )

        # Only a disagreement between two *valid* types is a conflict.
        # `concpet` against `concept` is one typo with one fix, already
        # reported above; calling it a conflict as well would make the
        # review queue read as if there were two problems.
        configured = sorted(t for t in by_type if t in config.entity_types)
        if len(configured) > 1:
            conflicting_notes = sorted({n for t in configured for n in by_type[t]})
            # The winner is read back off the graph, never re-derived: it
            # is what queries actually see, and it is the thing that tells
            # the user which of the two files is the one to edit.
            winner = tables.entities[slug].type
            losers = ", ".join(f"{t!r}" for t in configured if t != winner)
            findings.append(
                TypeFinding(
                    kind="conflicting_entity_type",
                    slug=slug,
                    detail=(
                        f"{slug!r} is declared in {', '.join(conflicting_notes)} "
                        f"with more than one type; the graph kept {winner!r} and "
                        f"ignored {losers}"
                    ),
                    note_ids=tuple(conflicting_notes),
                )
            )

    # Derived from the notes rather than by testing for the `unknown`
    # sentinel type: nothing stops a vault from configuring `unknown` as a
    # real entity type, and then the sentinel and a deliberate declaration
    # would be indistinguishable.
    for slug in tables.entities:
        if slug in declared:
            continue
        findings.append(
            TypeFinding(
                kind="reference_only_entity",
                slug=slug,
                detail=(
                    f"{slug!r} is only ever referenced by "
                    f"{', '.join(tables.entities[slug].note_ids)}, as an assertion "
                    f"or claim endpoint, and never extracted as an entity there, "
                    f"so it has no type"
                ),
                note_ids=tuple(tables.entities[slug].note_ids),
            )
        )
    return sorted(findings, key=lambda f: (f.kind, f.slug))
