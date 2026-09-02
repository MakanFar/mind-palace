"""Parsing, validation, and scaffolding of MINDPALACE.md."""

from __future__ import annotations

import re
from dataclasses import dataclass
from importlib import resources
from pathlib import Path

from mindpalace.atomic import atomic_write
from mindpalace.frontmatter import FrontMatterError, parse, render
from mindpalace.vault.paths import VaultPaths

SUPPORTED_SCHEMA_VERSIONS = {1}
REQUIRED_KEYS = {
    "schema_version",
    "entity_types",
    "edge_types",
    "thresholds",
    "embedder",
}
REQUIRED_THRESHOLDS = {
    "cluster_activation_entities",
    "abstain_bm25_floor",
    "abstain_cosine_floor",
    "community_lineage_jaccard",
}
# Added after v0 vaults were already scaffolded. A required key here would
# refuse every existing MINDPALACE.md on open; an optional one with a default
# lets an old vault keep working and a new one tune it.
OPTIONAL_THRESHOLDS = {
    "duplicate_cosine_floor": 0.92,
}
EDGE_TYPE_KEYS = {"directed", "cluster_weight", "domain", "range"}
TEMPLATE_HEADING = re.compile(r"^##\s+template:\s*(\S+)\s*$", re.MULTILINE)


class ConfigError(ValueError):
    """Raised for any invalid or missing configuration."""


@dataclass(frozen=True)
class EdgeType:
    """`domain` and `range` are entity-type signatures (docs/decisions/0001 §3).

    `None` means unconstrained. A symmetric type may carry `domain` only,
    applied to both ends: "range" has no meaning when the ends are
    interchangeable, and accepting it would let a vault declare a constraint
    that is enforced on whichever end happened to sort second.
    """

    name: str
    directed: bool
    cluster_weight: float
    domain: tuple[str, ...] | None = None
    range: tuple[str, ...] | None = None


@dataclass(frozen=True)
class Thresholds:
    cluster_activation_entities: int
    abstain_bm25_floor: float
    abstain_cosine_floor: float
    community_lineage_jaccard: float
    duplicate_cosine_floor: float = OPTIONAL_THRESHOLDS["duplicate_cosine_floor"]


@dataclass(frozen=True)
class Config:
    schema_version: int
    entity_types: list[str]
    edge_types: dict[str, EdgeType]
    thresholds: Thresholds
    embedder: dict
    templates: dict[str, str]

    def is_symmetric(self, edge_type: str) -> bool:
        try:
            return not self.edge_types[edge_type].directed
        except KeyError:
            raise ConfigError(f"unknown edge type {edge_type!r}") from None


def default_config_text() -> str:
    return (
        resources.files("mindpalace.templates")
        .joinpath("MINDPALACE.md")
        .read_text(encoding="utf-8")
    )


def load_config(path: Path) -> Config:
    try:
        data, body = parse(path.read_text(encoding="utf-8"))
    except FrontMatterError as exc:
        raise ConfigError(f"{path.name}: {exc}") from exc

    missing = REQUIRED_KEYS - data.keys()
    if missing:
        raise ConfigError(f"{path.name}: missing required key(s) {sorted(missing)}")
    unknown = data.keys() - REQUIRED_KEYS
    if unknown:
        raise ConfigError(f"{path.name}: unknown key(s) {sorted(unknown)}")

    version = data["schema_version"]
    if version not in SUPPORTED_SCHEMA_VERSIONS:
        raise ConfigError(
            f"{path.name}: schema_version {version} is unsupported "
            f"(supported: {sorted(SUPPORTED_SCHEMA_VERSIONS)})"
        )

    # Validate entity_types is a list, not a scalar
    if not isinstance(data["entity_types"], list):
        raise ConfigError(
            f"{path.name}: entity_types must be a list, got {type(data['entity_types']).__name__!r}"
        )

    edge_types = {}
    for name, spec in data["edge_types"].items():
        for field in ("directed", "cluster_weight"):
            if field not in spec:
                raise ConfigError(
                    f"{path.name}: edge type {name!r} is missing {field!r}"
                )

        # Check for unknown keys in edge type spec
        unknown_edge_fields = spec.keys() - EDGE_TYPE_KEYS
        if unknown_edge_fields:
            raise ConfigError(
                f"{path.name}: edge type {name!r} has unknown key(s) {sorted(unknown_edge_fields)}"
            )

        # Validate directed is a bool, not a coercible value
        if not isinstance(spec["directed"], bool):
            raise ConfigError(
                f"{path.name}: edge type {name!r} field 'directed' must be a boolean, got {type(spec['directed']).__name__!r}"
            )

        # Wrap cluster_weight coercion
        try:
            cluster_weight = float(spec["cluster_weight"])
        except (TypeError, ValueError) as exc:
            raise ConfigError(
                f"{path.name}: edge type {name!r} field 'cluster_weight' must be a float: {exc}"
            ) from exc

        signature = {}
        for end in ("domain", "range"):
            raw_end = spec.get(end)
            if raw_end is None:
                signature[end] = None
                continue
            if not isinstance(raw_end, list) or not all(
                isinstance(t, str) for t in raw_end
            ):
                raise ConfigError(
                    f"{path.name}: edge type {name!r} field {end!r} must be a "
                    f"list of entity types"
                )
            unknown_types = [t for t in raw_end if t not in data["entity_types"]]
            if unknown_types:
                raise ConfigError(
                    f"{path.name}: edge type {name!r} {end} names entity type(s) "
                    f"{unknown_types} that are not in entity_types"
                )
            signature[end] = tuple(raw_end)
        if not spec["directed"] and signature["range"] is not None:
            raise ConfigError(
                f"{path.name}: edge type {name!r} is symmetric, so it may declare "
                f"'domain' (applied to both ends) but not 'range'"
            )

        edge_types[name] = EdgeType(
            name=name,
            directed=spec["directed"],
            cluster_weight=cluster_weight,
            domain=signature["domain"],
            range=signature["range"],
        )

    raw_thresholds = data["thresholds"]
    missing_thresholds = REQUIRED_THRESHOLDS - raw_thresholds.keys()
    if missing_thresholds:
        raise ConfigError(
            f"{path.name}: thresholds missing {sorted(missing_thresholds)}"
        )

    # Check for unknown keys in thresholds
    unknown_threshold_fields = (
        raw_thresholds.keys() - REQUIRED_THRESHOLDS - OPTIONAL_THRESHOLDS.keys()
    )
    if unknown_threshold_fields:
        raise ConfigError(
            f"{path.name}: thresholds has unknown key(s) {sorted(unknown_threshold_fields)}"
        )
    thresholds = {k: raw_thresholds[k] for k in REQUIRED_THRESHOLDS}
    for key, default in OPTIONAL_THRESHOLDS.items():
        try:
            thresholds[key] = float(raw_thresholds.get(key, default))
        except (TypeError, ValueError) as exc:
            raise ConfigError(
                f"{path.name}: threshold {key!r} must be a float: {exc}"
            ) from exc

    return Config(
        schema_version=version,
        entity_types=list(data["entity_types"]),
        edge_types=edge_types,
        thresholds=Thresholds(**thresholds),
        embedder=dict(data["embedder"]),
        templates=_parse_templates(body),
    )


def _parse_templates(body: str) -> dict[str, str]:
    matches = list(TEMPLATE_HEADING.finditer(body))
    templates: dict[str, str] = {}
    for index, match in enumerate(matches):
        end = matches[index + 1].start() if index + 1 < len(matches) else len(body)
        templates[match.group(1)] = body[match.end() : end].strip()
    return templates


def _rewrite_front_matter(path: Path, mutate) -> None:
    """Apply `mutate(data)` to MINDPALACE.md's front-matter and write it back,
    leaving the template body byte-for-byte as it was.

    The result is re-validated through `load_config` before it is written, so
    a mutation that would produce an unloadable config is rejected instead of
    bricking the vault on its next open.
    """
    raw = path.read_text(encoding="utf-8")
    try:
        data, body = parse(raw)
    except FrontMatterError as exc:
        raise ConfigError(f"{path.name}: {exc}") from exc
    mutate(data)
    rendered = render(data, body)
    probe = path.with_name(f".{path.name}.probe")
    try:
        probe.write_text(rendered, encoding="utf-8")
        load_config(probe)
    finally:
        probe.unlink(missing_ok=True)
    atomic_write(path, rendered)


def add_edge_type(
    path: Path,
    name: str,
    *,
    directed: bool,
    cluster_weight: float = 1.0,
    domain: list[str] | None = None,
    range: list[str] | None = None,
) -> None:
    """Adopt a proposed edge type into the vocabulary (docs/decisions/0001 §1)."""

    def mutate(data: dict) -> None:
        if name in data["edge_types"]:
            raise ConfigError(f"{path.name}: edge type {name!r} already exists")
        spec: dict = {"directed": bool(directed), "cluster_weight": float(cluster_weight)}
        if domain:
            spec["domain"] = list(domain)
        if range:
            spec["range"] = list(range)
        data["edge_types"][name] = spec

    _rewrite_front_matter(path, mutate)


def add_entity_type(path: Path, name: str) -> None:
    def mutate(data: dict) -> None:
        if name in data["entity_types"]:
            raise ConfigError(f"{path.name}: entity type {name!r} already exists")
        data["entity_types"].append(name)

    _rewrite_front_matter(path, mutate)


def scaffold(paths: VaultPaths) -> None:
    for directory in paths.all_directories():
        directory.mkdir(parents=True, exist_ok=True)
    atomic_write(paths.mindpalace_md, default_config_text())
    atomic_write(paths.index_md, "# Index\n\n_No notes yet._\n")


def open_vault(root: Path, *, init: bool = False) -> tuple[VaultPaths, Config]:
    root = Path(root).expanduser().resolve()
    # Resolve both sides to catch symlinked paths (common on macOS)
    if root == Path.home().resolve() or root == Path(root.anchor).resolve():
        raise ConfigError(f"refusing to use {root} as a vault root")

    paths = VaultPaths(root)
    if paths.mindpalace_md.exists():
        return paths, load_config(paths.mindpalace_md)

    # Check non-emptiness first to tailor the error message. "Empty" ignores
    # the directories scaffolding itself creates: several servers can start on
    # one brand-new vault at once, and the one that loses the race would
    # otherwise mistake the winner's half-built `captures/`, `.graph/`, ... for
    # a foreign directory and refuse. Their names are all this ignores -- a
    # single unrelated file still means the vault is somebody else's.
    ours = {directory.name for directory in paths.all_directories()}
    ours.add(paths.index_md.name)
    is_empty = not root.exists() or all(
        entry.name in ours for entry in root.iterdir()
    )

    if not init:
        if is_empty:
            raise ConfigError(f"{root} has no MINDPALACE.md; re-run with --init")
        else:
            raise ConfigError(
                f"{root} is not a Mind Palace vault and is not empty; re-run with --init"
            )

    if not is_empty:
        raise ConfigError(
            f"{root} is not a Mind Palace vault and is not empty; refusing to --init"
        )

    scaffold(paths)
    return paths, load_config(paths.mindpalace_md)
