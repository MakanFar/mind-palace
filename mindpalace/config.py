"""Parsing, validation, and scaffolding of MINDPALACE.md."""

from __future__ import annotations

import re
from dataclasses import dataclass
from importlib import resources
from pathlib import Path

from mindpalace.atomic import atomic_write
from mindpalace.frontmatter import FrontMatterError, parse
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
TEMPLATE_HEADING = re.compile(r"^##\s+template:\s*(\S+)\s*$", re.MULTILINE)


class ConfigError(ValueError):
    """Raised for any invalid or missing configuration."""


@dataclass(frozen=True)
class EdgeType:
    name: str
    directed: bool
    cluster_weight: float


@dataclass(frozen=True)
class Thresholds:
    cluster_activation_entities: int
    abstain_bm25_floor: float
    abstain_cosine_floor: float
    community_lineage_jaccard: float


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
        unknown_edge_fields = spec.keys() - {"directed", "cluster_weight"}
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

        edge_types[name] = EdgeType(
            name=name,
            directed=spec["directed"],
            cluster_weight=cluster_weight,
        )

    raw_thresholds = data["thresholds"]
    missing_thresholds = REQUIRED_THRESHOLDS - raw_thresholds.keys()
    if missing_thresholds:
        raise ConfigError(
            f"{path.name}: thresholds missing {sorted(missing_thresholds)}"
        )

    # Check for unknown keys in thresholds
    unknown_threshold_fields = raw_thresholds.keys() - REQUIRED_THRESHOLDS
    if unknown_threshold_fields:
        raise ConfigError(
            f"{path.name}: thresholds has unknown key(s) {sorted(unknown_threshold_fields)}"
        )

    return Config(
        schema_version=version,
        entity_types=list(data["entity_types"]),
        edge_types=edge_types,
        thresholds=Thresholds(**{k: raw_thresholds[k] for k in REQUIRED_THRESHOLDS}),
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
