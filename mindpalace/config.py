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


def _validated_write(path: Path, rendered: str) -> None:
    """Write MINDPALACE.md only if the result loads. A rewrite that produced an
    unloadable config would brick the vault on its next open."""
    probe = path.with_name(f".{path.name}.probe")
    try:
        probe.write_text(rendered, encoding="utf-8")
        load_config(probe)
    finally:
        probe.unlink(missing_ok=True)
    atomic_write(path, rendered)


def _front_matter_span(lines: list[str]) -> tuple[int, int]:
    """(first, last) indices of the lines between the `---` delimiters."""
    if not lines or lines[0].strip() != "---":
        raise ConfigError("MINDPALACE.md has no front-matter block")
    for cursor in range(1, len(lines)):
        if lines[cursor].strip() == "---":
            return 1, cursor
    raise ConfigError("MINDPALACE.md: unterminated front-matter block")


def _yaml_flow(value: object) -> str:
    """One-line YAML for a scalar or a list, matching the template's style."""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (list, tuple)):
        return "[" + ", ".join(_yaml_flow(v) for v in value) + "]"
    return str(value)


def add_edge_type(
    path: Path,
    name: str,
    *,
    directed: bool,
    cluster_weight: float = 1.0,
    domain: list[str] | None = None,
    range: list[str] | None = None,
) -> None:
    """Adopt a proposed edge type into the vocabulary (docs/decisions/0001 §1).

    Edits the file textually -- one line appended to the `edge_types:` block
    -- rather than re-dumping the YAML, which would strip every comment the
    template ships and reflow every mapping. The result is validated by
    `load_config` before it is written.
    """
    config = load_config(path)
    if name in config.edge_types:
        raise ConfigError(f"{path.name}: edge type {name!r} already exists")
    raw = path.read_text(encoding="utf-8")
    lines = raw.splitlines(keepends=True)
    first, last = _front_matter_span(lines)
    # `range` is shadowed by the keyword argument of the same name.
    header = next(
        (i for i, line in enumerate(lines) if first <= i < last and re.match(r"^edge_types:\s*$", line)),
        None,
    )
    if header is None:
        raise ConfigError(f"{path.name}: edge_types must be a block mapping to add to")
    # The block runs until the next unindented, non-comment, non-blank line.
    stop = header + 1
    while stop < last and (
        lines[stop].startswith((" ", "\t")) or not lines[stop].strip() or lines[stop].lstrip().startswith("#")
    ):
        stop += 1
    # Insert after the last *item* line, not after trailing comments/blanks.
    insert_at = stop
    while insert_at > header + 1 and (
        not lines[insert_at - 1].strip() or lines[insert_at - 1].lstrip().startswith("#")
    ):
        insert_at -= 1
    sibling = lines[insert_at - 1] if insert_at - 1 > header else "  "
    indent = sibling[: len(sibling) - len(sibling.lstrip())] or "  "
    spec = [f"directed: {_yaml_flow(bool(directed))}", f"cluster_weight: {float(cluster_weight)}"]
    if domain:
        spec.append(f"domain: {_yaml_flow(list(domain))}")
    if range:
        spec.append(f"range: {_yaml_flow(list(range))}")
    lines.insert(insert_at, f"{indent}{name}: {{{', '.join(spec)}}}\n")
    _validated_write(path, "".join(lines))


def add_entity_type(path: Path, name: str) -> None:
    config = load_config(path)
    if name in config.entity_types:
        raise ConfigError(f"{path.name}: entity type {name!r} already exists")
    raw = path.read_text(encoding="utf-8")
    lines = raw.splitlines(keepends=True)
    first, last = _front_matter_span(lines)
    for i in range(first, last):
        flow = re.match(r"^(entity_types:\s*\[)(.*)(\]\s*)$", lines[i])
        if flow:
            items = flow.group(2).strip()
            joined = f"{items}, {name}" if items else name
            lines[i] = f"{flow.group(1)}{joined}{flow.group(3)}"
            break
        if re.match(r"^entity_types:\s*$", lines[i]):
            # Block style: append a `- name` line after the last item.
            stop = i + 1
            while stop < last and lines[stop].lstrip().startswith("-"):
                stop += 1
            sibling = lines[stop - 1] if stop - 1 > i else "  - x"
            indent = sibling[: len(sibling) - len(sibling.lstrip())] or "  "
            lines.insert(stop, f"{indent}- {name}\n")
            break
    else:
        raise ConfigError(f"{path.name}: could not find entity_types to add to")
    _validated_write(path, "".join(lines))


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
