import pytest

from mindpalace.config import (
    ConfigError,
    default_config_text,
    load_config,
    open_vault,
    scaffold,
)
from mindpalace.vault.paths import VaultPaths

MINIMAL = """---
schema_version: 1
entity_types: [concept]
edge_types:
  relates-to: {directed: false, cluster_weight: 1.0}
  supports: {directed: true, cluster_weight: 1.0}
thresholds:
  cluster_activation_entities: 150
  abstain_bm25_floor: 2.0
  abstain_cosine_floor: 0.35
  community_lineage_jaccard: 0.5
embedder: {kind: local, model: BAAI/bge-small-en-v1.5}
---

## template: extraction_next
Do the thing.

## template: report_next
Write the report.
"""


def write_config(tmp_path, text):
    path = tmp_path / "MINDPALACE.md"
    path.write_text(text)
    return path


def test_load_parses_structure_and_templates(tmp_path):
    config = load_config(write_config(tmp_path, MINIMAL))
    assert config.schema_version == 1
    assert config.entity_types == ["concept"]
    assert config.edge_types["relates-to"].directed is False
    assert config.edge_types["supports"].directed is True
    assert config.edge_types["relates-to"].cluster_weight == 1.0
    assert config.thresholds.cluster_activation_entities == 150
    assert config.thresholds.abstain_cosine_floor == 0.35
    assert config.embedder == {"kind": "local", "model": "BAAI/bge-small-en-v1.5"}
    assert config.templates["extraction_next"] == "Do the thing."
    assert config.templates["report_next"] == "Write the report."


def test_missing_required_key_names_the_key_and_file(tmp_path):
    broken = MINIMAL.replace("entity_types: [concept]\n", "")
    path = write_config(tmp_path, broken)
    with pytest.raises(ConfigError) as excinfo:
        load_config(path)
    assert "entity_types" in str(excinfo.value)
    assert "MINDPALACE.md" in str(excinfo.value)


def test_unknown_key_is_rejected_rather_than_ignored(tmp_path):
    broken = MINIMAL.replace("schema_version: 1", "schema_version: 1\nmystery: 3")
    with pytest.raises(ConfigError, match="mystery"):
        load_config(write_config(tmp_path, broken))


def test_unsupported_schema_version_is_rejected(tmp_path):
    broken = MINIMAL.replace("schema_version: 1", "schema_version: 99")
    with pytest.raises(ConfigError, match="schema_version"):
        load_config(write_config(tmp_path, broken))


def test_edge_type_missing_directed_is_rejected(tmp_path):
    broken = MINIMAL.replace(
        "relates-to: {directed: false, cluster_weight: 1.0}",
        "relates-to: {cluster_weight: 1.0}",
    )
    with pytest.raises(ConfigError, match="directed"):
        load_config(write_config(tmp_path, broken))


def test_default_config_is_itself_valid(tmp_path):
    config = load_config(write_config(tmp_path, default_config_text()))
    assert set(config.edge_types) == {
        "relates-to",
        "supports",
        "contradicts",
        "example-of",
        "part-of",
        "derived-from",
        "mentions",
    }
    assert config.edge_types["contradicts"].directed is False
    assert config.edge_types["supports"].directed is True
    assert "extraction_next" in config.templates


def test_scaffold_creates_layout_and_config(tmp_path):
    paths = VaultPaths(tmp_path)
    scaffold(paths)
    assert paths.mindpalace_md.exists()
    for directory in paths.all_directories():
        assert directory.is_dir()


def test_open_vault_requires_init_for_empty_directory(tmp_path):
    with pytest.raises(ConfigError, match="--init"):
        open_vault(tmp_path)


def test_open_vault_with_init_scaffolds_then_loads(tmp_path):
    paths, config = open_vault(tmp_path, init=True)
    assert paths.mindpalace_md.exists()
    assert config.schema_version == 1


def test_open_vault_refuses_non_empty_directory_without_config(tmp_path):
    (tmp_path / "unrelated.txt").write_text("hello")
    with pytest.raises(ConfigError, match="not a Mind Palace vault"):
        open_vault(tmp_path, init=True)


def test_open_vault_refuses_home_and_root(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    with pytest.raises(ConfigError, match="refusing"):
        open_vault(tmp_path, init=True)


# Regression tests for fixes (CRITICAL and IMPORTANT 1-4 plus Minor)

def test_open_vault_home_guard_with_symlinked_path(tmp_path, monkeypatch):
    """
    CRITICAL: $HOME guard must work even when path contains symlinks.
    On macOS, /var -> /private/var, so Path.home() might have symlinks.
    Both sides must be resolved before comparison.
    """
    import tempfile
    import os

    # Create a real directory and a symlink to it
    real_home = tmp_path / "real_home"
    real_home.mkdir()
    symlink_home = tmp_path / "symlink_home"
    try:
        symlink_home.symlink_to(real_home)
    except (OSError, NotImplementedError):
        # Skip on systems that don't support symlinks
        pytest.skip("symlinks not supported")

    # Set HOME to the symlinked path
    monkeypatch.setenv("HOME", str(symlink_home))

    # Verify that attempting to open the symlinked home directory still refuses
    with pytest.raises(ConfigError) as excinfo:
        open_vault(symlink_home, init=True)
    assert "refusing" in str(excinfo.value)

    # Verify no files were created
    assert not (real_home / "MINDPALACE.md").exists()
    assert not (real_home / "index.md").exists()


def test_edge_type_unknown_key_rejected(tmp_path):
    """IMPORTANT 1: unknown keys in edge_types specs are rejected, not silently dropped."""
    broken = MINIMAL.replace(
        "relates-to: {directed: false, cluster_weight: 1.0}",
        "relates-to: {directed: false, cluster_weight: 1.0, deprecated: true}",
    )
    path = write_config(tmp_path, broken)
    with pytest.raises(ConfigError) as excinfo:
        load_config(path)
    assert "deprecated" in str(excinfo.value)
    assert "MINDPALACE.md" in str(excinfo.value)


def test_thresholds_unknown_key_rejected(tmp_path):
    """IMPORTANT 1: unknown keys in thresholds are rejected, not silently dropped."""
    broken = MINIMAL.replace(
        "community_lineage_jaccard: 0.5",
        "community_lineage_jaccard: 0.5\n  experimental_threshold: 0.8",
    )
    path = write_config(tmp_path, broken)
    with pytest.raises(ConfigError) as excinfo:
        load_config(path)
    assert "experimental_threshold" in str(excinfo.value)
    assert "MINDPALACE.md" in str(excinfo.value)


def test_edge_type_directed_must_be_bool_not_string(tmp_path):
    """IMPORTANT 2: directed must be a bool, not a coercible value like a string."""
    # YAML parses directed: "false" as a string "false", not boolean False
    broken = MINIMAL.replace(
        'relates-to: {directed: false, cluster_weight: 1.0}',
        'relates-to: {directed: "false", cluster_weight: 1.0}',
    )
    path = write_config(tmp_path, broken)
    with pytest.raises(ConfigError) as excinfo:
        load_config(path)
    assert "directed" in str(excinfo.value)
    assert "relates-to" in str(excinfo.value)
    assert "MINDPALACE.md" in str(excinfo.value)


def test_edge_type_cluster_weight_invalid_float_wrapped(tmp_path):
    """IMPORTANT 3: invalid cluster_weight values raise ConfigError, not bare ValueError."""
    broken = MINIMAL.replace(
        "relates-to: {directed: false, cluster_weight: 1.0}",
        "relates-to: {directed: false, cluster_weight: not-a-number}",
    )
    path = write_config(tmp_path, broken)
    with pytest.raises(ConfigError) as excinfo:
        load_config(path)
    assert "cluster_weight" in str(excinfo.value)
    assert "relates-to" in str(excinfo.value)
    assert "MINDPALACE.md" in str(excinfo.value)


def test_entity_types_must_be_list_not_scalar(tmp_path):
    """IMPORTANT 4: entity_types must be a list, not a scalar that gets iterated."""
    broken = MINIMAL.replace(
        "entity_types: [concept]",
        "entity_types: concept",  # scalar instead of list
    )
    path = write_config(tmp_path, broken)
    with pytest.raises(ConfigError) as excinfo:
        load_config(path)
    assert "entity_types" in str(excinfo.value)
    assert "MINDPALACE.md" in str(excinfo.value)


def test_open_vault_error_message_tailored_for_non_empty_dir_without_init(tmp_path):
    """Minor: error message should be tailored based on whether directory is empty.

    The two candidate messages are distinguishable, so this must assert
    exactly the non-empty-dir wording and rule out the empty-dir wording --
    an `or` between the two would pass even if the tailoring branch fired
    backwards and returned the *empty*-dir message here instead, which is
    exactly the bug this test exists to catch.
    """
    (tmp_path / "unrelated.txt").write_text("hello")
    # Without init, error should mention the directory is not a vault
    with pytest.raises(ConfigError) as excinfo:
        open_vault(tmp_path, init=False)
    message = str(excinfo.value)
    assert "not a Mind Palace vault and is not empty" in message
    assert "no MINDPALACE.md" not in message


def test_open_vault_error_message_tailored_for_empty_dir_without_init(tmp_path):
    """The companion branch: an empty (or nonexistent) directory without
    --init gets the "no MINDPALACE.md" wording, not the "is not empty" one."""
    with pytest.raises(ConfigError) as excinfo:
        open_vault(tmp_path, init=False)
    message = str(excinfo.value)
    assert "no MINDPALACE.md" in message
    assert "is not empty" not in message


def test_init_scaffolds_over_another_sessions_half_built_vault(tmp_path):
    """Several servers may start on a brand-new vault at once, and they now
    all get as far as scaffolding it. The loser used to see the winner's
    freshly created `captures/`, `.graph/`, ... , conclude the directory was
    somebody else's, and refuse to --init -- so the very first launch of a new
    vault could still fail a session. Directories scaffolding itself creates
    are ours, not evidence of a foreign directory."""
    from mindpalace.vault.paths import VaultPaths

    for directory in VaultPaths(tmp_path).all_directories():
        directory.mkdir(parents=True, exist_ok=True)

    paths, config = open_vault(tmp_path, init=True)

    assert paths.mindpalace_md.exists()
    assert config.schema_version == 1


def test_init_still_refuses_a_directory_with_unrelated_content(tmp_path):
    """The loosening above must not extend to anything that is not ours: the
    guard exists so `--vault ~/Documents` cannot be scaffolded over."""
    (tmp_path / "captures").mkdir()
    (tmp_path / "tax-returns.pdf").write_text("not mine")

    with pytest.raises(ConfigError) as excinfo:
        open_vault(tmp_path, init=True)
    assert "refusing to --init" in str(excinfo.value)
