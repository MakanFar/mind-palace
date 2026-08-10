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
