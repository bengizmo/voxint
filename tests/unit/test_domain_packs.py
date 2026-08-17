from pathlib import Path

import pytest
import yaml

from voxint.config import Settings
from voxint.domain_packs.base import DomainPack, DomainPackError, load_default
from voxint.domain_packs.registry import (
    available_domain_packs,
    default_domain_pack,
    resolve_domain_pack_by_name,
    resolve_folder_pack_name,
    resolve_run_domain_pack,
)


def _write_pack(root: Path, name: str, **fields: object) -> Path:
    pack_dir = root / name
    pack_dir.mkdir(parents=True)
    manifest = {"name": name, **fields}
    (pack_dir / "manifest.yaml").write_text(yaml.safe_dump(manifest))
    return pack_dir


def test_generic_pack_loads() -> None:
    pack = load_default()
    assert pack.name == "generic"
    assert pack.vocabulary == ()
    assert "enhancement_context" in pack.prompt_fragments


# --- serialization round-trip (the per-run snapshot contract) ----------------


def test_to_mapping_from_mapping_round_trips() -> None:
    pack = DomainPack(
        name="interviews",
        description="Investigative interviews",
        vocabulary=("subpoena", "deposition"),
        name_seeds=("Jane Doe",),
        prompt_fragments={"enhancement_context": "Formal register.", "summary_context": "Neutral."},
    )
    restored = DomainPack.from_mapping(pack.to_mapping())
    assert restored == pack


def test_to_mapping_is_json_safe_types() -> None:
    pack = load_default()
    data = pack.to_mapping()
    assert isinstance(data["vocabulary"], list)
    assert isinstance(data["name_seeds"], list)
    assert isinstance(data["prompt_fragments"], dict)


def test_to_mapping_does_not_alias_live_dict() -> None:
    pack = DomainPack(name="p", prompt_fragments={"k": "v"})
    data = pack.to_mapping()
    data["prompt_fragments"]["k"] = "mutated"
    assert pack.prompt_fragments["k"] == "v"


@pytest.mark.parametrize(
    "bad",
    [
        {},  # no name
        {"name": ""},  # blank name
        {"name": "  "},  # whitespace-only name
        {"name": 3},  # non-string name
        {"name": "p", "description": 7},  # non-string description
        {"name": "p", "vocabulary": "notalist"},  # vocabulary not a list
        {"name": "p", "vocabulary": [1, 2]},  # vocabulary entries not strings
        {"name": "p", "name_seeds": {"a": "b"}},  # name_seeds not a list
        {"name": "p", "prompt_fragments": ["x"]},  # fragments not a mapping
        {"name": "p", "prompt_fragments": {"k": 5}},  # fragment value not a string
    ],
)
def test_from_mapping_rejects_corrupt_snapshots(bad: dict[str, object]) -> None:
    with pytest.raises(DomainPackError):
        DomainPack.from_mapping(bad)


def test_load_missing_manifest_raises(tmp_path: Path) -> None:
    with pytest.raises(DomainPackError):
        DomainPack.load(tmp_path / "nope")


def test_load_malformed_yaml_raises(tmp_path: Path) -> None:
    pack_dir = tmp_path / "broken"
    pack_dir.mkdir()
    (pack_dir / "manifest.yaml").write_text("name: [unclosed")
    with pytest.raises(DomainPackError):
        DomainPack.load(pack_dir)


# --- registry resolution -----------------------------------------------------


def test_default_pack_is_generic_when_unconfigured() -> None:
    settings = Settings(_env_file=None)
    assert default_domain_pack(settings).name == "generic"


def test_default_pack_honors_domain_pack_path(tmp_path: Path) -> None:
    _write_pack(tmp_path, "custom", vocabulary=["widget"])
    settings = Settings(_env_file=None, domain_pack_path=tmp_path / "custom")
    pack = default_domain_pack(settings)
    assert pack.name == "custom"
    assert pack.vocabulary == ("widget",)


def test_available_packs_includes_generic_by_default() -> None:
    settings = Settings(_env_file=None)
    packs = available_domain_packs(settings)
    assert set(packs) == {"generic"}


def test_available_packs_scans_packs_dir(tmp_path: Path) -> None:
    _write_pack(tmp_path, "podcast", vocabulary=["cold open"])
    _write_pack(tmp_path, "interview")
    settings = Settings(_env_file=None, domain_packs_dir=tmp_path)
    packs = available_domain_packs(settings)
    assert set(packs) == {"generic", "podcast", "interview"}
    assert packs["podcast"].vocabulary == ("cold open",)


def test_available_packs_missing_dir_raises(tmp_path: Path) -> None:
    settings = Settings(_env_file=None, domain_packs_dir=tmp_path / "absent")
    with pytest.raises(DomainPackError):
        available_domain_packs(settings)


def test_available_packs_duplicate_name_conflict_raises(tmp_path: Path) -> None:
    _write_pack(tmp_path, "dup", vocabulary=["a"])
    # A second pack folder whose manifest also claims name "dup" but differs.
    other = tmp_path / "dup2"
    other.mkdir()
    (other / "manifest.yaml").write_text(yaml.safe_dump({"name": "dup", "vocabulary": ["b"]}))
    settings = Settings(_env_file=None, domain_packs_dir=tmp_path)
    with pytest.raises(DomainPackError):
        available_domain_packs(settings)


def test_available_packs_same_pack_two_sources_is_idempotent(tmp_path: Path) -> None:
    # DOMAIN_PACK_PATH points at a pack that ALSO sits under DOMAIN_PACKS_DIR.
    _write_pack(tmp_path, "shared", vocabulary=["x"])
    settings = Settings(
        _env_file=None,
        domain_pack_path=tmp_path / "shared",
        domain_packs_dir=tmp_path,
    )
    packs = available_domain_packs(settings)
    assert packs["shared"].vocabulary == ("x",)


def test_resolve_by_name_returns_pack(tmp_path: Path) -> None:
    _write_pack(tmp_path, "podcast")
    settings = Settings(_env_file=None, domain_packs_dir=tmp_path)
    assert resolve_domain_pack_by_name("podcast", settings).name == "podcast"


def test_resolve_by_name_unknown_raises(tmp_path: Path) -> None:
    settings = Settings(_env_file=None, domain_packs_dir=tmp_path)
    with pytest.raises(DomainPackError):
        resolve_domain_pack_by_name("nonesuch", settings)


# --- folder → pack name resolution (pure) ------------------------------------


def test_folder_match_returns_mapped_name() -> None:
    mapping = {"podcasts": "podcast", "interviews": "interview"}
    assert resolve_folder_pack_name("podcasts/ep1.wav", mapping) == "podcast"
    assert resolve_folder_pack_name("interviews/subdir/a.wav", mapping) == "interview"


def test_folder_unmapped_returns_none() -> None:
    assert resolve_folder_pack_name("misc/x.wav", {"podcasts": "podcast"}) is None


def test_folder_empty_mapping_returns_none() -> None:
    assert resolve_folder_pack_name("podcasts/ep1.wav", {}) is None


def test_folder_longest_ancestor_wins() -> None:
    mapping = {"audio": "general", "audio/interviews": "interview"}
    assert resolve_folder_pack_name("audio/interviews/a.wav", mapping) == "interview"
    assert resolve_folder_pack_name("audio/other/a.wav", mapping) == "general"


def test_folder_matches_on_components_not_string_prefix() -> None:
    # "pod" must NOT match a file under "podcasts" — component-wise ancestry only.
    mapping = {"pod": "wrong"}
    assert resolve_folder_pack_name("podcasts/ep1.wav", mapping) is None


def test_folder_key_equal_to_parent_dir_matches() -> None:
    assert resolve_folder_pack_name("podcasts/ep1.wav", {"podcasts": "podcast"}) == "podcast"


# --- resolve_run_domain_pack (snapshot selection precedence) -----------------


def test_run_pack_explicit_name_wins(tmp_path: Path) -> None:
    _write_pack(tmp_path, "podcast")
    _write_pack(tmp_path, "interview")
    settings = Settings(_env_file=None, domain_packs_dir=tmp_path)
    snap = resolve_run_domain_pack(
        "podcasts/ep1.wav",
        settings=settings,
        folder_domain_packs={"podcasts": "podcast"},
        explicit_name="interview",
    )
    assert snap["name"] == "interview"


def test_run_pack_folder_mapping_used(tmp_path: Path) -> None:
    _write_pack(tmp_path, "podcast", vocabulary=["cold open"])
    settings = Settings(_env_file=None, domain_packs_dir=tmp_path)
    snap = resolve_run_domain_pack(
        "podcasts/ep1.wav",
        settings=settings,
        folder_domain_packs={"podcasts": "podcast"},
    )
    assert snap["name"] == "podcast"
    assert snap["vocabulary"] == ["cold open"]


def test_run_pack_unmapped_folder_uses_default() -> None:
    settings = Settings(_env_file=None)
    snap = resolve_run_domain_pack(
        "misc/x.wav", settings=settings, folder_domain_packs={"podcasts": "podcast"}
    )
    assert snap["name"] == "generic"


def test_run_pack_none_source_uses_default() -> None:
    settings = Settings(_env_file=None)
    snap = resolve_run_domain_pack(
        None, settings=settings, folder_domain_packs={"podcasts": "podcast"}
    )
    assert snap["name"] == "generic"


def test_run_pack_mapped_name_unresolvable_raises(tmp_path: Path) -> None:
    settings = Settings(_env_file=None, domain_packs_dir=tmp_path)
    with pytest.raises(DomainPackError):
        resolve_run_domain_pack(
            "podcasts/ep1.wav",
            settings=settings,
            folder_domain_packs={"podcasts": "ghost"},
        )


def test_run_pack_snapshot_round_trips_to_pack(tmp_path: Path) -> None:
    _write_pack(tmp_path, "podcast", vocabulary=["a"], name_seeds=["Jane"])
    settings = Settings(_env_file=None, domain_packs_dir=tmp_path)
    snap = resolve_run_domain_pack(
        "podcasts/ep1.wav",
        settings=settings,
        folder_domain_packs={"podcasts": "podcast"},
    )
    restored = DomainPack.from_mapping(snap)
    assert restored.name == "podcast"
    assert restored.name_seeds == ("Jane",)
