from voxint.domain_packs.base import load_default


def test_generic_pack_loads() -> None:
    pack = load_default()
    assert pack.name == "generic"
    assert pack.vocabulary == ()
    assert "enhancement_context" in pack.prompt_fragments
