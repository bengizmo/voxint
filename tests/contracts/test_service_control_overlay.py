"""The opt-in Docker socket is scoped to the API in every compute tier (#556)."""

import pytest
import yaml

from tests.contracts.conftest import REPO_ROOT


def test_service_control_overlay_shape() -> None:
    overlay = yaml.safe_load((REPO_ROOT / "compose.service-controls.yaml").read_text())
    assert set(overlay["services"]) == {"api"}
    api = overlay["services"]["api"]
    assert api["volumes"] == ["/var/run/docker.sock:/var/run/docker.sock"]
    assert api["group_add"] == ["${DOCKER_GID:-999}"]
    assert api["environment"]["VOXINT_SERVICE_CONTROL"] == "docker"


@pytest.mark.parametrize("tier", ["cpu", "gpu", "rocm", "metal"])
def test_socket_is_api_only_across_tiers(tier: str) -> None:
    # YAML-level contract: check all layers, including inherited YAML anchors.
    # Compose concatenates volume lists and merges environment mappings; this
    # overlay adds only API fields and cannot remove other services' mounts.
    layers = [
        yaml.safe_load((REPO_ROOT / filename).read_text())["services"]
        for filename in ("compose.yaml", f"compose.{tier}.yaml", "compose.service-controls.yaml")
    ]
    assert {"api", "worker", "beat"} <= layers[0].keys()
    if tier != "metal":
        assert {"whisper", "pyannote", "titanet"} <= layers[1].keys()
    api_environment = {}
    api_volumes = []
    for services in layers:
        for name, service in services.items():
            volumes = service.get("volumes", [])
            if name == "api":
                api_environment.update(service.get("environment", {}))
                api_volumes.extend(volumes)
            else:
                assert all("docker.sock" not in str(volume) for volume in volumes), name
                assert "VOXINT_SERVICE_CONTROL" not in service.get("environment", {}), name
    assert api_environment["VOXINT_SERVICE_CONTROL"] == "docker"
    assert api_environment.get("COMPUTE_TIER", "gpu") == tier
    assert api_volumes.count("/var/run/docker.sock:/var/run/docker.sock") == 1
    assert len(api_volumes) > 1  # Existing API storage survives the added socket.
