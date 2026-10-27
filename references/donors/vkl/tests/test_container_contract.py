import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_rocm_container_contract_is_present():
    dockerfile = (ROOT / "Dockerfile").read_text()
    compose = (ROOT / "compose.yaml").read_text()
    makefile = (ROOT / "Makefile").read_text()

    assert "rocm/pytorch:rocm7.2.2_ubuntu24.04_py3.12_pytorch_release_2.10.0" in dockerfile
    for service in ("test:", "gpu-smoke:", "bench:", "shell:"):
        assert service in compose
    assert "extends:" not in compose
    for gpu_option in (
        "privileged: true",
        "/dev/kfd",
        "/dev/dri",
        "keep-groups",
        "label=disable",
        "seccomp=unconfined",
        "ipc: host",
    ):
        assert gpu_option in compose
    for target in ("image:", "test:", "gpu-smoke:", "bench:", "shell:"):
        assert target in makefile
    assert "podman" in makefile


def test_test_service_uses_the_rootless_podman_gpu_contract():
    compose = (ROOT / "compose.yaml").read_text()
    test_service = compose.split("  gpu-smoke:", maxsplit=1)[0]

    for required in (
        "privileged: true",
        "ipc: host",
        "/dev/kfd",
        "/dev/dri",
        "keep-groups",
        "label=disable",
        "seccomp=unconfined",
    ):
        assert required in test_service


def test_bench_service_mounts_the_configurable_public_dataset_root_read_only():
    compose = (ROOT / "compose.yaml").read_text()
    bench_service = compose.split("  bench:", maxsplit=1)[1].split("  shell:", maxsplit=1)[0]

    assert "${VKL_DATASETS_ROOT:-/mnt/cps_persistent1_shared/datasets/public}:/datasets:ro" in bench_service


def test_container_rule_and_ghcr_publisher_are_documented():
    readme = (ROOT / "README.md").read_text().lower()
    publisher = (ROOT / ".github" / "workflows" / "publish-image.yml").read_text()

    assert "mandatory" in readme
    assert "make gpu-smoke" in readme
    assert "ghcr.io/bjoernellens1/vkl" in publisher
    assert "packages: write" in publisher
    assert "sha-" in publisher
    assert "latest" in publisher
    assert "target: ci-test" in publisher
    assert "load: false" in publisher


def test_ci_runs_container_contract_on_push_and_pull_request_without_publishing():
    ci = (ROOT / ".github" / "workflows" / "ci.yml").read_text()
    dockerfile = (ROOT / "Dockerfile").read_text()
    trigger_block = ci.split("permissions:", maxsplit=1)[0]

    assert "pull_request" in trigger_block
    assert "push" in trigger_block
    assert "target: ci-test" in ci
    assert "push: false" in ci
    assert re.search(r"FROM python:\d+\.\d+-slim AS ci-test", dockerfile)
    assert "torch==2.10.0" in dockerfile


def test_cuda_dockerfile_installs_pinned_triton_on_pytorch_cuda_base():
    dockerfile = (ROOT / "Dockerfile.cuda").read_text()

    assert re.search(r"^FROM (docker\.io/)?pytorch/pytorch:[^\s]+cuda[^\s]+", dockerfile, re.MULTILINE)
    assert "triton==3.6.0" in dockerfile
    assert "apt-get update && apt-get install --no-install-recommends -y python3.12-venv" in dockerfile
    assert "build-essential" in dockerfile
    assert "python -m venv --system-site-packages /opt/venv" in dockerfile
    assert 'ENV PATH="/opt/venv/bin:$PATH"' in dockerfile
    assert "pip install --no-cache-dir -e '.[dev]'" in dockerfile


def test_cuda_compose_requests_all_nvidia_gpus_without_rocm_devices():
    compose = (ROOT / "compose.cuda.yaml").read_text()

    assert "test-cuda:" in compose
    assert "bench-cuda:" in compose
    assert "gpus: all" in compose
    assert "/dev/kfd" not in compose
    assert "/dev/dri" not in compose


def test_cuda_compose_exposes_distinct_source_build_and_immutable_image_metadata():
    compose = (ROOT / "compose.cuda.yaml").read_text()

    for metadata in (
        "VKL_SOURCE_REVISION: ${VKL_SOURCE_REVISION:-unknown}",
        "VKL_BUILD_IMAGE_REVISION: ${VKL_BUILD_IMAGE_REVISION:-unknown}",
        "VKL_IMAGE_REFERENCE: ${VKL_CUDA_IMAGE:-unknown}",
        "VKL_IMAGE_DIGEST: ${VKL_IMAGE_DIGEST:-unavailable}",
    ):
        assert compose.count(metadata) == 2


def test_make_targets_read_runtime_image_revision_from_oci_label():
    makefile = (ROOT / "Makefile").read_text()

    label = 'org.opencontainers.image.revision'
    assert f'IMAGE_REVISION = $(shell $(PODMAN) image inspect --format \'{{{{ index .Config.Labels "{label}" }}}}\'' in makefile
    assert f'CUDA_IMAGE_REVISION = $(shell $(DOCKER) image inspect --format \'{{{{ index .Config.Labels "{label}" }}}}\'' in makefile
    assert "VKL_IMAGE_REVISION=$(IMAGE_REVISION)" in makefile
    assert "VKL_BUILD_IMAGE_REVISION=$(CUDA_IMAGE_REVISION)" in makefile
    assert "VKL_IMAGE_REVISION=$(CUDA_IMAGE_REVISION)" in makefile
    assert "VKL_BUILD_IMAGE_REVISION=$(VKL_SOURCE_REVISION)" not in makefile


def test_cuda_make_targets_resolve_image_digest_and_pass_provenance():
    makefile = (ROOT / "Makefile").read_text()

    assert "CUDA_IMAGE_DIGEST = $(shell $(DOCKER) image inspect" in makefile
    assert "VKL_SOURCE_REVISION=$(VKL_SOURCE_REVISION)" in makefile
    assert "VKL_IMAGE_DIGEST=$(CUDA_IMAGE_DIGEST)" in makefile


def test_cuda_make_targets_use_docker_compose():
    makefile = (ROOT / "Makefile").read_text()

    for target in ("image-cuda:", "test-cuda:", "bench-cuda:"):
        assert target in makefile
    assert "DOCKER_COMPOSE ?= $(DOCKER) compose" in makefile
