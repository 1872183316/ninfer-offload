"""tools.run: model resolution without network and GPU placement planning."""

import argparse
from pathlib import Path
from types import SimpleNamespace

import pytest

from tools import run

GIB = run.GIB


def facts(device, experts_bytes, experts, qwen4exp=False, name="m", mtp=False, mtp_device=0,
          mtp_host=0):
    return SimpleNamespace(device_bytes=device, expert_bytes=experts_bytes, host_bytes=0,
                           experts=experts, qwen4exp=qwen4exp, name=name, kv=lambda ctx: 0,
                           mtp=mtp, mtp_device_bytes=mtp_device, mtp_host_bytes=mtp_host,
                           mtp_kv=lambda ctx: 0)


def args(**overrides):
    values = dict(device=0, max_context=4096, offload=None, gpu_experts=None, threads=None,
                  mtp=None, draft_tokens=run.DEFAULT_DRAFT_TOKENS)
    values.update(overrides)
    return argparse.Namespace(**values)


@pytest.fixture
def machine(monkeypatch):
    monkeypatch.setattr(run, "gpu_memory", lambda device: (16 * GIB, 16 * GIB, "GPU"))
    monkeypatch.setattr(run, "available_memory", lambda: 512 * GIB)
    monkeypatch.setattr(run, "physical_cores", lambda: 12)


def value(options, flag):
    return options[options.index(flag) + 1]


def test_moe_that_fits_runs_without_offload(machine):
    options = run.plan(facts(4 * GIB, 6 * GIB, 128), args())
    assert "--moe-offload" not in options
    assert value(options, "--max-context") == "4096"


def test_moe_that_does_not_fit_offloads_as_many_experts_as_fit(machine):
    options = run.plan(facts(4 * GIB, 64 * GIB, 512), args())
    fixed = 4 * GIB + run.RUNTIME + run.HEADROOM
    assert value(options, "--moe-gpu-experts") == str(int((16 * GIB - fixed) / (64 * GIB / 512)))
    assert value(options, "--moe-threads") == "12"


def test_flash_next_always_offloads(machine):
    options = run.plan(facts(4 * GIB, 1 * GIB, 512, qwen4exp=True), args())
    assert "--moe-offload" in options
    assert value(options, "--moe-gpu-experts") == "512"


def test_explicit_choices_win(machine):
    options = run.plan(facts(4 * GIB, 64 * GIB, 512), args(gpu_experts=7, threads=3))
    assert value(options, "--moe-gpu-experts") == "7"
    assert value(options, "--moe-threads") == "3"


def test_dense_model_that_does_not_fit_is_rejected(machine):
    with pytest.raises(SystemExit, match="dense models cannot offload"):
        run.plan(facts(20 * GIB, 0, 0), args())


def test_offload_that_cannot_fit_dense_weights_is_rejected(machine):
    with pytest.raises(SystemExit, match="every expert on the CPU"):
        run.plan(facts(15 * GIB, 64 * GIB, 512), args())


def test_mtp_is_enabled_when_the_model_has_it_and_its_weights_are_planned(machine):
    plain = run.plan(facts(4 * GIB, 64 * GIB, 512, qwen4exp=True), args())
    assert "--spec" not in plain
    options = run.plan(facts(4 * GIB, 64 * GIB, 512, qwen4exp=True, mtp=True, mtp_device=GIB,
                             mtp_host=GIB), args())
    assert value(options, "--spec") == "mtp"
    assert value(options, "--draft-tokens") == str(run.DEFAULT_DRAFT_TOKENS)
    # The MTP device weights take room from the GPU experts.
    assert int(value(options, "--moe-gpu-experts")) < int(value(plain, "--moe-gpu-experts"))


def test_mtp_can_be_disabled_or_tuned(machine):
    model = facts(4 * GIB, 64 * GIB, 512, qwen4exp=True, mtp=True)
    assert "--spec" not in run.plan(model, args(mtp=False))
    assert value(run.plan(model, args(draft_tokens=3)), "--draft-tokens") == "3"
    with pytest.raises(SystemExit, match="no MTP component"):
        run.plan(facts(4 * GIB, 64 * GIB, 512), args(mtp=True))


def test_local_models_resolve_without_network(tmp_path, monkeypatch):
    def offline(*_):
        raise AssertionError("network used")

    monkeypatch.setattr(run, "list_files", offline)
    model = tmp_path / "x.ninfer"
    model.write_bytes(b"")
    assert run.resolve(str(model), tmp_path, 20) == model
    cached = tmp_path / "models" / "Qwen3.8-Flash-Next"
    cached.mkdir(parents=True)
    (cached / "qwen3_8_flash_next.ninfer").write_bytes(b"")
    (cached / "qwen3_8_flash_next.ninfer.part-0001").write_bytes(b"")
    assert run.resolve("Qwen/Qwen3.8-Flash-Next", tmp_path / "models", 20) == \
        cached / "qwen3_8_flash_next.ninfer"


def test_prebuilt_prefers_huggingface_only_when_it_answers(monkeypatch):
    sources = (("huggingface", "a/hf"), ("modelscope", "b/ms"))
    asked = []

    def files(hub, repo):
        asked.append(hub)
        return {"m.ninfer": 1}

    monkeypatch.setattr(run, "list_files", files)
    monkeypatch.delenv("HF_ENDPOINT", raising=False)
    monkeypatch.setattr(run, "reachable", lambda endpoint: False)
    assert run.prebuilt_files(sources)[:2] == ("modelscope", "b/ms")
    monkeypatch.setattr(run, "reachable", lambda endpoint: True)
    assert run.prebuilt_files(sources)[:2] == ("huggingface", "a/hf")
    assert asked == ["modelscope", "huggingface"]


def test_prebuilt_falls_back_when_a_source_fails(monkeypatch):
    def files(hub, repo):
        if hub == "huggingface":
            raise OSError("connection reset")
        return {"m.ninfer": 1, "m.ninfer.part-0001": 2, "README.md": 3}

    monkeypatch.setattr(run, "list_files", files)
    monkeypatch.setattr(run, "reachable", lambda endpoint: True)
    hub, repo, found = run.prebuilt_files((("huggingface", "a/hf"), ("modelscope", "b/ms")))
    assert (hub, repo) == ("modelscope", "b/ms")
    assert set(found) == {"m.ninfer", "m.ninfer.part-0001"}


def test_only_ninfer_files_are_fetched():
    files = {"README.md": 1, "a.ninfer": 5, "a.ninfer.part-0001": 5, "SHA256SUMS": 1,
             "model-00001.safetensors": 9}
    assert run.ninfer_files(files) == {"a.ninfer": 5, "a.ninfer.part-0001": 5, "SHA256SUMS": 1}
