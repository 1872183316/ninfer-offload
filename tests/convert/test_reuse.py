"""tools.convert --reuse: identical stored objects are copied, changed ones are converted."""

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys

from tools.artifact.reader import Artifact
from tools.artifact.schema import binding_parts

ROOT = Path(__file__).resolve().parents[2]
QUERY = "text/layers/0/attention/query"


def checkpoint(tmp_path: Path) -> Path:
    source = tmp_path / "source"
    source.mkdir()
    config = {
        "architectures": ["Qwen3_5ForCausalLM"], "hidden_size": 128, "vocab_size": 8,
        "num_hidden_layers": 1, "max_position_embeddings": 128,
        "layer_types": ["full_attention"], "num_attention_heads": 2, "num_key_value_heads": 1,
        "head_dim": 8, "intermediate_size": 24,
        "rope_parameters": {"partial_rotary_factor": 0.5, "mrope_section": [1, 1, 0]},
    }
    (source / "config.json").write_text(json.dumps(config))
    for name, value in {"tokenizer.json": {"model": {"vocab": {str(i): i for i in range(6)}}},
                        "tokenizer_config.json": {}, "generation_config.json": {}}.items():
        (source / name).write_text(json.dumps(value))
    (source / "chat_template.jinja").write_text("{{ messages }}")
    (tmp_path / "recipe.py").write_text("""import os
import torch
from tools.convert.sources.logical import array_source

def configure(model, recipe, sources):
    value = float(os.environ["VALUE"])
    for name, parameter in model.parameters.items():
        recipe.assign(name, source=array_source(
            torch.full(parameter.shape, value, dtype=torch.bfloat16), "test"))
    recipe.assign("text/layers/0/attention/query", format=os.environ["QUERY_FORMAT"],
                  method="grouped_absmax")
""")
    return source


def run(tmp_path, out, value, query_format, *extra):
    command = [sys.executable, "-B", "-m", "tools.convert", "--model", str(tmp_path / "source"),
               "--recipe", str(tmp_path / "recipe.py"), "--out", str(out), "--device", "cpu",
               *extra]
    env = {**os.environ, "PYTHONDONTWRITEBYTECODE": "1", "VALUE": value,
           "QUERY_FORMAT": query_format}
    return subprocess.run(command, cwd=ROOT, capture_output=True, text=True, env=env)


def objects(path: Path) -> dict[str, bytes]:
    with Artifact(path) as artifact:
        result = {}
        for name, binding in artifact.directory.bindings.items():
            (object_id, _, _), = binding_parts(binding, artifact.by_id)[:1]
            result[name] = artifact.read_object(object_id)
        return result


def test_reuse_copies_identical_objects_and_converts_changed_ones(tmp_path):
    checkpoint(tmp_path)
    first = tmp_path / "first.ninfer"
    result = run(tmp_path, first, "1.0", "q8_g32_fp16")
    assert result.returncode == 0, result.stderr
    # Different source values: copied objects keep the first conversion's bytes.
    second = tmp_path / "second.ninfer"
    result = run(tmp_path, second, "2.0", "q4_g64_fp16", "--reuse", str(first))
    assert result.returncode == 0, result.stderr
    report = json.loads(Path(str(second) + ".conversion.json").read_text())
    old, new = objects(first), objects(second)
    fresh = objects(tmp_path / "fresh.ninfer") if run(
        tmp_path, tmp_path / "fresh.ninfer", "2.0", "q4_g64_fp16").returncode == 0 else {}
    assert new[QUERY] == fresh[QUERY] != old[QUERY]
    copied = [name for name in old if name != QUERY]
    assert copied and all(new[name] == old[name] for name in copied)
    assert report["reused_objects"] >= 1


def test_reuse_rejects_a_different_recipe(tmp_path):
    checkpoint(tmp_path)
    first = tmp_path / "first.ninfer"
    assert run(tmp_path, first, "1.0", "q8_g32_fp16").returncode == 0
    (tmp_path / "other_recipe.py").write_text((tmp_path / "recipe.py").read_text())
    result = subprocess.run(
        [sys.executable, "-B", "-m", "tools.convert", "--model", str(tmp_path / "source"),
         "--recipe", str(tmp_path / "other_recipe.py"), "--out", str(tmp_path / "x.ninfer"),
         "--device", "cpu", "--reuse", str(first)],
        cwd=ROOT, capture_output=True, text=True,
        env={**os.environ, "VALUE": "1.0", "QUERY_FORMAT": "q8_g32_fp16"})
    assert result.returncode != 0 and "must match" in result.stderr
