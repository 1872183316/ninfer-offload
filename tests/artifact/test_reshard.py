from __future__ import annotations

import pytest

from tools.artifact.framing import HEADER, PART_MAGIC
from tools.artifact.reader import Artifact
from tools.artifact.reshard import reshard
from tools.artifact.schema import ResourceSpec
from tools.artifact.writer import ArtifactWriter


def _without_files(directory: dict) -> dict:
    return {key: value for key, value in directory.items() if key != "files"}


def test_reshard_preserves_payload_and_directory(tmp_path):
    source = tmp_path / "one.ninfer"
    payload = bytes(range(251)) * 110
    with ArtifactWriter(
        source,
        [ResourceSpec("data", len(payload))],
        components={"text": {"config": {}, "resources": {"data": "data"}}},
        bindings={},
        metadata={"name": "m"},
    ) as writer:
        writer.write_object("data", payload)
    with Artifact(source) as artifact:
        original = artifact.directory.to_json()
        assert len(artifact.directory.files) == 1

    split = tmp_path / "split" / "model.ninfer"
    paths = reshard(source, split, 12288)
    assert [p.name for p in paths] == [
        "model.ninfer",
        "model.ninfer.part-0001",
        "model.ninfer.part-0002",
        "model.ninfer.part-0003",
    ]
    assert all(p.stat().st_size <= 12288 for p in paths)
    with Artifact(split) as artifact:
        assert _without_files(artifact.directory.to_json()) == _without_files(original)
        assert artifact.read_object("data") == payload
        identity = artifact.artifact_id
    for index, part in enumerate(paths[1:], 1):
        assert HEADER.unpack(part.read_bytes()[:32]) == (PART_MAGIC, index, identity)

    joined = tmp_path / "joined.ninfer"
    reshard(split, joined, 32_000_000_000)
    with Artifact(joined) as artifact:
        assert artifact.directory.to_json() == original
        assert artifact.read_object("data") == payload
        assert artifact.artifact_id != identity


def test_reshard_keeps_existing_outputs(tmp_path):
    source = tmp_path / "one.ninfer"
    with ArtifactWriter(
        source,
        [ResourceSpec("data", 20000)],
        components={"text": {"config": {}, "resources": {"data": "data"}}},
        bindings={},
    ) as writer:
        writer.write_object("data", bytes(20000))
    out = tmp_path / "model.ninfer"
    existing = tmp_path / "model.ninfer.part-0002"
    existing.write_bytes(b"keep")
    with pytest.raises(FileExistsError):
        reshard(source, out, 12288)
    assert existing.read_bytes() == b"keep"
    assert not out.exists() and not (tmp_path / "model.ninfer.part-0001").exists()
    assert not list(tmp_path.glob(".*.tmp"))
