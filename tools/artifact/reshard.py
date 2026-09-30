"""Rewrite an artifact's file set with a different maximum file size.

  python -m tools.artifact.reshard model.ninfer out/model.ninfer --max-file-gb 50

The logical payload, objects, bindings, uses, metadata and provenance are copied unchanged; only
the `files` table, the continuation files and the artifact id are new. The result is opened
through its entry file like any converter output.
"""

from __future__ import annotations

import argparse
from bisect import bisect_right
import os
from pathlib import Path
import tempfile
from uuid import uuid4

from .file_io import Writeback
from .framing import HEADER, MAGIC, PART_MAGIC, PAYLOAD_ALIGNMENT
from .reader import Artifact
from .schema import ArtifactError
from .writer import layout_directory


def reshard(source: Path, out: Path, max_file_bytes: int) -> list[Path]:
    """Copy `source` into a new file set under `out`; return the published paths."""
    with Artifact(source) as artifact:
        description = artifact.directory.to_json()
        payload_bytes = artifact.payload_bytes
        directory, encoded, entry_start = layout_directory(
            out.name, description, payload_bytes, max_file_bytes=max_file_bytes
        )
        destinations = [out] + [out.parent / f.path for f in directory.files[1:]]
        for target in destinations:
            if target.exists():
                raise FileExistsError(target)
        artifact_id = uuid4().bytes
        prefixes = [0]
        for file in directory.files:
            prefixes.append(prefixes[-1] + file.payload_bytes)
        out.parent.mkdir(parents=True, exist_ok=True)
        writeback = Writeback()
        fds: list[int] = []
        temporary: list[Path] = []
        published: list[Path] = []
        try:
            for index, (target, file) in enumerate(zip(destinations, directory.files)):
                fd, name = tempfile.mkstemp(
                    prefix=f".{target.name}.", suffix=".tmp", dir=target.parent
                )
                fds.append(fd)
                temporary.append(Path(name))
                start = entry_start if index == 0 else PAYLOAD_ALIGNMENT
                os.ftruncate(fd, start + file.payload_bytes)
                header = HEADER.pack(
                    MAGIC if index == 0 else PART_MAGIC,
                    len(encoded) if index == 0 else index,
                    artifact_id,
                )
                os.pwrite(fd, header, 0)
                if index == 0:
                    os.pwrite(fd, encoded, HEADER.size)
            offset = 0
            for chunk in artifact.iter_range(0, payload_bytes):
                view = memoryview(chunk)
                while view:
                    index = bisect_right(prefixes, offset) - 1
                    count = min(len(view), prefixes[index + 1] - offset)
                    start = entry_start if index == 0 else PAYLOAD_ALIGNMENT
                    position = start + offset - prefixes[index]
                    part = view[:count]
                    while part:
                        written = os.pwrite(fds[index], part, position)
                        if written <= 0:
                            raise OSError(f"short write at file offset {position}")
                        writeback.written(fds[index], written)
                        part = part[written:]
                        position += written
                    view = view[count:]
                    offset += count
            if offset != payload_bytes:
                raise ArtifactError(f"copied {offset} of {payload_bytes} payload bytes")
            writeback.flush()
            while fds:
                os.close(fds.pop())
            for index in [*range(1, len(temporary)), 0]:
                os.link(temporary[index], destinations[index])
                published.append(destinations[index])
            for path in temporary:
                path.unlink()
            return destinations
        except BaseException:
            while fds:
                os.close(fds.pop())
            for path in (*temporary, *published):
                path.unlink(missing_ok=True)
            raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("source", type=Path, help="existing entry file")
    parser.add_argument("out", type=Path, help="new entry file (must not exist)")
    parser.add_argument("--max-file-gb", type=float, default=32.0,
                        help="maximum bytes per file in decimal GB (default 32)")
    args = parser.parse_args()
    if args.out.resolve() == args.source.resolve():
        raise SystemExit("out must differ from source")
    paths = reshard(args.source, args.out, int(args.max_file_gb * 1e9))
    for path in paths:
        print(f"{path}  {path.stat().st_size}")


if __name__ == "__main__":
    main()
