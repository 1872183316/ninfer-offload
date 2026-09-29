"""Safetensors checkpoint streamed shard-by-shard from a remote repository.

The converter's job order determines the order of source shards. A background fetcher downloads
shards ahead of that order within a disk budget, and a shard is deleted once no remaining job
reads it. This converts checkpoints larger than local disk without changing conversion results.
"""

from __future__ import annotations

import json
from math import prod
import os
from pathlib import Path
import re
import threading
import time
import urllib.request

import torch

from .safetensors import _DTYPES, SafetensorsSource, TensorInfo

# Source tensor names inside LogicalSource labels ("path:name", "concat(a,b)", "rows(...)").
_NAME = re.compile(r"(?:^|[:(,])((?:model|mtp|lm_head)[A-Za-z0-9_.]*)")


class StreamingSafetensorsSource(SafetensorsSource):
    """`root` holds config/index/tokenizer files and `shard_headers.json`; shards go to root."""

    def __init__(self, root: str | Path, url: str, budget_bytes: int, parallel: int = 4):
        super().__init__(root)
        headers = json.loads((self.root / "shard_headers.json").read_text())
        self._remote = {}
        for file, entry in headers.items():
            path = self.root / file
            infos = {}
            for name, item in entry["tensors"].items():
                begin, end = item["data_offsets"]
                infos[name] = TensorInfo(path, tuple(item["shape"]), item["dtype"],
                                         entry["data_offset"] + begin, end - begin)
            self._headers[path] = infos
            self._remote[path] = entry["data_offset"] + max(
                (t["data_offsets"][1] for t in entry["tensors"].values()), default=0)
        self.url = url.rstrip("/") + "/"
        self.budget = budget_bytes
        self.parallel = parallel
        self._lock = threading.Condition()
        self._ready: set[Path] = {p for p in self._remote if p.is_file()
                                  and p.stat().st_size == self._remote[p]}
        self._fetching: set[Path] = set()
        self._deleted: set[Path] = set()
        self._order: list[Path] = []      # first-use order of shards
        self._last_job: dict[Path, int] = {}
        self._job = -1
        self._error: BaseException | None = None
        self._stop = False
        self._single_pass: set[Path] = set()
        self.downloaded_bytes = 0

    # -- planning --------------------------------------------------------------------------
    def plan(self, jobs_files: list[list[Path]]) -> None:
        """jobs_files[i] lists the shards job i reads, in read order."""
        seen = set()
        for index, files in enumerate(jobs_files):
            for path in files:
                self._last_job[path] = index
                if path not in seen:
                    seen.add(path)
                    self._order.append(path)
        for _ in range(self.parallel):
            threading.Thread(target=self._fetch_loop, daemon=True).start()

    def files_of(self, labels: list[str]) -> list[Path]:
        files = []
        for label in labels:
            for name in _NAME.findall(label):
                path = self.weight_map.get(name)
                if path is not None and path not in files:
                    files.append(path)
        return files

    def begin_job(self, index: int) -> None:
        with self._lock:
            self._job = index
            for path in list(self._ready):
                if self._last_job.get(path, -1) < index:
                    self._release(path)
            self._lock.notify_all()

    # -- fetching --------------------------------------------------------------------------
    def _held_bytes(self) -> int:
        return sum(self._remote[p] for p in self._ready | self._fetching)

    def _next(self) -> Path | None:
        for path in self._order:
            if path in self._ready or path in self._fetching or path in self._deleted:
                continue
            if self._last_job.get(path, -1) < self._job:
                continue
            return path
        return None

    def _fetch_loop(self) -> None:
        while True:
            with self._lock:
                while True:
                    if self._stop or self._error:
                        return
                    path = self._next()
                    if path is None:
                        return
                    # The first needed shard may always proceed; later ones respect the budget.
                    first = all(p in self._ready or p in self._fetching or p in self._deleted
                                for p in self._order[: self._order.index(path)])
                    if first or self._held_bytes() + self._remote[path] <= self.budget:
                        self._fetching.add(path)
                        break
                    self._lock.wait(timeout=5)
            try:
                self._download(path)
            except BaseException as error:  # surface in the converter thread
                with self._lock:
                    self._error = error
                    self._lock.notify_all()
                return
            with self._lock:
                self._fetching.discard(path)
                self._ready.add(path)
                self._lock.notify_all()

    def _download(self, path: Path) -> None:
        part = path.with_suffix(path.suffix + ".part")
        size = self._remote[path]
        for attempt in range(20):
            have = part.stat().st_size if part.exists() else 0
            if have >= size:
                break
            try:
                request = urllib.request.Request(self.url + path.name,
                                                 headers={"Range": f"bytes={have}-{size - 1}"})
                with urllib.request.urlopen(request, timeout=120) as response, part.open("ab") as out:
                    while True:
                        chunk = response.read(1 << 22)
                        if not chunk:
                            break
                        out.write(chunk)
                        self.downloaded_bytes += len(chunk)
            except OSError:
                time.sleep(min(60, 5 * (attempt + 1)))
        if part.stat().st_size != size:
            raise OSError(f"{path.name}: download incomplete")
        part.replace(path)

    def _release(self, path: Path) -> None:
        if path in self._ready:
            self._ready.discard(path)
            self._deleted.add(path)
            for fd_path in [p for p in self._fds if p == path]:
                os.close(self._fds.pop(fd_path))
            path.unlink(missing_ok=True)

    # -- reads -----------------------------------------------------------------------------
    def read_flat(self, name: str, begin: int = 0, end: int | None = None):
        # Recipe preparation probes one value per input to validate readability before any job
        # runs. For a shard not yet fetched, answer that probe from the header's dtype; every
        # value that reaches an artifact is read from the complete shard during production.
        info = self.describe(name)
        count = (end if end is not None else prod(info.shape)) - begin
        if info.file not in self._ready and info.file not in self._fetching:
            dtype, word = _DTYPES[info.dtype]
            if count <= 1:
                return torch.zeros(max(count, 0), dtype=dtype)
            if count * word <= self._remote_read_limit:
                # Small tensors (norms, convolutions) of an unfetched shard: exact range read.
                offset = info.offset + begin * word
                request = urllib.request.Request(
                    self.url + info.file.name,
                    headers={"Range": f"bytes={offset}-{offset + count * word - 1}"},
                )
                for attempt in range(10):
                    try:
                        with urllib.request.urlopen(request, timeout=60) as response:
                            raw = response.read()
                        break
                    except OSError:
                        time.sleep(3 * (attempt + 1))
                if len(raw) != count * word:
                    raise OSError(f"{name}: short remote range read")
                return torch.frombuffer(bytearray(raw), dtype=dtype)
        return super().read_flat(name, begin, end)

    _remote_read_limit = 64 << 20

    def _file(self, path: Path) -> int:
        with self._lock:
            while path not in self._ready:
                if self._error:
                    raise self._error
                if path in self._deleted:
                    raise RuntimeError(f"{path.name} was released before a later read")
                self._lock.wait(timeout=5)
            # A single-pass shard earlier in this job's order is complete once a later one opens.
            position = self._order.index(path)
            for other in list(self._ready):
                if (other in self._single_pass and self._last_job.get(other, -1) == self._job
                        and self._order.index(other) < position):
                    self._release(other)
            self._lock.notify_all()
        return super()._file(path)

    def mark_single_pass(self, paths: list[Path]) -> None:
        """Shards read once, sequentially, by one job may be released as the job advances."""
        self._single_pass.update(paths)

    def close(self) -> None:
        with self._lock:
            self._stop = True
            self._lock.notify_all()
        super().close()
