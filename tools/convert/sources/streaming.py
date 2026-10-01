"""Safetensors checkpoint streamed from a remote repository in job order.

The converter's jobs determine which source tensors are read and in what order. A background
fetcher downloads exactly those byte ranges (adjacent tensors of one job in one shard share a
request) ahead of the converter, within a disk budget, and deletes each range once no remaining job
reads it. Local disk use is therefore bounded by the budget regardless of how the checkpoint lays
tensors out across shards, and conversion results are unchanged.
"""

from __future__ import annotations

from dataclasses import dataclass
import http.client
import json
from math import prod
import os
from pathlib import Path
import re
import shutil
import sys
import threading
import time
import urllib.request

import torch

from ..download import USER_AGENT
from .safetensors import _DTYPES, SafetensorsSource, TensorInfo

# Network failures worth retrying: socket errors and truncated HTTP bodies.
_TRANSIENT = (OSError, http.client.HTTPException)

# Source tensor names inside LogicalSource labels ("path:name", "concat(a,b)", "rows(...)").
_NAME = re.compile(r"(?:^|[:(,])((?:model|mtp|lm_head)[A-Za-z0-9_.]*)")

# Tensors of one job in one shard separated by at most this many bytes share a download.
_COALESCE_GAP = 1 << 20


@dataclass(eq=False)
class _Segment:
    shard: Path
    begin: int  # absolute byte range in the shard
    end: int
    job: int  # first job that reads it
    last_job: int
    order: int  # position in the fetch order
    state: str = "pending"  # pending | fetching | ready | released

    @property
    def size(self) -> int:
        return self.end - self.begin


class StreamingSafetensorsSource(SafetensorsSource):
    """`root` holds config/index/tokenizer files and `shard_headers.json`.

    Downloaded ranges live in `root/.stream/`. A shard already present in `root` with its full
    size is read locally.
    """

    def __init__(self, root: str | Path, url: str, budget_bytes: int, parallel: int = 4):
        super().__init__(root)
        headers = json.loads((self.root / "shard_headers.json").read_text())
        self._remote: dict[Path, int] = {}
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
        self._local = {p for p in self._remote
                       if p.is_file() and p.stat().st_size == self._remote[p]}
        self.url = url.rstrip("/") + "/"
        self.budget = budget_bytes
        self.parallel = parallel
        self.cache = self.root / ".stream"
        self._lock = threading.Condition()
        self._segments: list[_Segment] = []
        self._of: dict[str, _Segment] = {}  # tensor name -> segment holding it
        self._large_jobs: set[int] = set()  # jobs whose ranges are released as they advance
        self._job = -1
        self._error: BaseException | None = None
        self._stop = False
        self._waiting: set[_Segment] = set()
        self._fds_seg: dict[_Segment, int] = {}
        self.downloaded_bytes = 0
        self.refetched_bytes = 0
        self.peak_bytes = 0

    # -- planning --------------------------------------------------------------------------
    def tensors_of(self, labels: list[str]) -> list[str]:
        """Source tensor names read through `labels`, in read order."""
        names: list[str] = []
        for label in labels:
            for name in _NAME.findall(label):
                if name in self.weight_map and name not in names:
                    names.append(name)
        return names

    def plan(self, jobs_tensors: list[list[str]]) -> None:
        """jobs_tensors[i] lists the source tensors job i reads, in read order."""
        uses: dict[str, list[int]] = {}
        for index, names in enumerate(jobs_tensors):
            for name in names:
                uses.setdefault(name, []).append(index)
        for index, names in enumerate(jobs_tensors):
            job_bytes = 0
            current: _Segment | None = None
            for name in names:
                info = self.describe(name)
                if info.file in self._local or uses[name][0] != index:
                    continue
                last = uses[name][-1]
                job_bytes += info.bytes
                if (current is not None and current.shard == info.file
                        and current.last_job == last
                        and 0 <= info.offset - current.end <= _COALESCE_GAP):
                    current.end = info.offset + info.bytes
                else:
                    current = _Segment(info.file, info.offset, info.offset + info.bytes,
                                       index, last, len(self._segments))
                    self._segments.append(current)
                self._of[name] = current
            if job_bytes > self.budget // 4:
                self._large_jobs.add(index)
        self.cache.mkdir(exist_ok=True)
        for segment in self._segments:
            if self._path(segment).is_file():
                segment.state = "ready"
        for _ in range(self.parallel):
            threading.Thread(target=self._fetch_loop, daemon=True).start()

    def begin_job(self, index: int) -> None:
        with self._lock:
            self._job = index
            for segment in self._segments:
                if segment.state == "ready" and segment.last_job < index:
                    self._release(segment)
            self._lock.notify_all()

    # -- fetching --------------------------------------------------------------------------
    def _path(self, segment: _Segment) -> Path:
        return self.cache / f"{segment.shard.name}.{segment.begin}-{segment.end}"

    def _held_bytes(self) -> int:
        return sum(s.size for s in self._segments if s.state in ("fetching", "ready"))

    def _next(self) -> _Segment | None:
        for segment in self._waiting:
            if segment.state in ("pending", "released"):
                return segment
        for segment in self._segments:
            if segment.state == "pending" and segment.last_job >= self._job:
                return segment
        return None

    def _fetch_loop(self) -> None:
        while True:
            with self._lock:
                while True:
                    if self._stop or self._error:
                        return
                    segment = self._next()
                    # Only a range the converter is blocked on may exceed the budget.
                    if segment is not None and (
                            segment in self._waiting
                            or self._held_bytes() + segment.size <= self.budget):
                        if segment.state == "released":
                            self.refetched_bytes += segment.size
                        segment.state = "fetching"
                        self.peak_bytes = max(self.peak_bytes, self._held_bytes())
                        break
                    self._lock.wait(timeout=5)
            try:
                self._download(segment)
            except BaseException as error:  # surface in the converter thread
                with self._lock:
                    self._error = error
                    self._lock.notify_all()
                return
            with self._lock:
                segment.state = "ready"
                self._lock.notify_all()

    def _download(self, segment: _Segment) -> None:
        path = self._path(segment)
        part = path.with_name(path.name + ".part")
        for attempt in range(20):
            have = part.stat().st_size if part.exists() else 0
            if have >= segment.size:
                break
            try:
                request = urllib.request.Request(
                    self.url + segment.shard.name,
                    headers={**USER_AGENT,
                             "Range": f"bytes={segment.begin + have}-{segment.end - 1}"})
                with urllib.request.urlopen(request, timeout=120) as response, \
                        part.open("ab") as out:
                    if response.status != 206:
                        raise RuntimeError(f"{segment.shard.name}: server ignored the byte range")
                    while True:
                        chunk = response.read(1 << 22)
                        if not chunk:
                            break
                        out.write(chunk)
                        self.downloaded_bytes += len(chunk)
            except _TRANSIENT:
                time.sleep(min(60, 5 * (attempt + 1)))
        if not part.exists() or part.stat().st_size != segment.size:
            raise OSError(f"{segment.shard.name}[{segment.begin}:{segment.end}]: "
                          "download incomplete")
        part.replace(path)

    def _release(self, segment: _Segment) -> None:
        segment.state = "released"
        fd = self._fds_seg.pop(segment, None)
        if fd is not None:
            os.close(fd)
        self._path(segment).unlink(missing_ok=True)

    def _acquire(self, name: str, info: TensorInfo) -> _Segment:
        """Wait until the range holding `name` is on disk; called with the lock held."""
        segment = self._of.get(name)
        if segment is None or (segment.state == "released" and segment.last_job < self._job):
            # Unplanned read: fetch exactly this tensor for the current job.
            segment = _Segment(info.file, info.offset, info.offset + info.bytes,
                               self._job, self._job, len(self._segments))
            self._segments.append(segment)
            self._of[name] = segment
        elif segment.state == "released":
            # A large job read a range again after a later one; hold its ranges from now on.
            self._large_jobs.discard(self._job)
        self._waiting.add(segment)
        self._lock.notify_all()
        try:
            while segment.state != "ready":
                if self._error:
                    raise self._error
                self._lock.wait(timeout=5)
        finally:
            self._waiting.discard(segment)
        if self._job in self._large_jobs:
            # Ranges of a large job are read sequentially; earlier ones are complete.
            for other in self._segments:
                if (other.state == "ready" and other.job == self._job
                        and other.last_job == self._job and other.order < segment.order):
                    self._release(other)
            self._lock.notify_all()
        return segment

    # -- reads -----------------------------------------------------------------------------
    def read_flat(self, name: str, begin: int = 0, end: int | None = None):
        info = self.describe(name)
        if info.file in self._local:
            return super().read_flat(name, begin, end)
        elements = prod(info.shape)
        end = elements if end is None else end
        if not 0 <= begin <= end <= elements:
            raise ValueError(f"{name}: source element range [{begin},{end}) exceeds {info.shape}")
        dtype, word = _DTYPES[info.dtype]
        count = end - begin
        if count == 0:
            return torch.empty(0, dtype=dtype)
        if self._job < 0:
            # Recipe preparation probes one value per input to validate readability before any
            # job runs; answer it from the header's dtype. Other reads before the first job are
            # served exactly by a range request without touching the disk. Every value that
            # reaches an artifact is read through planned ranges during production.
            if count <= 1:
                return torch.zeros(count, dtype=dtype)
            offset = info.offset + begin * word
            return torch.frombuffer(bytearray(self._range(info.file, offset, count * word)),
                                    dtype=dtype)
        # Only the converter thread reads and releases ranges, so the file stays valid after
        # the lock is dropped.
        with self._lock:
            segment = self._acquire(name, info)
            fd = self._fds_seg.get(segment)
            if fd is None:
                fd = os.open(self._path(segment), os.O_RDONLY)
                self._fds_seg[segment] = fd
        raw = os.pread(fd, count * word, info.offset - segment.begin + begin * word)
        if len(raw) != count * word:
            raise ValueError(f"{name}: short source read")
        self.bytes_read += len(raw)
        return torch.frombuffer(bytearray(raw), dtype=dtype)

    def _range(self, shard: Path, offset: int, size: int) -> bytes:
        request = urllib.request.Request(
            self.url + shard.name,
            headers={**USER_AGENT, "Range": f"bytes={offset}-{offset + size - 1}"})
        for attempt in range(10):
            try:
                with urllib.request.urlopen(request, timeout=60) as response:
                    raw = response.read()
                if len(raw) == size:
                    return raw
            except _TRANSIENT:
                pass
            time.sleep(3 * (attempt + 1))
        raise OSError(f"{shard.name}: short remote range read")

    def close(self) -> None:
        with self._lock:
            self._stop = True
            self._lock.notify_all()
            for segment in list(self._fds_seg):
                os.close(self._fds_seg.pop(segment))
        if self._segments:
            print(f"streaming: downloaded {self.downloaded_bytes / 1e9:.1f} GB "
                  f"(refetched {self.refetched_bytes / 1e9:.1f} GB), "
                  f"peak range cache {self.peak_bytes / 1e9:.1f} GB", file=sys.stderr, flush=True)
        super().close()

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()
        if exc_type is None and self.cache.is_dir():
            # Keep downloaded ranges after a failure so that a rerun reuses them.
            shutil.rmtree(self.cache)
