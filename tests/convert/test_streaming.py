import functools
import http.server
import json
from pathlib import Path
import threading

import pytest
import torch
from safetensors.torch import save_file

from tools.convert.sources.streaming import StreamingSafetensorsSource


class _RangeHandler(http.server.SimpleHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def do_GET(self):
        path = Path(self.translate_path(self.path))
        data = path.read_bytes()
        begin, end = 0, len(data) - 1
        if "Range" in self.headers:
            spec = self.headers["Range"].removeprefix("bytes=")
            first, last = spec.split("-")
            begin, end = int(first), int(last)
        self.server.requests.append((path.name, begin, end))
        self.send_response(206 if "Range" in self.headers else 200)
        self.send_header("Content-Length", str(end - begin + 1))
        self.end_headers()
        self.wfile.write(data[begin:end + 1])


@pytest.fixture
def remote(tmp_path):
    """Three shards whose tensors are read by jobs in an order unrelated to the shard layout."""
    served = tmp_path / "remote"
    served.mkdir()
    tensors = {}
    layout = {
        "a.safetensors": ["model.l0.w", "model.l3.w", "model.table.0"],
        "b.safetensors": ["model.l1.w", "model.table.1", "model.l2.norm"],
        "c.safetensors": ["model.l2.w", "model.table.2", "model.l0.norm"],
    }
    generator = torch.Generator().manual_seed(0)
    for file, names in layout.items():
        shard = {n: torch.randn(256, 64, generator=generator).to(torch.bfloat16) for n in names}
        save_file(shard, served / file)
        tensors.update(shard)
    headers = {}
    weight_map = {}
    for file, names in layout.items():
        raw = (served / file).read_bytes()
        length = int.from_bytes(raw[:8], "little")
        header = json.loads(raw[8:8 + length])
        header.pop("__metadata__", None)
        headers[file] = {"data_offset": 8 + length, "tensors": header}
        weight_map.update({n: file for n in names})
    local = tmp_path / "local"
    local.mkdir()
    (local / "shard_headers.json").write_text(json.dumps(headers))
    (local / "model.safetensors.index.json").write_text(json.dumps({"weight_map": weight_map}))
    server = http.server.ThreadingHTTPServer(
        ("127.0.0.1", 0), functools.partial(_RangeHandler, directory=str(served)))
    server.requests = []
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield local, f"http://127.0.0.1:{server.server_port}/", tensors, server
    server.shutdown()


JOBS = [
    ["model.l0.w", "model.l0.norm"],
    ["model.l1.w"],
    ["model.l2.w", "model.l2.norm"],
    ["model.l3.w", "model.l0.norm"],
    ["model.table.0", "model.table.1", "model.table.2"],
]
TENSOR = 256 * 64 * 2


def _run(source, jobs, tensors, reads=None):
    source.plan(jobs)
    for index, names in enumerate(jobs):
        source.begin_job(index)
        for name in reads[index] if reads else names:
            # Read in two halves as the converter reads row chunks.
            half = 256 * 32
            values = torch.cat([source.read_flat(name, 0, half), source.read_flat(name, half)])
            assert torch.equal(values, tensors[name].reshape(-1)), name
            held = sum(p.stat().st_size for p in source.cache.iterdir())
            source.observed_peak = max(getattr(source, "observed_peak", 0), held)


def test_streaming_downloads_only_job_ranges_within_budget(remote):
    local, url, tensors, server = remote
    budget = 3 * TENSOR
    with StreamingSafetensorsSource(local, url, budget, parallel=3) as source:
        _run(source, JOBS, tensors)
        unique = {n for names in JOBS for n in names}
        assert source.downloaded_bytes == len(unique) * TENSOR
        assert source.refetched_bytes == 0
        assert source.peak_bytes <= budget
        assert source.observed_peak <= budget
    # Ranges, not whole shards: no request covers a full shard.
    assert all(name.endswith(".safetensors") for name, _, _ in server.requests)
    assert all(end - begin + 1 <= 2 * TENSOR for _, begin, end in server.requests)
    # A successful run leaves no cached ranges behind.
    assert not source.cache.exists()


def test_large_job_releases_ranges_as_it_advances(remote):
    local, url, tensors, _ = remote
    # The table job (3 tensors) exceeds a quarter of the budget and is read sequentially.
    budget = 2 * TENSOR
    with StreamingSafetensorsSource(local, url, budget, parallel=2) as source:
        _run(source, JOBS, tensors)
        assert source.refetched_bytes == 0
        assert source.observed_peak <= budget + TENSOR  # a waited range may exceed the budget


def test_out_of_order_read_in_large_job_refetches_correct_data(remote):
    local, url, tensors, _ = remote
    budget = 2 * TENSOR
    reads = [*JOBS[:4], ["model.table.0", "model.table.1", "model.table.0", "model.table.2"]]
    with StreamingSafetensorsSource(local, url, budget, parallel=2) as source:
        _run(source, JOBS, tensors, reads)
        assert source.refetched_bytes == TENSOR


def test_unplanned_read_is_fetched_on_demand(remote):
    local, url, tensors, _ = remote
    with StreamingSafetensorsSource(local, url, 4 * TENSOR, parallel=2) as source:
        jobs = [["model.l0.w"], ["model.l1.w"]]
        reads = [["model.l0.w", "model.l3.w"], ["model.l1.w"]]
        _run(source, jobs, tensors, reads)


def test_preparation_reads_use_direct_ranges(remote):
    local, url, tensors, _ = remote
    with StreamingSafetensorsSource(local, url, TENSOR, parallel=1) as source:
        assert source.read_flat("model.l2.norm", 0, 1).numel() == 1
        assert torch.equal(source.read_flat("model.l2.norm"), tensors["model.l2.norm"].reshape(-1))
        assert not source.cache.exists()
