"""Download a checkpoint's index for conversion, and optionally all of its weights.

Always downloads the small repository files (config, tokenizer, chat template, shard index) and
writes `shard_headers.json` with every shard's safetensors header, read through HTTP range
requests. That directory is enough for streaming conversion (`tools.convert --stream-url`).
With `--all` it also downloads every weight shard (resumable) for ordinary local conversion.

  python -m tools.convert.download \
      https://modelscope.cn/models/Qwen/Qwen3.8-Flash-Next/resolve/master/ models/flashnext-hf
"""

from __future__ import annotations

import argparse
import concurrent.futures
import http.client
import json
import struct
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

SMALL_FILES = (
    "config.json", "generation_config.json", "model.safetensors.index.json", "tokenizer.json",
    "tokenizer_config.json", "vocab.json", "merges.txt", "chat_template.jinja",
    "preprocessor_config.json", "video_preprocessor_config.json", "LICENSE",
)
_TRANSIENT = (OSError, http.client.HTTPException)
# Some mirrors (hf-mirror.com) reject the default Python-urllib agent.
USER_AGENT = {"User-Agent": "ninfer-offload/1.0"}


def repository_url(hub: str, repo: str, endpoint: str | None = None) -> str:
    """File base URL for `modelscope` or `huggingface` (optionally a mirror endpoint)."""

    if hub == "modelscope":
        return f"{(endpoint or 'https://modelscope.cn').rstrip('/')}/models/{repo}/resolve/master/"
    if hub == "huggingface":
        return f"{(endpoint or 'https://huggingface.co').rstrip('/')}/{repo}/resolve/main/"
    raise ValueError(f"unknown hub {hub!r}")


def fetch(url: str, start: int | None = None, end: int | None = None) -> bytes:
    headers = dict(USER_AGENT) if start is None else {**USER_AGENT, "Range": f"bytes={start}-{end}"}
    for attempt in range(5):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=headers),
                                        timeout=120) as response:
                return response.read()
        except urllib.error.HTTPError:
            raise
        except _TRANSIENT:
            if attempt == 4:
                raise
            time.sleep(3 * (attempt + 1))
    raise AssertionError


def shard_header(base: str, name: str) -> tuple[str, dict]:
    length = struct.unpack("<Q", fetch(base + name, 0, 7))[0]
    header = json.loads(fetch(base + name, 8, 8 + length - 1))
    header.pop("__metadata__", None)
    return name, {"data_offset": 8 + length, "tensors": header}


def shard_sizes(out: Path) -> dict[str, int]:
    """Exact byte size of every shard, from `shard_headers.json`."""

    headers = json.loads((out / "shard_headers.json").read_text())
    return {
        name: entry["data_offset"]
        + max((t["data_offsets"][1] for t in entry["tensors"].values()), default=0)
        for name, entry in headers.items()
    }


def prepare(base: str, out: Path) -> None:
    """Small repository files plus `shard_headers.json`; no weight data."""

    base = base.rstrip("/") + "/"
    out.mkdir(parents=True, exist_ok=True)
    for name in SMALL_FILES:
        try:
            (out / name).write_bytes(fetch(base + name))
            print(f"downloaded {name}")
        except urllib.error.HTTPError as error:
            print(f"skipped {name} ({error.code})")
    index_path = out / "model.safetensors.index.json"
    if not index_path.is_file():
        raise SystemExit(f"{base}model.safetensors.index.json not found: check the URL/repository")
    index = json.loads(index_path.read_text())
    shards = sorted(set(index["weight_map"].values()))
    headers = {}
    with concurrent.futures.ThreadPoolExecutor(8) as pool:
        for name, header in pool.map(lambda f: shard_header(base, f), shards):
            headers[name] = header
    (out / "shard_headers.json").write_text(json.dumps(headers))
    print(f"wrote shard_headers.json for {len(headers)} shards")


def _download(url: str, path: Path, size: int, progress) -> None:
    part = path.with_suffix(path.suffix + ".part")
    for attempt in range(20):
        have = part.stat().st_size if part.exists() else 0
        if have >= size:
            break
        try:
            request = urllib.request.Request(
                url, headers={**USER_AGENT, "Range": f"bytes={have}-{size - 1}"})
            with urllib.request.urlopen(request, timeout=120) as response, part.open("ab") as file:
                while chunk := response.read(1 << 22):
                    file.write(chunk)
                    progress(len(chunk))
        except _TRANSIENT:
            time.sleep(min(60, 5 * (attempt + 1)))
    if part.stat().st_size != size:
        raise OSError(f"{path.name}: download incomplete")
    part.replace(path)


def download_all(base: str, out: Path, parallel: int = 4) -> None:
    """Download every weight shard into `out`; complete shards are kept, `.part` files resume."""

    base = base.rstrip("/") + "/"
    sizes = shard_sizes(out)
    pending = {n: s for n, s in sizes.items()
               if not ((out / n).is_file() and (out / n).stat().st_size == s)}
    total = sum(pending.values())
    done = sum((out / (n + ".part")).stat().st_size for n in pending
               if (out / (n + ".part")).exists())
    lock = threading.Lock()
    started = time.monotonic()
    reported = 0.0

    def progress(count: int) -> None:
        nonlocal done, reported
        with lock:
            done += count
            now = time.monotonic()
            if now - reported >= 10:
                reported = now
                rate = done / max(now - started, 1e-9) / 1e6
                print(f"downloaded {done / 1e9:.1f} / {total / 1e9:.1f} GB ({rate:.1f} MB/s)",
                      flush=True)

    print(f"{len(sizes) - len(pending)} of {len(sizes)} shards already present; "
          f"{total / 1e9:.1f} GB to download")
    with concurrent.futures.ThreadPoolExecutor(parallel) as pool:
        for future in [pool.submit(_download, base + n, out / n, s, progress)
                       for n, s in pending.items()]:
            future.result()
    print("all shards downloaded")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("url", help="repository file base URL (…/resolve/master/ or …/resolve/main/)")
    parser.add_argument("out", type=Path, help="output directory, later passed as --model")
    parser.add_argument("--all", action="store_true", help="also download every weight shard")
    parser.add_argument("--parallel", type=int, default=4, help="parallel shard downloads")
    args = parser.parse_args()
    prepare(args.url, args.out)
    if args.all:
        download_all(args.url, args.out, args.parallel)


if __name__ == "__main__":
    main()
