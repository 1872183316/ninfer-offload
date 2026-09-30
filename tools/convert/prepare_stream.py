"""Prepare a directory for streaming conversion (`tools.convert --stream-url`).

Downloads the small repository files (config, tokenizer, chat template, shard index) and writes
`shard_headers.json` with every shard's safetensors header, read through HTTP range requests.
No weight data is downloaded.

  python -m tools.convert.prepare_stream \
      https://modelscope.cn/models/Qwen/Qwen3.8-Flash-Next/resolve/master/ flashnext-hf
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import struct
import urllib.error
import urllib.request
from pathlib import Path

SMALL_FILES = (
    "config.json", "generation_config.json", "model.safetensors.index.json", "tokenizer.json",
    "tokenizer_config.json", "vocab.json", "merges.txt", "chat_template.jinja",
    "preprocessor_config.json", "video_preprocessor_config.json", "LICENSE",
)


def fetch(url: str, start: int | None = None, end: int | None = None) -> bytes:
    headers = {} if start is None else {"Range": f"bytes={start}-{end}"}
    for attempt in range(5):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=headers),
                                        timeout=120) as response:
                return response.read()
        except urllib.error.HTTPError:
            raise
        except OSError:
            if attempt == 4:
                raise
    raise AssertionError


def shard_header(base: str, name: str) -> tuple[str, dict]:
    length = struct.unpack("<Q", fetch(base + name, 0, 7))[0]
    header = json.loads(fetch(base + name, 8, 8 + length - 1))
    header.pop("__metadata__", None)
    return name, {"data_offset": 8 + length, "tensors": header}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("url", help="repository file base URL (…/resolve/master/)")
    parser.add_argument("out", help="output directory, later passed as --model")
    args = parser.parse_args()
    base = args.url.rstrip("/") + "/"
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for name in SMALL_FILES:
        try:
            (out / name).write_bytes(fetch(base + name))
            print(f"downloaded {name}")
        except urllib.error.HTTPError as error:
            print(f"skipped {name} ({error.code})")
    index = json.loads((out / "model.safetensors.index.json").read_text())
    shards = sorted(set(index["weight_map"].values()))
    headers = {}
    with concurrent.futures.ThreadPoolExecutor(8) as pool:
        for name, header in pool.map(lambda f: shard_header(base, f), shards):
            headers[name] = header
    (out / "shard_headers.json").write_text(json.dumps(headers))
    print(f"wrote shard_headers.json for {len(headers)} shards")


if __name__ == "__main__":
    main()
