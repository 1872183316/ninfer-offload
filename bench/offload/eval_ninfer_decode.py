"""Decode speed of the NInfer CLI on the four evaluation prompts.

Run from the repository root; arguments after the model are passed to `ninfer`:
  python bench/offload/eval_ninfer_decode.py LABEL MODEL.ninfer --moe-offload --moe-threads 12 \
      --moe-gpu-experts 64 --moe-expert-stats bench/offload/stats_flash_next.txt
Prints one JSON line per run. Set NINFER_MEMORY_MAX (e.g. 86G) to run each process inside a
systemd user scope with that memory limit, which protects the host from swapping.
"""

import json
import os
import re
import subprocess
import sys

sys.path.insert(0, os.path.dirname(__file__))
from prompts import PROMPTS  # noqa: E402

BINARY = os.environ.get("NINFER_BIN", "./build-sm89/apps/ninfer")
REPS = int(os.environ.get("REPS", "2"))


def main() -> None:
    label, model, extra = sys.argv[1], sys.argv[2], sys.argv[3:]
    limit = []
    if os.environ.get("NINFER_MEMORY_MAX"):
        limit = ["systemd-run", "--user", "--scope", "-q", "-p",
                 f"MemoryMax={os.environ['NINFER_MEMORY_MAX']}", "-p", "MemorySwapMax=0"]
    for rep in range(REPS):
        for name, prompt in PROMPTS:
            cmd = limit + [BINARY, model, "--prompt", prompt, "--max-context", "2048",
                           "--kv-capacity", "2048", "--max-new", "512", "--no-thinking",
                           "--greedy"] + extra
            r = subprocess.run(cmd, capture_output=True, text=True)

            def get(key: str) -> str | None:
                m = re.search(r"summary\s+" + key + r"\s+([\d.]+)", r.stderr)
                return m.group(1) if m else None

            print(json.dumps({"cfg": label, "rep": rep, "prompt": name, "rc": r.returncode,
                              "gen_tokens": get("generated tokens"),
                              "decode_tps": get("decode speed"),
                              "prefill_tps": get("prefill speed"),
                              "head": r.stdout[:60].replace("\n", " ")},
                             ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
