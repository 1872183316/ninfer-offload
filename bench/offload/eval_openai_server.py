"""Decode speed of an OpenAI-compatible server (ninfer-serve, llama.cpp / ik_llama.cpp) on the
same four prompts, for comparisons with eval_ninfer_decode.py.

  python bench/offload/eval_openai_server.py PORT LABEL
"""

import json
import os
import sys
import time
import urllib.request

sys.path.insert(0, os.path.dirname(__file__))
from prompts import PROMPTS  # noqa: E402

port, label = sys.argv[1], sys.argv[2]
# ninfer-serve requires the served model id; llama-server accepts any.
with urllib.request.urlopen(f"http://127.0.0.1:{port}/v1/models", timeout=60) as resp:
    model = json.loads(resp.read())["data"][0]["id"]


def req(content, n):
    body = {"model": model, "messages": [{"role": "user", "content": content}], "max_tokens": n,
            "temperature": 0, "top_k": 1, "stream": False,
            "chat_template_kwargs": {"enable_thinking": False}}
    r = urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions",
                               data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    t0 = time.time()
    with urllib.request.urlopen(r, timeout=900) as resp:
        d = json.loads(resp.read())
    return d, time.time() - t0
req("你好", 16)  # warmup
for rep in range(2):
    for name, p in PROMPTS:
        d, wall = req(p, 512)
        t = d.get("timings", {})
        n = t.get("predicted_n") or d["usage"]["completion_tokens"]
        rec = {"cfg": label, "rep": rep, "prompt": name, "gen_tokens": n,
               "decode_tps": round(t.get("predicted_per_second", 0), 2),
               "draft_n": t.get("draft_n"), "draft_accepted": t.get("draft_n_accepted"),
               "wall_s": round(wall, 1),
               "head": d["choices"][0]["message"]["content"][:60].replace("\n", " ")}
        print(json.dumps(rec, ensure_ascii=False), flush=True)
