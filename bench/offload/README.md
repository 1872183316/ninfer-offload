# Offload evaluation

Scripts behind the decode-speed numbers in the top-level README and
[expert-offload.md](../../docs/maintainer/expert-offload.md). Run them from the repository root.

| File | Purpose |
|---|---|
| `prompts.py` | The four evaluation prompts (Chinese essay, code edit, English-to-Chinese translation, explanation). |
| `eval_ninfer_decode.py` | Runs the `ninfer` CLI on each prompt (greedy, 512 new tokens, 2048 context, two repetitions) and prints decode/prefill tok/s. |
| `eval_openai_server.py` | The same prompts against an OpenAI-compatible server such as llama.cpp's `llama-server`, for comparisons. |
| `calibrate.sh` | Six calibration prompts, disjoint from the evaluation set, producing routing counts for `--moe-expert-stats`. |
| `stats_flash_next.txt`, `stats_qwen3_6_35b_a3b.txt` | Routing counts measured with `calibrate.sh` (one line of per-expert counts per layer). |

```bash
NINFER_MEMORY_MAX=86G python bench/offload/eval_ninfer_decode.py r64 qwen3_8_flash_next.ninfer \
  --moe-offload --moe-threads 12 --moe-gpu-experts 64 \
  --moe-expert-stats bench/offload/stats_flash_next.txt
```

`NINFER_MEMORY_MAX` runs each process in a systemd user scope with that memory limit, so a model
that does not fit is stopped instead of driving the host into swap.
