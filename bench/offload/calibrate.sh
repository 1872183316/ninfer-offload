#!/bin/bash
# Routing statistics for --moe-expert-stats: runs six calibration prompts (disjoint from the
# evaluation prompts) with all experts on the host and sums the per-layer expert counts.
#   bench/offload/calibrate.sh MODEL.ninfer OUT_STATS.txt [ninfer binary]
# Run from the repository root. Set NINFER_MEMORY_MAX (e.g. 86G) for a systemd memory limit.
set -eu
MODEL=$1
OUT=$2
BIN=${3:-./build-sm89/apps/ninfer}
WORK=$(mktemp -d)
LIMIT=()
if [ -n "${NINFER_MEMORY_MAX:-}" ]; then
  LIMIT=(systemd-run --user --scope -q -p MemoryMax=$NINFER_MEMORY_MAX -p MemorySwapMax=0)
fi
i=0
while IFS= read -r p; do
  i=$((i+1))
  "${LIMIT[@]}" "$BIN" "$MODEL" --prompt "$p" --max-context 2048 --kv-capacity 2048 --max-new 384 \
    --no-thinking --greedy --moe-offload --moe-gpu-experts 0 --moe-threads "${THREADS:-12}" \
    --moe-record-stats "$WORK/c$i.txt" > "$WORK/c$i.out" 2> "$WORK/c$i.err" || echo "run $i failed"
done <<'PROMPTS'
写一个Python函数，实现快速排序，并解释它的时间复杂度和空间复杂度。
Explain the difference between a process and a thread in operating systems, with examples.
把这句话翻译成英文并分析语法：人工智能正在深刻地改变我们的工作方式和生活方式。
一个水池有甲乙两个进水管，甲管单独注满需要6小时，乙管单独注满需要4小时，两管同时打开需要多少小时注满？请写出详细步骤。
写一首关于秋天的七言绝句，然后逐句赏析。
What are the main causes of inflation, and how do central banks typically respond?
PROMPTS
python3 - "$WORK" "$OUT" <<'PY'
import glob, sys
work, out = sys.argv[1], sys.argv[2]
total = None
for f in sorted(glob.glob(work + "/c*.txt")):
    rows = [list(map(int, l.split())) for l in open(f) if l.strip()]
    total = rows if total is None else [[a + b for a, b in zip(r, s)] for r, s in zip(total, rows)]
open(out, "w").write("\n".join(" ".join(map(str, r)) for r in total) + "\n")
for layer in (0, len(total) // 2, len(total) - 1):
    r = sorted(total[layer], reverse=True)
    s = sum(r)
    print(f"layer {layer}: top64 share={sum(r[:64]) / s:.2f} top128 share={sum(r[:128]) / s:.2f}")
PY
rm -rf "$WORK"
