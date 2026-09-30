# Weight converter guide (tools/convert)

English | [中文](README.zh-CN.md)

`tools.convert` quantizes HuggingFace-format safetensors weights (BF16/FP16, …) and writes the
`.ninfer` file that NInfer loads.

1. [Setup](#1-setup)
2. [The easy way: the conversion wizard](#2-the-easy-way-the-conversion-wizard)
3. [Two ways to read the weights: streaming / full download](#3-two-ways-to-read-the-weights)
4. [Manual example: Qwen3.8-Flash-Next](#4-manual-example-qwen38-flash-next)
5. [Manual example: a checkpoint already on disk](#5-manual-example-a-checkpoint-already-on-disk)
6. [Choosing the precision (bits)](#6-choosing-the-precision-bits)
7. [All command-line options](#7-all-command-line-options)
8. [Custom recipes](#8-custom-recipes)
9. [After conversion: inspect and validate](#9-after-conversion-inspect-and-validate)
10. [Troubleshooting](#10-troubleshooting)

The lower-level recipe API is documented in [docs/weight-conversion.md](../../docs/weight-conversion.md).

---

## 1. Setup

| Item | Requirement |
|---|---|
| OS | Linux |
| Python | 3.11 |
| Packages | `torch`, `numpy` |
| GPU | Optional. A CUDA GPU makes quantization faster; CPU conversion also works |
| Working directory | **Run from the repository root** (commands are `python -m tools.convert…`) |

```bash
cd ninfer-offload                     # repository root
python3.11 -m venv ~/convert-venv && source ~/convert-venv/bin/activate
pip install torch numpy
```

The converter does not need a NInfer build; you run the resulting file with `ninfer`.

---

## 2. The easy way: the conversion wizard

```bash
python -m tools.convert.wizard
```

The wizard asks a few questions; pressing Enter accepts each default:

1. **Where the original weights are**: ModelScope, HuggingFace (or a mirror), or already on disk.
2. **How to read them** (online models): streaming (download while converting; least disk) or
   download everything first.
3. It detects the architecture and picks the official recipe; for regular Qwen3.6/3.8 models it
   also asks whether to include vision and MTP.
4. **Output precision**: several presets with their estimated file sizes, or custom bits per
   weight class (see [section 6](#6-choosing-the-precision-bits)).
5. The output path. The wizard checks free disk space and says how much is missing.
6. It prints the exact commands, then either **runs them now** or **saves them as a shell script**
   (useful for long runs under tmux/systemd).

The wizard only composes ordinary converter commands and shows them before running, so you can
repeat a conversion later without it.

---

## 3. Two ways to read the weights

| | Full download, then convert | Streaming |
|---|---|---|
| Use when | The disk holds both the original weights and the output | The original weights are larger than your free space, or you don't want to keep them |
| Disk usage | Original weights + output | Output + a bounded shard cache |
| Original weights | Kept; convert again at another precision without downloading | Deleted as used; another precision downloads again |

**How streaming works**: the converter orders its jobs from the recipe and works out which job
first and last reads each safetensors shard. Six background threads download shards ahead in that
order (HTTP range requests, resumable), keeping the cache under `--stream-budget-gb`; a shard is
deleted as soon as its last job finishes. The output is identical to a local conversion.

---

## 4. Manual example: Qwen3.8-Flash-Next

The official BF16 checkpoint is about 360 GB (131 shards); with the official recipe the output is
about 108 GB. Every shard is downloaded once, so the total time is dominated by your download speed
(about 7–8 hours at 15 MB/s).

### 4.1 Disk space

Streaming needs about 108 GB of output + an 80 GB shard cache (recommended) + ~10 GB margin,
**about 200 GB** in total. A smaller cache still works but is slower: the PLE n-gram table is
spread over most shards and is converted last, so those shards stay cached until the end; once the
cache is full, prefetching stops and downloads become serial.

### 4.2 Step 1: download the index (a few MB)

```bash
python -m tools.convert.download \
  https://modelscope.cn/models/Qwen/Qwen3.8-Flash-Next/resolve/master/ \
  models/flashnext-hf
```

This downloads `config.json`, the tokenizer, the chat template and other small files (files the
repository does not have print `skipped ... (404)`, which is fine) and reads every shard header into
`shard_headers.json`. It ends with `wrote shard_headers.json for 131 shards`.

With `--all` it then downloads every weight shard (resumable) for a
[full-download conversion](#5-manual-example-a-checkpoint-already-on-disk).

The URL is the prefix from which single files can be downloaded, ending in `/`:
- ModelScope: `https://modelscope.cn/models/<org>/<model>/resolve/master/`
- HuggingFace: `https://huggingface.co/<org>/<model>/resolve/main/` (mirrors use the same path)

### 4.3 Step 2: convert

```bash
python -m tools.convert \
  --model models/flashnext-hf \
  --recipe qwen3_8_flash_next \
  --out models/qwen3_8_flash_next.ninfer \
  --stream-url https://modelscope.cn/models/Qwen/Qwen3.8-Flash-Next/resolve/master/ \
  --stream-budget-gb 80 \
  --max-file-bytes 250000000000
```

- `--model` is the directory from step 1; shards are cached there too.
- `--max-file-bytes 250000000000` keeps a single output file. The default splits at 32 GB into
  `xxx.ninfer` + `xxx.ninfer.part-0001`, … (also fine; pass only the entry file to NInfer).
  To change the split of an existing artifact later (e.g. for upload limits), use
  `python -m tools.artifact.reshard xxx.ninfer out/xxx.ninfer --max-file-gb 50`; do not use `split`.
- Flash-Next currently supports only the `text` component; do not add `mtp`/`vision` or `--proposal`.
- Add `--dry-run` to print each weight class's format and the estimated size without downloading
  or converting anything.

### 4.4 Long runs

Conversion takes hours and stops if your remote session drops. Use tmux:

```bash
tmux new -s convert        # run the command inside; Ctrl-b d detaches; tmux attach -t convert returns
```

or a systemd user unit, which can also cap memory so the rest of the machine stays responsive:

```bash
loginctl enable-linger $USER          # keep user units running after logout (once)
systemd-run --user --unit ninfer-convert -p MemoryMax=40G -p MemorySwapMax=0 -p Nice=10 \
  --working-directory=$PWD \
  bash -c 'source ~/convert-venv/bin/activate; python -m tools.convert ...options... > convert.log 2>&1'
systemctl --user status ninfer-convert     # status; `stop` stops it; `reset-failed` before a restart
```

Set `MemoryMax` for your machine; Flash-Next converts within a 40G limit.

### 4.5 Reading the progress

```
[73/972] text/layers/12/moe/experts/gate (+1): q4_g64_fp16 (...)
```

`73/972` is the job number and total. Jobs differ greatly in size (the final PLE table is about
35 GB), so the job number is not a time estimate. The bytes written are more telling:

```bash
du -h  models/.qwen3_8_flash_next.ninfer.*.tmp   # written so far; about 108 GB at the end
du -sh models/flashnext-hf                       # shard cache
```

On success the temporary file is renamed to `qwen3_8_flash_next.ninfer`, and
`qwen3_8_flash_next.ninfer.conversion.json` records the sources, recipe, formats and timing.

### 4.6 After an interruption

- Complete shards are reused and `.part` files resume where they stopped.
- The output cannot be resumed; conversion restarts from job 1, and shards already deleted are
  downloaded again.
- Remove the leftover temporary output first: `rm models/.qwen3_8_flash_next.ninfer.*.tmp`.
- After success you may delete remaining `*.safetensors`/`*.part` in `models/flashnext-hf`; keep the
  tokenizer and config files.

---

## 5. Manual example: a checkpoint already on disk

When the original weights are local (or downloaded with `download --all`), omit the `--stream-*`
options:

```bash
python -m tools.convert.download \
  https://modelscope.cn/models/Qwen/Qwen3.8-Flash-Next/resolve/master/ models/flashnext-hf --all
python -m tools.convert --model models/flashnext-hf --recipe qwen3_8_flash_next \
  --out models/qwen3_8_flash_next.ninfer --max-file-bytes 250000000000
```

Qwen3.6-35B-A3B (MoE):

```bash
python -m tools.convert \
  --model /path/to/Qwen3.6-35B-A3B \
  --recipe qwen3_6_35b_a3b \
  --components text,vision,mtp \
  --resource chat_template.jinja=tools/chat_templates/qwen3_6.jinja \
  --out models/qwen3_6_35b_a3b.ninfer
```

Qwen3.6-27B (Dense):

```bash
python -m tools.convert \
  --model /path/to/Qwen3.6-27B \
  --recipe qwen3_6_27b \
  --components text,vision,mtp \
  --resource chat_template.jinja=tools/chat_templates/qwen3_6.jinja \
  --proposal \
  --out models/qwen3_6_27b.ninfer
```

Select only the components you need: `text` (default) for chat, `vision` for image input, `mtp`
for MTP speculative decoding.

---

## 6. Choosing the precision (bits)

### 6.1 Official recipes

Defined in [official_recipes.py](official_recipes.py):

| Recipe | Model | Precision |
|---|---|---|
| `qwen3_8_flash_next` | Qwen3.8-Flash-Next | expert gate/up **4-bit**, expert down **5-bit**, PLE table 5-bit, embedding 8-bit, output head 6-bit, other projections 8-bit; router, shared_score, GDN a/b stay BF16 |
| `qwen3_6_35b_a3b` | Qwen3.6-35B-A3B | experts 4-bit, expert down 5/6-bit, other projections 8-bit |
| `qwen3_6_27b` / `qwen3_8_27b` | Qwen3.6/3.8-27B | projections 4/5-bit, vocabulary 6/8-bit |
| `qwen3_6_27b_nvfp4` / `qwen3_8_27b_nvfp4` | needs `--source quantized=` official NVFP4 weights | NVFP4 (runs only on RTX 50-series) |

### 6.2 Changing the bits with `--precision`

Starting from the official recipe, `--precision` sets one width for each of three classes:

| Class | Meaning |
|---|---|
| `experts` | routed expert gate/up (the bulk of a MoE model) |
| `expert-down` | routed expert down |
| `linear` | other layer projections: attention, GDN, shared expert, Dense MLP |

The embedding, output head, PLE table and small weights the recipe keeps in BF16 (routers, …) are
unchanged. Widths: **4, 5, 6, 8**.

```bash
# higher-quality experts: gate/up 5-bit, down 6-bit
python -m tools.convert ... --recipe qwen3_8_flash_next --precision experts=5,expert-down=6
# check the size first
python -m tools.convert ... --precision experts=5,expert-down=6 --dry-run
```

Estimated Flash-Next sizes (`--dry-run` output):

| Choice | Estimated size |
|---|---|
| official recipe | 108 GB |
| `experts=4,expert-down=4` | 103 GB |
| `experts=4,expert-down=4,linear=4` | 101 GB |
| `experts=5,expert-down=6` | 123 GB |
| `experts=8,expert-down=8,linear=8` | 167 GB |

More bits means better quality and a larger file. With experts offloaded to system RAM, decode
speed is roughly inversely proportional to the expert bytes read per token.

The official recipes are the tested combinations. Other choices are checked only against the
formats the kernels accept and have not each been run; [test-run](#92-test-run) the result.

### 6.3 Why there are no 1-, 2- or 3-bit options

NInfer's GPU and CPU kernels implement 4/5/6/8-bit grouped integers (plus FP8 and NVFP4), and the
converter writes only those formats. Rounding to 1–3 bits with a per-group absolute maximum would
also badly damage model quality; usable low-bit quantization needs importance-weighted or codebook
methods. Supporting 1–3 bits requires new storage formats, a quantizer and GPU/CPU kernels — a
separate development effort not in the converter today.

### 6.4 Format names

| Format | Meaning | Average bits per weight |
|---|---|---|
| `q4_g64_fp16` | 4-bit integers, one FP16 scale per 64 | 4.25 |
| `q5_g64_fp16` | 5-bit, groups of 64 | 5.25 |
| `q6_g64_fp16` | 6-bit, groups of 64 | 6.25 |
| `q8_g32_fp16` | 8-bit, groups of 32 | 8.5 |
| `fp8_e4m3fn_row_bf16` | FP8 with one BF16 scale per row | ~8 |
| `bf16` | unquantized | 16 |

The `grouped_absmax` method scales each group by its absolute maximum and rounds to integers; it
needs no calibration data.

---

## 7. All command-line options

`python -m tools.convert --help` shows them at any time.

| Option | Default | Meaning |
|---|---|---|
| `--model DIR` | required | Model directory (config, tokenizer, weights; the index directory in streaming mode). The architecture is detected |
| `--recipe NAME or FILE` | required | Official recipe name, or `my_recipe.py` / `my_recipe.py:function` |
| `--out PATH` | required | Output path; **an existing file is an error, never overwritten** |
| `--precision` | none | Bits per class, e.g. `experts=4,expert-down=5,linear=8` |
| `--dry-run` | off | Print formats and the estimated size, then exit |
| `--override FILE` | none | Python file applied after the recipe and `--precision` |
| `--components` | `text` | Any of `text,vision,mtp,dflash,dflash2` |
| `--source NAME=PATH` | none | Extra source, repeatable (`quantized=` for NVFP4 recipes, `dflash2=`, …) |
| `--resource ROLE=PATH` | none | Replace a packaged resource, e.g. `chat_template.jinja=template.jinja` |
| `--proposal` | off | Add the speculative-decoding proposal head (Dense recipes) |
| `--proposal-rows` | 131072 | Proposal head rows |
| `--ranking PATH` | built in | Token ranking used by the proposal head |
| `--name` | none | Model name stored in metadata; does not affect computation |
| `--device` | `cuda` | Quantization device: `cuda`, `cuda:1`, `cpu` |
| `--rows-per-chunk` | 512 | Matrix rows processed at a time; lower it if memory is short |
| `--max-file-bytes` | 32000000000 | Per-file limit in bytes; larger outputs split into `.part-000N` |
| `--stream-url URL` | none | Streaming mode (needs `shard_headers.json` from `tools.convert.download`) |
| `--stream-budget-gb` | 20 | Shard cache limit in GB; a shard the current job waits for may exceed it |

---

## 8. Custom recipes

To change a few weights, write an override file such as `my_override.py`:

```python
def configure(model, recipe, sources):
    # layer 0 MLP down in 6-bit
    recipe.assign("text/layers/0/mlp/down", format="q6_g64_fp16", method="grouped_absmax")
```

```bash
python -m tools.convert ... --recipe qwen3_6_27b --override my_override.py
```

- Names accept `*` wildcards; a selector matching nothing is an error.
- To list every logical weight name:

  ```python
  def configure(model, recipe, sources):
      for name, p in model.parameters.items():
          print(name, p.shape)
      raise SystemExit
  ```

- When changing `format`, also give `method`.
- The converter only guarantees a valid file; whether a format combination runs depends on NInfer's
  operators, so test-run new combinations.

Grouping, shared weights, weights from another checkpoint and custom quantizers are covered in
[docs/weight-conversion.md](../../docs/weight-conversion.md).

---

## 9. After conversion: inspect and validate

### 9.1 Inspect (no inference)

```bash
python -m tools.artifact.inspect models/qwen3_8_flash_next.ninfer --objects --bindings
```

### 9.2 Test-run

```bash
./build-sm89/apps/ninfer models/qwen3_8_flash_next.ninfer --prompt "The capital of France is" \
  --no-thinking --max-context 2048 --kv-capacity 2048 --max-new 32 \
  --moe-offload --moe-threads 12 --moe-gpu-experts 0
```

The output should be fluent text. See the repository README for speed options such as
`--moe-gpu-experts` and `--moe-expert-stats`.

### 9.3 Numerical validation (optional)

An independent FP32 reference decodes the stored quantized weights and follows the official
mathematics layer by layer; compare it with NInfer:

```bash
echo "A plain English paragraph of about sixty words ..." > sample.txt
python -m tools.validate.qwen4exp_reference models/qwen3_8_flash_next.ninfer \
  --tokenizer models/flashnext-hf --text sample.txt --tokens 64
./build-sm89/apps/ninfer-perplexity models/qwen3_8_flash_next.ninfer --text sample.txt \
  --moe-offload --moe-threads 12
```

Reference result for the official recipe on a 59-token English paragraph: NInfer PPL 4.2346,
FP32 reference 4.2407. They should differ by a few parts per thousand. The reference supports
Flash-Next only and runs slowly on the CPU; a few dozen `--tokens` is enough.

---

## 10. Troubleshooting

| Symptom | Cause / fix |
|---|---|
| `No module named tools` | Not run from the repository root |
| `conversion output already exists` | Choose another output name or delete the old file |
| `shard_headers.json` not found | Run `tools.convert.download` first for streaming, or `--model` points elsewhere |
| `model.safetensors.index.json not found` | Wrong URL or repository; `<prefix>config.json` should download in a browser |
| `xxx.safetensors: download incomplete` | Repeated failures: a long network outage or a full disk. Check `df -h` and rerun (complete shards are reused) |
| Downloads slow down and stay slow | The shard cache is full and downloads became serial; free space and raise `--stream-budget-gb` |
| `... was released before a later read` | A shard was deleted too early (should not happen); delete that shard and rerun |
| `bits must be one of 4, 5, 6, 8` | 1–3 bits are not supported; see [6.3](#63-why-there-are-no-1--2--or-3-bit-options) |
| `CUDA out of memory` | Lower `--rows-per-chunk` (e.g. 128) or use `--device cpu` |
| The machine freezes during conversion | Out of memory; cap the converter with systemd-run `MemoryMax` (4.4) |
| The system disk fills up | Put `--model` and `--out` on a disk with enough space; check with `df -h` |
| Conversion succeeds but `ninfer` rejects a format | No operator for that combination; use the official recipe or other bits |
