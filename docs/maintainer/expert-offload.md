# Routed-expert offload

This fork runs MoE models whose routed experts exceed device memory. Routed expert banks stay in
host memory and are computed by host threads; a configurable subset of experts per layer also gets a
device replica. Everything else (embedding, attention/GDN, router, shared expert, head, KV, state)
remains device resident. Offload is a startup choice (`--moe-offload`, accepted by `ninfer`,
`ninfer-perplexity` and `ninfer-serve`); without it the native device-resident path is unchanged.

## Placement and binding

- `MoeOffloadOptions` (`include/ninfer/types.h`) selects offload, the number of device-resident
  experts per layer, host threads, and an optional routing-statistics file.
- `bind_moe` (`src/models/qwen3_5/load/text.cpp`) binds every Text-layer routed expert with
  `Residency::Host`. MTP keeps its experts on the device.
- Resident experts are ranked per layer by the statistics file (one line of per-expert routing
  counts per Text layer) or, without it, by lowest id. Their rows are requested from the host bank
  with `Binder::require_device_rows`; the materializer concatenates the selected row spans of each
  row-split plane into a standalone payload (storage-layouts section 3.7) and uploads it. The copy
  moves stored bytes only; codes and scales are unchanged.
- Replica slot `s` holds expert `resident[s]`: gate/up rows `[s*2I, (s+1)*2I)`, down rows
  `[s*H, (s+1)*H)`.

## Execution

`ops::hybrid_sparse_moe` (`src/ops/sparse_moe/hybrid/`) replaces `sparse_moe` for offloaded layers:

1. GPU: router logits, exact top-K (lower id wins ties), normalized weights, sigmoid shared gate;
   selections and the BF16 input are written to a mapped host mailbox.
2. `cuStreamWriteValue32` publishes the request; the Program-owned `HybridMoeHostRuntime` service
   thread computes the host-only selected experts with AVX2 row-split kernels
   (`src/ops/sparse_moe/host/`) and writes an FP32 partial.
3. Meanwhile the GPU computes the shared expert and the selected resident experts.
4. `cuStreamWaitValue32` waits for the host; a combine kernel adds both FP32 partials to the
   residual with one BF16 rounding.

All flag values are constants, so the sequence is CUDA Graph capturable. The mailbox is sized for
the largest Text call (one prefill chunk or one decode/verify batch). Reduction association between
device and host shares is a private choice; the oracle is the Sparse MoE formula in
[the model reference](qwen3_5-model.md#sparse-moe).

## Host kernels

Host kernels decode `row_split_k128_v1` Q4/Q5/Q6/Q8 exactly and accumulate in FP32 against FP32
activations (A16-equivalent). Q5/Q6 high planes are expanded with BMI2 `pext`/`pdep`; activations
are pre-permuted per G64 group so low and high nibbles meet contiguous vectors. The executor runs
gate/up+SwiGLU items and down items as two parallel phases on a persistent pool; each down item owns
its output rows. Kernels require AVX2, FMA, F16C and BMI2, checked at startup.

## Verification

- `ninfer_host_rowsplit_dot_test`, `ninfer_host_moe_test`: host kernels and executor against FP64
  logical decode.
- `ninfer_hybrid_sparse_moe_test`: complete hybrid Op against an FP64 oracle, including all-host,
  all-resident, mixed residency and top-10 geometry.
- `ninfer_host_moe_bench`: host throughput at real bank sizes with a DRAM read reference.

## Qwen3.8-Flash-Next (Qwen4-Exp)

`Architecture::Qwen4Exp` (`qwen4_exp_text`) extends the Qwen3.5 family:

- **Hyper-connection residual.** The residual is four BF16 streams `[4H,T]`. Each attention/GDN and
  MoE block reads a gated mix of the normalized streams and adds its output back to every stream
  with a per-stream injection weight (`ops::hc_*`, `include/ninfer/ops/hyper_connection.h`). The
  head mixer replaces the final RMSNorm. Mixers therefore read their input directly and write to a
  separate `[H,T]` output (`src/models/qwen3_5/execution/qwen4exp.cpp`).
- **Attention.** 24 query / 2 KV heads of dimension 256 (causal geometry `CausalD256H24Kv2`,
  group 12: small-T tiles hold at most four tokens). The QSA indexer selects every visible token
  while at most 2051 tokens are visible, so up to `--max-context 2051` Programs run dense causal
  attention unchanged.
- **QSA token selection** (`--max-context` above 2051; `include/ninfer/ops/qsa.h`). Each
  attention layer stores a raw 128-dim index key per token in an extra BF16 plane of the Main Text
  KV pool (one plane per attention layer, after the K/V planes), so pages, prefix reuse, Host KV
  and checkpoints carry it. Per call `qsa_index_append` stores the keys, `qsa_select` builds the
  pooled, normalized and rotated key of every complete 4-token block, scores it against the four
  normalized and rotated index queries (`sum_h relu(q_h.k)/sqrt(128)`) and keeps the top 512 blocks
  (lower index wins ties) plus the incomplete tail, and `qsa_attention` appends K/V and attends to
  exactly those tokens. The launch sequence depends only on the column count, so decode CUDA Graphs
  of all frontier profiles share one topology. The route requires the BF16 KV profile and text
  input (RoPE positions equal cache positions); other KV dtypes and `--vision` are rejected above
  2051. Its attention kernel is a scalar FP32 kernel: with routed experts on the CPU, attention
  time is small next to the MoE.
- **GatedDeltaNet** uses the 16K/48V-head geometry with a sigmoid output gate
  (`ops::gated_rmsnorm_sigmoid`).
- **PLE n-gram injection** (layer 1). The GPU hashes each token with its two-token history into 16
  table rows (`ops::ple_ngram_rows`); the host runtime gathers them from the host-mapped Q5 table
  (`Residency::HostMapped`, ~35 GB, never uploaded) through the MoE mailbox (`ops::ple_gather`);
  the GPU then applies the key/value projections, the stream gate and the dilated convolution
  (`ops::ple_gate`, `ops::ple_dilated_conv_silu`). The convolution and token history live in one
  extra pseudo layer of the Linear Attention state pool, so checkpoints and slot copies include it.
- Qwen4-Exp requires `--moe-offload`; MTP and other speculative backends are not implemented.

Conversion streams the byte ranges of the BF16 checkpoint that upcoming jobs read from ModelScope
(`tools/convert/sources/streaming.py`; disk use bounded by `--stream-budget-gb` plus the output):

```bash
python -m tools.convert --model <dir with config/tokenizer/shard_headers.json> \
  --recipe qwen3_8_flash_next --out qwen3_8_flash_next.ninfer --device cuda \
  --stream-url https://modelscope.cn/models/Qwen/Qwen3.8-Flash-Next/resolve/master/ \
  --stream-budget-gb 20 --max-file-bytes 250000000000
```

The recipe stores routed gate/up as Q4, routed down and the PLE table as Q5, the output head as Q6,
other projections as Q8, and the router, shared-expert score and GDN a/b controls as BF16
(108 GB total).

`tools/validate/qwen4exp_reference.py` is an independent FP32 forward pass over the artifact's
stored weights (exact decode of codes and scales) following the reference transformers model. On
a 59-token English paragraph `ninfer-perplexity --moe-offload --kv-dtype bf16` gives mean NLL
1.44330 against the reference's 1.44474; a 16-token greedy continuation matches the reference
argmax at 15 positions, the exception being a near tie (reference margin 0.29 nats).

Decode speed on the development host (Xeon E5-2673 v3 12C, DDR3 LRDIMM ~21 GB/s measured read,
RTX 4060 Ti 16 GB, PCIe 3.0 x8), `ninfer --moe-offload --moe-threads 12 --greedy --no-thinking`,
`--max-context 2048`, BF16 KV, four prompts (essay, code edit, translation, explanation; 112-512
generated tokens), two repetitions each, resident experts ranked by routing counts from six
disjoint calibration prompts (top 64 experts cover 31-54% of routings per layer):

| resident experts per layer | decode tok/s (per prompt) | prefill tok/s |
|---|---|---|
| 0 (5.16 GiB device) | 10.0-10.2 | 21.6-29.0 |
| 48 | 11.9-13.0 | 26.7-33.6 |
| 64 | 12.5-14.1 | 28.6-35.4 |

Repetitions differ by at most 0.1 tok/s. The routed experts occupy 69.2 GB (4.58 bits/weight), so
with no resident experts each token reads ~1.35 GB of routed-expert weights on the host
(~13.6 GB/s effective).

### QSA validation and cost

`ninfer_qsa_test` checks the three QSA Ops against a naive CPU oracle at the real geometry
(prefill chunks above and below the dense limit, batched decode with a dead row, verify-style
rows, single decode columns with token splits): index-key and K/V storage are byte-exact, block
selections equal the oracle exactly in every case, and attention outputs are within 0.69 of the
BF16 tolerance (`2^-8 |ref| + 2e-4`).

End to end (3,300-token English model card, `ninfer-perplexity --context 3328`, BF16 KV, all
experts on the host) the mean NLL is 0.73746 against 0.73321 for the FP32 reference with QSA and
0.73267 for the reference with dense attention. On this text QSA changes the reference by less
than the NInfer-versus-reference noise: over the first 2,050 tokens, where both reference variants
are mathematically identical, they still differ on 41 tokens by more than 0.02 nats (CPU reduction
order and expert-routing flips), and NInfer's per-token NLL differs from the reference by 0.084
nats on average. The dense NInfer route gives 1.12756 on those 2,050 tokens and the QSA route
1.12934 (reference 1.12134); on a 59-token paragraph the QSA route gives 1.44704 against 1.44330
dense. The Op-level selection test is therefore the decisive check of QSA semantics. Functionally,
a 3,352-token prompt whose passkey appears only in its first sentence is answered correctly.

Serving cost on the development host (`ninfer-serve`, 64 resident experts, greedy, 128 generated
tokens unless stated):

| configuration | decode tok/s | prefill tok/s |
|---|---|---|
| 1,777-token prompt, `--max-context 2048` (dense) | 11.83 / 12.02 | 34.1 |
| same prompt, `--max-context 4096` (QSA) | 11.69 / 11.87 | 33.7 |
| 3,327-token prompt, `--max-context 4096` (QSA) | 11.68 / 11.76 | 36.4 |
| four benchmark prompts, dense 2048 | 12.70-14.65 | |
| four benchmark prompts, QSA 4096 | 12.52-14.36 (0.9-2.0% slower, one prompt equal) | |

The first QSA attention kernel used one CTA per KV head and cost about 5% of decode speed at
1,777 tokens; splitting each column's tokens over up to 32 CTAs reduced that to about 1.2%. The
remainder is the index projection and the selection launches, which also run while every block is
selected.
