# Routed-expert offload

This fork runs MoE models whose routed experts exceed device memory. Routed expert banks stay in
host memory and are computed by host threads; a configurable number of device slots per layer holds
copies of selected experts, computed by the GPU. Everything else (embedding, attention/GDN, router,
shared expert, head, KV, state) remains device resident. Offload is a startup choice
(`--moe-offload`, accepted by `ninfer`, `ninfer-perplexity` and `ninfer-serve`); without it the
native device-resident path is unchanged.

## Placement and binding

- `MoeOffloadOptions` (`include/ninfer/types.h`) selects offload, the number of device slots per
  layer (`--moe-gpu-experts`), host threads, an optional routing-statistics file and whether the
  slots follow routing (`dynamic_residency`, default; `--moe-static-experts` keeps the startup
  choice).
- `bind_moe` (`src/models/qwen3_5/load/text.cpp`) binds every Text-layer routed expert with
  `Residency::Host`; the materializer page-locks those host objects (`HostPageLock`) so device
  copies from them are asynchronous DMA. The Qwen4-Exp MTP layer binds its experts the same way
  with no slots.
- The startup residents are ranked per layer by the statistics file (one line of per-expert
  routing counts per Text layer) or, without it, by lowest id; the Model data records only this
  list (`HybridSparseMoeWeights::resident`).
- The slots are Program-owned mutable state: the persistent layout reserves one row-split bank
  pair per offloaded layer (`ops::hybrid_moe_slot_bytes`), and `HybridMoeHostRuntime::add_layer`
  copies the startup residents into them. Slot `s` holds the stored rows of one expert: gate/up
  rows `[s*2I, (s+1)*2I)` and down rows `[s*H, (s+1)*H)`. Copies move stored bytes only; codes and
  scales are unchanged (storage-layouts section 3.7).

## Execution

`ops::hybrid_sparse_moe` (`src/ops/sparse_moe/hybrid/`) replaces `sparse_moe` for offloaded layers:

1. GPU: router logits (`router_logits_kernel`, one warp per router row over all SMs; its first
   block also copies the BF16 input to the mapped host mailbox), then per token exact top-K
   (lower id wins ties), normalized weights and the sigmoid shared gate (`route_kernel`). Each
   selection reads the layer's slot map once (mapped host memory, `int16[E]`); that value alone
   decides the side. Selections, with slot-held ones written as `-1 - id`, go to the mailbox.
   An earlier single-kernel route computed the 513 router rows in one block per token and took
   93-194 us per decode call on the development GPU; the split takes about 20 us.
2. `cuStreamWriteValue32` publishes the request; the Program-owned `HybridMoeHostRuntime` service
   thread computes the non-negative selections with AVX2 row-split kernels
   (`src/ops/sparse_moe/host/`) and writes an FP32 partial.
3. Meanwhile the GPU computes the shared expert and the slot-held selections.
4. `cuStreamWaitValue32` waits for the host; a combine kernel adds both FP32 partials to the
   residual with one BF16 rounding.

All flag values are constants and the map is read at run time, so the sequence is CUDA Graph
capturable and graphs stay valid while slots change. The mailbox is sized for the largest Text
call (one prefill chunk or one decode/verify batch). Reduction association between device and host
shares, and which side computes a selection, are private choices; the oracle is the Sparse MoE
formula in [the model reference](qwen3_5-model.md#sparse-moe).

## Expert cache (dynamic residency)

Routing is strongly local in time: the experts a layer chose for the last few dozen tokens are far
more likely to be chosen next than the statically most frequent ones. With dynamic residency a
cache thread of the host runtime keeps the slots on recently frequent experts:

- **Score.** Every call's selections (both sides) update a per-layer decayed frequency
  `f[e] <- f[e] * 2^(-T/16) + count` (half-life 16 routed token columns).
- **Admission.** After a call, host-computed experts with `f >= 3` replace the slot-held expert of
  lowest `f` if their own `f` is higher, at most two per layer and call. The threshold keeps the
  copy traffic small: every copy reads the expert from host memory once, the same DRAM traffic as
  computing it once on the CPU.
- **Replacement protocol.** (1) The victim's map entry is set to -1 and the current request count
  `n` is recorded (after a full fence). (2) Once the service thread has taken request `n+2`, every
  call that could have read the old entry has finished its slot kernels, because the calls share
  one stream and each ends with its request. (3) The new expert's plane spans are copied into the
  slot on a separate non-blocking stream (relaxed capture mode, so a concurrent graph capture is
  unaffected). (4) When the copy's event has completed, the new map entry is set. A slot is never
  read while it is being written, and a selection is never computed twice or not at all.
- Prefill chunks and verify batches update the scores like decode calls; a call simply uses
  whatever the map holds when it routes.

`HybridMoeHostRuntime::mapped_experts` and `replacements` expose the state for tests and
diagnostics.

### Expert cache results (Qwen3.8-Flash-Next)

Offline replay of a routing trace (2,826 decode tokens of the four benchmark prompts, 64 slots per
layer) predicted the hit rate of slot-held selections to rise from 32.7% with the
statistics-ranked static choice to 64-68% with this policy, at about 14 copies per token; plain LRU
admission reached 70-74% but needed 45-130 copies per token. `ninfer-serve` on the development host
(see below; 64 slots per layer, `--max-context 2048`, greedy, mean of two repetitions, tok/s):

| configuration | essay | code edit | translation | explanation |
|---|---|---|---|---|
| v0.3.0, no MTP (same day) | 14.37 | 12.61 | 13.33 | 13.80 |
| v0.3.0, MTP 1 draft | 14.74 | 15.16 | 15.77 | 16.65 |
| static slots (`--moe-static-experts`) | 15.34 | 13.56 | 14.28 | 14.77 |
| dynamic slots | 24.08 | 19.17 | 19.16 | 21.76 |
| dynamic slots, MTP 1 draft | 25.53 | 22.57 | 21.66 | 25.42 |
| dynamic slots, MTP 2 drafts | 22.90 | 24.11 | 20.99 | 25.84 |
| draft acceptance, dynamic, 1 draft | 58% | 97% | 85% | 84% |
| draft acceptance, dynamic, 2 drafts | 42% | 94% | 73% | 75% |

Repetitions differ by at most 1.3 tok/s (translation, 112 tokens). The static row is 7-8% faster
than v0.3.0 because of the split route kernel; dynamic slots add 34-57% to that without MTP and
29-63% with one draft (1.6-1.8x v0.3.0 without MTP). With
the cache, one draft is still the best default on prose and two drafts win on code (+7%) while
losing 10% on the essay. The resident set size of the server is unchanged (73.3 GB without MTP);
the copies use about 6 ms of PCIe time per token.

Accuracy. Which side computes a selection changes only reduction order, but in a top-10-of-512
router such rounding differences occasionally flip a later routing decision or a near-tied
token, as any change of placement does. Mean NLL of a 3,299-token English text (`--context 2048
--stride 1024`): all experts on the host 0.749313, static lowest-id slots 0.745008, static
statistics-ranked slots 0.747752, dynamic slots 0.744098 (two runs identical). Per-token NLL
differs between the static placements by 0.056-0.058 on average and between dynamic and its
static starting point by 0.019. A 256-token greedy CLI generation (code edit) was identical with
static and dynamic slots, and 59-token NLL was identical (1.440390).

Determinism. Slot changes are asynchronous, so the device/host split of a call depends on timing.
Greedy outputs through `ninfer-serve` can therefore differ between runs with dynamic slots (the
code-edit benchmark generated 385 and 399 tokens in two repetitions; static slots repeat exactly);
use `--moe-static-experts` where bitwise repeatability matters.

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
  all-slot, mixed residency and top-10 geometry, and two dynamic-residency sequences (48 calls of
  one and three columns, partly back to back) during which the cache replaces slots; every call
  is checked.
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
- Qwen4-Exp requires `--moe-offload`; `--spec mtp` is the only speculative backend (see below).

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

### MTP speculative decoding

An artifact converted with `--components text,mtp` (1.5 GB more; `--reuse` adds it to an existing
text-only conversion) enables `--spec mtp --draft-tokens N`. The predictor follows the reference
`Qwen4ExpMTP` module: `u = fc_embedding(rmsnorm(emb(x[t+1]))) + fc_hidden(rmsnorm(w[t]))` per
stream, where `w[t]` is the target's final wide stream `[4H]` before the head mixer, then one full
attention block with its own QSA indexer and a 512-expert MoE, its own head mixer and the shared
output head; the pair `(w[t], x[t+1])` uses RoPE position `t`
(`TextContext::qwen4exp_mtp_core`, `src/models/qwen3_5/execution/qwen4exp.cpp`). The MTP
condition is therefore the wide stream: prefill and verify capture it and the MTP hidden buffers,
checkpoint images and round state are `hc_count * H` wide.

- **Placement.** The MTP routed experts are host runtime layer 48 (after the 48 Text layers) with
  no GPU replicas; the other MTP weights (0.1 GiB) and the MTP KV pool (with its QSA index plane)
  are on the GPU.
- **Verify and state commit.** A round runs the target over `k+1` columns with
  `GdnStateAction::RecordForReplay`: GDN layers record their convolution inputs and recurrence
  inputs (`causal_conv1d_silu_record`, `gated_delta_net_replay_record`), the PLE layer records its
  gated convolution inputs and token ids, and after greedy acceptance on the device
  `GdnReplayFoldPlan` (registered for the 36-layer 16K/48V geometry) and `ops::ple_replay_fold`
  commit exactly the accepted columns into the state slots. Rejected columns never touch committed
  state.
- **Prefill.** Every prompt column runs the MTP stem and attention (appending its MTP K/V and
  index keys); only the final column, whose output feeds the first proposal, runs the MoE stage.
- `ninfer_hyper_connection_test` checks `ple_replay_fold` and `ninfer_causal_conv1d_silu_test` the
  record kernel against naive oracles; `ninfer_gdn_replay_fold_test` covers the Flash-Next fold
  geometry. `tools/validate/qwen4exp_mtp_reference.py` is an FP32 reference of the predictor.

Draft acceptance of the FP32 reference predictor on four engine-generated sequences (accepted
fraction at draft positions 1/2/3): translation 84/75/71%, code edit 95/94/94%, essay 57/52/49%,
explanation 85/81/81%.

Decode speed through `ninfer-serve` measured with v0.3.0 (static resident experts and the
single-block route kernel; same host and prompts as above, 64 resident experts, `--max-context
2048`, greedy, mean of two repetitions; repetitions differ by at most 0.5 tok/s). Current numbers
with the expert cache are in [Expert cache results](#expert-cache-results-qwen38-flash-next):

| prompt | no MTP | 1 draft | 2 drafts | 3 drafts |
|---|---|---|---|---|
| essay | 14.29 | 14.74 (+3%) | 13.61 (-5%) | 11.26 (-21%) |
| code edit | 12.61 | 15.16 (+20%) | 16.65 (+32%) | 17.55 (+39%) |
| translation | 13.30 | 15.77 (+19%) | 15.51 (+17%) | 14.97 (+13%) |
| explanation | 13.69 | 16.65 (+22%) | 17.49 (+28%) | 16.87 (+23%) |
| draft acceptance (essay / code / translation / explanation) | | 59/95/83/88% | 45/93/70/79% | 32/92/62/68% |

A round with more drafts reads more distinct host experts (the union of experts over 2/3/4
consecutive tokens is 1.67/2.25/2.79 times one token's), so extra drafts pay off only at high
acceptance. One draft was faster than no MTP on every prompt and is `ninfer-run`'s default; two or
three drafts are faster for code and slower for free-form prose. MTP adds about 0.1 GiB of device
memory and 1.4 GB of host memory.

Greedy MTP output is not bit-identical to greedy output without MTP: a verify round computes
`k+1` columns at once, which changes GEMM and MoE batching and occasionally flips a routing or a
top-1 choice. On the translation prompt all 112 tokens were identical for one and two drafts and
on the code-edit prompt (`ninfer`, 48 resident experts) 256 tokens were identical for two drafts; the essay diverged at token 71
(one draft) and 128 (two drafts). Teacher-forcing each full essay output through the FP32
reference, the engine's token differs from the reference argmax at 23 of 390 positions without MTP
(13 by more than 0.5 nats, mean NLL 0.435), 15 of 378 with one draft (5; 0.377) and 27 of 396 with
two drafts (15; 0.445). The divergence points are such engine-versus-reference disagreements (1.38
and 1.46 nats, one in each route), so MTP does not measurably change agreement with the
reference; the size of these disagreements in every route is a property of the offload engine's
numerics that this work did not investigate.

MTP works above the dense limit and with concurrent requests: with `--max-context 4096
--max-concurrency 2` and one draft, a 3,345-token prompt whose passkey appears only in its first
sentence is answered correctly alone and while a second request decodes; the concurrent code-edit
request decodes at 14.3 tok/s against 12.2 without MTP (alone: 15.4 against 12.7). Prefill of a
3,348-token prompt (three prompts per server, different first sentences) runs at 34.7/36.4/36.6
tok/s with MTP against 35.3/37.1/37.2 without, about 1-2% slower. An earlier version ran the MTP
MoE over every prompt column; it prefilled at 35.4-36.2 tok/s once warm but only 25.5-26.4 tok/s on
the first long prompt after startup (reproduced three times; not reproduced without MTP). That
penalty disappeared with the final-column MoE; its exact cause was not established.
