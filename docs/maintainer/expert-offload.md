# Routed-expert offload

This fork runs MoE models whose routed experts exceed device memory. Routed expert banks stay in
host memory and are computed by host threads; a configurable subset of experts per layer also gets a
device replica. Everything else (embedding, attention/GDN, router, shared expert, head, KV, state)
remains device resident. Offload is a startup choice (`--moe-offload`); without it the native
device-resident path is unchanged.

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
