#pragma once

#include "core/arena.h"
#include "core/tensor.h"
#include "core/weight.h"

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace ninfer::ops {

/**
 * Sparse MoE whose routed experts live in host memory, with a device cache of selected experts.
 *
 * Mathematics (per token column h), identical to the Qwen3.5/qwen4exp routed MoE:
 *   probs = softmax(W_router h); ids = topk(probs, K) (lower id wins an exact tie)
 *   w_e   = probs[e] / sum_{ids} probs
 *   y     = sum_{e in ids} w_e * W_down[e](SiLU(W_gate[e] h) * W_up[e] h)
 *         + sigmoid(W_shared_score h) * W_shared_down(SiLU(W_shared_gate h) * W_shared_up h)
 *   destination <- destination + y        (AddResidual, one BF16 rounding)
 *
 * Execution: the GPU routes, looks up each selected expert in the runtime's slot map, and publishes
 * the selection to the host runtime with every expert held by a device slot marked as taken. It
 * then computes the shared expert and the slot-held experts while the host computes the others,
 * waits for the host partial and merges both partials into destination. Each selection is
 * computed exactly once; which side computes it does not change the formula. The handoff uses
 * stream memory operations on mapped host memory, so the whole sequence is CUDA Graph capturable.
 */
struct HybridSparseMoeWeights {
    Weight router_shared_gate;     // BF16 [E+1, H]; row E is the shared-expert score
    Weight shared_gate_up;         // row-split [2*Is, H]
    Weight shared_down;            // row-split [H, Is]
    Weight host_gate_up;           // row-split [E*2*I, H] in host memory
    Weight host_down;              // row-split [E*H, I] in host memory
    // Experts placed in device slots at startup; its size is the layer's slot count.
    std::vector<std::int32_t> resident;
    std::int32_t experts      = 0;
    std::int32_t top_k        = 0;
    std::int32_t hidden       = 0;
    std::int32_t intermediate = 0;
    std::int32_t shared_intermediate = 0;
    std::int32_t layer        = 0; // index into the host runtime's layer table
};

inline constexpr std::int32_t kHybridMoeMaxExperts = 512;
inline constexpr std::int32_t kHybridMoeMaxTopK    = 16;
// Mailbox layer word of a PLE table gather request.
inline constexpr std::uint32_t kHybridPleRequest   = 0xffffffffU;

/**
 * Program-owned device slots of one layer. Slot s holds the gate/up rows [s*2I, (s+1)*2I) and the
 * down rows [s*H, (s+1)*H) of one routed expert, stored exactly as in the host banks.
 */
struct HybridMoeSlotBanks {
    Weight gate_up; // row-split [slots*2*I, H]
    Weight down;    // row-split [slots*H, I]
    std::int32_t slots = 0;
};

// Device bytes of the layer's slot banks (weights.resident.size() slots), 256-byte aligned.
[[nodiscard]] std::uint64_t hybrid_moe_slot_bytes(const HybridSparseMoeWeights& weights);
// Binds slot banks over `storage` of hybrid_moe_slot_bytes(weights) device bytes.
[[nodiscard]] HybridMoeSlotBanks hybrid_moe_slot_banks(const HybridSparseMoeWeights& weights,
                                                       void* storage);

enum class HybridMoeResidency {
    Static,  // the startup residents stay in their slots
    Dynamic, // slots follow recent routing; host banks must be page-locked
};

/**
 * Program-owned host side: worker pool, per-layer host banks and slot maps, the mapped mailbox and,
 * with dynamic residency, the expert cache. One runtime serves the layers registered with add_layer
 * in a single stream order.
 *
 * Dynamic residency keeps a decayed routing frequency per expert (half-life of 16 routed token
 * columns). An expert routed to the host whose frequency reaches 3 and exceeds that of the least
 * frequent slot-held expert replaces it, at most two per layer and call. A replacement first
 * unmaps the victim, waits until every call that may have read the old map has completed (two
 * further requests), copies the new expert's rows into the slot on a separate stream, and maps
 * it once the copy has completed.
 */
class HybridMoeHostRuntime {
public:
    HybridMoeHostRuntime(std::int32_t threads, std::int32_t max_tokens, std::int32_t max_hidden,
                         std::int32_t max_top_k, HybridMoeResidency residency);

    // Writes routing_counts() as one whitespace-separated line per registered layer.
    void write_routing_counts(const std::string& path) const;
    ~HybridMoeHostRuntime();
    HybridMoeHostRuntime(const HybridMoeHostRuntime&)            = delete;
    HybridMoeHostRuntime& operator=(const HybridMoeHostRuntime&) = delete;

    // Registers a layer's host banks and device slots and uploads its startup residents. Returns
    // the index stored in HybridSparseMoeWeights::layer.
    std::int32_t add_layer(const HybridSparseMoeWeights& weights, const HybridMoeSlotBanks& slots);

    // Registers the Qwen4-Exp n-gram table served by ple_gather. `table` is a host-addressable
    // row-split matrix [rows / packing, packing * head_dim].
    void set_ple_table(const Weight& table, std::int32_t heads, std::int32_t head_dim,
                       std::int32_t packing);

    // Optional routing statistics: counts[layer][expert] of selections observed by the host.
    [[nodiscard]] std::vector<std::vector<std::uint64_t>> routing_counts() const;
    // Experts currently mapped to device slots of `layer`, in ascending id order.
    [[nodiscard]] std::vector<std::int32_t> mapped_experts(std::int32_t layer) const;
    // Completed slot replacements since startup.
    [[nodiscard]] std::uint64_t replacements() const;

    struct Impl;
    [[nodiscard]] Impl& impl() noexcept { return *impl_; }

private:
    std::unique_ptr<Impl> impl_;
};

[[nodiscard]] std::size_t hybrid_sparse_moe_workspace_bytes(const HybridSparseMoeWeights& weights,
                                                            std::int32_t max_tokens);

void hybrid_sparse_moe(const Tensor& x, const HybridSparseMoeWeights& weights,
                       HybridMoeHostRuntime& runtime, Tensor& destination,
                       WorkspaceArena& workspace, cudaStream_t stream);

/**
 * Gathers Qwen4-Exp n-gram embedding rows from the host table registered with set_ple_table.
 * rows I32 [heads,T] holds logical table rows; out BF16 [heads*head_dim,T] receives
 *   out[j*head_dim + d, t] = table_row(rows[j,t])[d]
 * as the exact stored value rounded once to BF16. Rows outside the table read as zero. The
 * handoff uses the same mapped mailbox as hybrid_sparse_moe and is CUDA Graph capturable.
 */
void ple_gather(const Tensor& rows, HybridMoeHostRuntime& runtime, Tensor& out,
                cudaStream_t stream);

} // namespace ninfer::ops
