#pragma once

#include "ninfer/ops/hybrid_sparse_moe.h"
#include "ops/sparse_moe/host/host_moe.h"

#include <cuda.h>
#include <cuda_bf16.h>

#include <atomic>
#include <cstdint>
#include <mutex>
#include <optional>
#include <thread>
#include <vector>

namespace ninfer::ops {

// Device-visible addresses of the mapped mailbox.
struct HybridMailboxDevice {
    CUdeviceptr request_word = 0; // GPU writes 1; host consumes and writes 0
    CUdeviceptr done_word    = 0; // host writes 1; GPU waits then writes 0
    CUdeviceptr layer_word   = 0;
    CUdeviceptr tokens_word  = 0;
    int* ids                 = nullptr; // selected id, or -1 - id when a device slot computes it
    float* alpha             = nullptr;
    __nv_bfloat16* x         = nullptr;
    const float* out         = nullptr;
};

struct HybridMoeHostRuntime::Impl {
    Impl(std::int32_t threads, std::int32_t max_tokens, std::int32_t max_hidden,
         std::int32_t max_top_k, HybridMoeResidency residency);
    ~Impl();

    [[nodiscard]] HybridMailboxDevice device_mailbox() const noexcept { return device_; }

    struct Layer {
        host::HostExpertBanks banks;
        Weight host_gate_up;
        Weight host_down;
        HybridMoeSlotBanks slots;
        std::int32_t top_k = 0;
        std::vector<std::uint64_t> counts;
        // Mapped slot map [E]: the slot holding the expert, or -1. Written by the host only.
        volatile std::int16_t* map     = nullptr;
        const std::int16_t* device_map = nullptr;
        // Cache-thread state.
        std::vector<std::int32_t> owner;  // [slots] mapped expert, or -1 while being replaced
        std::vector<std::uint8_t> moving; // [E] expert is being copied into a slot
        std::vector<float> score;         // [E] decayed routing frequency
    };

    void serve();
    void gather_ple(std::int32_t tokens);

    // Expert cache (dynamic residency).
    struct Observation {
        std::int32_t layer  = 0;
        std::int32_t tokens = 0;
        std::vector<std::int32_t> ids;
    };
    struct Replacement {
        std::int32_t layer       = 0;
        std::int32_t slot        = 0;
        std::int32_t expert      = 0;
        std::uint64_t safe_after = 0; // requests to take before the slot may be overwritten
        cudaEvent_t copied       = nullptr;
    };
    void cache_loop();
    void observe(const Observation& observation, std::vector<Replacement>& pending);
    void copy_expert(const Layer& layer, std::int32_t expert, std::int32_t slot,
                     cudaStream_t stream) const;

    // Qwen4-Exp n-gram table: logical row r is packed row r / packing, columns
    // [(r % packing) * head_dim, +head_dim).
    struct PleTable {
        host::RowSplitMatrix table;
        std::int32_t heads    = 0;
        std::int32_t head_dim = 0;
        std::int32_t packing  = 0;
        std::int64_t rows     = 0;
    };
    std::optional<PleTable> ple;

    std::int32_t max_tokens      = 0;
    std::int32_t max_hidden      = 0;
    std::int32_t max_top_k       = 0;
    HybridMoeResidency residency = HybridMoeResidency::Static;
    int device                   = 0;

    // Mapped host allocation; words are separated by cache lines.
    void* host_base = nullptr;
    volatile std::uint32_t* request = nullptr;
    volatile std::uint32_t* done    = nullptr;
    volatile std::uint32_t* layer   = nullptr;
    volatile std::uint32_t* tokens  = nullptr;
    const std::int32_t* ids         = nullptr;
    const float* alpha              = nullptr;
    const __nv_bfloat16* x          = nullptr;
    float* out                      = nullptr;
    HybridMailboxDevice device_;

    std::vector<Layer> layers;
    std::vector<void*> maps; // mapped slot-map allocations, one per layer
    std::vector<float> x_f32;
    host::HostThreadPool pool;
    host::HostMoeExecutor executor;
    std::atomic<bool> stop{false};
    // Requests taken by the service thread. A call that routed before a map change has completed
    // once two further requests have been taken (one stream; each call ends with its request).
    std::atomic<std::uint64_t> received{0};
    std::atomic<std::uint64_t> replaced{0};
    std::mutex observed_mutex;
    std::vector<Observation> observed;
    std::thread service;
    std::thread cache;
};

} // namespace ninfer::ops
