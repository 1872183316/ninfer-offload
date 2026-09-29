#pragma once

#include "ninfer/ops/hybrid_sparse_moe.h"
#include "ops/sparse_moe/host/host_moe.h"

#include <cuda.h>
#include <cuda_bf16.h>

#include <atomic>
#include <cstdint>
#include <thread>
#include <vector>

namespace ninfer::ops {

// Device-visible addresses of the mapped mailbox.
struct HybridMailboxDevice {
    CUdeviceptr request_word = 0; // GPU writes 1; host consumes and writes 0
    CUdeviceptr done_word    = 0; // host writes 1; GPU waits then writes 0
    CUdeviceptr layer_word   = 0;
    CUdeviceptr tokens_word  = 0;
    int* ids                 = nullptr;
    float* alpha             = nullptr;
    __nv_bfloat16* x         = nullptr;
    const float* out         = nullptr;
};

struct HybridMoeHostRuntime::Impl {
    Impl(std::int32_t threads, std::int32_t max_tokens, std::int32_t max_hidden,
         std::int32_t max_top_k);
    ~Impl();

    [[nodiscard]] HybridMailboxDevice device_mailbox() const noexcept { return device_; }

    struct Layer {
        host::HostExpertBanks banks;
        std::int32_t top_k = 0;
        std::vector<std::uint8_t> on_host;
        std::vector<std::uint64_t> counts;
    };

    void serve();

    std::int32_t max_tokens = 0;
    std::int32_t max_hidden = 0;
    std::int32_t max_top_k  = 0;

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
    std::vector<float> x_f32;
    host::HostThreadPool pool;
    host::HostMoeExecutor executor;
    std::atomic<bool> stop{false};
    std::atomic<bool> serving{false};
    std::thread service;
};

} // namespace ninfer::ops
