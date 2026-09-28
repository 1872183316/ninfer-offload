#pragma once

#include "ops/sparse_moe/host/host_rowsplit_dot.h"

#include <atomic>
#include <cstdint>
#include <functional>
#include <memory>
#include <thread>
#include <vector>

namespace ninfer::ops::host {

// Routed expert banks of one MoE layer in host memory. Expert e owns gate rows
// [e*2I, e*2I+I), up rows [e*2I+I, (e+1)*2I) and down rows [e*H, (e+1)*H).
struct HostExpertBanks {
    RowSplitMatrix gate_up;
    RowSplitMatrix down;
    std::int32_t experts      = 0;
    std::int32_t hidden       = 0;
    std::int32_t intermediate = 0;
};

// One layer's host share of routed experts for T token columns.
struct HostMoeJob {
    std::int32_t tokens = 0;
    std::int32_t top_k  = 0;
    const float* x      = nullptr;           // [T][H] represented activations as FP32
    const std::int32_t* ids = nullptr;       // [T][top_k] selected logical expert ids
    const float* alpha  = nullptr;           // [T][top_k] normalized routing weights
    const std::uint8_t* on_host = nullptr;   // [E] nonzero: the host computes this expert
    float* out          = nullptr;           // [T][H] overwritten with the host partial sum
};

// Persistent worker pool. The calling thread participates in every parallel phase.
class HostThreadPool {
public:
    explicit HostThreadPool(std::int32_t threads);
    ~HostThreadPool();
    HostThreadPool(const HostThreadPool&)            = delete;
    HostThreadPool& operator=(const HostThreadPool&) = delete;

    [[nodiscard]] std::int32_t threads() const noexcept { return threads_; }

    // Runs fn(item) for every item in [0,items) and returns after all complete.
    void parallel_for(std::int32_t items, const std::function<void(std::int32_t)>& fn);

private:
    void worker_loop();
    void drain(std::uint32_t generation);

    std::int32_t threads_ = 1;
    std::vector<std::thread> workers_;
    const std::function<void(std::int32_t)>* fn_ = nullptr;
    std::int32_t items_                          = 0;
    std::uint32_t generation_                    = 0;
    alignas(64) std::atomic<std::uint64_t> epoch_{0};
    // (generation << 32) | next item. Claims are CAS-validated against the generation so a
    // worker left over from a finished phase can never take an item of the next one.
    alignas(64) std::atomic<std::uint64_t> next_{0};
    alignas(64) std::atomic<std::int32_t> done_{0};
    std::atomic<bool> stop_{false};
};

// Computes out[t] = sum over host experts e selected by token t of
//   alpha[t,e] * W_down[e](SiLU(W_gate[e] x_t) * (W_up[e] x_t))
// with exact stored-weight decode and FP32 arithmetic. Experts not marked on_host contribute
// nothing. Scratch storage is owned by the executor and reused across calls.
class HostMoeExecutor {
public:
    explicit HostMoeExecutor(HostThreadPool& pool) : pool_(pool) {}
    void run(const HostExpertBanks& banks, const HostMoeJob& job);

private:
    HostThreadPool& pool_;
    std::vector<float> x_prepared_;   // [U][Tu][Kp_gate_up]
    std::vector<float> act_prepared_; // [U][Tu][Kp_down]
    std::vector<std::int32_t> unique_;
    std::vector<std::int32_t> token_list_;  // [U][T] tokens using expert u
    std::vector<float> token_alpha_;        // [U][T]
    std::vector<std::int32_t> token_count_; // [U]
};

} // namespace ninfer::ops::host
