#include "ops/sparse_moe/host/host_moe.h"

#include <immintrin.h>

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace ninfer::ops::host {
namespace {

constexpr int kSpinIterations      = 1 << 16;
constexpr std::int32_t kGateRows   = 32; // phase-1 rows per work item
constexpr std::int32_t kDownRows   = 64; // phase-2 output rows per work item

float silu(float v) { return v / (1.0F + std::exp(-v)); }

} // namespace

HostThreadPool::HostThreadPool(std::int32_t threads) : threads_(std::max(1, threads)) {
    workers_.reserve(static_cast<std::size_t>(threads_ - 1));
    for (std::int32_t i = 1; i < threads_; ++i) workers_.emplace_back([this] { worker_loop(); });
}

HostThreadPool::~HostThreadPool() {
    stop_.store(true, std::memory_order_release);
    epoch_.fetch_add(1, std::memory_order_acq_rel);
    epoch_.notify_all();
    for (auto& w : workers_) w.join();
}

void HostThreadPool::drain(std::uint32_t generation) {
    for (;;) {
        std::uint64_t claim = next_.load(std::memory_order_acquire);
        for (;;) {
            if (static_cast<std::uint32_t>(claim >> 32) != generation) return;
            if (static_cast<std::int32_t>(claim & 0xffffffffU) >= items_) return;
            if (next_.compare_exchange_weak(claim, claim + 1, std::memory_order_acq_rel,
                                            std::memory_order_acquire)) {
                break;
            }
        }
        // The claimed item belongs to `generation`; its fields cannot change until it completes.
        (*fn_)(static_cast<std::int32_t>(claim & 0xffffffffU));
        done_.fetch_add(1, std::memory_order_release);
    }
}

void HostThreadPool::worker_loop() {
    std::uint64_t seen = 0;
    for (;;) {
        std::uint64_t e = epoch_.load(std::memory_order_acquire);
        for (int spins = 0; e == seen; e = epoch_.load(std::memory_order_acquire)) {
            if (++spins < kSpinIterations) {
                _mm_pause();
            } else {
                epoch_.wait(seen, std::memory_order_acquire);
            }
        }
        seen = e;
        if (stop_.load(std::memory_order_acquire)) return;
        drain(static_cast<std::uint32_t>(next_.load(std::memory_order_acquire) >> 32));
    }
}

void HostThreadPool::parallel_for(std::int32_t items,
                                  const std::function<void(std::int32_t)>& fn) {
    if (items <= 0) return;
    if (threads_ == 1 || items == 1) {
        for (std::int32_t i = 0; i < items; ++i) fn(i);
        return;
    }
    const std::uint32_t generation = ++generation_;
    fn_    = &fn;
    items_ = items;
    done_.store(0, std::memory_order_relaxed);
    next_.store(static_cast<std::uint64_t>(generation) << 32, std::memory_order_release);
    epoch_.fetch_add(1, std::memory_order_acq_rel);
    epoch_.notify_all();
    drain(generation);
    while (done_.load(std::memory_order_acquire) < items) _mm_pause();
}

void HostMoeExecutor::run(const HostExpertBanks& banks, const HostMoeJob& job) {
    const std::int32_t T = job.tokens;
    const std::int32_t H = banks.hidden;
    const std::int32_t I = banks.intermediate;
    if (T <= 0 || job.top_k <= 0 || !job.x || !job.ids || !job.alpha || !job.on_host || !job.out) {
        throw std::invalid_argument("host MoE: invalid job");
    }
    if (banks.gate_up.k != H || banks.down.k != I || banks.gate_up.rows != banks.experts * 2 * I ||
        banks.down.rows != banks.experts * H) {
        throw std::invalid_argument("host MoE: bank geometry mismatch");
    }

    // Unique host experts and the tokens that select them.
    unique_.clear();
    token_list_.assign(static_cast<std::size_t>(T) * T * job.top_k, 0);
    token_alpha_.assign(token_list_.size(), 0.0F);
    token_count_.assign(static_cast<std::size_t>(T) * job.top_k, 0);
    for (std::int32_t t = 0; t < T; ++t) {
        for (std::int32_t s = 0; s < job.top_k; ++s) {
            const std::int32_t e = job.ids[t * job.top_k + s];
            if (e < 0 || e >= banks.experts) throw std::invalid_argument("host MoE: invalid id");
            if (!job.on_host[e]) continue;
            auto it = std::find(unique_.begin(), unique_.end(), e);
            std::size_t u = static_cast<std::size_t>(it - unique_.begin());
            if (it == unique_.end()) unique_.push_back(e);
            const std::int32_t n = token_count_[u]++;
            token_list_[u * T + n]  = t;
            token_alpha_[u * T + n] = job.alpha[t * job.top_k + s];
        }
    }
    const std::int32_t U  = static_cast<std::int32_t>(unique_.size());
    const std::int32_t Kg = prepared_activation_floats(banks.gate_up);
    const std::int32_t Kd = prepared_activation_floats(banks.down);
    std::fill(job.out, job.out + static_cast<std::int64_t>(T) * H, 0.0F);
    if (U == 0) return;

    x_prepared_.resize(static_cast<std::size_t>(U) * T * Kg);
    act_prepared_.assign(static_cast<std::size_t>(U) * T * Kd, 0.0F);
    for (std::int32_t u = 0; u < U; ++u) {
        for (std::int32_t n = 0; n < token_count_[u]; ++n) {
            prepare_activation(banks.gate_up, job.x + static_cast<std::int64_t>(token_list_[u * T + n]) * H,
                               x_prepared_.data() + (static_cast<std::size_t>(u) * T + n) * Kg);
        }
    }

    // Phase 1: gate/up rows and SwiGLU, written in the down bank's prepared order.
    const std::int32_t gate_blocks = (I + kGateRows - 1) / kGateRows;
    pool_.parallel_for(U * gate_blocks, [&](std::int32_t item) {
        const std::int32_t u  = item / gate_blocks;
        const std::int32_t j0 = (item % gate_blocks) * kGateRows;
        const std::int32_t n  = std::min(kGateRows, I - j0);
        const std::int32_t Tu = token_count_[u];
        const std::int32_t e  = unique_[u];
        float gate[kGateRows * 8];
        float up[kGateRows * 8];
        const float* xs = x_prepared_.data() + static_cast<std::size_t>(u) * T * Kg;
        float* act      = act_prepared_.data() + static_cast<std::size_t>(u) * T * Kd;
        for (std::int32_t t0 = 0; t0 < Tu; t0 += 8) {
            const std::int32_t tt = std::min(8, Tu - t0);
            rows_dot(banks.gate_up, e * 2 * I + j0, n, xs + static_cast<std::size_t>(t0) * Kg, Kg,
                     tt, gate, tt);
            rows_dot(banks.gate_up, e * 2 * I + I + j0, n, xs + static_cast<std::size_t>(t0) * Kg,
                     Kg, tt, up, tt);
            for (std::int32_t r = 0; r < n; ++r) {
                const std::int32_t pos = prepared_position(banks.down, j0 + r);
                for (std::int32_t t = 0; t < tt; ++t) {
                    act[static_cast<std::size_t>(t0 + t) * Kd + pos] =
                        silu(gate[r * tt + t]) * up[r * tt + t];
                }
            }
        }
    });

    // Phase 2: each item owns output rows [r0, r0+n) for every token.
    const std::int32_t down_blocks = (H + kDownRows - 1) / kDownRows;
    pool_.parallel_for(down_blocks, [&](std::int32_t item) {
        const std::int32_t r0 = item * kDownRows;
        const std::int32_t n  = std::min(kDownRows, H - r0);
        float partial[kDownRows * 8];
        for (std::int32_t u = 0; u < U; ++u) {
            const std::int32_t Tu = token_count_[u];
            const std::int32_t e  = unique_[u];
            const float* act = act_prepared_.data() + static_cast<std::size_t>(u) * T * Kd;
            for (std::int32_t t0 = 0; t0 < Tu; t0 += 8) {
                const std::int32_t tt = std::min(8, Tu - t0);
                rows_dot(banks.down, e * H + r0, n, act + static_cast<std::size_t>(t0) * Kd, Kd, tt,
                         partial, tt);
                for (std::int32_t t = 0; t < tt; ++t) {
                    const std::int32_t token = token_list_[u * T + t0 + t];
                    const float a            = token_alpha_[u * T + t0 + t];
                    float* dst = job.out + static_cast<std::int64_t>(token) * H + r0;
                    for (std::int32_t r = 0; r < n; ++r) dst[r] += a * partial[r * tt + t];
                }
            }
        }
    });
}

} // namespace ninfer::ops::host
