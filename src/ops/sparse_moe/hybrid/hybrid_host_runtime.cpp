#include "ops/sparse_moe/hybrid/hybrid_host_runtime.h"

#include <cuda_runtime.h>
#include <immintrin.h>

#include <chrono>
#include <cstring>
#include <fstream>
#include <stdexcept>
#include <string>
#include <thread>

namespace ninfer::ops {
namespace {

constexpr std::size_t kLine = 64;

std::size_t align(std::size_t v, std::size_t a) { return (v + a - 1) / a * a; }

void check(cudaError_t status, const char* what) {
    if (status != cudaSuccess) {
        throw std::runtime_error(std::string("hybrid MoE host runtime: ") + what + ": " +
                                 cudaGetErrorString(status));
    }
}

float bf16_to_float(__nv_bfloat16 v) {
    std::uint16_t bits;
    std::memcpy(&bits, &v, sizeof(bits));
    const std::uint32_t word = static_cast<std::uint32_t>(bits) << 16;
    float out;
    std::memcpy(&out, &word, sizeof(out));
    return out;
}

} // namespace

HybridMoeHostRuntime::Impl::Impl(std::int32_t threads, std::int32_t max_tokens_,
                                 std::int32_t max_hidden_, std::int32_t max_top_k_)
    : max_tokens(max_tokens_), max_hidden(max_hidden_), max_top_k(max_top_k_), pool(threads),
      executor(pool) {
    if (!host::host_kernels_supported()) {
        throw std::runtime_error("MoE expert offload requires an x86 CPU with AVX2/FMA/F16C/BMI2");
    }
    const std::size_t T = static_cast<std::size_t>(max_tokens);
    const std::size_t H = static_cast<std::size_t>(max_hidden);
    const std::size_t K = static_cast<std::size_t>(max_top_k);
    std::size_t off     = 4 * kLine; // request, done, layer, tokens on separate lines
    const std::size_t ids_off   = off;
    off                         = align(off + T * K * sizeof(std::int32_t), kLine);
    const std::size_t alpha_off = off;
    off                         = align(off + T * K * sizeof(float), kLine);
    const std::size_t x_off     = off;
    off                         = align(off + T * H * sizeof(__nv_bfloat16), kLine);
    const std::size_t out_off   = off;
    off                         = align(off + T * H * sizeof(float), kLine);

    check(cudaHostAlloc(&host_base, off, cudaHostAllocMapped), "allocate mapped mailbox");
    std::memset(host_base, 0, off);
    void* device_base = nullptr;
    check(cudaHostGetDevicePointer(&device_base, host_base, 0), "map mailbox");
    auto* hb = static_cast<std::byte*>(host_base);
    auto* db = static_cast<std::byte*>(device_base);
    request  = reinterpret_cast<volatile std::uint32_t*>(hb);
    done     = reinterpret_cast<volatile std::uint32_t*>(hb + kLine);
    layer    = reinterpret_cast<volatile std::uint32_t*>(hb + 2 * kLine);
    tokens   = reinterpret_cast<volatile std::uint32_t*>(hb + 3 * kLine);
    ids      = reinterpret_cast<const std::int32_t*>(hb + ids_off);
    alpha    = reinterpret_cast<const float*>(hb + alpha_off);
    x        = reinterpret_cast<const __nv_bfloat16*>(hb + x_off);
    out      = reinterpret_cast<float*>(hb + out_off);
    device_.request_word = reinterpret_cast<CUdeviceptr>(db);
    device_.done_word    = reinterpret_cast<CUdeviceptr>(db + kLine);
    device_.layer_word   = reinterpret_cast<CUdeviceptr>(db + 2 * kLine);
    device_.tokens_word  = reinterpret_cast<CUdeviceptr>(db + 3 * kLine);
    device_.ids          = reinterpret_cast<int*>(db + ids_off);
    device_.alpha        = reinterpret_cast<float*>(db + alpha_off);
    device_.x            = reinterpret_cast<__nv_bfloat16*>(db + x_off);
    device_.out          = reinterpret_cast<const float*>(db + out_off);
    x_f32.resize(T * H);
    service = std::thread([this] { serve(); });
}

HybridMoeHostRuntime::Impl::~Impl() {
    stop.store(true, std::memory_order_release);
    if (service.joinable()) service.join();
    if (host_base) (void)cudaFreeHost(host_base);
}

void HybridMoeHostRuntime::Impl::serve() {
    using clock       = std::chrono::steady_clock;
    auto last_request = clock::now();
    std::uint32_t idle_polls = 0;
    bool sleeping            = false;
    while (!stop.load(std::memory_order_acquire)) {
        if (*request != 1U) {
            // Busy-poll while requests are recent; back off after one idle second.
            if (sleeping) {
                std::this_thread::sleep_for(std::chrono::microseconds(100));
            } else if ((++idle_polls & 0xfffU) == 0 &&
                       clock::now() - last_request > std::chrono::seconds(1)) {
                sleeping = true;
            } else {
                _mm_pause();
            }
            continue;
        }
        idle_polls        = 0;
        sleeping          = false;
        *request          = 0;
        const auto L      = static_cast<std::size_t>(*layer);
        const auto T      = static_cast<std::int32_t>(*tokens);
        auto& state       = layers.at(L);
        const auto H      = static_cast<std::int64_t>(state.banks.hidden);
        for (std::int64_t i = 0; i < T * H; ++i) {
            x_f32[static_cast<std::size_t>(i)] = bf16_to_float(x[i]);
        }
        const host::HostMoeJob job{T, state.top_k, x_f32.data(), ids, alpha,
                                   state.on_host.data(), out};
        executor.run(state.banks, job);
        for (std::int32_t i = 0; i < T * state.top_k; ++i) {
            ++state.counts[static_cast<std::size_t>(ids[i])];
        }
        _mm_sfence();
        *done        = 1U;
        last_request = clock::now();
    }
}

HybridMoeHostRuntime::HybridMoeHostRuntime(std::int32_t threads, std::int32_t max_tokens,
                                           std::int32_t max_hidden, std::int32_t max_top_k)
    : impl_(std::make_unique<Impl>(threads, max_tokens, max_hidden, max_top_k)) {}

HybridMoeHostRuntime::~HybridMoeHostRuntime() = default;

std::int32_t HybridMoeHostRuntime::add_layer(const HybridSparseMoeWeights& w) {
    if (w.hidden > impl_->max_hidden || w.top_k > impl_->max_top_k) {
        throw std::invalid_argument("hybrid MoE host runtime: layer exceeds mailbox geometry");
    }
    Impl::Layer layer;
    layer.banks.gate_up      = host::row_split_matrix(w.host_gate_up);
    layer.banks.down         = host::row_split_matrix(w.host_down);
    layer.banks.experts      = w.experts;
    layer.banks.hidden       = w.hidden;
    layer.banks.intermediate = w.intermediate;
    layer.top_k              = w.top_k;
    layer.on_host.resize(static_cast<std::size_t>(w.experts));
    for (std::int32_t e = 0; e < w.experts; ++e) {
        layer.on_host[static_cast<std::size_t>(e)] = w.slot_of_expert[static_cast<std::size_t>(e)] < 0;
    }
    layer.counts.assign(static_cast<std::size_t>(w.experts), 0);
    impl_->layers.push_back(std::move(layer));
    return static_cast<std::int32_t>(impl_->layers.size() - 1);
}

void HybridMoeHostRuntime::write_routing_counts(const std::string& path) const {
    std::ofstream out(path);
    if (!out) throw std::runtime_error("cannot write MoE routing counts to " + path);
    for (const auto& layer : impl_->layers) {
        for (std::size_t e = 0; e < layer.counts.size(); ++e) {
            out << (e ? " " : "") << layer.counts[e];
        }
        out << '\n';
    }
}

std::vector<std::vector<std::uint64_t>> HybridMoeHostRuntime::routing_counts() const {
    std::vector<std::vector<std::uint64_t>> out;
    for (const auto& layer : impl_->layers) out.push_back(layer.counts);
    return out;
}

} // namespace ninfer::ops
