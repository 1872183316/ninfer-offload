#include "ops/sparse_moe/hybrid/hybrid_host_runtime.h"

#include "core/weight_view.h"

#include <cuda_runtime.h>
#include <immintrin.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <stdexcept>
#include <string>
#include <thread>

namespace ninfer::ops {
namespace {

constexpr std::size_t kLine = 64;

// Expert cache policy (dynamic residency).
constexpr float kHalfLifeColumns = 16.0F; // routed token columns
constexpr float kAdmitFrequency  = 3.0F;
constexpr int kReplacementsPerCall = 2;
constexpr auto kCachePoll = std::chrono::microseconds(200);

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

std::int32_t selected_expert(std::int32_t id) { return id < 0 ? -1 - id : id; }

// Bytes per row of each row-split plane (storage-layouts section 3.7).
struct RowBytes {
    std::size_t codes  = 0;
    std::size_t high   = 0;
    std::size_t scales = 0;
};

RowBytes row_bytes(const Weight& w) {
    const auto groups = static_cast<std::size_t>(w.padded_shape[1] / w.group);
    const std::size_t high = w.qtype == QType::Q5_G64_FP16   ? 8
                             : w.qtype == QType::Q6_G64_FP16 ? 16
                                                             : 0;
    return {groups * 32, groups * high, groups * 2};
}

void copy_rows(const Weight& dst, std::int64_t dst_row, const Weight& src, std::int64_t src_row,
               std::int64_t rows, cudaStream_t stream) {
    const RowBytes rb = row_bytes(src);
    const auto plane  = [&](const void* d, const void* s, std::size_t bytes) {
        if (!bytes) return;
        check(cudaMemcpyAsync(static_cast<std::byte*>(const_cast<void*>(d)) + dst_row * bytes,
                              static_cast<const std::byte*>(s) + src_row * bytes,
                              static_cast<std::size_t>(rows) * bytes, cudaMemcpyHostToDevice,
                              stream),
              "copy expert rows");
    };
    plane(dst.qdata, src.qdata, rb.codes);
    plane(dst.qhigh, src.qhigh, rb.high);
    plane(dst.scales, src.scales, rb.scales);
}

bool page_locked(const void* p) {
    if (!p) return true;
    cudaPointerAttributes attributes{};
    if (cudaPointerGetAttributes(&attributes, p) != cudaSuccess) {
        (void)cudaGetLastError();
        return false;
    }
    return attributes.type == cudaMemoryTypeHost;
}

WeightGeometry slot_geometry(const Weight& bank, std::int64_t rows) {
    const std::array<std::uint64_t, 2> shape{static_cast<std::uint64_t>(rows),
                                             static_cast<std::uint64_t>(bank.k)};
    return weight_geometry(bank.qtype, QuantLayout::RowSplit, shape);
}

Weight slot_weight(const Weight& bank, std::int64_t rows, std::byte* data) {
    const WeightParent parent{slot_geometry(bank, rows), data, 0.0F};
    const WeightView view{parent.geometry.shape, {{&parent, 0, parent.geometry.elements}}};
    return native_weight(view);
}

} // namespace

std::uint64_t hybrid_moe_slot_bytes(const HybridSparseMoeWeights& w) {
    const auto slots = static_cast<std::int64_t>(w.resident.size());
    if (!slots) return 0;
    return align(slot_geometry(w.host_gate_up, slots * 2 * w.intermediate).bytes, 256) +
           align(slot_geometry(w.host_down, slots * w.hidden).bytes, 256);
}

HybridMoeSlotBanks hybrid_moe_slot_banks(const HybridSparseMoeWeights& w, void* storage) {
    HybridMoeSlotBanks out;
    out.slots = static_cast<std::int32_t>(w.resident.size());
    if (!out.slots) return out;
    auto* base       = static_cast<std::byte*>(storage);
    const auto gu    = slot_geometry(w.host_gate_up, std::int64_t{out.slots} * 2 * w.intermediate);
    out.gate_up      = slot_weight(w.host_gate_up, std::int64_t{out.slots} * 2 * w.intermediate, base);
    out.down         = slot_weight(w.host_down, std::int64_t{out.slots} * w.hidden,
                                   base + align(gu.bytes, 256));
    return out;
}

HybridMoeHostRuntime::Impl::Impl(std::int32_t threads, std::int32_t max_tokens_,
                                 std::int32_t max_hidden_, std::int32_t max_top_k_,
                                 HybridMoeResidency residency_)
    : max_tokens(max_tokens_), max_hidden(max_hidden_), max_top_k(max_top_k_),
      residency(residency_), pool(threads), executor(pool) {
    if (!host::host_kernels_supported()) {
        throw std::runtime_error("MoE expert offload requires an x86 CPU with AVX2/FMA/F16C/BMI2");
    }
    check(cudaGetDevice(&device), "query device");
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
    if (residency == HybridMoeResidency::Dynamic) cache = std::thread([this] { cache_loop(); });
}

HybridMoeHostRuntime::Impl::~Impl() {
    stop.store(true, std::memory_order_release);
    if (service.joinable()) service.join();
    if (cache.joinable()) cache.join();
    for (void* map : maps) (void)cudaFreeHost(map);
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
        idle_polls = 0;
        sleeping   = false;
        *request   = 0;
        received.fetch_add(1, std::memory_order_seq_cst);
        if (*layer == kHybridPleRequest) {
            gather_ple(static_cast<std::int32_t>(*tokens));
            _mm_sfence();
            *done        = 1U;
            last_request = clock::now();
            continue;
        }
        const auto L = static_cast<std::size_t>(*layer);
        const auto T = static_cast<std::int32_t>(*tokens);
        auto& state  = layers.at(L);
        const auto H = static_cast<std::int64_t>(state.banks.hidden);
        for (std::int64_t i = 0; i < T * H; ++i) {
            x_f32[static_cast<std::size_t>(i)] = bf16_to_float(x[i]);
        }
        const host::HostMoeJob job{T, state.top_k, x_f32.data(), ids, alpha, out};
        executor.run(state.banks, job);
        const std::int32_t n = T * state.top_k;
        for (std::int32_t i = 0; i < n; ++i) {
            ++state.counts[static_cast<std::size_t>(selected_expert(ids[i]))];
        }
        if (residency == HybridMoeResidency::Dynamic && state.slots.slots) {
            // Copied before completion: the next call overwrites the mailbox.
            Observation o{static_cast<std::int32_t>(L), T, std::vector<std::int32_t>(ids, ids + n)};
            const std::lock_guard lock(observed_mutex);
            observed.push_back(std::move(o));
        }
        _mm_sfence();
        *done        = 1U;
        last_request = clock::now();
    }
}

void HybridMoeHostRuntime::Impl::copy_expert(const Layer& l, std::int32_t expert,
                                             std::int32_t slot, cudaStream_t stream) const {
    const std::int64_t gu = 2LL * l.banks.intermediate;
    copy_rows(l.slots.gate_up, slot * gu, l.host_gate_up, expert * gu, gu, stream);
    copy_rows(l.slots.down, std::int64_t{slot} * l.banks.hidden, l.host_down,
              std::int64_t{expert} * l.banks.hidden, l.banks.hidden, stream);
}

void HybridMoeHostRuntime::Impl::observe(const Observation& o, std::vector<Replacement>& pending) {
    auto& l = layers.at(static_cast<std::size_t>(o.layer));
    const float decay = std::exp2(-static_cast<float>(o.tokens) / kHalfLifeColumns);
    for (auto& s : l.score) s *= decay;
    for (const auto id : o.ids) l.score[static_cast<std::size_t>(selected_expert(id))] += 1.0F;

    // Host-computed experts of this call that became frequent enough, most frequent first.
    std::vector<std::int32_t> candidates;
    for (const auto id : o.ids) {
        if (id < 0 || l.moving[id] || l.map[id] >= 0 || l.score[id] < kAdmitFrequency ||
            std::find(candidates.begin(), candidates.end(), id) != candidates.end()) {
            continue;
        }
        candidates.push_back(id);
    }
    std::sort(candidates.begin(), candidates.end(),
              [&](std::int32_t a, std::int32_t b) { return l.score[a] > l.score[b]; });
    int admitted = 0;
    for (const auto expert : candidates) {
        if (admitted == kReplacementsPerCall) break;
        std::int32_t victim = -1;
        for (std::int32_t s = 0; s < l.slots.slots; ++s) {
            const auto owner = l.owner[static_cast<std::size_t>(s)];
            if (owner >= 0 && (victim < 0 || l.score[owner] < l.score[l.owner[victim]])) victim = s;
        }
        if (victim < 0 || l.score[l.owner[victim]] >= l.score[expert]) break;
        l.map[l.owner[victim]]   = -1;
        l.owner[victim]          = -1;
        l.moving[expert]         = 1;
        // The unmapping must be visible before the request count it is measured against is read.
        std::atomic_thread_fence(std::memory_order_seq_cst);
        pending.push_back({o.layer, victim, expert, received.load(std::memory_order_seq_cst) + 2});
        ++admitted;
    }
}

void HybridMoeHostRuntime::Impl::cache_loop() {
    // Copies on this thread must not interfere with stream capture on the execution thread.
    cudaStreamCaptureMode mode = cudaStreamCaptureModeRelaxed;
    cudaStream_t stream        = nullptr;
    if (cudaSetDevice(device) != cudaSuccess ||
        cudaThreadExchangeStreamCaptureMode(&mode) != cudaSuccess ||
        cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking) != cudaSuccess) {
        std::fprintf(stderr, "hybrid MoE: expert cache disabled (CUDA setup failed)\n");
        return;
    }
    struct CopyGroup {
        cudaEvent_t event = nullptr;
        std::vector<Replacement> replacements;
    };
    std::vector<Replacement> pending;
    std::vector<CopyGroup> copying;
    std::vector<Observation> batch;
    bool failed = false;
    while (!stop.load(std::memory_order_acquire) && !failed) {
        {
            const std::lock_guard lock(observed_mutex);
            batch.swap(observed);
        }
        for (const auto& o : batch) observe(o, pending);
        const bool idle = batch.empty() && pending.empty() && copying.empty();
        batch.clear();

        // Slots that no call can still read receive their new expert.
        const auto now = received.load(std::memory_order_seq_cst);
        CopyGroup group;
        for (auto it = pending.begin(); it != pending.end();) {
            if (now < it->safe_after) {
                ++it;
                continue;
            }
            try {
                copy_expert(layers[static_cast<std::size_t>(it->layer)], it->expert, it->slot,
                            stream);
            } catch (const std::exception& e) {
                std::fprintf(stderr, "hybrid MoE: expert cache stopped: %s\n", e.what());
                failed = true;
                break;
            }
            group.replacements.push_back(*it);
            it = pending.erase(it);
        }
        if (!group.replacements.empty() && !failed) {
            if (cudaEventCreateWithFlags(&group.event, cudaEventDisableTiming) != cudaSuccess ||
                cudaEventRecord(group.event, stream) != cudaSuccess) {
                std::fprintf(stderr, "hybrid MoE: expert cache stopped: event failure\n");
                failed = true;
            } else {
                copying.push_back(std::move(group));
            }
        }
        // Completed copies become visible to routing.
        for (auto it = copying.begin(); it != copying.end() && !failed;) {
            const auto status = cudaEventQuery(it->event);
            if (status == cudaErrorNotReady) break; // groups complete in stream order
            if (status != cudaSuccess) {
                std::fprintf(stderr, "hybrid MoE: expert cache stopped: %s\n",
                             cudaGetErrorString(status));
                failed = true;
                break;
            }
            for (const auto& r : it->replacements) {
                auto& l                 = layers[static_cast<std::size_t>(r.layer)];
                l.owner[r.slot]         = r.expert;
                l.moving[r.expert]      = 0;
                l.map[r.expert]         = static_cast<std::int16_t>(r.slot);
            }
            replaced.fetch_add(it->replacements.size(), std::memory_order_relaxed);
            (void)cudaEventDestroy(it->event);
            it = copying.erase(it);
        }
        if (idle) std::this_thread::sleep_for(kCachePoll);
        else std::this_thread::sleep_for(kCachePoll / 4);
    }
    (void)cudaStreamSynchronize(stream);
    for (auto& g : copying) (void)cudaEventDestroy(g.event);
    (void)cudaStreamDestroy(stream);
}

void HybridMoeHostRuntime::Impl::gather_ple(std::int32_t T) {
    const auto& p        = *ple;
    const std::int64_t n = static_cast<std::int64_t>(T) * p.heads;
    const std::int64_t width = static_cast<std::int64_t>(p.heads) * p.head_dim;
    // Row j of token t lands at out[t * width + (j % heads) * head_dim].
    pool.parallel_for(static_cast<std::int32_t>(n), [&](std::int32_t i) {
        const std::int64_t row = ids[i];
        float* dst = out + (i / p.heads) * width + static_cast<std::int64_t>(i % p.heads) * p.head_dim;
        if (row < 0 || row >= p.rows) {
            std::fill(dst, dst + p.head_dim, 0.0F);
            return;
        }
        host::decode_row_range(p.table, static_cast<std::int32_t>(row / p.packing),
                               static_cast<std::int32_t>(row % p.packing) * p.head_dim, p.head_dim,
                               dst);
    });
}

void HybridMoeHostRuntime::set_ple_table(const Weight& table, std::int32_t heads,
                                         std::int32_t head_dim, std::int32_t packing) {
    if (heads <= 0 || head_dim <= 0 || packing <= 0 || heads > impl_->max_top_k ||
        heads * head_dim > impl_->max_hidden || table.k != packing * head_dim) {
        throw std::invalid_argument("hybrid host runtime: PLE table exceeds mailbox geometry");
    }
    Impl::PleTable p;
    p.table    = host::row_split_matrix(table);
    p.heads    = heads;
    p.head_dim = head_dim;
    p.packing  = packing;
    p.rows     = static_cast<std::int64_t>(table.n) * packing;
    impl_->ple = p;
}

HybridMoeHostRuntime::HybridMoeHostRuntime(std::int32_t threads, std::int32_t max_tokens,
                                           std::int32_t max_hidden, std::int32_t max_top_k,
                                           HybridMoeResidency residency)
    : impl_(std::make_unique<Impl>(threads, max_tokens, max_hidden, max_top_k, residency)) {}

HybridMoeHostRuntime::~HybridMoeHostRuntime() = default;

std::int32_t HybridMoeHostRuntime::add_layer(const HybridSparseMoeWeights& w,
                                             const HybridMoeSlotBanks& slots) {
    if (w.hidden > impl_->max_hidden || w.top_k > impl_->max_top_k ||
        w.experts > kHybridMoeMaxExperts) {
        throw std::invalid_argument("hybrid MoE host runtime: layer exceeds mailbox geometry");
    }
    if (slots.slots != static_cast<std::int32_t>(w.resident.size()) ||
        (slots.slots && (slots.gate_up.n != slots.slots * 2 * w.intermediate ||
                         slots.down.n != slots.slots * w.hidden ||
                         slots.gate_up.qtype != w.host_gate_up.qtype ||
                         slots.down.qtype != w.host_down.qtype))) {
        throw std::invalid_argument("hybrid MoE host runtime: slot banks differ from the layer");
    }
    if (impl_->residency == HybridMoeResidency::Dynamic && slots.slots) {
        for (const Weight* bank : {&w.host_gate_up, &w.host_down}) {
            if (!page_locked(bank->qdata) || !page_locked(bank->qhigh) ||
                !page_locked(bank->scales)) {
                throw std::invalid_argument(
                    "hybrid MoE host runtime: dynamic residency requires page-locked host banks");
            }
        }
    }
    Impl::Layer layer;
    layer.banks.gate_up      = host::row_split_matrix(w.host_gate_up);
    layer.banks.down         = host::row_split_matrix(w.host_down);
    layer.banks.experts      = w.experts;
    layer.banks.hidden       = w.hidden;
    layer.banks.intermediate = w.intermediate;
    layer.host_gate_up       = w.host_gate_up;
    layer.host_down          = w.host_down;
    layer.slots              = slots;
    layer.top_k              = w.top_k;
    layer.counts.assign(static_cast<std::size_t>(w.experts), 0);
    layer.owner.assign(static_cast<std::size_t>(slots.slots), -1);
    layer.moving.assign(static_cast<std::size_t>(w.experts), 0);
    layer.score.assign(static_cast<std::size_t>(w.experts), 0.0F);

    void* map = nullptr;
    check(cudaHostAlloc(&map, static_cast<std::size_t>(w.experts) * sizeof(std::int16_t),
                        cudaHostAllocMapped),
          "allocate slot map");
    impl_->maps.push_back(map);
    void* device_map = nullptr;
    check(cudaHostGetDevicePointer(&device_map, map, 0), "map slot map");
    layer.map        = static_cast<volatile std::int16_t*>(map);
    layer.device_map = static_cast<const std::int16_t*>(device_map);
    for (std::int32_t e = 0; e < w.experts; ++e) layer.map[e] = -1;

    cudaStream_t stream = nullptr;
    check(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking), "create upload stream");
    try {
        for (std::int32_t s = 0; s < slots.slots; ++s) {
            const auto e = w.resident[static_cast<std::size_t>(s)];
            if (e < 0 || e >= w.experts || layer.map[e] >= 0) {
                throw std::invalid_argument("hybrid MoE host runtime: invalid resident experts");
            }
            impl_->copy_expert(layer, e, s, stream);
            layer.owner[static_cast<std::size_t>(s)] = e;
            layer.map[e]                             = static_cast<std::int16_t>(s);
        }
        check(cudaStreamSynchronize(stream), "upload resident experts");
    } catch (...) {
        (void)cudaStreamSynchronize(stream);
        (void)cudaStreamDestroy(stream);
        throw;
    }
    check(cudaStreamDestroy(stream), "destroy upload stream");
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

std::vector<std::int32_t> HybridMoeHostRuntime::mapped_experts(std::int32_t layer) const {
    const auto& l = impl_->layers.at(static_cast<std::size_t>(layer));
    std::vector<std::int32_t> out;
    for (std::int32_t e = 0; e < l.banks.experts; ++e) {
        if (l.map[e] >= 0) out.push_back(e);
    }
    return out;
}

std::uint64_t HybridMoeHostRuntime::replacements() const {
    return impl_->replaced.load(std::memory_order_relaxed);
}

} // namespace ninfer::ops
