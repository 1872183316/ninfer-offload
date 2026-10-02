// Hybrid (device slots + host-offloaded experts) sparse MoE against a naive FP64 oracle, with fixed
// slots and with slots that the expert cache replaces between and during calls.
#include "ninfer/ops/hybrid_sparse_moe.h"

#include "core/arena.h"
#include "ops/quantized_weight.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <numeric>
#include <thread>
#include <vector>

using namespace ninfer;
using namespace ninfer::test;
using namespace ninfer::test::quantized_weight;

namespace {

#define CUDA_OK(x)                                                                                 \
    do {                                                                                           \
        const cudaError_t e = (x);                                                                 \
        if (e != cudaSuccess) {                                                                    \
            std::printf("CUDA error %s at %s:%d\n", cudaGetErrorString(e), __FILE__, __LINE__);    \
            std::exit(1);                                                                          \
        }                                                                                          \
    } while (0)

std::uint16_t to_bf16(float v) {
    std::uint32_t bits;
    std::memcpy(&bits, &v, 4);
    const std::uint32_t rounding = 0x7fffU + ((bits >> 16) & 1U);
    return static_cast<std::uint16_t>((bits + rounding) >> 16);
}

float from_bf16(std::uint16_t v) {
    const std::uint32_t bits = static_cast<std::uint32_t>(v) << 16;
    float f;
    std::memcpy(&f, &bits, 4);
    return f;
}

double hashed(std::uint64_t seed, std::uint64_t i) {
    return static_cast<double>(detail::mix64(seed * 0x9e3779b97f4a7c15ULL + i) % 20001) / 10000.0 -
           1.0;
}

struct DeviceBytes {
    void* ptr = nullptr;
    explicit DeviceBytes(std::size_t n) { CUDA_OK(cudaMalloc(&ptr, n ? n : 1)); }
    DeviceBytes(const void* host, std::size_t n) : DeviceBytes(n) {
        CUDA_OK(cudaMemcpy(ptr, host, n, cudaMemcpyHostToDevice));
    }
    ~DeviceBytes() { cudaFree(ptr); }
};

struct Case {
    std::int32_t E, K, H, I, Is;
    std::int32_t slots; // startup residents: experts 0, E/slots, 2E/slots, ...
    ops::HybridMoeResidency residency;
    std::uint32_t seed;
};

// One call: T token columns from input pattern `set` (set 0 and 1 favour opposite experts).
struct Call {
    std::int32_t T;
    int set;
};

struct Fixture {
    Case c;
    PackedWeight gu, dn, sgu, sdn;
    std::vector<std::uint16_t> router;
    std::vector<std::int32_t> perm;

    explicit Fixture(const Case& c_) : c(c_) {
        PatternedWeightOptions options;
        options.row_split_scale = RowSplitScalePattern::Small;
        options.row_split_codes = RowSplitCodePattern::Hashed;
        gu  = make_patterned_weight(QType::Q4_G64_FP16, c.E * 2 * c.I, c.H, c.seed, options);
        dn  = make_patterned_weight(QType::Q5_G64_FP16, c.E * c.H, c.I, c.seed + 1, options);
        sgu = make_patterned_weight(QType::Q8_G32_FP16, 2 * c.Is, c.H, c.seed + 2, options);
        sdn = make_patterned_weight(QType::Q8_G32_FP16, c.H, c.Is, c.seed + 3, options);
        // Router: column 0 carries a distinct per-expert score; the rest is small noise.
        perm.resize(static_cast<std::size_t>(c.E));
        std::iota(perm.begin(), perm.end(), 0);
        for (std::int32_t i = c.E - 1; i > 0; --i) {
            std::swap(perm[i], perm[detail::mix64(c.seed + i) % static_cast<std::uint64_t>(i + 1)]);
        }
        router.resize(static_cast<std::size_t>(c.E + 1) * c.H);
        for (std::int32_t r = 0; r <= c.E; ++r) {
            for (std::int32_t k = 0; k < c.H; ++k) {
                float v = static_cast<float>(hashed(c.seed + 7, r * c.H + k)) * 0.002F;
                if (k == 0) v = r < c.E ? 0.25F * static_cast<float>(perm[r]) / c.E * 8.0F : 0.3F;
                router[r * c.H + k] = to_bf16(v);
            }
        }
    }

    // Tokens differ in the column-0 sign so they pick different expert sets.
    void inputs(const Call& call, std::vector<std::uint16_t>& x,
                std::vector<std::uint16_t>& residual) const {
        x.resize(static_cast<std::size_t>(call.T) * c.H);
        residual.resize(x.size());
        for (std::int32_t t = 0; t < call.T; ++t) {
            for (std::int32_t k = 0; k < c.H; ++k) {
                float v = static_cast<float>(hashed(c.seed + 11 + t + 97 * call.set, k)) * 0.5F;
                if (k == 0) v = ((t + call.set) % 2 ? -1.0F : 1.0F) * (1.0F + 0.25F * t);
                x[t * c.H + k]        = to_bf16(v);
                residual[t * c.H + k] = to_bf16(static_cast<float>(hashed(c.seed + 13 + t, k)));
            }
        }
    }

    // Worst pointwise error ratio and relative L2 error of `got` against the FP64 oracle.
    std::pair<double, double> compare(const std::vector<std::uint16_t>& x,
                                      const std::vector<std::uint16_t>& residual,
                                      const std::vector<std::uint16_t>& got, std::int32_t T) const {
        const auto E = c.E, K = c.K, H = c.H, I = c.I, Is = c.Is;
        double err2 = 0, ref2 = 0, worst = 0;
        for (std::int32_t t = 0; t < T; ++t) {
            std::vector<double> logit(static_cast<std::size_t>(E + 1));
            for (std::int32_t r = 0; r <= E; ++r) {
                double s = 0;
                for (std::int32_t k = 0; k < H; ++k)
                    s += from_bf16(router[r * H + k]) * from_bf16(x[t * H + k]);
                logit[r] = s;
            }
            std::vector<std::int32_t> order(static_cast<std::size_t>(E));
            std::iota(order.begin(), order.end(), 0);
            std::stable_sort(order.begin(), order.end(),
                             [&](std::int32_t a, std::int32_t b) { return logit[a] > logit[b]; });
            double z = 0;
            for (std::int32_t s = 0; s < K; ++s) z += std::exp(logit[order[s]] - logit[order[0]]);
            std::vector<double> y(static_cast<std::size_t>(H), 0.0);
            const auto expert = [&](const PackedWeight& g, const PackedWeight& d, std::int64_t grow,
                                    std::int64_t drow, std::int32_t width, double scale) {
                std::vector<double> act(static_cast<std::size_t>(width));
                for (std::int32_t j = 0; j < width; ++j) {
                    double a = 0, b = 0;
                    for (std::int32_t k = 0; k < H; ++k) {
                        const double xv = from_bf16(x[t * H + k]);
                        a += logical_weight_fp64(g, static_cast<std::int32_t>(grow + j), k) * xv;
                        b += logical_weight_fp64(g, static_cast<std::int32_t>(grow + width + j), k) *
                             xv;
                    }
                    act[j] = a / (1 + std::exp(-a)) * b;
                }
                for (std::int32_t r = 0; r < H; ++r) {
                    double s = 0;
                    for (std::int32_t j = 0; j < width; ++j)
                        s += logical_weight_fp64(d, static_cast<std::int32_t>(drow + r), j) * act[j];
                    y[r] += scale * s;
                }
            };
            for (std::int32_t s = 0; s < K; ++s) {
                const std::int32_t e = order[s];
                expert(gu, dn, static_cast<std::int64_t>(e) * 2 * I, static_cast<std::int64_t>(e) * H,
                       I, std::exp(logit[e] - logit[order[0]]) / z);
            }
            expert(sgu, sdn, 0, 0, Is, 1.0 / (1.0 + std::exp(-logit[E])));
            for (std::int32_t r = 0; r < H; ++r) {
                const double ref = from_bf16(residual[t * H + r]) + y[r];
                const double e   = from_bf16(got[t * H + r]) - ref;
                err2 += e * e;
                ref2 += ref * ref;
                // One BF16 rounding of the result (<= |ref| 2^-8) plus the FP32 accumulation and
                // shared-expert gross bound used by the native SparseMoe A16 criterion (4e-3).
                worst = std::max(worst, std::fabs(e) / (std::fabs(ref) * std::ldexp(1.0, -8) + 4e-3));
            }
        }
        return {worst, std::sqrt(err2 / std::max(ref2, 1e-300))};
    }
};

// Runs `calls` in order and checks every output. With dynamic residency, `expect_moved` requires
// that the cache replaced slots and finally maps experts favoured by the last input set.
int check(const Case& c, const std::vector<Call>& calls, bool expect_moved) {
    const Fixture f(c);
    std::vector<std::uint8_t> host_gu = f.gu.payload;
    std::vector<std::uint8_t> host_dn = f.dn.payload;
    const HostPageLock lock_gu(host_gu.data(), host_gu.size());
    const HostPageLock lock_dn(host_dn.data(), host_dn.size());
    DeviceBytes d_router(f.router.data(), f.router.size() * 2);
    DeviceBytes d_sgu(f.sgu.payload.data(), f.sgu.payload.size());
    DeviceBytes d_sdn(f.sdn.payload.data(), f.sdn.payload.size());

    ops::HybridSparseMoeWeights w;
    w.router_shared_gate.qdata = d_router.ptr;
    w.router_shared_gate.qtype = QType::BF16;
    w.shared_gate_up           = f.sgu.device_weight(d_sgu.ptr);
    w.shared_down              = f.sdn.device_weight(d_sdn.ptr);
    w.host_gate_up             = f.gu.device_weight(host_gu.data());
    w.host_down                = f.dn.device_weight(host_dn.data());
    for (std::int32_t s = 0; s < c.slots; ++s) w.resident.push_back(s * (c.E / c.slots));
    w.experts             = c.E;
    w.top_k               = c.K;
    w.hidden              = c.H;
    w.intermediate        = c.I;
    w.shared_intermediate = c.Is;

    std::int32_t max_t = 1;
    for (const auto& call : calls) max_t = std::max(max_t, call.T);
    DeviceBytes slot_storage(ops::hybrid_moe_slot_bytes(w));
    ops::HybridMoeHostRuntime runtime(6, max_t, c.H, c.K, c.residency);
    w.layer = runtime.add_layer(w, ops::hybrid_moe_slot_banks(w, slot_storage.ptr));
    const auto initial = runtime.mapped_experts(w.layer);

    DeviceArena workspace(ops::hybrid_sparse_moe_workspace_bytes(w, max_t) + (1 << 16));
    cudaStream_t stream;
    CUDA_OK(cudaStreamCreate(&stream));
    DeviceBytes d_x(static_cast<std::size_t>(max_t) * c.H * 2);
    DeviceBytes d_dst(static_cast<std::size_t>(max_t) * c.H * 2);
    double worst = 0, rel = 0;
    std::vector<std::uint16_t> x, residual, got;
    for (const auto& call : calls) {
        f.inputs(call, x, residual);
        CUDA_OK(cudaMemcpy(d_x.ptr, x.data(), x.size() * 2, cudaMemcpyHostToDevice));
        CUDA_OK(cudaMemcpy(d_dst.ptr, residual.data(), residual.size() * 2, cudaMemcpyHostToDevice));
        Tensor tx(d_x.ptr, DType::BF16, {c.H, call.T});
        Tensor td(d_dst.ptr, DType::BF16, {c.H, call.T});
        ops::hybrid_sparse_moe(tx, w, runtime, td, workspace, stream);
        CUDA_OK(cudaStreamSynchronize(stream));
        got.resize(residual.size());
        CUDA_OK(cudaMemcpy(got.data(), d_dst.ptr, got.size() * 2, cudaMemcpyDeviceToHost));
        const auto [w_call, rel_call] = f.compare(x, residual, got, call.T);
        worst = std::max(worst, w_call);
        rel   = std::max(rel, rel_call);
        // Lets the cache thread act between some calls; others follow back to back.
        if (call.set == 1) std::this_thread::sleep_for(std::chrono::microseconds(300));
    }
    CUDA_OK(cudaStreamDestroy(stream));

    bool moved_ok = true;
    const auto final_mapped = runtime.mapped_experts(w.layer);
    if (expect_moved) {
        // The last calls favour the experts of set 1; their top choices must now hold slots.
        moved_ok = runtime.replacements() > 0 && final_mapped != initial;
    } else {
        moved_ok = runtime.replacements() == 0 && final_mapped == initial;
    }
    const bool ok = rel < 4e-3 && worst <= 1.0 && moved_ok;
    std::printf("%s E=%d K=%d H=%d I=%d Is=%d slots=%d %s calls=%zu replacements=%llu "
                "max_relL2=%.3g worst_pointwise=%.3f\n",
                ok ? "ok  " : "FAIL", c.E, c.K, c.H, c.I, c.Is, c.slots,
                c.residency == ops::HybridMoeResidency::Dynamic ? "dynamic" : "static",
                calls.size(), static_cast<unsigned long long>(runtime.replacements()), rel, worst);
    return ok ? 0 : 1;
}

} // namespace

int main() {
    int count = 0;
    if (cudaGetDeviceCount(&count) != cudaSuccess || count == 0) {
        std::printf("SKIP: no CUDA device\n");
        return 77;
    }
    using R = ops::HybridMoeResidency;
    int failures = 0;
    failures += check({64, 8, 512, 128, 128, 16, R::Static, 21}, {{1, 0}}, false); // mixed
    failures += check({64, 8, 512, 128, 128, 16, R::Static, 22}, {{3, 0}}, false);
    failures += check({64, 8, 512, 128, 128, 0, R::Static, 23}, {{2, 0}}, false);  // all host
    failures += check({64, 8, 512, 128, 128, 64, R::Static, 24}, {{2, 0}}, false); // all slots
    failures += check({96, 10, 768, 160, 160, 32, R::Static, 25}, {{5, 0}}, false); // top-10
    // Dynamic: slots move towards the experts of set 0, then of set 1, while every call (decode,
    // multi-column and back-to-back) must still equal the oracle.
    std::vector<Call> calls;
    for (int i = 0; i < 24; ++i) calls.push_back({1, 0});
    for (int i = 0; i < 24; ++i) calls.push_back({i % 3 ? 1 : 3, 1});
    failures += check({64, 8, 512, 128, 128, 8, R::Dynamic, 26}, calls, true);
    failures += check({96, 10, 768, 160, 160, 12, R::Dynamic, 27}, calls, true);
    std::printf("%s (%d failures)\n", failures ? "FAILED" : "PASSED", failures);
    return failures ? 1 : 0;
}
