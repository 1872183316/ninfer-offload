// Hybrid (device-resident + host-offloaded experts) sparse MoE against a naive FP64 oracle.
#include "ninfer/ops/hybrid_sparse_moe.h"

#include "core/arena.h"
#include "ops/quantized_weight.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <numeric>
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
    explicit DeviceBytes(const void* host, std::size_t n) {
        CUDA_OK(cudaMalloc(&ptr, n));
        CUDA_OK(cudaMemcpy(ptr, host, n, cudaMemcpyHostToDevice));
    }
    ~DeviceBytes() { cudaFree(ptr); }
};

// Independent materialization of selected rows as a standalone row-split payload (section 3.7).
PackedWeight replica_rows(const PackedWeight& parent, const std::vector<std::int32_t>& rows) {
    const auto spec     = detail::quant_spec(parent.weight.qtype);
    const auto gpr      = parent.weight.padded_shape[1] / spec.group_size;
    const std::size_t cb = static_cast<std::size_t>(gpr) * detail::nibble_bytes_per_group(spec);
    const std::size_t hb = static_cast<std::size_t>(gpr) * detail::high_bytes_per_group(spec);
    const std::size_t sb = static_cast<std::size_t>(gpr) * 2;
    const std::size_t n  = rows.size();
    PackedWeight out     = parent;
    out.code_plane_bytes   = n * cb;
    out.high_plane_offset  = detail::align_up_size(n * cb, 256);
    out.high_plane_bytes   = n * hb;
    out.scale_plane_offset = out.high_plane_offset + detail::align_up_size(n * hb, 256);
    out.scale_plane_bytes  = n * sb;
    out.payload.assign(out.scale_plane_offset + n * sb, 0);
    for (std::size_t i = 0; i < n; ++i) {
        const std::size_t r = static_cast<std::size_t>(rows[i]);
        std::memcpy(out.payload.data() + i * cb, parent.payload.data() + r * cb, cb);
        if (hb) {
            std::memcpy(out.payload.data() + out.high_plane_offset + i * hb,
                        parent.payload.data() + parent.high_plane_offset + r * hb, hb);
        }
        std::memcpy(out.payload.data() + out.scale_plane_offset + i * sb,
                    parent.payload.data() + parent.scale_plane_offset + r * sb, sb);
    }
    out.weight.n               = static_cast<std::int32_t>(n);
    out.weight.shape[0]        = static_cast<std::int32_t>(n);
    out.weight.padded_shape[0] = static_cast<std::int32_t>(n);
    out.weight.payload_bytes   = out.payload.size();
    return out;
}

int check(std::int32_t E, std::int32_t K, std::int32_t H, std::int32_t I, std::int32_t Is,
          std::int32_t T, int resident_modulo, std::uint32_t seed) {
    PatternedWeightOptions options;
    options.row_split_scale = RowSplitScalePattern::Small;
    options.row_split_codes = RowSplitCodePattern::Hashed;
    const PackedWeight gu   = make_patterned_weight(QType::Q4_G64_FP16, E * 2 * I, H, seed, options);
    const PackedWeight dn   = make_patterned_weight(QType::Q5_G64_FP16, E * H, I, seed + 1, options);
    const PackedWeight sgu  = make_patterned_weight(QType::Q8_G32_FP16, 2 * Is, H, seed + 2, options);
    const PackedWeight sdn  = make_patterned_weight(QType::Q8_G32_FP16, H, Is, seed + 3, options);

    // Router: column 0 carries a distinct per-expert score; the rest is small noise.
    std::vector<std::int32_t> perm(static_cast<std::size_t>(E));
    std::iota(perm.begin(), perm.end(), 0);
    for (std::int32_t i = E - 1; i > 0; --i) {
        std::swap(perm[i], perm[detail::mix64(seed + i) % static_cast<std::uint64_t>(i + 1)]);
    }
    std::vector<std::uint16_t> router(static_cast<std::size_t>(E + 1) * H);
    for (std::int32_t r = 0; r <= E; ++r) {
        for (std::int32_t k = 0; k < H; ++k) {
            float v = static_cast<float>(hashed(seed + 7, r * H + k)) * 0.002F;
            if (k == 0) v = r < E ? 0.25F * static_cast<float>(perm[r]) / E * 8.0F : 0.3F;
            router[r * H + k] = to_bf16(v);
        }
    }
    // Tokens differ in column 0 sign pattern so they pick different expert sets.
    std::vector<std::uint16_t> x(static_cast<std::size_t>(T) * H);
    std::vector<std::uint16_t> residual(static_cast<std::size_t>(T) * H);
    for (std::int32_t t = 0; t < T; ++t) {
        for (std::int32_t k = 0; k < H; ++k) {
            float v = static_cast<float>(hashed(seed + 11 + t, k)) * 0.5F;
            if (k == 0) v = (t % 2 ? -1.0F : 1.0F) * (1.0F + 0.25F * t);
            x[t * H + k]        = to_bf16(v);
            residual[t * H + k] = to_bf16(static_cast<float>(hashed(seed + 13 + t, k)));
        }
    }

    // Resident subset and its replicas.
    std::vector<std::int16_t> slot(static_cast<std::size_t>(E), -1);
    std::vector<std::int32_t> gu_rows;
    std::vector<std::int32_t> dn_rows;
    std::int16_t next = 0;
    for (std::int32_t e = 0; e < E; ++e) {
        if (resident_modulo && e % resident_modulo == 0) {
            slot[e] = next++;
            for (std::int32_t r = 0; r < 2 * I; ++r) gu_rows.push_back(e * 2 * I + r);
            for (std::int32_t r = 0; r < H; ++r) dn_rows.push_back(e * H + r);
        }
    }

    DeviceBytes d_router(router.data(), router.size() * 2);
    DeviceBytes d_sgu(sgu.payload.data(), sgu.payload.size());
    DeviceBytes d_sdn(sdn.payload.data(), sdn.payload.size());
    std::vector<std::uint8_t> host_gu = gu.payload;
    std::vector<std::uint8_t> host_dn = dn.payload;

    ops::HybridSparseMoeWeights w;
    w.router_shared_gate.qdata = d_router.ptr;
    w.router_shared_gate.qtype = QType::BF16;
    w.shared_gate_up           = sgu.device_weight(d_sgu.ptr);
    w.shared_down              = sdn.device_weight(d_sdn.ptr);
    w.host_gate_up             = gu.device_weight(host_gu.data());
    w.host_down                = dn.device_weight(host_dn.data());
    w.slot_of_expert           = slot;
    w.experts                  = E;
    w.top_k                    = K;
    w.hidden                   = H;
    w.intermediate             = I;
    w.shared_intermediate      = Is;

    std::unique_ptr<DeviceBytes> d_rgu;
    std::unique_ptr<DeviceBytes> d_rdn;
    PackedWeight rgu;
    PackedWeight rdn;
    if (next) {
        rgu   = replica_rows(gu, gu_rows);
        rdn   = replica_rows(dn, dn_rows);
        d_rgu = std::make_unique<DeviceBytes>(rgu.payload.data(), rgu.payload.size());
        d_rdn = std::make_unique<DeviceBytes>(rdn.payload.data(), rdn.payload.size());
        w.device_gate_up = rgu.device_weight(d_rgu->ptr);
        w.device_down    = rdn.device_weight(d_rdn->ptr);
    }

    ops::HybridMoeHostRuntime runtime(6, 16, H, K);
    w.layer = runtime.add_layer(w);

    DeviceBytes d_x(x.data(), x.size() * 2);
    DeviceBytes d_dst(residual.data(), residual.size() * 2);
    DeviceArena workspace(ops::hybrid_sparse_moe_workspace_bytes(w, T) + (1 << 16));
    cudaStream_t stream;
    CUDA_OK(cudaStreamCreate(&stream));
    Tensor tx(d_x.ptr, DType::BF16, {H, T});
    Tensor td(d_dst.ptr, DType::BF16, {H, T});
    ops::hybrid_sparse_moe(tx, w, runtime, td, workspace, stream);
    CUDA_OK(cudaStreamSynchronize(stream));
    std::vector<std::uint16_t> got(residual.size());
    CUDA_OK(cudaMemcpy(got.data(), d_dst.ptr, got.size() * 2, cudaMemcpyDeviceToHost));
    CUDA_OK(cudaStreamDestroy(stream));

    // FP64 oracle.
    double err2 = 0;
    double ref2 = 0;
    double worst = 0;
    double max_abs = 0;
    for (std::int32_t t = 0; t < T; ++t) {
        std::vector<double> logit(static_cast<std::size_t>(E + 1));
        for (std::int32_t r = 0; r <= E; ++r) {
            double s = 0;
            for (std::int32_t k = 0; k < H; ++k) s += from_bf16(router[r * H + k]) * from_bf16(x[t * H + k]);
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
                double a = 0;
                double b = 0;
                for (std::int32_t k = 0; k < H; ++k) {
                    const double xv = from_bf16(x[t * H + k]);
                    a += logical_weight_fp64(g, static_cast<std::int32_t>(grow + j), k) * xv;
                    b += logical_weight_fp64(g, static_cast<std::int32_t>(grow + width + j), k) * xv;
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
            max_abs = std::max(max_abs, std::fabs(e));
        }
    }
    const double rel = std::sqrt(err2 / std::max(ref2, 1e-300));
    const bool ok    = rel < 4e-3 && worst <= 1.0;
    std::printf("%s E=%d K=%d H=%d I=%d Is=%d T=%d resident=1/%d relL2=%.3g max_abs=%.3g worst_pointwise=%.3f\n",
                ok ? "ok  " : "FAIL", E, K, H, I, Is, T, resident_modulo, rel, max_abs, worst);
    return ok ? 0 : 1;
}

} // namespace

int main() {
    int count = 0;
    if (cudaGetDeviceCount(&count) != cudaSuccess || count == 0) {
        std::printf("SKIP: no CUDA device\n");
        return 77;
    }
    int failures = 0;
    failures += check(64, 8, 512, 128, 128, 1, 4, 21);   // mixed residency
    failures += check(64, 8, 512, 128, 128, 3, 4, 22);
    failures += check(64, 8, 512, 128, 128, 2, 0, 23);   // everything on the host
    failures += check(64, 8, 512, 128, 128, 2, 1, 24);   // everything resident
    failures += check(96, 10, 768, 160, 160, 5, 3, 25);  // Flash-Next-like: top-10, I % 64 != 0
    std::printf("%s (%d failures)\n", failures ? "FAILED" : "PASSED", failures);
    return failures ? 1 : 0;
}
