// Runtime-geometry linear route (shapes without a tuned schedule) against an FP64 oracle.
#include "ninfer/ops/linear.h"

#include "core/arena.h"
#include "ops/direct_bf16_weight.h"
#include "ops/quantized_weight.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <functional>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

using namespace ninfer;
using namespace ninfer::test;
using namespace ninfer::test::quantized_weight;

namespace {

float bf16_value(std::uint16_t bits) {
    const std::uint32_t word = static_cast<std::uint32_t>(bits) << 16;
    float f;
    std::memcpy(&f, &word, 4);
    return f;
}

std::uint16_t input_bits(std::uint32_t seed, std::int64_t i) {
    const std::uint64_t h = detail::mix64((static_cast<std::uint64_t>(seed) << 32) ^ i);
    const float v = static_cast<float>(static_cast<std::int64_t>(h % 4001) - 2000) / 1000.0F;
    std::uint32_t word;
    std::memcpy(&word, &v, 4);
    return static_cast<std::uint16_t>(word >> 16);
}

int check(const char* label, const Weight& host_view, const std::vector<std::uint8_t>& payload,
          const std::function<double(std::int32_t, std::int32_t)>& weight_at, std::int32_t t,
          std::uint32_t seed) {
    const std::int32_t n = host_view.n;
    const std::int32_t k = host_view.k;
    void* d_payload      = nullptr;
    cudaMalloc(&d_payload, payload.size());
    cudaMemcpy(d_payload, payload.data(), payload.size(), cudaMemcpyHostToDevice);
    Weight w = host_view;
    const auto shift = [&](const void* p) -> const void* {
        return p ? static_cast<const std::uint8_t*>(d_payload) +
                       (static_cast<const std::uint8_t*>(p) - payload.data())
                 : nullptr;
    };
    w.payload = d_payload;
    w.qdata   = shift(host_view.qdata);
    w.qhigh   = shift(host_view.qhigh);
    w.scales  = shift(host_view.scales);

    std::vector<std::uint16_t> x(static_cast<std::size_t>(k) * t);
    for (std::size_t i = 0; i < x.size(); ++i) x[i] = input_bits(seed, static_cast<std::int64_t>(i));
    void* d_x   = nullptr;
    void* d_out = nullptr;
    cudaMalloc(&d_x, x.size() * 2);
    cudaMalloc(&d_out, static_cast<std::size_t>(n) * t * 2);
    cudaMemcpy(d_x, x.data(), x.size() * 2, cudaMemcpyHostToDevice);
    Tensor tx(d_x, DType::BF16, {k, t});
    Tensor to(d_out, DType::BF16, {n, t});
    ops::linear(tx, w, to, nullptr);
    std::vector<std::uint16_t> out(static_cast<std::size_t>(n) * t);
    const cudaError_t status = cudaDeviceSynchronize();
    cudaMemcpy(out.data(), d_out, out.size() * 2, cudaMemcpyDeviceToHost);
    cudaFree(d_payload);
    cudaFree(d_x);
    cudaFree(d_out);
    if (status != cudaSuccess) {
        std::printf("FAIL %s: %s\n", label, cudaGetErrorString(status));
        return 1;
    }
    int failures = 0;
    double worst = 0;
    // Oracle rows: all rows for small N, a strided sample otherwise.
    const std::int32_t stride = n > 2048 ? 37 : 1;
    for (std::int32_t r = 0; r < n; r += stride) {
        for (std::int32_t c = 0; c < t; ++c) {
            double ref = 0;
            double abs = 0;
            for (std::int32_t i = 0; i < k; ++i) {
                const double p = weight_at(r, i) * bf16_value(x[static_cast<std::size_t>(c) * k + i]);
                ref += p;
                abs += std::fabs(p);
            }
            const double got = bf16_value(out[static_cast<std::size_t>(c) * n + r]);
            const double lim = std::ldexp(std::fabs(ref), -8) + 4.0 * k * std::ldexp(abs, -24) + 1e-30;
            const double err = std::fabs(got - ref);
            worst            = std::max(worst, err / lim);
            if (!(err <= lim)) {
                if (failures < 3)
                    std::printf("  FAIL %s r=%d t=%d got=%.6g ref=%.6g\n", label, r, c, got, ref);
                ++failures;
            }
        }
    }
    std::printf("%s %s N=%d K=%d T=%d worst=%.3f\n", failures ? "FAIL" : "ok  ", label, n, k, t,
                worst);
    return failures;
}

int quantized(QType q, const char* label, std::int32_t n, std::int32_t k, std::int32_t t) {
    PatternedWeightOptions options;
    options.row_split_scale = RowSplitScalePattern::Small;
    options.row_split_codes = RowSplitCodePattern::Hashed;
    const PackedWeight packed = make_patterned_weight(q, n, k, 1000U + n + k, options);
    const Weight view         = packed.device_weight(const_cast<std::uint8_t*>(packed.payload.data()));
    return check(label, view, packed.payload,
                 [&](std::int32_t r, std::int32_t c) { return logical_weight_fp64(packed, r, c); },
                 t, 7U + t);
}

int bf16(std::int32_t n, std::int32_t k, std::int32_t t) {
    const auto host = direct_bf16_weight::make_patterned(n, k, 77U);
    std::vector<std::uint8_t> payload(host.bits.size() * 2);
    std::memcpy(payload.data(), host.bits.data(), payload.size());
    const Weight view = host.device_weight(payload.data());
    return check("BF16", view, payload,
                 [&](std::int32_t r, std::int32_t c) {
                     return static_cast<double>(bf16_value(host.bits[static_cast<std::size_t>(r) * k + c]));
                 },
                 t, 11U + t);
}

} // namespace

int main() {
    int count = 0;
    if (cudaGetDeviceCount(&count) != cudaSuccess || count == 0) {
        std::printf("SKIP: no CUDA device\n");
        return 77;
    }
    int failures = 0;
    // Qwen4-Exp projection shapes: hyper-connection down/up/inject, GDN/attention inputs and
    // outputs, PLE key/value, indexer and head.
    for (const std::int32_t t : {1, 3, 8, 13}) {
        failures += quantized(QType::Q8_G32_FP16, "Q8", 320, 10240, t);
        failures += quantized(QType::Q8_G32_FP16, "Q8", 10240, 320, t);
        failures += quantized(QType::Q8_G32_FP16, "Q8", 4, 10240, t);
        failures += quantized(QType::Q8_G32_FP16, "Q8", 640, 2560, t);
        failures += quantized(QType::Q8_G32_FP16, "Q8", 2560, 6144, t);
        failures += quantized(QType::Q4_G64_FP16, "Q4", 1280, 2560, t);
        failures += quantized(QType::Q5_G64_FP16, "Q5", 2560, 640, t);
        failures += quantized(QType::Q6_G64_FP16, "Q6", 1000, 2560, t);
        failures += bf16(48, 2560, t);
    }
    std::printf("%s (%d failures)\n", failures ? "FAILED" : "PASSED", failures);
    return failures ? 1 : 0;
}
