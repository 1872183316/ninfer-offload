// Host routed-expert executor against a naive FP64 oracle of the MoE routed sum.
#include "ops/sparse_moe/host/host_moe.h"

#include "ops/quantized_weight.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <vector>

using namespace ninfer;
using namespace ninfer::test;
using namespace ninfer::test::quantized_weight;

namespace {

float input_value(std::uint32_t seed, std::int32_t t, std::int32_t k) {
    const std::uint64_t h = detail::mix64((static_cast<std::uint64_t>(seed) << 40) ^
                                          (static_cast<std::uint64_t>(t) << 24) ^
                                          static_cast<std::uint64_t>(k));
    const float v = static_cast<float>(static_cast<std::int64_t>(h % 65536) - 32768) / 16384.0F;
    return detail::bf16_to_f32(static_cast<std::uint16_t>(detail::float_bits(v) >> 16));
}

int check(QType gu_type, QType down_type, std::int32_t E, std::int32_t H, std::int32_t I,
          std::int32_t top_k, std::int32_t T, std::int32_t threads, std::uint32_t seed) {
    PatternedWeightOptions options;
    options.row_split_scale = RowSplitScalePattern::Small;
    options.row_split_codes = RowSplitCodePattern::Hashed;
    const PackedWeight gu   = make_patterned_weight(gu_type, E * 2 * I, H, seed, options);
    const PackedWeight dn   = make_patterned_weight(down_type, E * H, I, seed + 1, options);
    std::vector<std::uint8_t> gu_bytes = gu.payload;
    std::vector<std::uint8_t> dn_bytes = dn.payload;

    ops::host::HostExpertBanks banks;
    banks.gate_up      = ops::host::row_split_matrix(gu.device_weight(gu_bytes.data()));
    banks.down         = ops::host::row_split_matrix(dn.device_weight(dn_bytes.data()));
    banks.experts      = E;
    banks.hidden       = H;
    banks.intermediate = I;

    std::vector<float> x(static_cast<std::size_t>(T) * H);
    for (std::int32_t t = 0; t < T; ++t)
        for (std::int32_t k = 0; k < H; ++k) x[t * H + k] = input_value(seed, t, k);
    std::vector<std::int32_t> ids(static_cast<std::size_t>(T) * top_k);
    std::vector<float> alpha(ids.size());
    for (std::int32_t t = 0; t < T; ++t) {
        for (std::int32_t s = 0; s < top_k; ++s) {
            // Distinct experts per token; tokens overlap so shared experts batch across columns.
            ids[t * top_k + s]   = (s * 5 + t * 3) % E;
            alpha[t * top_k + s] = 0.05F + 0.1F * static_cast<float>((s + t) % 7);
        }
    }
    // Selections of some experts are taken elsewhere (negative ids) and contribute nothing.
    const auto on_host = [](std::int32_t e) { return (e % 3) != 1; };
    std::vector<std::int32_t> job_ids(ids.size());
    for (std::size_t i = 0; i < ids.size(); ++i) job_ids[i] = on_host(ids[i]) ? ids[i] : -1 - ids[i];

    std::vector<float> out(static_cast<std::size_t>(T) * H, NAN);
    ops::host::HostThreadPool pool(threads);
    ops::host::HostMoeExecutor exec(pool);
    ops::host::HostMoeJob job{T, top_k, x.data(), job_ids.data(), alpha.data(), out.data()};
    exec.run(banks, job);

    // FP64 oracle with exact logical weight decode.
    int failures  = 0;
    double worst  = 0.0;
    double ref_l2 = 0.0;
    double err_l2 = 0.0;
    for (std::int32_t t = 0; t < T; ++t) {
        std::vector<double> y(static_cast<std::size_t>(H), 0.0);
        std::vector<double> yabs(static_cast<std::size_t>(H), 0.0);
        for (std::int32_t s = 0; s < top_k; ++s) {
            const std::int32_t e = ids[t * top_k + s];
            if (!on_host(e)) continue;
            std::vector<double> act(static_cast<std::size_t>(I));
            for (std::int32_t j = 0; j < I; ++j) {
                double g = 0.0;
                double u = 0.0;
                for (std::int32_t k = 0; k < H; ++k) {
                    g += logical_weight_fp64(gu, e * 2 * I + j, k) * x[t * H + k];
                    u += logical_weight_fp64(gu, e * 2 * I + I + j, k) * x[t * H + k];
                }
                act[j] = g / (1.0 + std::exp(-g)) * u;
            }
            for (std::int32_t r = 0; r < H; ++r) {
                double d = 0.0;
                double a = 0.0;
                for (std::int32_t j = 0; j < I; ++j) {
                    const double p = logical_weight_fp64(dn, e * H + r, j) * act[j];
                    d += p;
                    a += std::fabs(p);
                }
                y[r] += alpha[t * top_k + s] * d;
                yabs[r] += alpha[t * top_k + s] * a;
            }
        }
        for (std::int32_t r = 0; r < H; ++r) {
            const double got = out[t * H + r];
            const double err = std::fabs(got - y[r]);
            // FP32 gate/up dots propagate a relative error ~H*u through SwiGLU into the down dot.
            const double lim = 4.0 * (H + I) * std::ldexp(1.0, -24) * yabs[r] + 1e-6;
            worst            = std::max(worst, err / lim);
            ref_l2 += y[r] * y[r];
            err_l2 += err * err;
            if (!(err <= lim)) {
                if (failures < 5)
                    std::printf("  FAIL t=%d r=%d got=%.8g ref=%.8g lim=%.3g\n", t, r, got, y[r],
                                lim);
                ++failures;
            }
        }
    }
    std::printf("%s E=%d H=%d I=%d k=%d T=%d threads=%d worst_err/bound=%.3f relL2=%.3g\n",
                failures ? "FAIL" : "ok  ", E, H, I, top_k, T, threads, worst,
                std::sqrt(err_l2 / std::max(ref_l2, 1e-300)));
    return failures;
}

} // namespace

int main() {
    if (!ops::host::host_kernels_supported()) {
        std::printf("SKIP: CPU lacks AVX2/FMA/F16C/BMI2\n");
        return 77;
    }
    int failures = 0;
    // 35B-A3B profile (Q4 gate/up, Q5 down) and Flash-Next-like widths (H 2560, I 640, top-10).
    failures += check(QType::Q4_G64_FP16, QType::Q5_G64_FP16, 12, 2048, 512, 8, 1, 4, 11);
    failures += check(QType::Q4_G64_FP16, QType::Q5_G64_FP16, 12, 2048, 512, 8, 3, 7, 12);
    failures += check(QType::Q4_G64_FP16, QType::Q6_G64_FP16, 12, 2048, 512, 8, 2, 12, 13);
    failures += check(QType::Q8_G32_FP16, QType::Q8_G32_FP16, 12, 2048, 512, 8, 2, 3, 14);
    failures += check(QType::Q4_G64_FP16, QType::Q5_G64_FP16, 14, 2560, 640, 10, 1, 12, 15);
    failures += check(QType::Q4_G64_FP16, QType::Q5_G64_FP16, 14, 2560, 640, 10, 5, 12, 16);
    failures += check(QType::Q4_G64_FP16, QType::Q5_G64_FP16, 14, 2560, 640, 10, 9, 1, 17);
    std::printf("%s (%d failures)\n", failures ? "FAILED" : "PASSED", failures);
    return failures ? 1 : 0;
}
