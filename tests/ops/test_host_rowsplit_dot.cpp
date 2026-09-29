// Host AVX2 row-split dot products against the independent FP64 logical-decode oracle.
#include "ops/sparse_moe/host/host_rowsplit_dot.h"

#include "ops/quantized_weight.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>

using namespace ninfer;
using namespace ninfer::test;
using namespace ninfer::test::quantized_weight;

namespace {

const char* name(QType q) {
    switch (q) {
    case QType::Q4_G64_FP16:
        return "Q4";
    case QType::Q5_G64_FP16:
        return "Q5";
    case QType::Q6_G64_FP16:
        return "Q6";
    case QType::Q8_G32_FP16:
        return "Q8";
    default:
        return "?";
    }
}

float input_value(std::uint32_t seed, std::int32_t t, std::int32_t k) {
    const std::uint64_t h = detail::mix64((static_cast<std::uint64_t>(seed) << 40) ^
                                          (static_cast<std::uint64_t>(t) << 24) ^
                                          static_cast<std::uint64_t>(k));
    // Represented BF16 activations in [-4, 4).
    const float v = static_cast<float>(static_cast<std::int64_t>(h % 65536) - 32768) / 8192.0F;
    return detail::bf16_to_f32(static_cast<std::uint16_t>(detail::float_bits(v) >> 16));
}

// FP32 dot of length K with any association: |err| <= K * u * sum|w x| (u = 2^-24), doubled
// for the scale product and the final horizontal reduction.
double bound(std::int32_t k, double abs_sum) { return 2.0 * k * std::ldexp(1.0, -24) * abs_sum; }

int check(QType qtype, std::int32_t n, std::int32_t k, std::int32_t tokens, std::uint32_t seed) {
    PatternedWeightOptions options;
    options.row_split_scale = RowSplitScalePattern::Small;
    options.row_split_codes = RowSplitCodePattern::Hashed;
    const PackedWeight packed = make_patterned_weight(qtype, n, k, seed, options);
    std::vector<std::uint8_t> payload = packed.payload;
    const Weight w                    = packed.device_weight(payload.data());
    const auto m                      = ops::host::row_split_matrix(w);

    const std::int32_t stride = ops::host::prepared_activation_floats(m);
    std::vector<float> logical(static_cast<std::size_t>(tokens) * k);
    std::vector<float> prepared(static_cast<std::size_t>(tokens) * stride);
    for (std::int32_t t = 0; t < tokens; ++t) {
        for (std::int32_t c = 0; c < k; ++c) logical[t * k + c] = input_value(seed, t, c);
        ops::host::prepare_activation(m, logical.data() + t * k, prepared.data() + t * stride);
    }

    // A row span starting mid-matrix exercises parent row addressing.
    const std::int32_t row_begin = n / 3;
    const std::int32_t rows      = n - row_begin;
    std::vector<float> out(static_cast<std::size_t>(rows) * tokens, NAN);
    ops::host::rows_dot(m, row_begin, rows, prepared.data(), stride, tokens, out.data(), tokens);

    int failures     = 0;
    double worst     = 0.0;
    for (std::int32_t r = 0; r < rows; ++r) {
        for (std::int32_t t = 0; t < tokens; ++t) {
            double ref = 0.0;
            double abs = 0.0;
            for (std::int32_t c = 0; c < k; ++c) {
                const double p = logical_weight_fp64(packed, row_begin + r, c) *
                                 static_cast<double>(logical[t * k + c]);
                ref += p;
                abs += std::fabs(p);
            }
            const double err = std::fabs(static_cast<double>(out[r * tokens + t]) - ref);
            const double lim = bound(k, abs) + 1e-30;
            worst            = std::max(worst, err / lim);
            if (!(err <= lim)) {
                if (failures < 5) {
                    std::printf("  FAIL %s n=%d k=%d T=%d row=%d t=%d got=%.9g ref=%.9g lim=%.3g\n",
                                name(qtype), n, k, tokens, row_begin + r, t,
                                static_cast<double>(out[r * tokens + t]), ref, lim);
                }
                ++failures;
            }
        }
    }
    std::printf("%s %s n=%d k=%d T=%d worst_err/bound=%.3f\n", failures ? "FAIL" : "ok  ",
                name(qtype), n, k, tokens, worst);
    return failures;
}

// Exact decode of a column span: every value must equal the logical weight bit for bit (a stored
// code times its binary16 scale is exact in FP32).
int check_decode(QType qtype, std::int32_t n, std::int32_t k, std::uint32_t seed) {
    PatternedWeightOptions options;
    options.row_split_scale = RowSplitScalePattern::Small;
    options.row_split_codes = RowSplitCodePattern::Hashed;
    const PackedWeight packed = make_patterned_weight(qtype, n, k, seed, options);
    std::vector<std::uint8_t> payload = packed.payload;
    const Weight w                    = packed.device_weight(payload.data());
    const auto m                      = ops::host::row_split_matrix(w);
    int failures                      = 0;
    // PLE table geometry: packed rows of 8 x 160 columns, spans crossing G64 group boundaries.
    const std::int32_t span = 160;
    std::vector<float> out(span);
    for (std::int32_t row = 0; row < n; row += 3) {
        for (std::int32_t column = 0; column + span <= k; column += span) {
            ops::host::decode_row_range(m, row, column, span, out.data());
            for (std::int32_t i = 0; i < span; ++i) {
                const double ref = logical_weight_fp64(packed, row, column + i);
                if (static_cast<double>(out[i]) != ref) {
                    if (failures < 5) {
                        std::printf("  FAIL decode %s row=%d col=%d got=%.9g ref=%.9g
",
                                    name(qtype), row, column + i, static_cast<double>(out[i]), ref);
                    }
                    ++failures;
                }
            }
        }
    }
    std::printf("%s decode %s n=%d k=%d
", failures ? "FAIL" : "ok  ", name(qtype), n, k);
    return failures;
}

} // namespace

int main() {
    if (!ops::host::host_kernels_supported()) {
        std::printf("SKIP: CPU lacks AVX2/FMA/F16C/BMI2\n");
        return 77;
    }
    int failures = 0;
    const QType qtypes[] = {QType::Q4_G64_FP16, QType::Q5_G64_FP16, QType::Q6_G64_FP16,
                            QType::Q8_G32_FP16};
    // Model K values: 35B-A3B hidden 2048 and expert width 512; Flash-Next hidden 2560 and
    // expert width 640 (K_pad 768, partial physical padding group).
    const std::int32_t ks[] = {2048, 512, 2560, 640, 200};
    for (const QType q : qtypes) {
        for (const std::int32_t k : ks) {
            for (const std::int32_t tokens : {1, 2, 3, 4, 5, 8}) {
                failures += check(q, 48, k, tokens, 0x5eedU + static_cast<std::uint32_t>(k));
            }
        }
    }
    for (const QType q : qtypes) failures += check_decode(q, 24, 1280, 0xab1eU);
    std::printf("%s (%d failures)\n", failures ? "FAILED" : "PASSED", failures);
    return failures ? 1 : 0;
}
