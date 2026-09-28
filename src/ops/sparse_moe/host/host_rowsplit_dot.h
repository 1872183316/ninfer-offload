#pragma once

#include "core/weight.h"

#include <cstddef>
#include <cstdint>

namespace ninfer::ops::host {

// Host view of a row_split_k128_v1 matrix [N,K] stored in host memory. Rows are addressed by
// their parent index; a routed-expert bank is one matrix whose expert e owns a fixed row span.
struct RowSplitMatrix {
    const std::uint8_t* codes  = nullptr;
    const std::uint8_t* high   = nullptr; // Q5/Q6 only
    const std::uint16_t* scales = nullptr;
    QType qtype                = QType::Q4_G64_FP16;
    std::int32_t rows          = 0;
    std::int32_t k             = 0;
    std::int32_t groups_per_row = 0;
};

// Builds the host view from a Weight whose plane pointers address host memory.
[[nodiscard]] RowSplitMatrix row_split_matrix(const Weight& weight);

// Number of floats in a prepared activation for matrix m (covers K_pad).
[[nodiscard]] std::int32_t prepared_activation_floats(const RowSplitMatrix& m);

// Converts a logical FP32 activation of length m.k into the kernel order for m's code layout.
// For G64 nibble codecs every group is stored as its 32 even lanes followed by its 32 odd lanes,
// so the low and high nibbles of one code byte meet contiguous activation vectors. Padding lanes
// are zero. The transform is a permutation; it performs no arithmetic.
void prepare_activation(const RowSplitMatrix& m, const float* x, float* prepared);

// Position of logical activation column k inside a prepared activation for m.
[[nodiscard]] inline std::int32_t prepared_position(const RowSplitMatrix& m, std::int32_t k) {
    if (m.qtype == QType::Q8_G32_FP16) return k;
    const std::int32_t base = k & ~63;
    const std::int32_t lane = k & 63;
    return base + (lane & 1) * 32 + (lane >> 1);
}

// out[r * out_stride + t] = sum_k W[row_begin + r, k] * x_t[k] for r in [0,rows), t in [0,tokens).
// prepared holds `tokens` prepared activations separated by prepared_stride floats. Stored codes
// and binary16 scales are decoded exactly; products and sums use FP32 FMA in a fixed order.
void rows_dot(const RowSplitMatrix& m, std::int32_t row_begin, std::int32_t rows,
              const float* prepared, std::int32_t prepared_stride, std::int32_t tokens,
              float* out, std::int32_t out_stride);

// True when the running CPU supports the AVX2/FMA/F16C/BMI2 instructions the kernels require.
[[nodiscard]] bool host_kernels_supported() noexcept;

} // namespace ninfer::ops::host
