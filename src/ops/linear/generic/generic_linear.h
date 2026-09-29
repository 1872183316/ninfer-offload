#pragma once

#include "core/tensor.h"
#include "core/weight.h"

#include <cuda_runtime.h>

namespace ninfer::ops::detail {

// Runtime-geometry route for BF16 and row-split Q4/Q5/Q6/Q8 weights whose (N,K) has no tuned
// registered schedule. A16: exact stored-weight decode, FP32 accumulation, one BF16 rounding.
[[nodiscard]] bool generic_linear_supported(const Weight& w) noexcept;
void generic_linear(const Tensor& x, const Weight& w, Tensor& out, cudaStream_t stream);

} // namespace ninfer::ops::detail
