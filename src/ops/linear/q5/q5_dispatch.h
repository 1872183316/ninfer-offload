#pragma once

#include "core/weight.h"
#include "ninfer/ops/linear.h"
#include "ops/linear/q5/q5_launch.h"

#include <cstdint>

namespace ninfer::ops::detail {

Q5Launch select_q5_a16_launch(std::int32_t n, std::int32_t k, std::int32_t t);
// True when (n,k) has a tuned registered schedule.
[[nodiscard]] bool q5_shape_registered(std::int32_t n, std::int32_t k) noexcept;
Q5Launch select_q5_launch(std::int32_t n, std::int32_t k, std::int32_t t, LinearPolicy policy);

void q5_dispatch(const Tensor& x, const Weight& w, Tensor& out, LinearPolicy policy,
                 cudaStream_t stream);

} // namespace ninfer::ops::detail
