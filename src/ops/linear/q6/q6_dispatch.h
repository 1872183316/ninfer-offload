#pragma once

#include "core/weight.h"
#include "ninfer/ops/linear.h"
#include "ops/linear/q6/q6_launch.h"

#include <cstdint>

namespace ninfer::ops::detail {

Q6Launch select_q6_a16_launch(std::int32_t n, std::int32_t k, std::int32_t t);
// True when (n,k) has a tuned registered schedule.
[[nodiscard]] bool q6_shape_registered(std::int32_t n, std::int32_t k) noexcept;
Q6Launch select_q6_launch(std::int32_t n, std::int32_t k, std::int32_t t, LinearPolicy policy);

void q6_dispatch(const Tensor& x, const Weight& w, Tensor& out, LinearPolicy policy,
                 cudaStream_t stream);

} // namespace ninfer::ops::detail
