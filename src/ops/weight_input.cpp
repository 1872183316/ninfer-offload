#include "ninfer/ops/weight_input.h"

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace ninfer::ops {
namespace {

void require(bool condition, const char* message) {
    if (!condition) { throw std::invalid_argument(message); }
}

const std::vector<std::uint64_t>& matrix(const WeightInput& input) {
    require(input.weight.shape.size() == 2, "projection weight must be a matrix");
    const auto elements = weight_element_count(input.weight.shape);
    require(!input.weight.parts.empty(), "projection weight has no parent regions");
    std::uint64_t covered = 0;
    for (const auto& part : input.weight.parts) {
        require(part.parent && part.begin < part.end &&
                    part.end <= part.parent->geometry.elements &&
                    part.end - part.begin <= elements - covered,
                "projection region exceeds its parent or logical matrix");
        covered += part.end - part.begin;
    }
    require(covered == elements, "projection regions do not cover its logical matrix");
    return input.weight.shape;
}

WeightView concatenate_rows(std::span<const WeightInput> inputs) {
    require(!inputs.empty(), "projection bank is empty");
    const auto columns = matrix(inputs.front())[1];
    WeightView out{{0, columns}, {}};
    for (const auto& input : inputs) {
        const auto& shape = matrix(input);
        require(shape[1] == columns &&
                    shape[0] <= std::numeric_limits<std::uint64_t>::max() - out.shape[0],
                "projection bank has inconsistent columns or overflowing rows");
        out.shape[0] += shape[0];
        out.parts.insert(out.parts.end(), input.weight.parts.begin(), input.weight.parts.end());
    }
    return out;
}

bool contiguous(const WeightView& view) {
    if (view.parts.empty()) { return false; }
    const auto* parent = view.parts.front().parent;
    auto end           = view.parts.front().begin;
    for (const auto& part : view.parts) {
        if (!parent || part.parent != parent || part.begin != end || part.end <= part.begin) {
            return false;
        }
        end = part.end;
    }
    return true;
}

LinearPolicy common_policy(std::span<const WeightInput> inputs) {
    auto result = LinearPolicy::AllowA4;
    for (const auto& input : inputs) {
        switch (input.policy) {
        case LinearPolicy::A16Only:
            result = LinearPolicy::A16Only;
            break;
        case LinearPolicy::AllowA8:
            if (result == LinearPolicy::AllowA4) { result = LinearPolicy::AllowA8; }
            break;
        case LinearPolicy::AllowA4:
            break;
        default:
            throw std::invalid_argument("invalid projection activation policy");
        }
    }
    return result;
}

SingleProjectionWeight single(std::span<const WeightInput> inputs) {
    auto view         = concatenate_rows(inputs);
    const auto region = contiguous_weight_region(view);
    const auto policy = common_policy(inputs);
    float divisor     = 0;
    if (region.parent->geometry.format == QType::NVFP4) {
        require(std::isfinite(region.parent->weight_scale_divisor) &&
                    region.parent->weight_scale_divisor > 0,
                "NVFP4 parent requires a positive weight divisor");
        for (const auto& input : inputs) {
            if (!input.activation_input_divisor) {
                require(!allows_a4(policy), "NVFP4 A4 native input requires an activation divisor");
                continue;
            }
            require(std::isfinite(*input.activation_input_divisor) &&
                        *input.activation_input_divisor > 0,
                    "NVFP4 native input requires a positive activation divisor");
            if (divisor == 0) {
                // A16 does not read this auxiliary; retain a stored positive value for the ABI.
                divisor = *input.activation_input_divisor;
            } else if (allows_a4(policy)) {
                // Current NVFP4 consumers use A16 for AllowA8 as well. Only their A4 route
                // quantizes the shared activation and therefore requires a common divisor.
                require(std::bit_cast<std::uint32_t>(divisor) ==
                            std::bit_cast<std::uint32_t>(*input.activation_input_divisor),
                        "shared NVFP4 native input requires identical activation divisors");
            }
        }
        // The legacy native ABI validates this field even on A16, where it is never read.
        // Keep that ABI detail out of the artifact's optional Use auxiliaries.
        if (divisor == 0) { divisor = 1.0F; }
    }
    return {native_weight(view, divisor), policy};
}

ProjectionWeights input_projection(std::span<const WeightInput, 4> inputs, bool attention) {
    const auto& q      = matrix(inputs[0]);
    const auto& k      = matrix(inputs[1]);
    const auto& third  = matrix(inputs[2]);
    const auto& fourth = matrix(inputs[3]);
    const bool dense =
        attention ? q == std::vector<std::uint64_t>{6144, 5120} &&
                        k == std::vector<std::uint64_t>{1024, 5120} && third == q && fourth == k
                  : q == std::vector<std::uint64_t>{2048, 5120} && k == q &&
                        third == std::vector<std::uint64_t>{6144, 5120} && fourth == third;
    const bool moe =
        attention ? q == std::vector<std::uint64_t>{4096, 2048} &&
                        k == std::vector<std::uint64_t>{512, 2048} && third == q && fourth == k
                  : q == std::vector<std::uint64_t>{2048, 2048} && k == q &&
                        third == std::vector<std::uint64_t>{4096, 2048} && fourth == third;
    require(dense || moe, "input projection: unsupported logical projection geometry");
    const auto joined = concatenate_rows(inputs);
    if (contiguous(joined)) {
        auto result       = single(inputs);
        const auto format = result.weight.qtype;
        const bool supported =
            (moe && format == QType::Q8_G32_FP16) ||
            (dense && (format == QType::NVFP4 || format == QType::FP8_E4M3FN_ROW_BF16 ||
                       (attention && format == QType::BF16)));
        require(supported, "input projection: unsupported single-parent format");
        return result;
    }
    require(dense, "input projection: unsupported multi-parent geometry");
    const auto first  = single(inputs.first<2>());
    const auto second = single(inputs.last<2>());
    require(first.weight.qtype == QType::Q4_G64_FP16 && second.weight.qtype == QType::Q5_G64_FP16,
            "input projection: paired native form requires Q4 and Q5");
    return PairedProjectionWeights{first.weight, second.weight};
}

} // namespace

SingleProjectionWeight prepare_linear_weight(const WeightInput& input) {
    return single({&input, 1});
}

SingleProjectionWeight prepare_linear_weight(std::span<const WeightInput> rows) {
    return single(rows);
}

SingleProjectionWeight prepare_attn_input_proj_weights(const WeightInput& query,
                                                       const WeightInput& key,
                                                       const WeightInput& value) {
    const auto& q = matrix(query);
    require((q == std::vector<std::uint64_t>{4096, 2048} ||
             q == std::vector<std::uint64_t>{4096, 5120}) &&
                matrix(key) == std::vector<std::uint64_t>{1024, q[1]} &&
                matrix(value) == matrix(key),
            "QKV input projection: unsupported logical geometry");
    const std::array inputs{query, key, value};
    auto result = single(inputs);
    require(result.weight.qtype == QType::Q8_G32_FP16,
            "QKV input projection: native form requires Q8");
    return result;
}

ProjectionWeights prepare_attn_input_proj_weights(const WeightInput& query, const WeightInput& key,
                                                  const WeightInput& gate,
                                                  const WeightInput& value) {
    const std::array inputs{query, key, gate, value};
    return input_projection(inputs, true);
}

ProjectionWeights prepare_gdn_input_proj_weights(const WeightInput& query, const WeightInput& key,
                                                 const WeightInput& value, const WeightInput& z) {
    const std::array inputs{query, key, value, z};
    return input_projection(inputs, false);
}

ProjectionWeights prepare_gdn_gating_proj_weights(const WeightInput& a, const WeightInput& b) {
    const auto& shape = matrix(a);
    require(shape == matrix(b) && (shape == std::vector<std::uint64_t>{48, 5120} ||
                                   shape == std::vector<std::uint64_t>{32, 2048}),
            "GDN control: unsupported A/B geometry");
    const std::array inputs{a, b};
    if (contiguous(concatenate_rows(inputs))) {
        auto result = single(inputs);
        require(result.weight.qtype == QType::BF16, "GDN control requires BF16 weights");
        return result;
    }
    require(shape[0] == 48, "GDN control: this geometry requires a combined parent");
    const auto first  = prepare_linear_weight(a);
    const auto second = prepare_linear_weight(b);
    require(first.weight.qtype == QType::BF16 && second.weight.qtype == QType::BF16,
            "GDN control requires BF16 weights");
    return PairedProjectionWeights{first.weight, second.weight};
}

SingleProjectionWeight prepare_linear_swiglu_weight(const WeightInput& gate,
                                                    const WeightInput& up) {
    require(matrix(gate) == matrix(up), "SwiGLU gate and up geometry differs");
    const std::array inputs{gate, up};
    return single(inputs);
}

SparseMoeWeights
prepare_sparse_moe_weights(const WeightInput& router, const WeightInput& shared_score,
                           std::span<const WeightInput> expert_gate_up,
                           std::span<const WeightInput> expert_down, const WeightInput& shared_gate,
                           const WeightInput& shared_up, const WeightInput& shared_down) {
    require(expert_gate_up.size() == 512 && expert_down.size() == 256,
            "SparseMoe native input requires 256 experts");
    require(matrix(router) == std::vector<std::uint64_t>{256, 2048} &&
                matrix(shared_score) == std::vector<std::uint64_t>{1, 2048},
            "SparseMoe router geometry differs");
    for (const auto& input : expert_gate_up) {
        require(matrix(input) == std::vector<std::uint64_t>{512, 2048},
                "SparseMoe gate/up geometry differs");
    }
    for (const auto& input : expert_down) {
        require(matrix(input) == std::vector<std::uint64_t>{2048, 512},
                "SparseMoe down geometry differs");
    }
    require(matrix(shared_gate) == std::vector<std::uint64_t>{512, 2048} &&
                matrix(shared_up) == matrix(shared_gate) &&
                matrix(shared_down) == std::vector<std::uint64_t>{2048, 512},
            "SparseMoe shared expert geometry differs");
    const std::array router_inputs{router, shared_score};
    const auto router_bank  = single(router_inputs);
    const auto gate_up_bank = single(expert_gate_up);
    const auto down_bank    = single(expert_down);
    const auto shared       = prepare_linear_swiglu_weight(shared_gate, shared_up);
    const auto down         = prepare_linear_weight(shared_down);
    const auto gate_format  = gate_up_bank.weight.qtype;
    const auto down_format  = down_bank.weight.qtype;
    require(router_bank.weight.qtype == QType::BF16 && shared.weight.qtype == QType::Q8_G32_FP16 &&
                down.weight.qtype == QType::Q8_G32_FP16 &&
                ((gate_format == QType::Q4_G64_FP16 &&
                  (down_format == QType::Q5_G64_FP16 || down_format == QType::Q6_G64_FP16)) ||
                 (gate_format == QType::Q8_G32_FP16 && down_format == QType::Q8_G32_FP16)),
            "SparseMoe native bank formats are unsupported");
    return {router_bank.weight, gate_up_bank.weight, down_bank.weight, shared.weight, down.weight};
}

} // namespace ninfer::ops

namespace ninfer::ops {

HybridSparseMoeWeights prepare_hybrid_sparse_moe_weights(
    const WeightInput& router, const WeightInput& shared_score,
    std::span<const WeightInput> expert_gate_up, std::span<const WeightInput> expert_down,
    const WeightInput& shared_gate, const WeightInput& shared_up, const WeightInput& shared_down,
    std::span<const std::int32_t> resident, const WeightInput* device_gate_up,
    const WeightInput* device_down, std::int32_t top_k) {
    const auto experts = expert_down.size();
    require(experts > 0 && experts <= kHybridMoeMaxExperts && expert_gate_up.size() == 2 * experts,
            "hybrid MoE: expert count is unsupported");
    require(top_k > 0 && top_k <= kHybridMoeMaxTopK && static_cast<std::size_t>(top_k) <= experts,
            "hybrid MoE: top-k is unsupported");
    const auto hidden       = matrix(router)[1];
    const auto intermediate = matrix(expert_down.front())[1];
    require(matrix(router)[0] == experts && matrix(shared_score) == std::vector<std::uint64_t>{1, hidden},
            "hybrid MoE: router geometry differs");
    const std::array router_inputs{router, shared_score};
    const auto router_bank  = single(router_inputs);
    const auto gate_up_bank = single(expert_gate_up);
    const auto down_bank    = single(expert_down);
    const auto shared       = prepare_linear_swiglu_weight(shared_gate, shared_up);
    const auto shared_d     = prepare_linear_weight(shared_down);
    const auto row_split    = [](const Weight& w) {
        return w.layout == QuantLayout::RowSplit &&
               (w.qtype == QType::Q4_G64_FP16 || w.qtype == QType::Q5_G64_FP16 ||
                w.qtype == QType::Q6_G64_FP16 || w.qtype == QType::Q8_G32_FP16);
    };
    require(router_bank.weight.qtype == QType::BF16 && row_split(gate_up_bank.weight) &&
                row_split(down_bank.weight) && row_split(shared.weight) && row_split(shared_d.weight),
            "hybrid MoE: bank formats are unsupported");
    require(gate_up_bank.weight.n == static_cast<std::int32_t>(2 * experts * intermediate) &&
                gate_up_bank.weight.k == static_cast<std::int32_t>(hidden) &&
                down_bank.weight.n == static_cast<std::int32_t>(experts * hidden) &&
                down_bank.weight.k == static_cast<std::int32_t>(intermediate),
            "hybrid MoE: expert bank geometry differs");

    HybridSparseMoeWeights out;
    out.router_shared_gate  = router_bank.weight;
    out.shared_gate_up      = shared.weight;
    out.shared_down         = shared_d.weight;
    out.host_gate_up        = gate_up_bank.weight;
    out.host_down           = down_bank.weight;
    out.experts             = static_cast<std::int32_t>(experts);
    out.top_k               = top_k;
    out.hidden              = static_cast<std::int32_t>(hidden);
    out.intermediate        = static_cast<std::int32_t>(intermediate);
    out.shared_intermediate = shared_d.weight.k;
    out.slot_of_expert.assign(experts, -1);
    for (std::size_t s = 0; s < resident.size(); ++s) {
        const auto e = resident[s];
        require(e >= 0 && static_cast<std::size_t>(e) < experts &&
                    out.slot_of_expert[static_cast<std::size_t>(e)] < 0,
                "hybrid MoE: invalid resident expert list");
        out.slot_of_expert[static_cast<std::size_t>(e)] = static_cast<std::int16_t>(s);
    }
    if (!resident.empty()) {
        require(device_gate_up && device_down, "hybrid MoE: resident experts need replicas");
        out.device_gate_up = prepare_linear_weight(*device_gate_up).weight;
        out.device_down    = prepare_linear_weight(*device_down).weight;
        require(out.device_gate_up.qtype == out.host_gate_up.qtype &&
                    out.device_down.qtype == out.host_down.qtype &&
                    out.device_gate_up.n ==
                        static_cast<std::int32_t>(resident.size() * 2 * intermediate) &&
                    out.device_down.n == static_cast<std::int32_t>(resident.size() * hidden),
                "hybrid MoE: replica geometry differs");
    }
    return out;
}

} // namespace ninfer::ops
