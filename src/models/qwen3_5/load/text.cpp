#include "models/qwen3_5/load/bindings.h"

#include <algorithm>
#include <fstream>
#include <limits>
#include <numeric>
#include <optional>
#include <set>
#include <span>
#include <sstream>

namespace ninfer::models::qwen3_5::loading {

AttentionWeights bind_attention(Bindings& b, const TextConfig& config, const std::string& p) {
    const auto& a = config.attention.value();
    const auto h  = config.hidden_size;
    const auto q  = a.query_width();
    const auto k  = a.key_width();
    AttentionWeights out;
    out.query      = b.parameter(p + "attention/query", {q, h}, {p + "mixer_input"});
    out.key        = b.parameter(p + "attention/key", {k, h}, {p + "mixer_input"});
    out.gate       = b.parameter(p + "attention/gate", {q, h}, {p + "mixer_input"});
    out.value      = b.parameter(p + "attention/value", {k, h}, {p + "mixer_input"});
    out.query_norm = b.direct(p + "attention/query_norm", {a.head_dim});
    out.key_norm   = b.direct(p + "attention/key_norm", {a.head_dim});
    out.output     = b.parameter(p + "attention/output", {h, q}, {p + "attention/gated_output"});
    return out;
}

DenseWeights bind_dense(Bindings& b, std::uint64_t h, std::uint64_t intermediate,
                        const std::string& p, bool draft) {
    const auto input = p + (draft ? "mlp_input" : "ffn_input");
    return {b.parameter(p + "mlp/gate", {intermediate, h}, {input}),
            b.parameter(p + "mlp/up", {intermediate, h}, {input}),
            b.parameter(p + "mlp/down", {h, intermediate},
                        {p + (draft ? "mlp_product" : "mlp/product")})};
}

namespace {

GdnWeights bind_gdn(Bindings& b, const TextConfig& text, const std::string& p) {
    const auto& g    = text.gdn.value();
    const auto h     = text.hidden_size;
    const auto k     = g.key_width();
    const auto v     = g.value_width();
    const auto heads = g.linear_num_value_heads;
    GdnWeights out;
    out.query        = b.parameter(p + "gdn/query", {k, h}, {p + "mixer_input"});
    out.key          = b.parameter(p + "gdn/key", {k, h}, {p + "mixer_input"});
    out.value        = b.parameter(p + "gdn/value", {v, h}, {p + "mixer_input"});
    out.z            = b.parameter(p + "gdn/z", {v, h}, {p + "mixer_input"});
    out.a_projection = b.parameter(p + "gdn/a_projection", {heads, h}, {p + "mixer_input"});
    out.b_projection = b.parameter(p + "gdn/b_projection", {heads, h}, {p + "mixer_input"});
    out.a_log        = b.direct(p + "gdn/a_log", {heads}, QType::FP32);
    out.dt_bias      = b.direct(p + "gdn/dt_bias", {heads}, QType::FP32);
    out.convolution =
        b.direct(p + "gdn/convolution", {g.linear_conv_kernel_dim, g.conv_channels()});
    out.norm   = b.direct(p + "gdn/norm", {g.linear_value_head_dim});
    out.output = b.parameter(p + "gdn/output", {h, v}, {p + "gdn/gated_output"});
    return out;
}

// Row ranges of `parameters` inside their single row-split parent, in the given order.
artifact::ObjectHandle parameter_rows(const Bindings& b, std::span<const WeightId> parameters,
                                      std::uint64_t columns,
                                      std::vector<artifact::RowRange>& rows) {
    std::optional<artifact::ObjectHandle> object;
    for (const auto id : parameters) {
        for (const auto& part : b.at(id).reference.binding.parts) {
            if (object && part.object.index != object->index) {
                throw artifact::ArtifactError("offloaded expert bank spans several objects");
            }
            object = part.object;
            if (part.begin % columns || part.end % columns) {
                throw artifact::ArtifactError("offloaded expert rows are not row aligned");
            }
            const std::uint64_t begin = part.begin / columns;
            const std::uint64_t count = (part.end - part.begin) / columns;
            if (!rows.empty() && rows.back().begin + rows.back().count == begin) {
                rows.back().count += count;
            } else {
                rows.push_back({begin, count});
            }
        }
    }
    if (!object) { throw artifact::ArtifactError("offloaded expert has no parent"); }
    return *object;
}

MoeWeights bind_moe(Bindings& b, const TextConfig& config, const std::string& prefix,
                    const MoeOffloadOptions* offload, const std::vector<std::int32_t>* resident) {
    const auto& moe   = std::get<MoeConfig>(config.ffn);
    const auto p      = prefix + "moe/";
    const auto input  = prefix + "ffn_input";
    const auto h      = config.hidden_size;
    const auto ir     = moe.moe_intermediate_size;
    const auto shared = moe.shared_expert_intermediate_size;
    MoeWeights out;
    out.router       = b.parameter(p + "router", {moe.num_experts, h}, {input});
    out.shared_score = b.parameter(p + "shared_score", {1, h}, {input});
    out.experts.reserve(moe.num_experts);
    const auto residency = offload ? artifact::Residency::Host : artifact::Residency::Device;
    for (std::uint32_t e = 0; e < moe.num_experts; ++e) {
        const auto ep = p + "experts/" + std::to_string(e) + "/";
        out.experts.push_back(
            {b.parameter(ep + "gate", {ir, h}, {input}, {}, residency),
             b.parameter(ep + "up", {ir, h}, {input}, {}, residency),
             b.parameter(ep + "down", {h, ir}, {ep + "product"}, {}, residency)});
    }
    if (offload) {
        MoeOffloadWeights o;
        if (resident) { o.resident = *resident; }
        if (!o.resident.empty()) {
            std::vector<WeightId> gate_up;
            std::vector<WeightId> down;
            for (const auto e : o.resident) {
                const auto& expert = out.experts.at(static_cast<std::size_t>(e));
                gate_up.push_back(expert.gate);
                gate_up.push_back(expert.up);
                down.push_back(expert.down);
            }
            std::vector<artifact::RowRange> gate_up_rows;
            std::vector<artifact::RowRange> down_rows;
            const auto gate_up_object = parameter_rows(b, gate_up, h, gate_up_rows);
            const auto down_object    = parameter_rows(b, down, ir, down_rows);
            const std::uint64_t count = o.resident.size();
            o.device_gate_up = b.replica(p + "offload/device_gate_up", gate_up_object,
                                         std::move(gate_up_rows), {count * 2 * ir, h});
            o.device_down    = b.replica(p + "offload/device_down", down_object,
                                         std::move(down_rows), {count * h, ir});
        }
        out.offload = std::move(o);
    }
    out.shared = {b.parameter(p + "shared/gate", {shared, h}, {input}),
                  b.parameter(p + "shared/up", {shared, h}, {input}),
                  b.parameter(p + "shared/down", {h, shared}, {p + "shared/product"})};
    return out;
}

} // namespace

BlockWeights bind_block(Bindings& b, const TextConfig& config, const std::string& p,
                        MixerKind mixer, const MoeOffloadOptions* offload,
                        const std::vector<std::int32_t>* resident) {
    BlockWeights out;
    out.input_norm          = b.direct(p + "input_norm", {config.hidden_size});
    out.post_attention_norm = b.direct(p + "post_attention_norm", {config.hidden_size});
    if (mixer == MixerKind::FullAttention) {
        out.mixer = bind_attention(b, config, p);
    } else {
        out.mixer = bind_gdn(b, config, p);
    }
    if (const auto* dense = std::get_if<DenseConfig>(&config.ffn)) {
        out.ffn = bind_dense(b, config.hidden_size, dense->intermediate_size, p);
    } else {
        out.ffn = bind_moe(b, config, p, offload, resident);
    }
    return out;
}

namespace {

// Per Text layer, the experts given a device replica: the `resident_experts` most frequently
// routed experts from the stats file (one line of counts per Text layer), or the lowest ids.
std::vector<std::vector<std::int32_t>> resident_experts(const MoeOffloadOptions& options,
                                                        const TextConfig& config) {
    const auto* moe = std::get_if<MoeConfig>(&config.ffn);
    if (!moe) { throw artifact::ArtifactError("MoE offload requires a MoE model"); }
    const std::uint32_t experts  = moe->num_experts;
    const std::uint32_t resident = std::min(options.resident_experts, experts);
    std::vector<std::vector<std::uint64_t>> counts;
    if (!options.expert_stats.empty()) {
        std::ifstream in(options.expert_stats);
        if (!in) { throw artifact::ArtifactError("cannot read MoE expert stats file"); }
        std::string line;
        while (std::getline(in, line)) {
            std::istringstream fields(line);
            std::vector<std::uint64_t> row;
            for (std::uint64_t value; fields >> value;) { row.push_back(value); }
            if (row.empty()) { continue; }
            if (row.size() != experts) {
                throw artifact::ArtifactError("MoE expert stats width differs from the model");
            }
            counts.push_back(std::move(row));
        }
        if (counts.size() != config.num_hidden_layers) {
            throw artifact::ArtifactError("MoE expert stats layer count differs from the model");
        }
    }
    std::vector<std::vector<std::int32_t>> out(config.num_hidden_layers);
    for (std::uint32_t layer = 0; layer < config.num_hidden_layers; ++layer) {
        std::vector<std::int32_t> order(experts);
        std::iota(order.begin(), order.end(), 0);
        if (!counts.empty()) {
            const auto& c = counts[layer];
            std::stable_sort(order.begin(), order.end(),
                             [&](std::int32_t a, std::int32_t b) { return c[a] > c[b]; });
        }
        order.resize(resident);
        std::sort(order.begin(), order.end());
        out[layer] = std::move(order);
    }
    return out;
}

} // namespace

TextWeights bind_text(Bindings& b, const TextConfig& config, const LoadOptions& options) {
    TextWeights out;
    out.token_embedding =
        b.parameter("text/token_embedding", {config.vocab_size, config.hidden_size});
    std::vector<std::string> head_inputs{"text/final_hidden"};
    if (options.speculative != SpeculativeBackend::None && !options.proposal_enabled()) {
        head_inputs.push_back(std::string(options.speculative_component()) + "/final_hidden");
    }
    out.output_head = b.parameter("text/output_head", {config.vocab_size, config.hidden_size},
                                  std::move(head_inputs));
    out.final_norm  = b.direct("text/final_norm", {config.hidden_size});
    out.layers.reserve(config.num_hidden_layers);
    const auto* offload = options.moe_offload.enabled ? &options.moe_offload : nullptr;
    const auto resident = offload ? resident_experts(*offload, config)
                                  : std::vector<std::vector<std::int32_t>>{};
    for (std::uint32_t i = 0; i < config.num_hidden_layers; ++i) {
        out.layers.push_back(bind_block(b, config, "text/layers/" + std::to_string(i) + "/",
                                        config.layer_types[i], offload,
                                        offload ? &resident.at(i) : nullptr));
    }
    return out;
}

ProposalWeights bind_proposal(Bindings& b, const artifact::Proposal& proposal,
                              const TextConfig& target, const LoadOptions& options,
                              std::uint32_t public_tokens) {
    const auto rows = proposal.indexed ? proposal.rows : target.vocab_size;
    if (rows > std::numeric_limits<std::uint32_t>::max() ||
        (proposal.indexed && rows > public_tokens)) {
        throw artifact::ArtifactError("proposal rows exceed the output domain");
    }
    ProposalWeights out;
    out.rows = static_cast<std::uint32_t>(rows);
    out.head = b.parameter("proposal/head", {rows, target.hidden_size},
                           {std::string(options.speculative_component()) + "/final_hidden"});
    if (proposal.indexed) {
        out.token_ids = b.direct("proposal/token_ids", {rows}, QType::INT32);
        out.global_token_ids =
            b.binder.values(b.at(*out.token_ids).reference.binding, QType::INT32).integers();
        std::set<std::int32_t> unique;
        for (const auto id : out.global_token_ids) {
            if (id < 0 || std::uint32_t(id) >= public_tokens || !unique.insert(id).second) {
                throw artifact::ArtifactError(
                    "proposal token IDs must be unique and in the public domain");
            }
        }
    }
    return out;
}

} // namespace ninfer::models::qwen3_5::loading
