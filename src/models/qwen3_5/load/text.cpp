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
        out.offload = std::move(o);
    }
    out.shared = {b.parameter(p + "shared/gate", {shared, h}, {input}),
                  b.parameter(p + "shared/up", {shared, h}, {input}),
                  b.parameter(p + "shared/down", {h, shared}, {p + "shared/product"})};
    return out;
}

} // namespace

HyperWeights bind_hyper(Bindings& b, const TextConfig& config, const std::string& p, bool inject) {
    const auto h    = config.hidden_size;
    const auto wide = std::uint64_t(config.hyper->count) * h;
    const auto rank = config.hyper->low_rank;
    HyperWeights out;
    out.norm = b.direct(p + "norm", {wide});
    out.down = b.parameter(p + "down", {rank, wide}, {p + "input"});
    out.up   = b.parameter(p + "up", {wide, rank}, {p + "mix"});
    if (inject) { out.inject = b.parameter(p + "inject", {config.hyper->count, wide}, {p + "input"}); }
    return out;
}

BlockWeights bind_qwen4exp_block(Bindings& b, const TextConfig& config, const std::string& p,
                                 MixerKind mixer, std::uint32_t layer,
                                 const MoeOffloadOptions* offload,
                                 const std::vector<std::int32_t>* resident) {
    BlockWeights out;
    out.attn_hc = bind_hyper(b, config, p + "attn_hc/", true);
    out.ffn_hc  = bind_hyper(b, config, p + "ffn_hc/", true);
    if (config.ple && config.ple->layer == layer) {
        const auto& ple = *config.ple;
        const auto h    = config.hidden_size;
        const auto wide = std::uint64_t(config.hyper->count) * h;
        PleWeights w;
        w.key         = b.parameter(p + "ple/key", {wide, ple.embed_dim}, {p + "ple/embedding"});
        w.value       = b.parameter(p + "ple/value", {h, ple.embed_dim}, {p + "ple/embedding"});
        w.norm_key    = b.direct(p + "ple/norm_key", {wide});
        w.norm_query  = b.direct(p + "ple/norm_query", {wide});
        w.norm_conv   = b.direct(p + "ple/norm_conv", {wide});
        w.convolution = b.direct(p + "ple/convolution", {ple.conv_kernel, wide});
        // The n-gram table is only gathered row by row: map it instead of loading it.
        w.table = b.parameter(p + "ple/table",
                              {ple.rows / ple.table_packing,
                               std::uint64_t(ple.table_packing) * ple.head_dim},
                              {}, {}, artifact::Residency::HostMapped);
        out.ple = w;
    }
    if (mixer == MixerKind::FullAttention) {
        out.mixer = bind_attention(b, config, p);
        const auto& ix = *config.indexer;
        out.indexer    = IndexerWeights{
            b.parameter(p + "indexer/query_key",
                        {std::uint64_t(ix.heads + 1) * ix.head_dim, config.hidden_size},
                        {p + "mixer_input"}),
            b.direct(p + "indexer/query_norm", {ix.head_dim}),
            b.direct(p + "indexer/key_norm", {ix.head_dim})};
    } else {
        out.mixer = bind_gdn(b, config, p);
    }
    out.ffn = bind_moe(b, config, p, offload, resident);
    return out;
}

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
    const bool qwen4exp = config.architecture == Architecture::Qwen4Exp;
    if (qwen4exp) {
        out.head_hc = bind_hyper(b, config, "text/head_hc/", false);
    } else {
        out.final_norm = b.direct("text/final_norm", {config.hidden_size});
    }
    out.layers.reserve(config.num_hidden_layers);
    const auto* offload = options.moe_offload.enabled ? &options.moe_offload : nullptr;
    const auto resident = offload ? resident_experts(*offload, config)
                                  : std::vector<std::vector<std::int32_t>>{};
    for (std::uint32_t i = 0; i < config.num_hidden_layers; ++i) {
        const auto prefix  = "text/layers/" + std::to_string(i) + "/";
        const auto* chosen = offload ? &resident.at(i) : nullptr;
        out.layers.push_back(qwen4exp ? bind_qwen4exp_block(b, config, prefix, config.layer_types[i],
                                                            i, offload, chosen)
                                      : bind_block(b, config, prefix, config.layer_types[i],
                                                   offload, chosen));
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
