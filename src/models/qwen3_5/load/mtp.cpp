#include "models/qwen3_5/load/bindings.h"

namespace ninfer::models::qwen3_5::loading {

MtpWeights bind_mtp(Bindings& b, const TextConfig& config, const TextWeights& target,
                    const LoadOptions& options) {
    const auto h = config.hidden_size;
    MtpWeights out;
    if (config.architecture == Architecture::Qwen4Exp) {
        const auto wide    = std::uint64_t(config.hyper->count) * h;
        out.embedding_norm = b.direct("mtp/embedding_norm", {h});
        out.hidden_norm    = b.direct("mtp/hidden_norm", {wide});
        out.fc_embedding   = b.parameter("mtp/fc_embedding", {h, h}, {"mtp/embedding_input"});
        out.fc_hidden      = b.parameter("mtp/fc_hidden", {h, h}, {"mtp/hidden_input"});
        out.head_hc        = bind_hyper(b, config, "mtp/head_hc/", false);
        // The MTP block has no PLE (its layer index lies past the Text stack). Its routed
        // experts stay in host memory like the Text experts; the device MoE has no route for
        // this geometry, and the layer has no routing statistics to rank resident experts.
        if (!options.moe_offload.enabled) {
            throw artifact::ArtifactError("Qwen4-Exp MTP requires --moe-offload");
        }
        static const std::vector<std::int32_t> no_resident;
        out.layer = bind_qwen4exp_block(b, config, "mtp/layers/0/", MixerKind::FullAttention,
                                        config.num_hidden_layers, &options.moe_offload,
                                        &no_resident);
        out.token_embedding = target.token_embedding;
        out.output_head     = target.output_head;
        return out;
    }
    out.input_projection = b.parameter("mtp/input_projection", {h, 2ULL * h}, {"mtp/stem_input"});
    out.embedding_norm   = b.direct("mtp/embedding_norm", {h});
    out.hidden_norm      = b.direct("mtp/hidden_norm", {h});
    out.final_norm       = b.direct("mtp/final_norm", {h});
    out.layer            = bind_block(b, config, "mtp/layers/0/", MixerKind::FullAttention);
    out.token_embedding  = target.token_embedding;
    out.output_head      = target.output_head;
    return out;
}

} // namespace ninfer::models::qwen3_5::loading
