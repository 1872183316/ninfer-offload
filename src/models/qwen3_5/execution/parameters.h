#pragma once

#include "models/qwen3_5/model.h"
#include "ninfer/ops/weight_input.h"

#include <array>
#include <memory>
#include <limits>
#include <optional>
#include <stdexcept>
#include <variant>
#include <vector>

namespace ninfer::models::qwen3_5::execution {

using LinearParameters = ops::SingleProjectionWeight;

[[nodiscard]] inline std::int32_t dimension(std::uint64_t value) {
    if (value > std::uint64_t(std::numeric_limits<std::int32_t>::max())) {
        throw std::overflow_error("model dimension exceeds the Tensor integer domain");
    }
    return static_cast<std::int32_t>(value);
}

struct DenseParameters {
    LinearParameters gate_up;
    LinearParameters down;
};

using FfnParameters =
    std::variant<DenseParameters, ops::SparseMoeWeights, ops::HybridSparseMoeWeights>;

struct AttentionParameters {
    ops::ProjectionWeights projection;
    Tensor query_norm, key_norm;
    LinearParameters output;
};

struct GdnParameters {
    ops::ProjectionWeights projection;
    ops::ProjectionWeights control;
    Tensor a_log, dt_bias, convolution, norm;
    LinearParameters output;
};

// Qwen4-Exp mixers read the hyper-connection input directly (no residual norm) and write their
// block output to a separate BF16 [H,T] tensor. Every projection is an independent linear.
struct Qwen4AttentionParameters {
    LinearParameters query, key, gate, value, output;
    Tensor query_norm, key_norm;
    // QSA indexer: query|key projection [(heads+1)*head_dim, H] and per-head norms.
    LinearParameters index_query_key;
    Tensor index_query_norm, index_key_norm;
};

struct Qwen4GdnParameters {
    LinearParameters qkv, z, a, b, output; // qkv rows are query | key | value
    Tensor a_log, dt_bias, convolution, norm;
};

struct HyperParameters {
    Tensor norm;
    LinearParameters down, up;
    std::optional<LinearParameters> inject;
};

struct PleParameters {
    LinearParameters key, value;
    Tensor norm_key, norm_query, norm_conv, convolution;
    Weight table; // host-mapped; gathered by the host runtime
};

struct BlockParameters {
    Tensor input_norm, post_attention_norm;
    std::variant<AttentionParameters, GdnParameters, Qwen4AttentionParameters, Qwen4GdnParameters>
        mixer;
    FfnParameters ffn;
    ops::SparseMoeHints projection_prefetch;
    std::optional<HyperParameters> attn_hc, ffn_hc;
    std::optional<PleParameters> ple;
};

struct TextParameters {
    Weight token_embedding;
    LinearParameters output_head;
    Tensor final_norm;                      // Qwen3.5 only
    std::optional<HyperParameters> head_hc; // Qwen4-Exp only
    std::vector<BlockParameters> layers;
};

struct MtpProjectionParameters {
    LinearParameters packed;
    // Dense MTP projects K/V and Q/gate independently in its incremental path.
    // MoE MTP uses its existing complete-parent Attention projection.
    std::optional<std::array<LinearParameters, 4>> rows;
};

struct MtpParameters {
    LinearParameters input_projection;
    Tensor embedding_norm, hidden_norm, input_norm, post_attention_norm, final_norm;
    MtpProjectionParameters projection;
    Tensor query_norm, key_norm;
    LinearParameters output;
    FfnParameters ffn;
    LinearParameters output_head;
};

// Qwen4-Exp MTP: stem u_s = fc_embedding(norm(embedding)) + fc_hidden(norm(wide)_s), one
// full-attention block on the wide stream, and the head mixer before the shared output head.
struct Qwen4MtpParameters {
    Tensor embedding_norm, hidden_norm;
    LinearParameters fc_embedding, fc_hidden;
    HyperParameters head_hc;
    BlockParameters layer;
    LinearParameters output_head;
};

struct NormParameters {
    Tensor weight, bias;
};

struct VisionBlockParameters {
    NormParameters norm1, norm2;
    LinearParameters qkv;
    Tensor qkv_bias;
    LinearParameters output, fc1, fc2;
    Tensor output_bias, fc1_bias, fc2_bias;
};

struct VisionParameters {
    LinearParameters patch_embedding;
    Tensor patch_embedding_bias, position_embedding;
    std::vector<VisionBlockParameters> layers;
    NormParameters merger_norm;
    LinearParameters merger_fc1, merger_fc2;
    Tensor merger_fc1_bias, merger_fc2_bias;
};

struct DynamicConvParameters {
    Tensor base_kernel;
    LinearParameters kernel_projection;
};

struct DraftBlockParameters {
    Tensor input_norm, post_attention_norm;
    LinearParameters query_key_value, context_key, context_value;
    Tensor query_norm, key_norm;
    LinearParameters output;
    DenseParameters mlp;
    std::optional<DynamicConvParameters> attention_conv, mlp_conv;
};

struct SelectorParameters {
    LinearParameters hidden_projection;
    Tensor predecessor_codebook, successor_codebook;
};

struct DraftParameters {
    LinearParameters feature_projection;
    Tensor context_norm, final_norm;
    std::vector<DraftBlockParameters> layers;
    std::optional<SelectorParameters> selector;
    LinearParameters output_head;
};

struct ProposalParameters {
    LinearParameters head;
    std::optional<Tensor> token_ids;
    std::uint32_t rows = 0;
};

// Cold native preparation for the fixed model implementation. This owner is stable before
// startup sizing, execution, or Graph capture; all weight addresses borrow the source Model.
// Shape-dependent kernel selection and scratch remain with the calling implementation and Op.
class Parameters {
public:
    explicit Parameters(const Model& source);
    Parameters(const Parameters&)            = delete;
    Parameters& operator=(const Parameters&) = delete;
    Parameters(Parameters&&)                 = delete;
    Parameters& operator=(Parameters&&)      = delete;

    const Model& model;
    TextParameters text;
    std::optional<MtpParameters> mtp;
    std::optional<Qwen4MtpParameters> qwen4_mtp;
    std::optional<VisionParameters> vision;
    std::optional<DraftParameters> draft;
    std::optional<ProposalParameters> proposal;
};

} // namespace ninfer::models::qwen3_5::execution
