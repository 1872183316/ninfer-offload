// Qwen4-Exp (Qwen3.8-Flash-Next) Text execution: hyper-connection residual streams, the PLE n-gram
// injection, and mixers that read the mixed stream input directly. Mathematics follows the
// reference Qwen4ExpTextDecoderLayer; see docs/maintainer/expert-offload.md.

#include "models/qwen3_5/program/internal.h"
#include "models/qwen3_5/execution/text.h"
#include "models/qwen3_5/execution/ffn.h"
#include "models/qwen3_5/execution/scoped_value.h"

#include "core/nvtx.h"
#include "ninfer/ops/causal_conv1d_silu.h"
#include "ninfer/ops/embedding.h"
#include "ninfer/ops/gated_delta_net.h"
#include "ninfer/ops/gated_rmsnorm.h"
#include "ninfer/ops/gdn_gating.h"
#include "ninfer/ops/hybrid_sparse_moe.h"
#include "ninfer/ops/hyper_connection.h"
#include "ninfer/ops/linear.h"
#include "ninfer/ops/residual_add.h"
#include "ninfer/ops/rmsnorm.h"
#include "ninfer/ops/scalar.h"

#include <cuda_runtime.h>

#include <cmath>
#include <stdexcept>
#include <string>

namespace ninfer::models::qwen3_5::execution {
namespace {

void project(const Tensor& x, const LinearParameters& p, Tensor& out, WorkspaceArena& work,
             cudaStream_t stream) {
    ops::linear(x, p.weight, out, p.policy, work, stream);
}

// Copies rows [begin, begin + rows) of a contiguous [C,T] matrix into a contiguous [rows,T].
void copy_rows(const Tensor& source, std::int32_t begin, Tensor& out, cudaStream_t stream) {
    const std::size_t element = dtype_size(source.dtype);
    const std::size_t width   = static_cast<std::size_t>(out.ne[0]) * element;
    CUDA_CHECK(cudaMemcpy2DAsync(out.data, width,
                                 static_cast<const std::byte*>(source.data) + begin * element,
                                 static_cast<std::size_t>(source.nb[1]), width,
                                 static_cast<std::size_t>(out.numel() / out.ne[0]),
                                 cudaMemcpyDeviceToDevice, stream));
}

Tensor zeroed(WorkspaceArena& work, std::int32_t rows, std::int32_t columns, cudaStream_t stream) {
    Tensor out = work.alloc(DType::BF16, {rows, columns});
    CUDA_CHECK(cudaMemsetAsync(out.data, 0, out.bytes(), stream));
    return out;
}

} // namespace

void TextContext::hyper_mix(const HyperParameters& w, const Tensor& wide, Tensor& input,
                            Tensor* injection) {
    cudaStream_t s    = ctx_.stream;
    const int T       = wide.ne[1];
    const auto count  = static_cast<std::int32_t>(config_.hyper->count);
    const auto hidden = dimension(config_.hidden_size);
    auto scope        = work_.scope();
    Tensor xn         = work_.alloc(DType::BF16, {count * hidden, T});
    ops::hc_grouped_rmsnorm(wide, w.norm, hidden, config_.rms_norm_eps, xn, s);
    Tensor low = work_.alloc(DType::BF16, {dimension(config_.hyper->low_rank), T});
    project(xn, w.down, low, work_, s);
    ops::hc_scaled_silu(low, 1.0F / static_cast<float>(count), s);
    Tensor mix = work_.alloc(DType::BF16, {count * hidden, T});
    project(low, w.up, mix, work_, s);
    ops::hc_gated_mean(mix, xn, count, input, s);
    if (injection != nullptr) {
        if (!w.inject) { throw std::logic_error("hyper-connection mixer has no injection"); }
        Tensor inject = work_.alloc(DType::BF16, {count, T});
        project(xn, *w.inject, inject, work_, s);
        ops::hc_injection_weights(inject, count, *injection, s);
    }
}

void TextContext::qwen4exp_attention(const Qwen4AttentionParameters& p, const Tensor& h, Tensor& y,
                                     int fidx) {
    cudaStream_t s   = ctx_.stream;
    const int T      = h.ne[1];
    const auto& a    = *config_.attention;
    const auto dim   = dimension(a.head_dim);
    const auto heads = dimension(a.num_attention_heads);
    const auto kv    = dimension(a.num_key_value_heads);
    Tensor q         = work_.alloc(DType::BF16, {dim * heads, T});
    Tensor gate      = work_.alloc(DType::BF16, {dim * heads, T});
    Tensor k         = work_.alloc(DType::BF16, {dim * kv, T});
    Tensor v         = work_.alloc(DType::BF16, {dim * kv, T});
    project(h, p.query, q, work_, s);
    project(h, p.gate, gate, work_, s);
    project(h, p.key, k, work_, s);
    project(h, p.value, v, work_, s);
    Tensor qh = q.view({dim, heads, T});
    Tensor gh = gate.view({dim, heads, T});
    Tensor kh = k.view({dim, kv, T});
    Tensor vh = v.view({dim, kv, T});
    Tensor selected;
    if (batch_text_kv_->has_index_planes()) {
        // QSA: store the raw index keys, then select the attended blocks of every column.
        const ops::QsaGeometry g = qsa_geometry();
        const auto id            = dimension(g.index_dim);
        const auto ih            = dimension(g.index_heads);
        Tensor qk                = work_.alloc(DType::BF16, {(ih + 1) * id, T});
        project(h, p.index_query_key, qk, work_, s);
        const QsaRows rows = qsa_rows(T);
        selected = work_.alloc(DType::I32, {dimension(g.block_topk), rows.positions.ne[0],
                                            rows.positions.ne[1]});
        Tensor pages       = batch_text_kv_->index_pages(static_cast<std::uint32_t>(fidx));
        const Tensor tables = batch_text_kv_->batch_layer_view(fidx).block_tables;
        ops::qsa_index_append(qk.slice(0, ih * id, id), rows.positions, rows.valid,
                              rows.table_rows, tables, pages, s);
        ops::qsa_select(qk.slice(0, 0, ih * id), p.index_query_norm, p.index_key_norm,
                        rows.positions, rows.valid, rows.table_rows, tables, pages, g,
                        active_causal_attention_envelope_->max_visible_keys, work_, selected, s);
        active_qsa_selected_ = &selected;
    }
    Tensor o             = attention_core(p.query_norm, p.key_norm, qh, gh, kh, vh, fidx);
    active_qsa_selected_ = nullptr;
    project(o.view({dim * heads, T}), p.output, y, work_, s);
}

void TextContext::qwen4exp_gdn(const Qwen4GdnParameters& p, const Tensor& h, Tensor& y, int gidx,
                               Phase ph) {
    cudaStream_t s    = ctx_.stream;
    const int T       = h.ne[1];
    const auto& g     = *config_.gdn;
    const auto kw     = dimension(g.key_width());
    const auto vw     = dimension(g.value_width());
    const auto kd     = dimension(g.linear_key_head_dim);
    const auto vd     = dimension(g.linear_value_head_dim);
    const auto kheads = dimension(g.linear_num_key_heads);
    const auto vheads = dimension(g.linear_num_value_heads);
    const float scale = static_cast<float>(1.0 / std::sqrt(static_cast<double>(kd)));

    Tensor qkv = work_.alloc(DType::BF16, {2 * kw + vw, T});
    Tensor z   = work_.alloc(DType::BF16, {vw, T});
    Tensor a   = work_.alloc(DType::BF16, {vheads, T});
    Tensor b   = work_.alloc(DType::BF16, {vheads, T});
    project(h, p.qkv, qkv, work_, s);
    project(h, p.z, z, work_, s);
    project(h, p.a, a, work_, s);
    project(h, p.b, b, work_, s);
    Tensor decay = work_.alloc(DType::FP32, {vheads, T});
    Tensor beta  = work_.alloc(DType::FP32, {vheads, T});
    ops::gdn_gating(a, b, p.a_log, p.dt_bias, decay, beta, s);

    Tensor qc = work_.alloc(DType::BF16, {kw, T});
    Tensor kc = work_.alloc(DType::BF16, {kw, T});
    Tensor vc = work_.alloc(DType::BF16, {vw, T});
    Tensor o  = work_.alloc(DType::BF16, {vd, vheads, T});
    const auto layer = static_cast<std::uint32_t>(gidx);
    if (ph == Phase::Verify) {
        if (active_sequence_batch_ == 0 || active_linear_state_source_slots_ == nullptr) {
            throw std::logic_error("Verify GDN requires an explicit sequence batch and state slots");
        }
        const std::int32_t batch = active_sequence_batch_;
        const std::int32_t width = active_sequence_width_;
        Tensor conv_states       = state_.layer_view(layer).conv;
        const Tensor valid = active_valid_columns_ != nullptr ? *active_valid_columns_ : Tensor{};
        if (gdn_state_action_ == GdnStateAction::RecordForReplay) {
            // Speculative verify: evaluate from the source state without writing it and record
            // the transitions; the commit fold replays the accepted prefix.
            if (replay_records_ == nullptr || width * batch != T) {
                throw std::logic_error("Qwen4-Exp replay-record GDN binding is incomplete");
            }
            GdnReplayRecordLayer records = replay_records_->layer(gidx, batch);
            Tensor mixed = work_.alloc(DType::BF16, {2 * kw + vw, width, batch});
            ops::causal_conv1d_silu_record(qkv.view({2 * kw + vw, width, batch}), p.convolution,
                                           conv_states, valid, *active_linear_state_source_slots_,
                                           records.conv, mixed, s);
            Tensor mixed_flat = mixed.view({2 * kw + vw, T});
            copy_rows(mixed_flat, 0, qc, s);
            copy_rows(mixed_flat, kw, kc, s);
            copy_rows(mixed_flat, 2 * kw, vc, s);
            Tensor ob = o.view({vd, vheads, width, batch});
            ops::gated_delta_net_replay_record(
                qc.view({kd, kheads, width, batch}), kc.view({kd, kheads, width, batch}),
                vc.view({vd, vheads, width, batch}), decay.view({vheads, width, batch}),
                beta.view({vheads, width, batch}), scale, state_.layer_view(layer).recurrent,
                valid, *active_linear_state_source_slots_, records.key, records.value,
                records.gate, ob, s);
        } else {
            if (gdn_state_action_ != GdnStateAction::UpdateInPlace ||
                active_linear_state_destination_slots_ == nullptr || width != 1) {
                throw std::logic_error("Qwen4-Exp GDN decode updates width-one rows in place");
            }
            Tensor mixed = work_.alloc(DType::BF16, {2 * kw + vw, 1, batch});
            ops::causal_conv1d_silu_snapshot(qkv.view({2 * kw + vw, 1, batch}), p.convolution,
                                             conv_states, valid,
                                             *active_linear_state_source_slots_,
                                             *active_linear_state_destination_slots_, mixed, s);
            Tensor mixed_flat = mixed.view({2 * kw + vw, T});
            copy_rows(mixed_flat, 0, qc, s);
            copy_rows(mixed_flat, kw, kc, s);
            copy_rows(mixed_flat, 2 * kw, vc, s);
            Tensor states = state_.layer_view(layer).recurrent;
            Tensor ob     = o.view({vd, vheads, 1, batch});
            ops::gated_delta_net_batch_update(
                qc.view({kd, kheads, 1, batch}), kc.view({kd, kheads, 1, batch}),
                vc.view({vd, vheads, 1, batch}), decay.view({vheads, 1, batch}),
                beta.view({vheads, 1, batch}), scale, /*normalize_qk=*/true, states,
                *active_linear_state_source_slots_, *active_linear_state_destination_slots_, ob,
                s);
        }
    } else {
        Tensor conv_in  = state_.conv_slot(layer, linear_state_source_slot_);
        Tensor conv_out = state_.conv_slot(layer, linear_state_destination_slot_);
        ops::causal_conv1d_silu_split(qkv, p.convolution, conv_in, conv_out, qc, kc, vc, s);
        Tensor state_in  = state_.recurrent_slot(layer, linear_state_source_slot_);
        Tensor state_out = state_.recurrent_slot(layer, linear_state_destination_slot_);
        ops::gated_delta_net(qc.view({kd, kheads, T}), kc.view({kd, kheads, T}),
                             vc.view({vd, vheads, T}), decay, beta, scale,
                             /*normalize_qk=*/true, work_, state_in, state_out, o,
                             ctx_.execution_view());
    }
    Tensor on = work_.alloc(DType::BF16, {vd, vheads, T});
    Tensor zh = z.view({vd, vheads, T});
    if (config_.gdn_sigmoid_gate) {
        ops::gated_rmsnorm_sigmoid(o, p.norm, zh, config_.rms_norm_eps, on, s);
    } else {
        ops::gated_rmsnorm(o, p.norm, zh, config_.rms_norm_eps, on, s);
    }
    project(on.view({vw, T}), p.output, y, work_, s);
}

void TextContext::qwen4exp_ple(const PleParameters& p, Tensor& wide, Phase ph) {
    cudaStream_t s      = ctx_.stream;
    const auto& ple     = *config_.ple;
    const int T         = wide.ne[1];
    const auto count    = static_cast<std::int32_t>(config_.hyper->count);
    const auto hidden   = dimension(config_.hidden_size);
    const auto heads    = static_cast<std::int32_t>(ple.heads());
    if (active_ids_ == nullptr || host_moe_ == nullptr) {
        throw std::logic_error("Qwen4-Exp PLE requires token ids and the host runtime");
    }

    // The PLE state lives in the recurrent region of the pseudo state layer after every GDN layer.
    const Tensor pool = state_.layer_view(config_.linear_attention_layers).recurrent;
    const ops::PleStateView state{
        .base                 = pool.data,
        .slot_pitch_bytes     = pool.nb[3],
        .conv_offset_bytes    = 0,
        .history_offset_bytes = static_cast<std::int64_t>(config_.ple_conv_state_bytes()),
    };
    std::int32_t width = T;
    std::int32_t batch = 1;
    Tensor valid;
    Tensor source, destination;
    const bool record = ph == Phase::Verify && gdn_state_action_ == GdnStateAction::RecordForReplay;
    if (ph == Phase::Verify) {
        if (active_linear_state_source_slots_ == nullptr) {
            throw std::logic_error("Qwen4-Exp PLE verify requires source state slots");
        }
        width  = active_sequence_width_;
        batch  = active_sequence_batch_;
        valid  = active_valid_columns_ != nullptr ? *active_valid_columns_ : Tensor{};
        source = *active_linear_state_source_slots_;
        if (record) {
            // Speculative verify reads the source state only; an empty destination makes the
            // state Ops write nothing, and the commit folds the accepted prefix.
            if (replay_records_ == nullptr || replay_records_->ple_conv.data == nullptr) {
                throw std::logic_error("Qwen4-Exp PLE verify has no replay records");
            }
        } else {
            if (active_linear_state_destination_slots_ == nullptr ||
                gdn_state_action_ != GdnStateAction::UpdateInPlace) {
                throw std::logic_error("Qwen4-Exp PLE decode updates rows in place");
            }
            destination = *active_linear_state_destination_slots_;
        }
    } else {
        source      = work_.alloc(DType::I32, {1});
        destination = work_.alloc(DType::I32, {1});
        ops::set_i32_scalar(source, linear_state_source_slot_, s);
        ops::set_i32_scalar(destination, linear_state_destination_slot_, s);
    }
    ops::PleHash hash;
    hash.ngram_size      = static_cast<std::int32_t>(ple.ngram_size);
    hash.heads_per_ngram = static_cast<std::int32_t>(ple.heads_per_ngram);
    hash.eos             = static_cast<std::int32_t>(ple.eos_token_id);
    if (ple.ngram_size > 4 || heads > 32) { throw std::logic_error("PLE geometry exceeds hash"); }
    for (std::size_t i = 0; i < ple.multipliers.size(); ++i) {
        hash.multipliers[i] = ple.multipliers[i];
    }
    for (std::int32_t j = 0; j < heads; ++j) {
        hash.vocab[j]  = ple.head_vocab_sizes[static_cast<std::size_t>(j)];
        hash.offset[j] = ple.head_offsets[static_cast<std::size_t>(j)];
    }

    Tensor rows = work_.alloc(DType::I32, {heads, width, batch});
    ops::ple_ngram_rows(active_ids_->view({width, batch}), valid, hash, state, source, destination,
                        rows, s);
    Tensor embedding = work_.alloc(DType::BF16, {dimension(ple.embed_dim), T});
    ops::ple_gather(rows.view({heads, T}), *host_moe_, embedding, s);

    Tensor key   = work_.alloc(DType::BF16, {count * hidden, T});
    Tensor value = work_.alloc(DType::BF16, {hidden, T});
    project(embedding, p.key, key, work_, s);
    project(embedding, p.value, value, work_, s);
    Tensor key_n   = work_.alloc(DType::BF16, {count * hidden, T});
    Tensor query_n = work_.alloc(DType::BF16, {count * hidden, T});
    ops::hc_grouped_rmsnorm(key, p.norm_key, hidden, config_.rms_norm_eps, key_n, s);
    ops::hc_grouped_rmsnorm(wide, p.norm_query, hidden, config_.rms_norm_eps, query_n, s);
    Tensor gated = work_.alloc(DType::BF16, {count * hidden, T});
    ops::ple_gate(key_n, query_n, value, count, gated, s);
    Tensor gated_n = key_n; // key_n is dead; reuse its storage
    ops::hc_grouped_rmsnorm(gated, p.norm_conv, hidden, config_.rms_norm_eps, gated_n, s);
    if (record) {
        // Raw convolution inputs and token ids of every column, row-major [.., width, batch].
        CUDA_CHECK(cudaMemcpyAsync(replay_records_->ple_conv.data, gated_n.data, gated_n.bytes(),
                                   cudaMemcpyDeviceToDevice, s));
        CUDA_CHECK(cudaMemcpyAsync(replay_records_->ple_ids.data, active_ids_->data,
                                   static_cast<std::size_t>(T) * sizeof(std::int32_t),
                                   cudaMemcpyDeviceToDevice, s));
    }
    Tensor conv = query_n; // query_n is dead as well
    ops::ple_dilated_conv_silu(gated_n.view({count * hidden, width, batch}), p.convolution,
                               static_cast<std::int32_t>(ple.ngram_size), valid, state, source,
                               destination, conv, s);
    // hidden += gated + conv, with the reference's two BF16 roundings.
    ops::residual_add(conv, gated, s);
    ops::residual_add(gated, wide, s);
}

void TextContext::qwen4exp_layers(Tensor& x, Phase ph) {
    cudaStream_t s     = ctx_.stream;
    const int T        = x.ne[1];
    const auto count   = static_cast<std::int32_t>(config_.hyper->count);
    const auto hidden  = dimension(config_.hidden_size);
    const bool prefill = ph == Phase::Prefill;
    Tensor wide        = work_.alloc(DType::BF16, {count * hidden, T});
    ops::hc_expand(x, count, wide, s);
    for (std::size_t layer = 0; layer < parameters_.text.layers.size(); ++layer) {
        const auto& block  = parameters_.text.layers[layer];
        const bool full    = config_.layer_types[layer] == MixerKind::FullAttention;
        const auto compact = static_cast<int>(config_.compact_layer_indices[layer]);
        nvtx::ScopedRange layer_range(
            full ? (prefill ? nvtx::Name::PrefillLayerFull : nvtx::Name::VerifyLayerFull)
                 : (prefill ? nvtx::Name::PrefillLayerGdn : nvtx::Name::VerifyLayerGdn),
            full ? nvtx::Category::Attention : nvtx::Category::Gdn, layer);
        try {
            if (block.ple) {
                auto scope = work_.scope();
                qwen4exp_ple(*block.ple, wide, ph);
            }
            {
                auto scope      = work_.scope();
                Tensor input    = work_.alloc(DType::BF16, {hidden, T});
                Tensor weights  = work_.alloc(DType::FP32, {count, T});
                hyper_mix(*block.attn_hc, wide, input, &weights);
                Tensor y = work_.alloc(DType::BF16, {hidden, T});
                if (const auto* a = std::get_if<Qwen4AttentionParameters>(&block.mixer)) {
                    qwen4exp_attention(*a, input, y, compact);
                } else {
                    qwen4exp_gdn(std::get<Qwen4GdnParameters>(block.mixer), input, y, compact, ph);
                }
                ops::hc_combine(wide, y, weights, s);
            }
            {
                auto scope     = work_.scope();
                Tensor input   = work_.alloc(DType::BF16, {hidden, T});
                Tensor weights = work_.alloc(DType::FP32, {count, T});
                hyper_mix(*block.ffn_hc, wide, input, &weights);
                Tensor y = zeroed(work_, hidden, T, s);
                ffn(input, block.ffn, y, next_projection_hints(static_cast<int>(layer)), work_, s,
                    false, host_moe_);
                ops::hc_combine(wide, y, weights, s);
            }
        } catch (const std::exception& error) {
            throw std::runtime_error("text/layers/" + std::to_string(layer) +
                                     (prefill ? " prefill" : " verify") +
                                     " columns=" + std::to_string(T) + ": " + error.what());
        }
    }
    if (wide_capture_ != nullptr) {
        if (wide_capture_->dtype != DType::BF16 || wide_capture_->ne[0] != count * hidden ||
            wide_capture_->ne[1] != T || !wide_capture_->is_contiguous()) {
            throw std::logic_error("Qwen4-Exp wide capture must be BF16 [hc*H,T]");
        }
        CUDA_CHECK(cudaMemcpyAsync(wide_capture_->data, wide.data, wide.bytes(),
                                   cudaMemcpyDeviceToDevice, s));
    }
    hyper_mix(*parameters_.text.head_hc, wide, x, nullptr);
}

void TextContext::qwen4exp_mtp_core(const Tensor& ids, const Tensor& hidden,
                                    const Tensor& positions, const Tensor& rope_positions,
                                    ops::CausalAttentionExecutionEnvelope envelope,
                                    Tensor& mtp_hidden, const Tensor* input_embeddings,
                                    int output_columns) {
    cudaStream_t s    = ctx_.stream;
    const auto& m     = *parameters_.qwen4_mtp;
    const int T       = static_cast<int>(ids.numel());
    if (output_columns < 0 || output_columns > T) {
        throw std::invalid_argument("Qwen4-Exp MTP output columns must be in [0,T]");
    }
    const auto count  = static_cast<std::int32_t>(config_.hyper->count);
    const auto H      = dimension(config_.hidden_size);
    const auto wide   = count * H;
    Tensor flat_ids   = ids.view({T});
    Tensor stream_in  = hidden.view({wide, T});
    Tensor u          = mtp_hidden.view({wide, T});

    // Stem: u_s = fc_embedding(norm(embedding)) + fc_hidden(norm(wide)_s), one BF16 rounding.
    {
        auto scope = work_.scope();
        Tensor emb;
        if (input_embeddings != nullptr) {
            emb = input_embeddings->view({H, T});
        } else {
            emb = work_.alloc(DType::BF16, {H, T});
            ops::embedding(flat_ids, *embed_, emb, s);
        }
        Tensor e = work_.alloc(DType::BF16, {H, T});
        ops::rmsnorm(emb, m.embedding_norm, config_.rms_norm_eps, true, e, s);
        Tensor hn = work_.alloc(DType::BF16, {wide, T});
        ops::rmsnorm(stream_in, m.hidden_norm, config_.rms_norm_eps, true, hn, s);
        Tensor fe = work_.alloc(DType::BF16, {H, T});
        project(e, m.fc_embedding, fe, work_, s);
        Tensor fh = work_.alloc(DType::BF16, {H, count * T});
        project(hn.view({H, count * T}), m.fc_hidden, fh, work_, s);
        ops::hc_expand(fe, count, u, s);
        ops::residual_add(fh.view({wide, T}), u, s);
    }

    // One full-attention block: attention against the MTP KV cache, routed experts on the host.
    const Tensor& rows = active_backend_kv_table_rows_ != nullptr ? *active_backend_kv_table_rows_
                                                                  : io_.backend_kv_table_row;
    ScopedValue<const Tensor*> cache_binding(active_cache_positions_, &positions);
    ScopedValue<const Tensor*> rope_binding(active_rope_positions_, &rope_positions);
    ScopedValue<const ops::CausalAttentionExecutionEnvelope*> envelope_binding(
        active_causal_attention_envelope_, &envelope);
    ScopedValue<const qwen3_5::PagedKVCache*> cache(batch_text_kv_, batch_mtp_kv_);
    ScopedValue<const Tensor*> table_rows(active_kv_table_rows_, &rows);
    const auto& block = m.layer;
    {
        auto scope     = work_.scope();
        Tensor input   = work_.alloc(DType::BF16, {H, T});
        Tensor weights = work_.alloc(DType::FP32, {count, T});
        hyper_mix(*block.attn_hc, u, input, &weights);
        Tensor y = work_.alloc(DType::BF16, {H, T});
        qwen4exp_attention(std::get<Qwen4AttentionParameters>(block.mixer), input, y, 0);
        ops::hc_combine(u, y, weights, s);
    }
    // The MoE is per column, and earlier columns contribute to later ones only through the K/V
    // appended above, so columns without an output skip it.
    if (output_columns == 0) { return; }
    {
        auto scope     = work_.scope();
        const int N    = output_columns;
        Tensor tail    = u.slice(1, T - N, N);
        Tensor input   = work_.alloc(DType::BF16, {H, N});
        Tensor weights = work_.alloc(DType::FP32, {count, N});
        hyper_mix(*block.ffn_hc, tail, input, &weights);
        Tensor y = zeroed(work_, H, N, s);
        ffn(input, block.ffn, y, {}, work_, s, false, host_moe_);
        ops::hc_combine(tail, y, weights, s);
    }
}

} // namespace ninfer::models::qwen3_5::execution
