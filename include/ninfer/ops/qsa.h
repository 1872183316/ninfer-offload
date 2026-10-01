#pragma once

#include "core/paged_kv_cache.h"
#include "core/tensor.h"

#include <cuda_runtime.h>

#include <array>
#include <cstddef>
#include <cstdint>

namespace ninfer::ops {

/**
 * Qwen4-Exp QSA (query-selected sparse attention) on full-attention layers.
 *
 * Every Op works on B independent rows of W columns. positions is contiguous device I32 [W,B]
 * holding the absolute cache position p of each column, kv_table_rows is contiguous device I32
 * [B] selecting the block-table row of each row, and valid_columns is contiguous device I32 [B]
 * or empty (every column live). Column j of row b is live when j < valid_columns[b]; dead
 * columns read and write no cache and produce zero outputs. Live positions of one row are
 * sequential. A column sees the cache positions [0,p]; text RoPE positions equal cache positions.
 *
 * Indexer (C = compress, K = block_topk, Hi = index heads, Di = index_dim, R = rotary_dim):
 *
 *   raw key    r_t = index key of token t (BF16 [Di], stored by qsa_index_append)
 *   block i    = tokens [C*i, C*i+C), complete when C*i+C-1 <= p; nb = (p+1) / C
 *   pooled     u_i = BF16(mean_FP32(r_{C*i..C*i+C-1}))
 *   key        k_i = BF16(RoPE(BF16(rmsnorm(u_i) * (1 + w_key)), C*i))
 *   query      q_h = BF16(RoPE(BF16(rmsnorm(raw_q_h) * (1 + w_query)), p))
 *   score      s_i = sum_h max(0, <q_h, k_i>) / sqrt(Di)
 *   selection  nb <= K: every block; otherwise the K blocks with the largest score, a lower block
 *              index winning ties. Selected tokens are the tokens of the selected blocks plus the
 *              incomplete tail [C*nb, p].
 *
 * rmsnorm uses eps and the mean square over Di. RoPE is split-half NeoX over dimensions [0,R)
 * with angle pos * inv_freq[i], inv_freq[i] = FP32(theta^(-2i/R)), evaluated in FP32 and rounded
 * once to BF16; dimensions [R,Di) are unchanged. The BF16 roundings above are semantic boundaries
 * of the reference model. The oracle evaluates the formulas naively from the represented inputs
 * with those boundaries; selections are compared exactly except where the oracle scores of the
 * last selected and first rejected blocks differ by less than the numerical tolerance.
 *
 * The registered geometry is Hi=4, Di=128, R=64, C=4, K=512, attention [256,24,2] with the BF16
 * cache profile (BF16 K, FP16 V) and page-major planes [X,64,H,N].
 */
struct QsaGeometry {
    std::int32_t index_heads = 4;
    std::int32_t index_dim   = 128;
    std::int32_t rotary_dim  = 64;
    std::int32_t compress    = 4;
    std::int32_t block_topk  = 512;
    float theta              = 10000000.0F;
    float eps                = 1e-6F;
};

/**
 * Store the raw index key of every live column. key is BF16 [Di,W*B] with contiguous rows and
 * any column stride (a row slice of the index projection output is accepted). pages is the BF16 index plane [Di,64,1,N] of the layer and block_tables the I32
 * [L,rows] table matrix. The key is copied bit for bit to position p.
 */
void qsa_index_append(const Tensor& key, const Tensor& positions, const Tensor& valid_columns,
                      const Tensor& kv_table_rows, const Tensor& block_tables, Tensor& pages,
                      cudaStream_t stream);

/**
 * Select the attended blocks of every live column. raw_query is BF16 [Hi*Di,W*B] (head h in rows
 * [h*Di,(h+1)*Di), contiguous rows, any column stride); query_norm/key_norm are BF16 [Di]. selected is
 * contiguous I32 [K,W,B]: the selected block indices in ascending order followed by -1. Dead
 * columns are filled with -1. max_visible is a host promise bounding p+1 over live columns.
 */
void qsa_select(const Tensor& raw_query, const Tensor& query_norm, const Tensor& key_norm,
                const Tensor& positions, const Tensor& valid_columns, const Tensor& kv_table_rows,
                const Tensor& block_tables, const Tensor& pages, const QsaGeometry& geometry,
                std::uint32_t max_visible, WorkspaceArena& workspace, Tensor& selected,
                cudaStream_t stream);

// Transient workspace of qsa_select for B rows and W*B columns under max_visible.
[[nodiscard]] std::size_t qsa_select_workspace_bytes(const QsaGeometry& geometry,
                                                     std::uint32_t max_visible,
                                                     std::int32_t rows, std::int32_t columns);

/**
 * Append K/V of every live column and attend to its selected tokens.
 *
 * q/out are contiguous BF16 [256,24,W,B]; k/v are contiguous BF16 [256,2,W,B] (q and k already
 * normalized and rotated). The BF16 cache profile stores K bit for bit and V as FP16_RNE. For a
 * live column with selected token set S (see above; the current token is written before any
 * column reads it),
 *
 *   out[:,h] = sum_{t in S} softmax_t(scale * <q_h, K_t>) * V_t,  K/V head = h / 12.
 *
 * The oracle evaluates the formula in FP64 from the represented cache values; out is compared
 * after its BF16 storage rounding. Dead columns write exact zeros. Calls with few columns split
 * each column's tokens over several CTAs and combine the partial softmax states; the split count
 * depends only on the column count.
 */
void qsa_attention(const Tensor& q, const Tensor& k, const Tensor& v, const Tensor& positions,
                   const Tensor& valid_columns, const Tensor& kv_table_rows,
                   const Tensor& selected, const QsaGeometry& geometry, float scale,
                   PagedKVBatchLayerView cache, WorkspaceArena& workspace, Tensor& out,
                   cudaStream_t stream);

// Transient workspace of qsa_attention for W*B columns (token-split partials of small calls).
[[nodiscard]] std::size_t qsa_attention_workspace_bytes(std::int32_t columns);

} // namespace ninfer::ops
