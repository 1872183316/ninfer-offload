// Qwen4-Exp QSA ops against a naive CPU oracle: exact index/KV storage, block selection with the
// reference BF16 boundaries, and FP64 sparse attention over the represented cache.
#include "ninfer/ops/qsa.h"

#include "core/arena.h"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <functional>
#include <iterator>
#include <numeric>
#include <vector>

using namespace ninfer;

namespace {

int failures = 0;

constexpr int kDi = 128, kHi = 4, kD = 256, kHq = 24, kHkv = 2, kP = 64, kC = 4, kK = 512;

float bf(std::uint16_t b) {
    const std::uint32_t w = static_cast<std::uint32_t>(b) << 16;
    float f;
    std::memcpy(&f, &w, 4);
    return f;
}

std::uint16_t tobf(float f) {
    std::uint32_t w;
    std::memcpy(&w, &f, 4);
    return static_cast<std::uint16_t>((w + 0x7fffU + ((w >> 16) & 1U)) >> 16);
}

double rbf(double v) { return bf(tobf(static_cast<float>(v))); }

std::uint64_t mix(std::uint64_t v) {
    v += 0x9e3779b97f4a7c15ULL;
    v = (v ^ (v >> 30)) * 0xbf58476d1ce4e5b9ULL;
    v = (v ^ (v >> 27)) * 0x94d049bb133111ebULL;
    return v ^ (v >> 31);
}

float uniform(std::uint64_t seed, std::uint64_t i, float range) {
    return (static_cast<float>(mix(seed * 1315423911ULL + i) % 20001) / 10000.0F - 1.0F) * range;
}

template <class T>
struct Dev {
    T* p = nullptr;
    std::size_t n;
    explicit Dev(const std::vector<T>& h) : n(h.size()) {
        cudaMalloc(&p, n * sizeof(T));
        cudaMemcpy(p, h.data(), n * sizeof(T), cudaMemcpyHostToDevice);
    }
    ~Dev() { cudaFree(p); }
    std::vector<T> get() const {
        std::vector<T> h(n);
        cudaDeviceSynchronize();
        cudaMemcpy(h.data(), p, n * sizeof(T), cudaMemcpyDeviceToHost);
        return h;
    }
};

void check(bool ok, const char* what) {
    if (!ok) {
        std::printf("FAIL %s\n", what);
        ++failures;
    }
}

struct Case {
    const char* name;
    int width;
    std::vector<int> start;  // first position of each row
    std::vector<int> valid;  // empty: all live
    int max_visible;
};

void run(const Case& cs) {
    const int B = static_cast<int>(cs.start.size());
    const int W = cs.width;
    const int columns = W * B;
    const int L = (cs.max_visible + kP - 1) / kP;
    const int N = L * B + 3;
    const ops::QsaGeometry g{};

    // Random page permutation per row.
    std::vector<int> pages(N);
    std::iota(pages.begin(), pages.end(), 0);
    for (int i = N - 1; i > 0; --i) std::swap(pages[i], pages[mix(i + 77) % (i + 1)]);
    std::vector<std::int32_t> tables(static_cast<std::size_t>(L) * (B + 1), -1);
    for (int b = 0; b < B; ++b)
        for (int l = 0; l < L; ++l) tables[static_cast<std::size_t>(b + 1) * L + l] = pages[b * L + l];
    std::vector<std::int32_t> table_rows(B);
    for (int b = 0; b < B; ++b) table_rows[b] = b + 1;  // row 0 is unused

    std::vector<std::int32_t> positions(columns);
    for (int b = 0; b < B; ++b)
        for (int j = 0; j < W; ++j) positions[b * W + j] = cs.start[b] + j;
    auto live = [&](int c) { return cs.valid.empty() || c % W < cs.valid[c / W]; };

    // Cache planes with random history.
    const std::size_t ipl = static_cast<std::size_t>(kDi) * kP * N;
    const std::size_t kpl = static_cast<std::size_t>(kD) * kP * kHkv * N;
    std::vector<std::uint16_t> index(ipl), kc(kpl), vc(kpl);
    for (std::size_t i = 0; i < ipl; ++i) index[i] = tobf(uniform(1, i, 1.0F));
    for (std::size_t i = 0; i < kpl; ++i) {
        kc[i] = tobf(uniform(2, i, 1.0F));
        vc[i] = __half_as_ushort(__float2half_rn(uniform(3, i, 1.0F)));
    }
    auto iaddr = [&](int b, int pos, int d) {
        const int pg = tables[static_cast<std::size_t>(table_rows[b]) * L + pos / kP];
        return (static_cast<std::size_t>(pg) * kP + pos % kP) * kDi + d;
    };
    auto kvaddr = [&](int b, int pos, int h, int d) {
        const int pg = tables[static_cast<std::size_t>(table_rows[b]) * L + pos / kP];
        return ((static_cast<std::size_t>(pg) * kHkv + h) * kP + pos % kP) * kD + d;
    };

    // New activations.
    const int qk_rows = (kHi + 1) * kDi;  // the index projection output [640,T]
    std::vector<std::uint16_t> qk(static_cast<std::size_t>(qk_rows) * columns);
    for (std::size_t i = 0; i < qk.size(); ++i) qk[i] = tobf(uniform(4, i, 2.0F));
    std::vector<std::uint16_t> qn(kDi), kn(kDi);
    for (int d = 0; d < kDi; ++d) {
        qn[d] = tobf(uniform(5, d, 0.5F));
        kn[d] = tobf(uniform(6, d, 0.5F));
    }
    std::vector<std::uint16_t> q(static_cast<std::size_t>(kD) * kHq * columns),
        k(static_cast<std::size_t>(kD) * kHkv * columns), v(k.size());
    for (std::size_t i = 0; i < q.size(); ++i) q[i] = tobf(uniform(7, i, 1.0F));
    for (std::size_t i = 0; i < k.size(); ++i) {
        k[i] = tobf(uniform(8, i, 1.0F));
        v[i] = tobf(uniform(9, i, 1.0F));
    }

    Dev<std::uint16_t> d_index(index), d_kc(kc), d_vc(vc), d_qk(qk), d_qn(qn), d_kn(kn), d_q(q),
        d_k(k), d_v(v);
    Dev<std::int32_t> d_tables(tables), d_rows(table_rows), d_pos(positions);
    std::vector<std::int32_t> valid = cs.valid;
    Dev<std::int32_t> d_valid(valid.empty() ? std::vector<std::int32_t>{0} : valid);
    Dev<std::int32_t> d_sel(std::vector<std::int32_t>(static_cast<std::size_t>(kK) * columns, 7));
    Dev<std::uint16_t> d_out(std::vector<std::uint16_t>(q.size(), 0x7fc0));

    Tensor t_pos(d_pos.p, DType::I32, {W, B});
    Tensor t_valid = valid.empty() ? Tensor() : Tensor(d_valid.p, DType::I32, {B});
    Tensor t_rows(d_rows.p, DType::I32, {B});
    Tensor t_tables(d_tables.p, DType::I32, {L, B + 1});
    Tensor t_index(d_index.p, DType::BF16, {kDi, kP, 1, N});
    Tensor t_qk(d_qk.p, DType::BF16, {qk_rows, columns});
    Tensor t_iq  = t_qk.slice(0, 0, kHi * kDi);
    Tensor t_ik  = t_qk.slice(0, kHi * kDi, kDi);
    Tensor t_qn(d_qn.p, DType::BF16, {kDi});
    Tensor t_kn(d_kn.p, DType::BF16, {kDi});
    Tensor t_sel(d_sel.p, DType::I32, {kK, W, B});
    Tensor t_q(d_q.p, DType::BF16, {kD, kHq, W, B});
    Tensor t_k(d_k.p, DType::BF16, {kD, kHkv, W, B});
    Tensor t_v(d_v.p, DType::BF16, {kD, kHkv, W, B});
    Tensor t_out(d_out.p, DType::BF16, {kD, kHq, W, B});
    PagedKVBatchLayerView cache{
        .k_pages      = Tensor(d_kc.p, DType::BF16, {kD, kP, kHkv, N}),
        .v_pages      = Tensor(d_vc.p, DType::FP16, {kD, kP, kHkv, N}),
        .block_tables = t_tables,
        .head_dim     = kD,
        .num_kv_heads = kHkv,
        .storage      = KvCacheStorage::BFloat16};

    WorkspaceArena workspace(ops::qsa_select_workspace_bytes(g, cs.max_visible, B, columns) + 256);
    ops::qsa_index_append(t_ik, t_pos, t_valid, t_rows, t_tables, t_index, nullptr);
    ops::qsa_select(t_iq, t_qn, t_kn, t_pos, t_valid, t_rows, t_tables, t_index, g,
                    cs.max_visible, workspace, t_sel, nullptr);
    WorkspaceArena attention_workspace(ops::qsa_attention_workspace_bytes(columns) + 256);
    ops::qsa_attention(t_q, t_k, t_v, t_pos, t_valid, t_rows, t_sel, g, 1.0F / 16.0F, cache,
                       attention_workspace, t_out, nullptr);
    if (cudaDeviceSynchronize() != cudaSuccess) {
        std::printf("FAIL %s: CUDA error %s\n", cs.name, cudaGetErrorString(cudaGetLastError()));
        ++failures;
        return;
    }

    // Reference storage after the appends.
    for (int c = 0; c < columns; ++c) {
        if (!live(c)) continue;
        const int b = c / W, pos = positions[c];
        for (int d = 0; d < kDi; ++d)
            index[iaddr(b, pos, d)] = qk[static_cast<std::size_t>(c) * qk_rows + kHi * kDi + d];
        for (int h = 0; h < kHkv; ++h)
            for (int d = 0; d < kD; ++d) {
                const std::size_t src = (static_cast<std::size_t>(c) * kHkv + h) * kD + d;
                kc[kvaddr(b, pos, h, d)] = k[src];
                vc[kvaddr(b, pos, h, d)] = __half_as_ushort(__float2half_rn(bf(v[src])));
            }
    }
    check(d_index.get() == index, "index plane after append");
    check(d_kc.get() == kc, "K plane after append");
    check(d_vc.get() == vc, "V plane after append");

    // Selection oracle.
    double inv_freq[32];
    for (int i = 0; i < 32; ++i)
        inv_freq[i] = static_cast<float>(std::pow(static_cast<double>(g.theta), -2.0 * i / 64.0));
    auto rope = [&](const std::vector<double>& x, int pos) {
        std::vector<double> y(x);
        for (int i = 0; i < 32; ++i) {
            const double phi = static_cast<float>(static_cast<float>(pos) *
                                                  static_cast<float>(inv_freq[i]));
            const double c = std::cos(phi), s = std::sin(phi);
            y[i]      = rbf(x[i] * c - x[i + 32] * s);
            y[i + 32] = rbf(x[i + 32] * c + x[i] * s);
        }
        return y;
    };
    auto norm = [&](std::vector<double> x, const std::vector<std::uint16_t>& w) {
        double ss = 0;
        for (double e : x) ss += e * e;
        const double inv = 1.0 / std::sqrt(ss / kDi + g.eps);
        for (int d = 0; d < kDi; ++d) x[d] = rbf(x[d] * inv * (1.0 + bf(w[d])));
        return x;
    };
    const std::vector<std::int32_t> sel = d_sel.get();
    int checked = 0, tolerated = 0;
    for (int c = 0; c < columns; ++c) {
        const std::int32_t* got = &sel[static_cast<std::size_t>(c) * kK];
        if (!live(c)) {
            check(std::all_of(got, got + kK, [](int x) { return x == -1; }), "dead column");
            continue;
        }
        const int b = c / W, p = positions[c], nb = (p + 1) / kC;
        std::vector<int> want;
        std::vector<double> score(nb, 0.0);
        if (nb <= kK) {
            for (int i = 0; i < nb; ++i) want.push_back(i);
        } else {
            std::vector<std::vector<double>> qh(kHi);
            for (int h = 0; h < kHi; ++h) {
                std::vector<double> x(kDi);
                for (int d = 0; d < kDi; ++d)
                    x[d] = bf(qk[static_cast<std::size_t>(c) * qk_rows + h * kDi + d]);
                qh[h] = rope(norm(x, qn), p);
            }
            for (int i = 0; i < nb; ++i) {
                std::vector<double> u(kDi);
                for (int d = 0; d < kDi; ++d) {
                    float sum = 0.0F;
                    for (int t = 0; t < kC; ++t) sum += bf(index[iaddr(b, kC * i + t, d)]);
                    u[d] = rbf(sum / kC);
                }
                const std::vector<double> key = rope(norm(u, kn), kC * i);
                for (int h = 0; h < kHi; ++h) {
                    double dot = 0;
                    for (int d = 0; d < kDi; ++d) dot += qh[h][d] * key[d];
                    score[i] += std::max(dot, 0.0);
                }
                score[i] /= std::sqrt(static_cast<double>(kDi));
            }
            std::vector<int> order(nb);
            std::iota(order.begin(), order.end(), 0);
            std::stable_sort(order.begin(), order.end(),
                             [&](int a, int z) { return score[a] > score[z]; });
            want.assign(order.begin(), order.begin() + kK);
            std::sort(want.begin(), want.end());
        }
        std::vector<int> have;
        for (int i = 0; i < kK && got[i] >= 0; ++i) have.push_back(got[i]);
        bool ok = std::is_sorted(have.begin(), have.end()) && have.size() == want.size() &&
                  std::all_of(got + have.size(), got + kK, [](int x) { return x == -1; });
        if (ok && have != want) {
            // Accept differences confined to near ties at the selection boundary.
            std::vector<double> sorted(score);
            std::sort(sorted.begin(), sorted.end(), std::greater<>());
            const double edge = sorted[kK - 1];
            const double tol  = 4e-3 * std::fabs(edge) + 1e-6;
            std::vector<int> diff;
            std::set_symmetric_difference(have.begin(), have.end(), want.begin(), want.end(),
                                          std::back_inserter(diff));
            for (int i : diff) ok = ok && std::fabs(score[i] - edge) <= tol;
            if (ok) ++tolerated;
        }
        if (!ok) std::printf("  %s column %d: selection mismatch\n", cs.name, c);
        check(ok, "selection");
        ++checked;
    }

    // Attention oracle over the GPU's (validated) selection.
    const std::vector<std::uint16_t> out = d_out.get();
    double worst = 0;
    int bad = 0;
    for (int c = 0; c < columns; ++c) {
        const int b = c / W, p = positions[c], nb = (p + 1) / kC;
        std::vector<int> tokens;
        if (live(c)) {
            for (int i = 0; i < std::min(nb, kK); ++i)
                for (int t = 0; t < kC; ++t) tokens.push_back(sel[static_cast<std::size_t>(c) * kK + i] * kC + t);
            for (int t = nb * kC; t <= p; ++t) tokens.push_back(t);
        }
        for (int h = 0; h < kHq; ++h) {
            const int kvh = h / (kHq / kHkv);
            std::vector<double> ref(kD, 0.0);
            if (!tokens.empty()) {
                std::vector<double> s(tokens.size());
                double m = -1e300;
                for (std::size_t t = 0; t < tokens.size(); ++t) {
                    double dot = 0;
                    for (int d = 0; d < kD; ++d)
                        dot += bf(q[((static_cast<std::size_t>(c) * kHq + h) * kD) + d]) *
                               bf(kc[kvaddr(b, tokens[t], kvh, d)]);
                    s[t] = dot / 16.0;
                    m    = std::max(m, s[t]);
                }
                double l = 0;
                for (double& e : s) l += (e = std::exp(e - m));
                for (std::size_t t = 0; t < tokens.size(); ++t)
                    for (int d = 0; d < kD; ++d)
                        ref[d] += s[t] / l *
                                  __half2float(__ushort_as_half(vc[kvaddr(b, tokens[t], kvh, d)]));
            }
            for (int d = 0; d < kD; ++d) {
                const double got = bf(out[(static_cast<std::size_t>(c) * kHq + h) * kD + d]);
                const double lim = std::ldexp(1.0, -8) * std::fabs(ref[d]) + 2e-4;
                const double e   = std::fabs(got - ref[d]);
                worst            = std::max(worst, e / lim);
                if (!(e <= lim) && bad++ < 3)
                    std::printf("  %s out[c=%d,h=%d,d=%d] got %.6g ref %.6g\n", cs.name, c, h, d,
                                got, ref[d]);
            }
        }
    }
    check(bad == 0, "attention output");
    std::printf("%s: %d columns, %d near-tie selections, attention worst %.3f of tolerance\n",
                cs.name, checked, tolerated, worst);
}

} // namespace

int main() {
    // Long prefill chunk: selection active (nb 750..765 > 512) across a page boundary.
    run({"prefill_sparse", 64, {3000}, {}, 4096});
    // Short prompt: every block plus the tail is attended.
    run({"prefill_dense", 37, {1990}, {}, 2048});
    // Batched decode rows at different depths with one dead row (dummy position).
    run({"decode_batch", 1, {5000, 2052, 100, 7000}, {1, 1, 1, 0}, 8192});
    // Single decode columns: 32 token splits, most of them empty at a short context.
    run({"decode_single_short", 1, {100}, {}, 2048});
    run({"decode_single_long", 1, {6000}, {}, 8192});
    // Multi-token verify-style rows with partial validity.
    run({"verify_batch", 3, {4100, 2500}, {3, 2}, 4608});
    if (failures != 0) {
        std::printf("%d failure(s)\n", failures);
        return 1;
    }
    std::printf("ok\n");
    return 0;
}
