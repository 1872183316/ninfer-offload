// Qwen4-Exp hyper-connection and PLE ops against naive FP64 / exact CPU references.
#include "ninfer/ops/gated_rmsnorm.h"
#include "ninfer/ops/hyper_connection.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

using namespace ninfer;

namespace {

int failures = 0;

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

std::uint64_t mix(std::uint64_t v) {
    v += 0x9e3779b97f4a7c15ULL;
    v = (v ^ (v >> 30)) * 0xbf58476d1ce4e5b9ULL;
    v = (v ^ (v >> 27)) * 0x94d049bb133111ebULL;
    return v ^ (v >> 31);
}

std::vector<std::uint16_t> random_bf16(std::size_t n, std::uint64_t seed, float range) {
    std::vector<std::uint16_t> v(n);
    for (std::size_t i = 0; i < n; ++i) {
        v[i] = tobf((static_cast<float>(mix(seed * 1315423911ULL + i) % 20001) / 10000.0F - 1.0F) *
                    range);
    }
    return v;
}

template <class T>
struct Dev {
    T* p = nullptr;
    std::size_t n;
    explicit Dev(const std::vector<T>& h) : n(h.size()) {
        cudaMalloc(&p, n * sizeof(T));
        cudaMemcpy(p, h.data(), n * sizeof(T), cudaMemcpyHostToDevice);
    }
    explicit Dev(std::size_t count) : n(count) {
        cudaMalloc(&p, n * sizeof(T));
        cudaMemset(p, 0, n * sizeof(T));
    }
    ~Dev() { cudaFree(p); }
    std::vector<T> get() const {
        std::vector<T> h(n);
        cudaDeviceSynchronize();
        cudaMemcpy(h.data(), p, n * sizeof(T), cudaMemcpyDeviceToHost);
        return h;
    }
};

void expect(const char* label, const std::vector<std::uint16_t>& got, const std::vector<double>& ref,
            double rel = std::ldexp(1.0, -8), double abs = 1e-3) {
    double worst = 0;
    int bad      = 0;
    for (std::size_t i = 0; i < ref.size(); ++i) {
        const double e   = std::fabs(bf(got[i]) - ref[i]);
        const double lim = rel * std::fabs(ref[i]) + abs;
        worst            = std::max(worst, e / lim);
        if (!(e <= lim)) {
            if (bad < 3) std::printf("  %s[%zu] got %.6g ref %.6g\n", label, i, bf(got[i]), ref[i]);
            ++bad;
        }
    }
    std::printf("%s %s worst=%.3f\n", bad ? "FAIL" : "ok  ", label, worst);
    failures += bad;
}

double sigm(double v) { return 1.0 / (1.0 + std::exp(-v)); }

void hyper_pieces() {
    const int H = 2560, C = 4, T = 3;
    const auto x  = random_bf16(std::size_t(C) * H * T, 1, 3.0F);
    const auto w  = random_bf16(std::size_t(C) * H, 2, 0.5F);
    Dev<std::uint16_t> dx(x), dw(w), dout(std::size_t(C) * H * T);
    Tensor tx(dx.p, DType::BF16, {C * H, T}), tw(dw.p, DType::BF16, {C * H}),
        to(dout.p, DType::BF16, {C * H, T});
    ops::hc_grouped_rmsnorm(tx, tw, H, 1e-6F, to, nullptr);
    std::vector<double> ref(x.size());
    for (int t = 0; t < T; ++t)
        for (int s = 0; s < C; ++s) {
            double ss = 0;
            for (int h = 0; h < H; ++h) ss += std::pow(bf(x[(t * C + s) * H + h]), 2);
            const double inv = 1.0 / std::sqrt(ss / H + 1e-6);
            for (int h = 0; h < H; ++h) {
                const auto i = std::size_t(t * C + s) * H + h;
                ref[i]       = bf(x[i]) * inv * (1.0 + bf(w[s * H + h]));
            }
        }
    const auto xn = dout.get();
    expect("hc_grouped_rmsnorm", xn, ref);

    // gated mean over the normalized streams with a random pre-sigmoid gate
    const auto g = random_bf16(x.size(), 3, 4.0F);
    Dev<std::uint16_t> dg(g), dm(std::size_t(H) * T);
    Tensor tg(dg.p, DType::BF16, {C * H, T}), tm(dm.p, DType::BF16, {H, T});
    ops::hc_gated_mean(tg, to, C, tm, nullptr);
    std::vector<double> mref(std::size_t(H) * T);
    for (int t = 0; t < T; ++t)
        for (int h = 0; h < H; ++h) {
            double sum = 0;
            for (int s = 0; s < C; ++s) {
                const auto i = std::size_t(t * C + s) * H + h;
                sum += sigm(bf(g[i])) * bf(xn[i]);
            }
            mref[std::size_t(t) * H + h] = sum / C;
        }
    expect("hc_gated_mean", dm.get(), mref);

    // scaled SiLU
    auto y = random_bf16(320 * T, 4, 6.0F);
    Dev<std::uint16_t> dy(y);
    Tensor ty(dy.p, DType::BF16, {320, T});
    ops::hc_scaled_silu(ty, 0.25F, nullptr);
    std::vector<double> yref(y.size());
    for (std::size_t i = 0; i < y.size(); ++i) {
        const double v = bf(y[i]) * 0.25;
        yref[i]        = v * sigm(v);
    }
    expect("hc_scaled_silu", dy.get(), yref);

    // injection weights and combine
    const auto inj = random_bf16(std::size_t(C) * T, 5, 8.0F);
    const auto blk = random_bf16(std::size_t(H) * T, 6, 2.0F);
    Dev<std::uint16_t> dinj(inj), dblk(blk), dx2(x);
    Dev<float> dwt(std::size_t(C) * T);
    Tensor tinj(dinj.p, DType::BF16, {C, T}), twt(dwt.p, DType::FP32, {C, T}),
        tblk(dblk.p, DType::BF16, {H, T}), tx2(dx2.p, DType::BF16, {C * H, T});
    ops::hc_injection_weights(tinj, C, twt, nullptr);
    ops::hc_combine(tx2, tblk, twt, nullptr);
    std::vector<double> cref(x.size());
    for (int t = 0; t < T; ++t)
        for (int s = 0; s < C; ++s) {
            const double wt = 2.0 * sigm(bf(inj[t * C + s]) / C);
            for (int h = 0; h < H; ++h) {
                const auto i = std::size_t(t * C + s) * H + h;
                cref[i]      = bf(x[i]) + bf(blk[std::size_t(t) * H + h]) * wt;
            }
        }
    expect("hc_combine", dx2.get(), cref);

    // PLE gate
    const auto key = random_bf16(x.size(), 7, 2.0F);
    Dev<std::uint16_t> dkey(key), dgate(x.size());
    Tensor tkey(dkey.p, DType::BF16, {C * H, T}), tgate(dgate.p, DType::BF16, {C * H, T});
    ops::ple_gate(tkey, to, tblk, C, tgate, nullptr);
    std::vector<double> gref(x.size());
    for (int t = 0; t < T; ++t)
        for (int s = 0; s < C; ++s) {
            double dot = 0;
            for (int h = 0; h < H; ++h) {
                const auto i = std::size_t(t * C + s) * H + h;
                dot += bf(key[i]) * bf(xn[i]);
            }
            double v = dot / std::sqrt(double(H));
            v        = std::copysign(std::sqrt(std::max(std::fabs(v), 1e-6)), v);
            for (int h = 0; h < H; ++h) {
                gref[std::size_t(t * C + s) * H + h] = sigm(v) * bf(blk[std::size_t(t) * H + h]);
            }
        }
    expect("ple_gate", dgate.get(), gref, std::ldexp(1.0, -8), 2e-3);

    // sigmoid-gated RMSNorm (Qwen4-Exp GDN output norm): 48 heads of 128
    const auto o  = random_bf16(128 * 48 * T, 8, 3.0F);
    const auto z  = random_bf16(o.size(), 9, 5.0F);
    const auto nw = random_bf16(128, 10, 1.0F);
    Dev<std::uint16_t> dox(o), dz(z), dnw(nw), don(o.size());
    Tensor tox(dox.p, DType::BF16, {128, 48, T}), tz(dz.p, DType::BF16, {128, 48, T}),
        tnw(dnw.p, DType::BF16, {128}), ton(don.p, DType::BF16, {128, 48, T});
    ops::gated_rmsnorm_sigmoid(tox, tnw, tz, 1e-6F, ton, nullptr);
    std::vector<double> nref(o.size());
    for (std::size_t r = 0; r < o.size() / 128; ++r) {
        double ss = 0;
        for (int d = 0; d < 128; ++d) ss += std::pow(bf(o[r * 128 + d]), 2);
        const double inv = 1.0 / std::sqrt(ss / 128 + 1e-6);
        for (int d = 0; d < 128; ++d) {
            nref[r * 128 + d] = bf(o[r * 128 + d]) * inv * bf(nw[d]) * sigm(bf(z[r * 128 + d]));
        }
    }
    expect("gated_rmsnorm_sigmoid", don.get(), nref);
}

// Literal port of the reference shift-right-ignore-eos hashing over a whole history.
std::vector<std::int32_t> reference_rows(const std::vector<std::int64_t>& tokens,
                                         const ops::PleHash& h) {
    const int context = h.ngram_size - 1;
    std::vector<std::int64_t> history(context, h.eos);
    history.insert(history.end(), tokens.begin(), tokens.end());
    const int n = static_cast<int>(history.size());
    std::vector<std::int32_t> rows;
    for (int p = context; p < n; ++p) {
        int previous_eos = -1;
        for (int q = 0; q < p; ++q)
            if (history[q] == h.eos) previous_eos = q;
        std::uint64_t mixed = static_cast<std::uint64_t>(history[p]) * h.multipliers[0];
        for (int order = 2; order <= h.ngram_size; ++order) {
            const int shift = order - 1;
            const bool valid = p - (previous_eos + 1) >= shift && p - shift >= 0;
            const std::uint64_t shifted = valid ? history[p - shift] : h.eos;
            mixed ^= shifted * h.multipliers[order - 1];
            for (int j = 0; j < h.heads_per_ngram; ++j) {
                const int head = (order - 2) * h.heads_per_ngram + j;
                rows.push_back(static_cast<std::int32_t>(mixed % h.vocab[head] + h.offset[head]));
            }
        }
    }
    return rows;
}

void ngram() {
    ops::PleHash h;
    h.ngram_size      = 3;
    h.heads_per_ngram = 8;
    h.eos             = 248044;
    const std::uint64_t mult[3] = {23703573157769ULL, 20109073645365ULL, 8052911324071ULL};
    for (int i = 0; i < 3; ++i) h.multipliers[i] = mult[i];
    std::uint64_t off = 0;
    for (int j = 0; j < 16; ++j) {
        h.vocab[j]  = 20000003ULL + 20ULL * j + (j % 3);
        h.offset[j] = off;
        off += h.vocab[j];
    }
    // Two sequences, fed in chunks of 5 then 1 (decode) then 3, with eos inside.
    const std::vector<std::vector<std::int64_t>> seqs = {
        {11, 900, 248044, 77, 5, 248044, 248044, 3001, 12},
        {248044, 42, 43, 44, 45, 46, 47, 48, 49}};
    const int slots = 4, pitch = 64;
    Dev<std::int32_t> state(slots * pitch / 4);
    ops::PleStateView view{state.p, pitch, 0, 0};
    std::vector<std::int32_t> src = {0, 1}, dst = {2, 3};
    int begin = 0;
    std::vector<std::vector<std::int32_t>> got(2);
    for (int chunk : {5, 1, 3}) {
        std::vector<std::int32_t> ids(chunk * 2);
        for (int b = 0; b < 2; ++b)
            for (int t = 0; t < chunk; ++t) ids[b * chunk + t] = static_cast<std::int32_t>(seqs[b][begin + t]);
        Dev<std::int32_t> dids(ids), dsrc(src), ddst(dst), drows(std::size_t(16) * chunk * 2);
        Tensor tids(dids.p, DType::I32, {chunk, 2}), tsrc(dsrc.p, DType::I32, {2}),
            tdst(ddst.p, DType::I32, {2}), trows(drows.p, DType::I32, {16, chunk, 2});
        ops::ple_ngram_rows(tids, Tensor{}, h, view, tsrc, tdst, trows, nullptr);
        const auto r = drows.get();
        for (int b = 0; b < 2; ++b)
            got[b].insert(got[b].end(), r.begin() + b * chunk * 16, r.begin() + (b + 1) * chunk * 16);
        std::swap(src, dst);
        begin += chunk;
    }
    int bad = 0;
    for (int b = 0; b < 2; ++b) {
        const auto ref = reference_rows(seqs[b], h);
        bad += ref != got[b];
    }
    std::printf("%s ple_ngram_rows (chunked, eos-aware)\n", bad ? "FAIL" : "ok  ");
    failures += bad;
}

void conv() {
    const int C = 64, K = 4, D = 3, hist = 9, W = 7;
    const auto x = random_bf16(std::size_t(C) * W, 11, 2.0F);
    const auto w = random_bf16(std::size_t(K) * C, 12, 1.0F);
    Dev<std::uint16_t> dw(w);
    Tensor tw(dw.p, DType::BF16, {C, K});
    const std::int64_t pitch = hist * C * 2;
    // one pass over 7 columns vs 4 + 3 through the slot state
    Dev<std::uint16_t> state_a(std::size_t(hist) * C * 2), state_b(std::size_t(hist) * C * 2);
    ops::PleStateView va{state_a.p, pitch, 0, 0}, vb{state_b.p, pitch, 0, 0};
    std::vector<std::int32_t> s0 = {0}, s1 = {1};
    Dev<std::int32_t> d0(s0), d1(s1);
    Tensor t0(d0.p, DType::I32, {1}), t1(d1.p, DType::I32, {1});
    Dev<std::uint16_t> dx(x), dfull(x.size());
    Tensor tx(dx.p, DType::BF16, {C, W, 1}), tfull(dfull.p, DType::BF16, {C, W, 1});
    ops::ple_dilated_conv_silu(tx, tw, D, Tensor{}, va, t0, t0, tfull, nullptr);
    std::vector<std::uint16_t> xa(x.begin(), x.begin() + 4 * C), xb(x.begin() + 4 * C, x.end());
    Dev<std::uint16_t> dxa(xa), dxb(xb), dpa(xa.size()), dpb(xb.size());
    Tensor txa(dxa.p, DType::BF16, {C, 4, 1}), txb(dxb.p, DType::BF16, {C, 3, 1}),
        tpa(dpa.p, DType::BF16, {C, 4, 1}), tpb(dpb.p, DType::BF16, {C, 3, 1});
    ops::ple_dilated_conv_silu(txa, tw, D, Tensor{}, vb, t0, t1, tpa, nullptr);
    ops::ple_dilated_conv_silu(txb, tw, D, Tensor{}, vb, t1, t0, tpb, nullptr);
    std::vector<double> ref(x.size());
    for (int t = 0; t < W; ++t)
        for (int c = 0; c < C; ++c) {
            double acc = 0;
            for (int k = 0; k < K; ++k) {
                const int src = t - (K - 1 - k) * D;
                if (src >= 0) acc += bf(w[k * C + c]) * bf(x[std::size_t(src) * C + c]);
            }
            ref[std::size_t(t) * C + c] = acc * sigm(acc);
        }
    expect("ple_dilated_conv_silu (one pass)", dfull.get(), ref);
    auto split = dpa.get();
    const auto second = dpb.get();
    split.insert(split.end(), second.begin(), second.end());
    expect("ple_dilated_conv_silu (4+3 via state)", split, ref);
}

} // namespace

int main() {
    int count = 0;
    if (cudaGetDeviceCount(&count) != cudaSuccess || count == 0) {
        std::printf("SKIP: no CUDA device\n");
        return 77;
    }
    hyper_pieces();
    ngram();
    conv();
    std::printf("%s (%d failures)\n", failures ? "FAILED" : "PASSED", failures);
    return failures ? 1 : 0;
}
