// Host routed-expert throughput at real bank sizes, with a DRAM read-bandwidth reference.
//
// usage: ninfer_host_moe_bench [threads=12] [tokens=1] [hidden=2048] [inter=512] [experts=256]
//                              [top_k=8] [gate_up=q4] [down=q5] [layers=8] [shared=0]
// Each measured call touches a different layer's banks, so selected rows stream from DRAM.
#include "ops/sparse_moe/host/host_moe.h"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

using namespace ninfer;
using namespace ninfer::ops::host;

namespace {

struct Codec {
    int group;
    int code;
    int high;
};

Codec codec(QType q) {
    switch (q) {
    case QType::Q4_G64_FP16:
        return {64, 32, 0};
    case QType::Q5_G64_FP16:
        return {64, 32, 8};
    case QType::Q6_G64_FP16:
        return {64, 32, 16};
    default:
        return {32, 32, 0};
    }
}

QType parse(const char* s) {
    if (!std::strcmp(s, "q4")) return QType::Q4_G64_FP16;
    if (!std::strcmp(s, "q5")) return QType::Q5_G64_FP16;
    if (!std::strcmp(s, "q6")) return QType::Q6_G64_FP16;
    return QType::Q8_G32_FP16;
}

struct Bank {
    std::vector<std::uint8_t> codes;
    std::vector<std::uint8_t> high;
    std::vector<std::uint16_t> scales;
    RowSplitMatrix m;
    std::size_t bytes = 0;
};

Bank make_bank(QType q, std::int32_t rows, std::int32_t k, std::mt19937_64& rng) {
    const Codec c   = codec(q);
    const int gpr   = (k + 127) / 128 * 128 / c.group;
    const auto grps = static_cast<std::size_t>(rows) * gpr;
    Bank b;
    b.codes.resize(grps * c.code);
    b.high.resize(grps * c.high);
    b.scales.assign(grps, 0x2000); // binary16 2^-7
    for (auto& v : b.codes) v = static_cast<std::uint8_t>(rng());
    for (auto& v : b.high) v = static_cast<std::uint8_t>(rng());
    b.m.codes          = b.codes.data();
    b.m.high           = c.high ? b.high.data() : nullptr;
    b.m.scales         = b.scales.data();
    b.m.qtype          = q;
    b.m.rows           = rows;
    b.m.k              = k;
    b.m.groups_per_row = gpr;
    b.bytes            = b.codes.size() + b.high.size() + b.scales.size() * 2;
    return b;
}

double now() {
    return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch())
        .count();
}

double read_bandwidth(HostThreadPool& pool) {
    const std::size_t n = std::size_t{1} << 28; // 2 GiB of uint64
    std::vector<std::uint64_t> buf(n, 1);
    std::vector<std::uint64_t> sums(64, 0);
    const std::int32_t items = 256;
    double best              = 0;
    for (int rep = 0; rep < 3; ++rep) {
        const double t0 = now();
        pool.parallel_for(items, [&](std::int32_t i) {
            const std::size_t chunk = n / items;
            std::uint64_t s         = 0;
            for (std::size_t j = i * chunk; j < (i + 1) * chunk; ++j) s += buf[j];
            sums[i % 64] += s;
        });
        const double dt = now() - t0;
        best            = std::max(best, n * 8.0 / dt / 1e9);
    }
    return best;
}

} // namespace

int main(int argc, char** argv) {
    const int threads = argc > 1 ? std::atoi(argv[1]) : 12;
    const int T       = argc > 2 ? std::atoi(argv[2]) : 1;
    const int H       = argc > 3 ? std::atoi(argv[3]) : 2048;
    const int I       = argc > 4 ? std::atoi(argv[4]) : 512;
    const int E       = argc > 5 ? std::atoi(argv[5]) : 256;
    const int K       = argc > 6 ? std::atoi(argv[6]) : 8;
    const QType gq    = parse(argc > 7 ? argv[7] : "q4");
    const QType dq    = parse(argc > 8 ? argv[8] : "q5");
    const int layers  = argc > 9 ? std::atoi(argv[9]) : 8;
    // shared=1: every token selects token 0's experts (measures extra tokens per expert).
    const bool shared = argc > 10 && std::atoi(argv[10]) != 0;

    if (!host_kernels_supported()) {
        std::printf("CPU lacks AVX2/FMA/F16C/BMI2\n");
        return 1;
    }
    HostThreadPool pool(threads);
    std::printf("threads=%d DRAM read bandwidth: %.1f GB/s\n", threads, read_bandwidth(pool));

    std::mt19937_64 rng(1);
    std::vector<Bank> gu;
    std::vector<Bank> dn;
    std::vector<HostExpertBanks> banks;
    for (int l = 0; l < layers; ++l) {
        gu.push_back(make_bank(gq, E * 2 * I, H, rng));
        dn.push_back(make_bank(dq, E * H, I, rng));
    }
    for (int l = 0; l < layers; ++l) banks.push_back({gu[l].m, dn[l].m, E, H, I});
    const double expert_bytes = (gu[0].bytes + dn[0].bytes) / static_cast<double>(E);

    std::vector<float> x(static_cast<std::size_t>(T) * H, 0.25F);
    std::vector<std::int32_t> ids(static_cast<std::size_t>(T) * K);
    std::vector<float> alpha(ids.size(), 1.0F / K);
    std::vector<std::uint8_t> on_host(E, 1);
    std::vector<float> out(static_cast<std::size_t>(T) * H);
    HostMoeExecutor exec(pool);

    const int iters   = 400;
    double total_time = 0;
    double total_b    = 0;
    std::uniform_int_distribution<int> pick(0, E - 1);
    for (int it = -40; it < iters; ++it) {
        std::vector<int> uniq;
        for (int t = 0; t < T; ++t) {
            for (int s = 0; s < K; ++s) {
                if (shared && t > 0) {
                    ids[t * K + s] = ids[s];
                    continue;
                }
                int e;
                do { e = pick(rng); } while ([&] {
                    for (int q = 0; q < s; ++q)
                        if (ids[t * K + q] == e) return true;
                    return false;
                }());
                ids[t * K + s] = e;
                if (std::find(uniq.begin(), uniq.end(), e) == uniq.end()) uniq.push_back(e);
            }
        }
        const HostMoeJob job{T, K, x.data(), ids.data(), alpha.data(), on_host.data(), out.data()};
        const double t0 = now();
        exec.run(banks[(it + 40) % layers], job);
        const double dt = now() - t0;
        if (it >= 0) {
            total_time += dt;
            total_b += uniq.size() * expert_bytes;
        }
    }
    std::printf("T=%d H=%d I=%d E=%d top_k=%d banks=%s/%s: %.1f us/layer, %.1f GB/s effective, "
                "%.2f MB/layer\n",
                T, H, I, E, K, argc > 7 ? argv[7] : "q4", argc > 8 ? argv[8] : "q5",
                total_time / iters * 1e6, total_b / total_time / 1e9, total_b / iters / 1e6);
    return 0;
}
