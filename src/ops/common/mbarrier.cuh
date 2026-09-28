#pragma once

#include "ops/common/memory.cuh"

#include <cstdint>

namespace ninfer::ops {

__device__ __forceinline__ void cta_mbarrier_init(std::uint64_t* barrier, std::uint32_t arrivals) {
    
#if __CUDA_ARCH__ >= 900
    asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;"
                 :
                 : "r"(smem_addr(barrier)), "r"(arrivals)
                 : "memory");
#else
    __trap(); // SM90+ only; never dispatched on sm_89
#endif

}

__device__ __forceinline__ void cta_mbarrier_wait(std::uint64_t* barrier, std::uint32_t phase) {
    constexpr std::uint32_t kSuspendTicks = 0x989680;
    
#if __CUDA_ARCH__ >= 900
    asm volatile("{\n"
                 ".reg .pred done;\n"
                 "wait_loop_%=: \n"
                 "mbarrier.try_wait.parity.shared::cta.b64 done, [%0], %1, %2;\n"
                 "@done bra wait_done_%=;\n"
                 "bra wait_loop_%=;\n"
                 "wait_done_%=: \n"
                 "}\n"
                 :
                 : "r"(smem_addr(barrier)), "r"(phase), "r"(kSuspendTicks)
                 : "memory");
#else
    __trap(); // SM90+ only; never dispatched on sm_89
#endif

}

__device__ __forceinline__ void cta_mbarrier_arrive(std::uint64_t* barrier) {
    
#if __CUDA_ARCH__ >= 900
    asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];" : : "r"(smem_addr(barrier)) : "memory");
#else
    __trap(); // SM90+ only; never dispatched on sm_89
#endif

}

__device__ __forceinline__ void cta_mbarrier_arrive_expect_tx(std::uint64_t* barrier,
                                                              std::uint32_t bytes) {
    
#if __CUDA_ARCH__ >= 900
    asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;"
                 :
                 : "r"(smem_addr(barrier)), "r"(bytes)
                 : "memory");
#else
    __trap(); // SM90+ only; never dispatched on sm_89
#endif

}

__device__ __forceinline__ void cta_mbarrier_fence_init() {
    
#if __CUDA_ARCH__ >= 900
    asm volatile("fence.mbarrier_init.release.cluster;" : : : "memory");
#else
    __trap(); // SM90+ only; never dispatched on sm_89
#endif

}

} // namespace ninfer::ops
