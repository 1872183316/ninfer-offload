#pragma once

#include "ninfer/types.h"

namespace ninfer::test {

// NVFP4 weights and the NVFP4/K8V4 KV encodings exist only in sm_120a builds; other targets
// reject them before execution, so their oracle cases do not apply there.
inline constexpr bool kTargetHasNvfp4 = NINFER_TARGET_SM >= 120;

[[nodiscard]] inline bool kv_storage_on_target(KvCacheStorage storage) noexcept {
    return kTargetHasNvfp4 || (storage != KvCacheStorage::Nvfp4Group16 &&
                               storage != KvCacheStorage::Fp8KeyNvfp4Value);
}

} // namespace ninfer::test
