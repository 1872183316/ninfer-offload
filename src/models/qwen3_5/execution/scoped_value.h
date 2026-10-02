#pragma once

namespace ninfer::models::qwen3_5::execution {

// Binds `slot` to `value` for a scope and restores the previous value on exit.
template <class T>
class ScopedValue {
public:
    ScopedValue(T& slot, T value) : slot_(slot), previous_(slot) { slot_ = value; }

    ScopedValue(const ScopedValue&)            = delete;
    ScopedValue& operator=(const ScopedValue&) = delete;

    ~ScopedValue() { slot_ = previous_; }

private:
    T& slot_;
    T previous_;
};

} // namespace ninfer::models::qwen3_5::execution
