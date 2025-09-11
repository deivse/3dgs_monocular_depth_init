#ifndef NATIVE_MODULES_SUBSAMPLING_SRC_STACK
#define NATIVE_MODULES_SUBSAMPLING_SRC_STACK

#include <stdexcept>
#include <vector>

namespace mdi {

template<typename T>
class Stack
{
    std::vector<T> _data;

public:
    void push(T&& item) { _data.push_back(std::forward<T>(item)); }

    template<typename... Args>
    void emplace(Args&&... args) {
        _data.emplace_back(std::forward<Args>(args)...);
    }

    T pop() {
        if (_data.empty()) {
            throw std::runtime_error("Pop from empty stack");
        }
        T item = _data.back();
        _data.pop_back();
        return item;
    }

    bool empty() const { return _data.empty(); }

    size_t size() const { return _data.size(); }
};
} // namespace mdi

#endif /* NATIVE_MODULES_SUBSAMPLING_SRC_STACK */
