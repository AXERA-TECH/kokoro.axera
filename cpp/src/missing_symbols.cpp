#include <cstdint>
#include <cstring>

// 提供缺失的符号实现
extern "C" {
    // 为char类型实现查找函数
    uint64_t __std_find_first_of_trivial_pos_1(const char* const first, uint64_t size, const char* const s_first, uint64_t s_size) {
        for (uint64_t i = 0; i < size; ++i) {
            for (uint64_t j = 0; j < s_size; ++j) {
                if (first[i] == s_first[j]) {
                    return i;
                }
            }
        }
        return size;
    }

    // 为wchar_t类型实现查找函数
    uint64_t __std_find_first_of_trivial_pos_2(const wchar_t* const first, uint64_t size, const wchar_t* const s_first, uint64_t s_size) {
        for (uint64_t i = 0; i < size; ++i) {
            for (uint64_t j = 0; j < s_size; ++j) {
                if (first[i] == s_first[j]) {
                    return i;
                }
            }
        }
        return size;
    }
}
