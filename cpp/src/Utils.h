#pragma once
#include <string>
#include <vector>
#include <sstream>
#include <iostream>
#include <codecvt>
#include <locale>
#include <cassert>

// 简单的日志宏替代 absl/log
#define LOG_INFO std::cout << "[INFO] "
#define LOG_WARNING std::cerr << "[WARN] "
#define CHECK(condition) \
    if (!(condition)) { \
        std::cerr << "[FATAL] Check failed: " << #condition << " at " << __FILE__ << ":" << __LINE__ << std::endl; \
        std::terminate(); \
    }

namespace BasicStringUtil {

    // 简单的 UTF-8 <-> UTF-16 转换
    // 注意: std::codecvt_utf8_utf16 在 C++17 被标记为 deprecated，但它是目前最便携的标准方案。
    
    inline void u8tou16(const char* src, size_t len, std::u16string& dst) {
        if (len == 0) { dst = u""; return; }
        try {
            std::wstring_convert<std::codecvt_utf8_utf16<char16_t>, char16_t> converter;
            dst = converter.from_bytes(src, src + len);
        } catch (...) {
            dst = u"";
        }
    }

    inline void u16tou8(const char16_t* src, size_t len, std::string& dst) {
        if (len == 0) { dst = ""; return; }
        try {
            std::wstring_convert<std::codecvt_utf8_utf16<char16_t>, char16_t> converter;
            dst = converter.to_bytes(src, src + len);
        } catch (...) {
            dst = "";
        }
    }

    inline void SplitString(const std::string& str, char delimiter, std::vector<std::string>* result) {
        std::stringstream ss(str);
        std::string item;
        while (std::getline(ss, item, delimiter)) {
            if (!item.empty()) {
                result->push_back(item);
            }
        }
    }
    
    inline void SplitString(const char* str, size_t len, char delimiter, std::vector<std::string>* result) {
        std::string s(str, len);
        SplitString(s, delimiter, result);
    }

    inline std::string DigitToChinese(char c) {
        switch(c) {
            case '0': return "零";
            case '1': return "一";
            case '2': return "二";
            case '3': return "三";
            case '4': return "四";
            case '5': return "五";
            case '6': return "六";
            case '7': return "七";
            case '8': return "八";
            case '9': return "九";
            default: return "";
        }
    }

    // 简单的数字转中文逻辑
    // 支持整数和小数
    // 例如: 123 -> 一百二十三, 3.14 -> 三点一四
    // 增强: 支持 IP 地址或版本号 (1.2.3.4) -> 一点二点三点四
    inline std::string NumberToChinese(const std::string& num_str) {
        if (num_str.empty()) return "";
        
        // Check for multiple dots -> IP/Version -> Read digits one by one
        int dot_count = 0;
        for (char c : num_str) {
            if (c == '.') dot_count++;
        }
        
        if (dot_count > 1) {
            std::string res;
            for (char c : num_str) {
                if (c == '.') {
                    res += "点";
                } else if (isdigit(c)) {
                    res += DigitToChinese(c);
                } else {
                    // Should not happen if regex is correct, but safe fallback
                    res += c;
                }
            }
            return res;
        }
        
        std::string res;
        size_t start = 0;
        if (num_str[0] == '-') {
            res += "负";
            start = 1;
        } else if (num_str[0] == '+') {
            start = 1;
        }

        size_t dot_pos = num_str.find('.');
        std::string integer_part = num_str.substr(start, dot_pos - start);
        std::string decimal_part;
        if (dot_pos != std::string::npos) {
            decimal_part = num_str.substr(dot_pos + 1);
        }

        // 处理整数部分
        if (integer_part.empty()) {
            res += "零";
        } else {
            // 简单的中文数字读法实现
            // 分组：每4位一组 (个, 万, 亿)
            // 由于 C++ 处理 UTF-8 字符串比较麻烦，这里尽量简化逻辑
            // 也可以选择直接按位读（如果是编号），但通常 TTS 需要数值读法。
            // 简单实现：如果太长（>12位），按位读；否则按数值读。
            
            if (integer_part.length() > 12) {
                for (char c : integer_part) {
                    res += DigitToChinese(c);
                }
            } else {
                // 数值读法
                const char* units[] = {"", "十", "百", "千"};
                const char* big_units[] = {"", "万", "亿", "兆"};
                
                // 去除前导零
                size_t first_nonzero = integer_part.find_first_not_of('0');
                if (first_nonzero == std::string::npos) {
                    res += "零";
                } else {
                    std::string s = integer_part.substr(first_nonzero);
                    int len = s.length();
                    
                    // 倒序处理，每4位一组
                    int group_count = (len + 3) / 4;
                    
                    bool zero_flag = false; // 前面是否有零需要补
                    
                    for (int i = 0; i < group_count; ++i) {
                        int group_idx = group_count - 1 - i; // 当前处理的是第几组（从高位到低位）
                        int start_idx = std::max(0, len - (i + 1) * 4);
                        int end_idx = len - i * 4;
                        std::string group_str = s.substr(start_idx, end_idx - start_idx);
                        
                        std::string group_res;
                        bool group_has_value = false;
                        bool last_is_zero = false;
                        
                        int g_len = group_str.length();
                        for (int j = 0; j < g_len; ++j) {
                            char digit = group_str[j];
                            int unit_idx = g_len - 1 - j;
                            
                            if (digit == '0') {
                                last_is_zero = true;
                            } else {
                                if (last_is_zero) {
                                    group_res += "零";
                                    last_is_zero = false;
                                }
                                // 处理 "一十" -> "十" 的情况 (仅在首位且值为1且单位为十)
                                // 但如果是 "一百一十"，中间的 "一" 不能省。
                                // 这里的逻辑简化：如果是整个数字的开头，且是十位，且是1，则省去一
                                // e.g. 12 -> 十二, 112 -> 一百一十二
                                if (digit == '1' && unit_idx == 1 && group_res.empty() && zero_flag == false && i == 0 && g_len == 2) {
                                     // Don't output "一", just unit
                                } else {
                                    group_res += DigitToChinese(digit);
                                }
                                group_res += units[unit_idx];
                                group_has_value = true;
                            }
                        }
                        
                        if (group_has_value) {
                            if (zero_flag && group_res.find("零") != 0) {
                                // 如果前面组有遗留零，或者本组开头不是零（但实际上前面可能有空档），通常由 last_is_zero 控制组内零
                                // 跨组零比较复杂。简单策略：如果上一组有值，本组不是从千位开始（即不满4位），补零
                                // 这里简化：如果之前有非零组，且当前组不满4位，或者高位是0，需要补零。
                                // 为了简化代码，暂不处理复杂的跨组补零，除了最简单的。
                            }
                             // 实际上跨组零通常在 "1001" -> "一千零一"
                             // 如果 s = 10001 (len=5), group 1 = '1', group 0 = '0001'
                             // group 1 res = "一", big_unit = "万"
                             // group 0: '0'->zero, '0'->zero, '0'->zero, '1'->'一'
                             // 应该输出 "一万零一"
                             // 我们可以在每组输出前，如果组内开头是0，且前面已经有输出了，加个零
                             if (!res.empty() && group_str[0] == '0') {
                                 // check if we already ended with zero
                                 // UTF-8 check is hard, just append and fix later or assume
                                 // simple: append "零"
                                 if (res.substr(res.length() - 3) != "零")
                                     res += "零";
                             }
                             
                             res += group_res;
                             res += big_units[i]; // big_units index is reversed? No, big_units index should be `i` from the end
                             // Wait, my `i` loop is from 0 (lowest group) to group_count-1 (highest).
                             // Actually I want to process from Highest to Lowest.
                             // Let's restart the loop logic above to be clearer.
                        }
                    }
                    
                    // Re-implementation with correct order: High to Low
                    res = ""; // clear and restart
                    if (num_str[0] == '-') res += "负";
                    
                    int remaining = len;
                    bool need_zero = false;
                    
                    for (int i = 0; i < len; ++i) {
                        int digit = s[i] - '0';
                        int pos = len - 1 - i; // 10^pos
                        int unit_idx = pos % 4;
                        int big_unit_idx = pos / 4;
                        
                        if (digit == 0) {
                            if (unit_idx == 0 && big_unit_idx > 0 && (need_zero || (i > 0 && (s[i-1]-'0')!=0) )) {
                                // End of a big unit group (万, 亿)
                                // If the whole group was 0, we don't add unit. 
                                // But we need to check if this group had any value.
                                // Complex. Let's fallback to a simpler recursive or chunk-based approach.
                            }
                            need_zero = true;
                        } else {
                            if (need_zero) {
                                res += "零";
                                need_zero = false;
                            }
                            // 处理 "一十" -> "十" (仅在值在10-19之间)
                            // 10-19: s.length() == 2, i=0, digit=1
                            if (digit == 1 && unit_idx == 1 && len == 2 && i == 0) {
                                // skip "一"
                            } else {
                                res += DigitToChinese(s[i]);
                            }
                            res += units[unit_idx];
                        }
                        
                        if (unit_idx == 0 && big_unit_idx > 0) {
                            // Check if this 4-digit group had any non-zero
                            // Look back up to 4 chars
                            bool group_has_value = false;
                            int start_chk = std::max(0, i - 3);
                            for(int k=start_chk; k<=i; ++k) if(s[k] != '0') group_has_value = true;
                            
                            if (group_has_value) {
                                res += big_units[big_unit_idx];
                                need_zero = false; // Unit added, reset zero pending? 
                                // No, "10001" -> One Wan Zero One. 
                                // If we finish "Wan", next is "0001". next digit is 0, so need_zero becomes true.
                            }
                        }
                    }
                }
            }
        }

        // 处理小数部分
        if (!decimal_part.empty()) {
            res += "点";
            for (char c : decimal_part) {
                res += DigitToChinese(c);
            }
        }
        
        return res;
    }
}

class TextProcessor {
public:
    virtual ~TextProcessor() = default;
    // Return pairs of (word, pos_tag)
    virtual std::vector<std::pair<std::string, std::string>> cut(const std::string& text) = 0;
    virtual std::vector<std::string> word_to_pinyin(const std::string& word) = 0;
    virtual std::string convert_numbers(const std::string& text) = 0;
};
