#pragma once

#include <vector>
#include <string>
#include <algorithm>
#include <sstream>
#include <cctype>
#include <regex>
#include <algorithm>
#include <iostream>
#include <codecvt>
#include <locale>


using namespace std;

// 判断是否是UTF-8字符的后续字节
inline bool is_utf8_continuation_byte(unsigned char c) {
    return (c & 0xC0) == 0x80;
}

// 计算UTF-8字符串的字符数（非字节数）
size_t utf8_strlen(const string& str) {
    size_t len = 0;
    for (size_t i = 0; i < str.size(); ) {
        unsigned char c = str[i];
        if ((c & 0x80) == 0) { // ASCII字符
            i += 1;
        } else if ((c & 0xE0) == 0xC0) { // 2字节UTF-8
            i += 2;
        } else if ((c & 0xF0) == 0xE0) { // 3字节UTF-8（包括大部分中文）
            i += 3;
        } else if ((c & 0xF8) == 0xF0) { // 4字节UTF-8
            i += 4;
        } else {
            i++; // 无效UTF-8，跳过
        }
        len++;
    }
    return len;
}

// 合并短句的英文版本
vector<string> merge_short_sentences_en(const vector<string>& sens) {
    vector<string> sens_out;
    for (const auto& s : sens) {
        // 如果前一个句子太短（<=2个单词），就与当前句子合并
        if (!sens_out.empty()) {
            istringstream iss(sens_out.back());
            int word_count = distance(std::istream_iterator<string>(iss), istream_iterator<string>());
            if (word_count <= 2) {
                sens_out.back() += " " + s;
                continue;
            }
        }
        sens_out.push_back(s);
    }
    
    // 处理最后一个句子如果太短的情况
    if (!sens_out.empty() && sens_out.size() > 1) {
        istringstream iss(sens_out.back());
        int word_count = distance(istream_iterator<string>(iss), istream_iterator<string>());
        if (word_count <= 2) {
            sens_out[sens_out.size()-2] += " " + sens_out.back();
            sens_out.pop_back();
        }
    }
    
    return sens_out;
}

// 合并短句的中文版本
vector<string> merge_short_sentences_zh(const vector<string>& sens) {
    vector<string> sens_out;
    for (const auto& s : sens) {
        // 如果前一个句子太短（<=2个字符），就与当前句子合并
        if (!sens_out.empty() && utf8_strlen(sens_out.back()) <= 2) {
            sens_out.back() += " " + s;
        } else {
            sens_out.push_back(s);
        }
    }
    
    // 处理最后一个句子如果太短的情况
    if (!sens_out.empty() && sens_out.size() > 1 && utf8_strlen(sens_out.back()) <= 2) {
        sens_out[sens_out.size()-2] += " " + sens_out.back();
        sens_out.pop_back();
    }
    
    return sens_out;
}

// 替换字符串中的子串
string replace_all(const string& input, const string& from, const string& to) {
    string result = input;
    size_t pos = 0;
    while ((pos = result.find(from, pos)) != string::npos) {
        result.replace(pos, from.length(), to);
        pos += to.length();
    }
    return result;
}

// 分割拉丁语系文本（英文、法文、西班牙文等）
vector<string> split_sentences_latin(const string& text) {
    string processed = text;
    
    // 替换中文标点为英文标点
    processed = replace_all(processed, "。", ".");
    processed = replace_all(processed, "！", ".");
    processed = replace_all(processed, "？", ".");
    processed = replace_all(processed, "；", ".");
    processed = replace_all(processed, "，", ",");
    processed = replace_all(processed, "“", "\"");
    processed = replace_all(processed, "”", "\"");
    processed = replace_all(processed, "‘", "'");
    processed = replace_all(processed, "’", "'");
    
    // 移除特定字符
    string chars_to_remove = "<>()[]\"«»";
    for (char c : chars_to_remove) {
        processed.erase(remove(processed.begin(), processed.end(), c), processed.end());
    }
    
    // 分割句子（简化版，按句号分割）
    vector<string> sentences;
    size_t start = 0;
    size_t end = processed.find('.');
    
    while (end != string::npos) {
        string sentence = processed.substr(start, end - start);
        // 去除前后空白
        sentence.erase(sentence.begin(), find_if(sentence.begin(), sentence.end(), [](int ch) { return !isspace(ch); }));
        sentence.erase(find_if(sentence.rbegin(), sentence.rend(), [](int ch) { return !isspace(ch); }).base(), sentence.end());
        if (!sentence.empty()) {
            sentences.push_back(sentence);
        }
        start = end + 1;
        end = processed.find('.', start);
    }
    
    // 添加最后一部分
    if (start < processed.size()) {
        string sentence = processed.substr(start);
        sentence.erase(sentence.begin(), find_if(sentence.begin(), sentence.end(), [](int ch) { return !isspace(ch); }));
        sentence.erase(find_if(sentence.rbegin(), sentence.rend(), [](int ch) { return !isspace(ch); }).base(), sentence.end());
        if (!sentence.empty()) {
            sentences.push_back(sentence);
        }
    }
    
    return sentences;
}

// 分割中文文本
vector<string> split_sentences_zh(const string& text) {
    string processed = text;
    
    // 替换中文标点为英文标点
    processed = replace_all(processed, "。", ".");
    processed = replace_all(processed, "！", ".");
    processed = replace_all(processed, "？", ".");
    processed = replace_all(processed, "；", ".");
    processed = replace_all(processed, "，", ",");
    
    // 将文本中的换行符、空格和制表符替换为空格
    processed = replace_all(processed, "\n", " ");
    processed = replace_all(processed, "\t", " ");
    processed = replace_all(processed, "  ", " "); // 多个空格合并为一个
    
    // 在标点符号后添加一个特殊标记用于分割
    string punctuation = ".,!?;";
    for (char c : punctuation) {
        string from(1, c);
        string to = from + " $#!";
        processed = replace_all(processed, from, to);
    }
    
    // 分割句子
    vector<string> sentences;
    size_t start = 0;
    size_t end = processed.find("$#!");
    
    while (end != string::npos) {
        string sentence = processed.substr(start, end - start);
        // 去除前后空白
        sentence.erase(sentence.begin(), find_if(sentence.begin(), sentence.end(), [](int ch) { return !isspace(ch); }));
        sentence.erase(find_if(sentence.rbegin(), sentence.rend(), [](int ch) { return !isspace(ch); }).base(), sentence.end());
        if (!sentence.empty()) {
            sentences.push_back(sentence);
        }
        start = end + 3; // "$#!" 长度为3
        end = processed.find("$#!", start);
    }
    
    // 添加最后一部分
    if (start < processed.size()) {
        string sentence = processed.substr(start);
        sentence.erase(sentence.begin(), find_if(sentence.begin(), sentence.end(), [](int ch) { return !isspace(ch); }));
        sentence.erase(find_if(sentence.rbegin(), sentence.rend(), [](int ch) { return !isspace(ch); }).base(), sentence.end());
        if (!sentence.empty()) {
            sentences.push_back(sentence);
        }
    }
    
    return sentences;
}

// // 主分割函数
// vector<string> split_sentence(const string& text, const string& language_str = "a") {
//     if (language_str == "a" || language_str == "b") {
//         return split_sentences_latin(text);
//     } else {
//         return split_sentences_zh(text);
//     }
// }

std::vector<int> intersperse(const std::vector<int>& lst) {
    std::vector<int> result(lst.size() * 2, 0);
    for (size_t i = 0; i < result.size(); i+=2) {
        result[i] = lst[i / 2];
    }
    return result;
}

// 辅助函数：在UTF-8中文字符之间插入空格
std::string insert_spaces_between_chinese_chars(const std::string& text) {
    if (text.empty()) return text;
    
    std::string result;
    result.reserve(text.length() * 2);  // 预分配空间，最多可能翻倍
    
    // 遍历UTF-8字符串，识别中文字符
    for (size_t i = 0; i < text.length(); ) {
        unsigned char c = static_cast<unsigned char>(text[i]);
        
        // 检查是否是中文字符的UTF-8编码
        // 常用汉字范围：0xE4B880-0xE9BEA5 (一-龥)
        if (i + 2 < text.length() && 
            (c == 0xE4 || c == 0xE5 || c == 0xE6 || 
             c == 0xE7 || c == 0xE8 || c == 0xE9)) {
            
            // 可能是中文字符（3字节UTF-8）
            // 复制完整的UTF-8字符
            result.push_back(text[i]);
            result.push_back(text[i+1]);
            result.push_back(text[i+2]);
            
            // 如果不是最后一个字符，在后面添加空格
            if (i + 3 < text.length()) {
                // 检查下一个字符是否是标点或空格
                unsigned char next_c = static_cast<unsigned char>(text[i+3]);
                if (!isspace(next_c) && next_c != 0xE3 && next_c != 0xEF) {
                    // 下一个字符不是空格、中文标点，则添加空格
                    result.push_back(' ');
                }
            }
            
            i += 3;
        }
        // 检查是否是中文标点（3字节）
        else if (i + 2 < text.length() && c == 0xE3) {
            // 中文标点如：。、等
            result.push_back(text[i]);
            result.push_back(text[i+1]);
            result.push_back(text[i+2]);
            i += 3;
        }
        // 检查是否是中文标点（3字节，EF开头）
        else if (i + 2 < text.length() && c == 0xEF) {
            // 中文标点如：！？等
            result.push_back(text[i]);
            result.push_back(text[i+1]);
            result.push_back(text[i+2]);
            i += 3;
        }
        // ASCII字符
        else if (c <= 0x7F) {
            result.push_back(text[i]);
            i++;
        }
        // 其他UTF-8字符（2字节或4字节）
        else if ((c & 0xE0) == 0xC0 && i + 1 < text.length()) {
            // 2字节UTF-8
            result.push_back(text[i]);
            result.push_back(text[i+1]);
            i += 2;
        }
        else if ((c & 0xF0) == 0xE0 && i + 2 < text.length()) {
            // 3字节UTF-8
            result.push_back(text[i]);
            result.push_back(text[i+1]);
            result.push_back(text[i+2]);
            i += 3;
        }
        else if ((c & 0xF8) == 0xF0 && i + 3 < text.length()) {
            // 4字节UTF-8
            result.push_back(text[i]);
            result.push_back(text[i+1]);
            result.push_back(text[i+2]);
            result.push_back(text[i+3]);
            i += 4;
        }
        else {
            // 无法识别，直接复制
            result.push_back(text[i]);
            i++;
        }
    }
    
    return result;
}


std::vector<std::string> split_sentence(const std::string& text, const std::string& lang_code = "a") {
    std::vector<std::string> result;
    if (text.empty()) {
        result.push_back(text);
        return result;
    }

    // 第一步：如果是中文/日文，在字之间插入空格
    std::string processed_text = text;
    if (lang_code == "z" || lang_code == "j") {
        processed_text = insert_spaces_between_chinese_chars(text);
        // std::cout << "处理后的文本: " << processed_text << std::endl;
    }
    
    try {
        // 根据语言代码选择正则表达式模式
        std::wstring wpattern;
        std::wstring wtext;
        
        // 将UTF-8字符串转换为宽字符字符串（支持中文）
        std::wstring_convert<std::codecvt_utf8<wchar_t>> converter;
        wtext = converter.from_bytes(processed_text);
        
        if (lang_code == "z" || lang_code == "j") {
            // 中文/日文标点：使用宽字符正则
            // 注意：中文标点直接写，不需要转义（除了反斜杠）
            wpattern = LR"(([。！？；，、："''（）【】《》…\n]))";
        } else {
            // 英文标点
            wpattern = LR"(([.!?;,:\n]))";
        }
        
        std::wregex pattern(wpattern);
        std::wsregex_token_iterator it(wtext.begin(), wtext.end(), pattern, {-1, 0});
        std::wsregex_token_iterator end;
        
        std::vector<std::wstring> wsentences;
        while (it != end) {
            wsentences.push_back(*it);
            ++it;
        }
        
        // 一句话带一个标点
        for (size_t i = 0; i + 1 < wsentences.size(); i += 2) {
            std::wstring wsentence;
            
            if (i + 1 < wsentences.size()) {
                wsentence = wsentences[i] + wsentences[i + 1];
            } else {
                wsentence = wsentences[i];
            }
            
            // 去除首尾空白字符（宽字符版本）
            auto start = wsentence.find_first_not_of(L" \t\n\r\f\v");
            auto end_pos = wsentence.find_last_not_of(L" \t\n\r\f\v");
            
            if (start != std::wstring::npos && end_pos != std::wstring::npos) {
                wsentence = wsentence.substr(start, end_pos - start + 1);
                
                if (!wsentence.empty()) {
                    // 转换回UTF-8
                    std::wstring_convert<std::codecvt_utf8<wchar_t>> conv;
                    result.push_back(conv.to_bytes(wsentence));
                }
            }
        }
        
        // 处理最后没有标点的文本片段
        if (wsentences.size() % 2 == 1 && wsentences.size() > 0) {
            std::wstring wlast_text = wsentences.back();
            
            // 去除首尾空白字符
            auto start = wlast_text.find_first_not_of(L" \t\n\r\f\v");
            auto end_pos = wlast_text.find_last_not_of(L" \t\n\r\f\v");
            
            if (start != std::wstring::npos && end_pos != std::wstring::npos) {
                wlast_text = wlast_text.substr(start, end_pos - start + 1);
                
                if (!wlast_text.empty()) {
                    std::wstring end_punctuation = (lang_code == "z" || lang_code == "j") ? L"。" : L".";
                    wlast_text += end_punctuation;
                    
                    // 转换回UTF-8
                    std::wstring_convert<std::codecvt_utf8<wchar_t>> conv;
                    result.push_back(conv.to_bytes(wlast_text));
                }
            }
        }
        
    } catch (const std::regex_error& e) {
        std::cerr << "正则表达式错误: " << e.what() << std::endl;
        result.push_back(processed_text);
    } catch (const std::exception& e) {
        std::cerr << "错误: " << e.what() << std::endl;
        result.push_back(processed_text);
    }
    
    // 如果没有分割出任何句子，返回原文本
    if (result.empty()) {
        result.push_back(processed_text);
    }

    // 句子间添加空格
    if (result.size() > 1) {
        for (size_t i = 0; i < result.size() - 1; i++) {
            result[i] += "...";
        }
    }

    return result;
}