#pragma once

#include <string>
#include <vector>
#include <unordered_map>
#include "Utils.h"

class PinyinFinder {
public:
    using UnicodeCharT = char16_t;
    using UnicodeStr = std::u16string;

    PinyinFinder();
    ~PinyinFinder();

    bool init(const std::string& singleCharacterDictPath, const std::string& wordsDictPath);

    void find_best_pinyin(const std::string& phrasestr, std::vector<std::string>& pinyins);

private:
    std::unordered_map<UnicodeStr, std::string> word_pinyin_dict_;
    static const int kMaxChars = 8; // 最大匹配长度，通常不需要太大
};
