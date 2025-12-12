#pragma once
#include <string>
#include <vector>
#include <unordered_set>
#include <utility>
#include <memory>
#include <functional>

class ToneSandhi {
public:
    using PinyinProvider = std::function<std::vector<std::string>(const std::string&)>;

    ToneSandhi();
    
    void setPinyinProvider(PinyinProvider provider);

    // Core logic: pre-merge
    std::vector<std::pair<std::string, std::string>> pre_merge_for_modify(
        const std::vector<std::pair<std::string, std::string>>& seg);

    // Core logic: modify tone
    std::vector<std::string> modified_tone(const std::string& word, const std::string& pos, 
                                           std::vector<std::string> finals);

private:
    // Internal sandhi rules
    std::vector<std::string> _bu_sandhi(const std::string& word, std::vector<std::string> finals);
    std::vector<std::string> _yi_sandhi(const std::string& word, std::vector<std::string> finals);
    std::vector<std::string> _neural_sandhi(const std::string& word, const std::string& pos, std::vector<std::string> finals);
    std::vector<std::string> _three_sandhi(const std::string& word, std::vector<std::string> finals);

    // Merge rules
    std::vector<std::pair<std::string, std::string>> _merge_bu(const std::vector<std::pair<std::string, std::string>>& seg);
    std::vector<std::pair<std::string, std::string>> _merge_yi(const std::vector<std::pair<std::string, std::string>>& seg);
    std::vector<std::pair<std::string, std::string>> _merge_reduplication(const std::vector<std::pair<std::string, std::string>>& seg);
    std::vector<std::pair<std::string, std::string>> _merge_er(const std::vector<std::pair<std::string, std::string>>& seg);
    
    // Helpers
    bool _all_tone_three(const std::vector<std::string>& finals);
    std::vector<std::string> _split_word(const std::string& word); 

    std::unordered_set<std::string> must_neural_tone_words;
    std::unordered_set<std::string> must_not_neural_tone_words;
    std::string punc;
    PinyinProvider pinyin_provider;
};
