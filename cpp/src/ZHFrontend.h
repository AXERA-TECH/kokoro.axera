#pragma once
#include <string>
#include <vector>
#include <memory>
#include <unordered_set>
#include "ToneSandhi.h"
#include "Utils.h"

struct MToken {
    std::string text;
    std::string tag;
    std::string whitespace;
    std::vector<std::string> phonemes; 
};

class ZHFrontend {
public:
    ZHFrontend(std::shared_ptr<TextProcessor> processor, const std::string& unk = "?");

    std::vector<MToken> operator()(const std::string& text, bool with_erhua = true);

private:
    std::string unk;
    std::shared_ptr<TextProcessor> processor;
    ToneSandhi tone_modifier;
    
    std::unordered_set<std::string> must_erhua;
    std::unordered_set<std::string> not_erhua;
    std::unordered_set<std::string> punc;

    struct InitFinal {
        std::vector<std::string> initials;
        std::vector<std::string> finals;
    };
    
    InitFinal _get_initials_finals(const std::string& word);
    InitFinal _merge_erhua(const std::vector<std::string>& initials, 
                           const std::vector<std::string>& finals, 
                           const std::string& word, const std::string& pos);
};
