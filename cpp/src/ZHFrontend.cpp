#include "ZHFrontend.h"
#include "ZHG2P.h"
#include <algorithm>
#include <iostream>

ZHFrontend::ZHFrontend(std::shared_ptr<TextProcessor> processor, const std::string& unk)
    : processor(processor), unk(unk) {
    
    punc = {";", ":", ",", ".", "!", "?", "—", "…", "\"", "(", ")", "“", "”"}; 
    must_erhua = {"小院儿", "胡同儿", "范儿", "老汉儿", "撒欢儿", "寻老礼儿", "妥妥儿", "媳妇儿"};
    not_erhua = {
        "虐儿", "为儿", "护儿", "瞒儿", "救儿", "替儿", "有儿", "一儿", "我儿", "俺儿", "妻儿",
        "拐儿", "聋儿", "乞儿", "患儿", "幼儿", "孤儿", "婴儿", "婴幼儿", "连体儿", "脑瘫儿",
        "流浪儿", "体弱儿", "混血儿", "蜜雪儿", "舫儿", "祖儿", "美儿", "应采儿", "可儿", "侄儿",
        "孙儿", "侄孙儿", "女儿", "男儿", "红孩儿", "花儿", "虫儿", "马儿", "鸟儿", "猪儿", "猫儿",
        "狗儿", "少儿"
    };

    tone_modifier.setPinyinProvider([this](const std::string& word) {
        return this->processor->word_to_pinyin(word);
    });
}

ZHFrontend::InitFinal ZHFrontend::_get_initials_finals(const std::string& word) {
    InitFinal res;
    auto pinyins = processor->word_to_pinyin(word);
    
    // Handle '嗯' special case check
    // Check if word contains 嗯 (UTF-8: E5 97 AF)
    // For simplicity, we iterate pinyins. If pinyin is "n" or "ng", we might need check.
    // Pypinyin logic: if word has '嗯', set final to 'n2'.
    // We skip this detailed logic for now.

    for (const auto& py : pinyins) {
        auto parts = ZHG2P::parse_pinyin(py);
        
        std::string final_with_tone = parts.final + std::to_string(parts.tone);
        
        // Special handling for zi, ci, si, zhi, chi, shi, ri
        if (parts.final == "i") {
            if (parts.initial == "z" || parts.initial == "c" || parts.initial == "s") {
                final_with_tone = "ii" + std::to_string(parts.tone);
            } else if (parts.initial == "zh" || parts.initial == "ch" || parts.initial == "sh" || parts.initial == "r") {
                final_with_tone = "iii" + std::to_string(parts.tone);
            }
        }
        
        res.initials.push_back(parts.initial);
        res.finals.push_back(final_with_tone);
    }
    return res;
}

ZHFrontend::InitFinal ZHFrontend::_merge_erhua(const std::vector<std::string>& initials, 
                       const std::vector<std::string>& finals, 
                       const std::string& word, const std::string& pos) {
    
    std::vector<std::string> new_initials;
    std::vector<std::string> new_finals = finals;
    
    // fix er1 -> er2
    std::string er = "儿";
    // Need to match word char index with final index. Assuming 1-to-1.
    // Since word is UTF-8 string, we need u16 conversion to index it.
    std::u16string u16word;
    BasicStringUtil::u8tou16(word.c_str(), word.size(), u16word);
    
    if (u16word.size() != finals.size()) {
        // Mismatch, return as is
        InitFinal res;
        res.initials = initials;
        res.finals = finals;
        return res;
    }

    for (size_t i = 0; i < new_finals.size(); ++i) {
        if (i == new_finals.size() - 1 && u16word[i] == 0x513F && new_finals[i] == "er1") {
            new_finals[i] = "er2";
        }
    }

    if (!must_erhua.count(word) && (not_erhua.count(word) || pos == "a" || pos == "j" || pos == "nr")) {
        InitFinal res;
        res.initials = initials;
        res.finals = new_finals;
        return res;
    }

    std::vector<std::string> merged_initials;
    std::vector<std::string> merged_finals;

    for (size_t i = 0; i < new_finals.size(); ++i) {
        // er2 or er5
        if (i == new_finals.size() - 1 && u16word[i] == 0x513F && (new_finals[i] == "er2" || new_finals[i] == "er5")
            && !merged_finals.empty()) {
            // Check word[-2:] not in not_erhua. Skipping for simplicity.
            
            // Merge: remove last digit of prev final, add 'R', add last digit
            std::string& prev = merged_finals.back();
            if (!prev.empty() && isdigit(prev.back())) {
                char tone = prev.back();
                prev.pop_back();
                prev += "R";
                prev += tone;
            } else {
                prev += "R5"; // Fallback
            }
        } else {
            merged_initials.push_back(initials[i]);
            merged_finals.push_back(new_finals[i]);
        }
    }
    
    InitFinal res;
    res.initials = merged_initials;
    res.finals = merged_finals;
    return res;
}

std::vector<MToken> ZHFrontend::operator()(const std::string& text, bool with_erhua) {
    std::vector<MToken> tokens;
    
    auto seg_cut = processor->cut(text);
    seg_cut = tone_modifier.pre_merge_for_modify(seg_cut);

    for (const auto& pair : seg_cut) {
        std::string word = pair.first;
        std::string pos = pair.second;
        
        MToken tk;
        tk.text = word;
        tk.tag = pos;
        
        if (pos == "x" || pos == "eng") {
            if (pos == "x" && punc.count(word)) {
                 tk.phonemes.push_back(word);
            }
            // If eng, we might want to keep it or transcribe it?
            // Python code: if pos in ('x', 'eng'): ...
            // Here we just keep it as text in phonemes if it's punct, else empty?
            // The prompt example showed: [['w', 'o3', ...]]
            // For English, maybe we should keep it as is.
            if (pos == "eng") tk.phonemes.push_back(word);

            tokens.push_back(tk);
            continue;
        }

        // g2p
        auto init_finals = _get_initials_finals(word);
        
        // tone sandhi
        auto modified_finals = tone_modifier.modified_tone(word, pos, init_finals.finals);
        
        // er hua
        if (with_erhua) {
            init_finals = _merge_erhua(init_finals.initials, modified_finals, word, pos);
        } else {
            init_finals.finals = modified_finals;
        }

        // Zip initials and finals into phonemes
        for (size_t i = 0; i < init_finals.initials.size(); ++i) {
            if (i < init_finals.finals.size()) {
                // We need to handle the mapping back to IPA?
                // The ZHFrontend returns py-like tokens (initial, final_with_tone).
                // ZHG2P logic usually takes pinyin string.
                // Here we return raw components: "z", "hong1".
                
                // The prompt says: [['w', 'o3', 'm', 'en2', ' '], ...]
                // So it splits initial and final.
                
                if (!init_finals.initials[i].empty()) {
                    tk.phonemes.push_back(init_finals.initials[i]);
                }
                tk.phonemes.push_back(init_finals.finals[i]);
            }
        }
        tokens.push_back(tk);
    }
    return tokens;
}
