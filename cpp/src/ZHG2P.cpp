#include "ZHG2P.h"
#include <iostream>
#include <regex>
#include <sstream>
#include <algorithm>
#include <unordered_set>

// ==========================================
// Data Tables from transcription.py
// ==========================================

static const std::unordered_map<std::string, std::vector<std::string>> INITIAL_MAPPING = {
    {"b", {"p"}}, {"c", {"ʦʰ"}}, {"ch", {"ʈʂʰ"}}, {"d", {"t"}},
    {"f", {"f"}}, {"g", {"k"}}, {"h", {"x"}}, {"j", {"ʨ"}},
    {"k", {"kʰ"}}, {"l", {"l"}}, {"m", {"m"}}, {"n", {"n"}},
    {"p", {"pʰ"}}, {"q", {"ʨʰ"}}, {"r", {"ɻ"}}, {"s", {"s"}},
    {"sh", {"ʂ"}}, {"t", {"tʰ"}}, {"x", {"ɕ"}}, {"z", {"ʦ"}},
    {"zh", {"ʈʂ"}}
};

static const std::unordered_map<std::string, std::vector<std::string>> FINAL_MAPPING = {
    {"a", {"a0"}}, {"ai", {"ai̯0"}}, {"an", {"a0", "n"}}, {"ang", {"a0", "ŋ"}},
    {"ao", {"au̯0"}}, {"e", {"ɤ0"}}, {"ei", {"ei̯0"}}, {"en", {"ə0", "n"}},
    {"eng", {"ə0", "ŋ"}}, {"er", {"ɚ0"}}, {"i", {"i0"}}, {"ia", {"j", "a0"}},
    {"ian", {"j", "ɛ0", "n"}}, {"iang", {"j", "a0", "ŋ"}}, {"iao", {"j", "au̯0"}},
    {"ie", {"j", "e0"}}, {"in", {"i0", "n"}}, {"ing", {"i0", "ŋ"}},
    {"iong", {"j", "ʊ0", "ŋ"}}, {"iou", {"j", "ou̯0"}}, {"ong", {"ʊ0", "ŋ"}},
    {"ou", {"ou̯0"}}, {"o", {"o0"}}, {"u", {"u0"}}, {"ua", {"w", "a0"}},
    {"uai", {"w", "ai̯0"}}, {"uan", {"w", "a0", "n"}}, {"uang", {"w", "a0", "ŋ"}},
    {"ui", {"w", "ei̯0"}}, {"un", {"w", "ə0", "n"}}, {"ueng", {"w", "ə0", "ŋ"}},
    {"uo", {"w", "o0"}}, {"ue", {"ɥ", "e0"}}, {"uen", {"w", "ə0", "n"}}, {"uei", {"w", "ei̯0"}},
    {"ü", {"y0"}}, {"üe", {"ɥ", "e0"}}, {"üan", {"ɥ", "ɛ0", "n"}}, {"ün", {"y0", "n"}},
    {"van", {"ɥ", "ɛ0", "n"}}, {"vn", {"y0", "n"}}, {"ve", {"ɥ", "e0"}}, {"v", {"y0"}},
    // ZHFrontend special finals for apical vowels
    {"ii", {"ɹ̩0"}},   // for z, c, s
    {"iii", {"ɻ̩0"}}   // for zh, ch, sh, r
};

static const std::unordered_map<std::string, std::vector<std::string>> FINAL_MAPPING_ZH_CH_SH_R = {
    {"i", {"ɻ̩0"}} 
};

static const std::unordered_map<std::string, std::vector<std::string>> FINAL_MAPPING_Z_C_S = {
    {"i", {"ɹ̩0"}} 
};

static const std::unordered_map<int, std::string> TONE_MAPPING = {
    {1, "˥"}, {2, "˧˥"}, {3, "˧˩˧"}, {4, "˥˩"}, {5, ""}
};

static const std::unordered_map<std::string, std::pair<std::string, int>> TONE_VOWELS = {
    {u8"ā", {u8"a", 1}}, {u8"á", {u8"a", 2}}, {u8"ǎ", {u8"a", 3}}, {u8"à", {u8"a", 4}},
    {u8"ē", {u8"e", 1}}, {u8"é", {u8"e", 2}}, {u8"ě", {u8"e", 3}}, {u8"è", {u8"e", 4}},
    {u8"ī", {u8"i", 1}}, {u8"í", {u8"i", 2}}, {u8"ǐ", {u8"i", 3}}, {u8"ì", {u8"i", 4}},
    {u8"ō", {u8"o", 1}}, {u8"ó", {u8"o", 2}}, {u8"ǒ", {u8"o", 3}}, {u8"ò", {u8"o", 4}},
    {u8"ū", {u8"u", 1}}, {u8"ú", {u8"u", 2}}, {u8"ǔ", {u8"u", 3}}, {u8"ù", {u8"u", 4}},
    {u8"ǖ", {u8"v", 1}}, {u8"ǘ", {u8"v", 2}}, {u8"ǚ", {u8"v", 3}}, {u8"ǜ", {u8"v", 4}},
    {u8"ń", {u8"n", 2}}, {u8"ň", {u8"n", 3}}, {u8"ǹ", {u8"n", 4}},
    {u8"ḿ", {u8"m", 2}}, {u8"m̀", {u8"m", 4}} 
};

// ==========================================
// Utility Functions
// ==========================================

static const std::unordered_map<char, std::string> LETTER_TO_IPA = {
    {'A', "ei̯"}, {'B', "pi"}, {'C', "si"}, {'D', "ti"}, {'E', "i"},
    {'F', "ef"}, {'G', "tʂi"}, {'H', "ei̯tʂ"}, {'I', "ai̯"}, {'J', "tʂei̯"},
    {'K', "kʰei̯"}, {'L', "el"}, {'M', "em"}, {'N', "en"}, {'O', "ou̯"},
    {'P', "pʰi"}, {'Q', "kʰju"}, {'R', "aɻ"}, {'S', "es"}, {'T', "tʰi"},
    {'U', "ju"}, {'V', "vi"}, {'W', "tʌplju"}, {'X', "eks"}, {'Y', "wai̯"},
    {'Z', "zi"},
    {'a', "ei̯"}, {'b', "pi"}, {'c', "si"}, {'d', "ti"}, {'e', "i"},
    {'f', "ef"}, {'g', "tʂi"}, {'h', "ei̯tʂ"}, {'i', "ai̯"}, {'j', "tʂei̯"},
    {'k', "kʰei̯"}, {'l', "el"}, {'m', "em"}, {'n', "en"}, {'o', "ou̯"},
    {'p', "pʰi"}, {'q', "kʰju"}, {'r', "aɻ"}, {'s', "es"}, {'t', "tʰi"},
    {'u', "ju"}, {'v', "vi"}, {'w', "tʌplju"}, {'x', "eks"}, {'y', "wai̯"},
    {'z', "zi"}
};

static std::string replace_all(std::string str, const std::string& from, const std::string& to) {
    if (from.empty()) return str;
    size_t start_pos = 0;
    while((start_pos = str.find(from, start_pos)) != std::string::npos) {
        str.replace(start_pos, from.length(), to);
        start_pos += to.length();
    }
    return str;
}

static std::string join(const std::vector<std::string>& vec, const std::string& delim) {
    std::string res;
    for (size_t i = 0; i < vec.size(); ++i) {
        if (i > 0) res += delim;
        res += vec[i];
    }
    return res;
}

static std::string trim(const std::string& str) {
    size_t first = str.find_first_not_of(" \t\n\r");
    if (std::string::npos == first) return "";
    size_t last = str.find_last_not_of(" \t\n\r");
    return str.substr(first, (last - first + 1));
}

// ==========================================
// ZHG2P Implementation
// ==========================================

ZHG2P::ZHG2P(std::shared_ptr<TextProcessor> proc, const std::string& ver, const std::string& u, const std::string& eng_dict_path)
    : processor(std::move(proc)), version(ver), unk(u) {
    if (version == "1.1") {
        frontend = std::make_unique<ZHFrontend>(processor, unk);
    }
    if (!eng_dict_path.empty()) {
        std::cout << "Loading English G2P dict from " << eng_dict_path << "..." << std::endl;
        eng_g2p = std::make_unique<EnG2P>(eng_dict_path);
    }
}

std::string ZHG2P::retone(std::string p) {
    p = replace_all(p, "˧˩˧", "↓");
    p = replace_all(p, "˧˥", "↗");
    p = replace_all(p, "˥˩", "↘");
    p = replace_all(p, "˥", "→");
    
    // ɨ handling
    p = replace_all(p, "\u027B\u0329", "ɨ"); 
    p = replace_all(p, "\u0279\u0329", "ɨ"); 
    p = replace_all(p, "ɻ̩", "ɨ"); 
    p = replace_all(p, "ɹ̩", "ɨ"); 

    return p;
}

ZHG2P::PinyinParts ZHG2P::parse_pinyin(const std::string& raw_pinyin) {
    PinyinParts parts;
    parts.tone = 5;
    std::string pinyin = trim(raw_pinyin);
    
    if (pinyin.empty()) return parts;

    // Normalize pinyin (handle tone marks like ā -> a, tone=1)
    std::string base = "";
    int detected_tone = 5;
    
    for (size_t i = 0; i < pinyin.length(); ) {
        bool matched = false;
        // Check against TONE_VOWELS keys
        // Iterate map - not efficient but works given small map and short pinyin string
        for (const auto& kv : TONE_VOWELS) {
            if (pinyin.compare(i, kv.first.length(), kv.first) == 0) {
                 if (detected_tone == 5) detected_tone = kv.second.second; 
                 base += kv.second.first;
                 i += kv.first.length();
                 matched = true;
                 break;
            }
        }
        if (!matched) {
            base += pinyin[i];
            i++;
        }
    }

    // Check explicitly written number tone (e.g. zhong1) - overrides mark if present (unlikely mixed)
    if (!base.empty()) {
        char last = base.back();
        if (isdigit(static_cast<unsigned char>(last))) {
            parts.tone = last - '0';
            base.pop_back();
        } else {
            parts.tone = detected_tone;
        }
    }

    base = replace_all(base, "v", "ü");

    // Handle y and w (standard pinyin normalization)
    if (base.rfind("yi", 0) == 0) {
        base = base.substr(1); // yi -> i
    } else if (base.rfind("y", 0) == 0) {
        if (base.length() > 1 && base[1] == 'u') {
             base = "ü" + base.substr(2); // yu -> ü
        } else {
             base = "i" + base.substr(1); // ya -> ia, you -> iou
        }
    } else if (base.rfind("wu", 0) == 0) {
        base = base.substr(1); // wu -> u
    } else if (base.rfind("w", 0) == 0) {
        base = "u" + base.substr(1); // wa -> ua, wo -> uo, wei -> uei
    }

    std::string p_initial = "";
    
    static const std::vector<std::string> multi_initials = {"zh", "ch", "sh"};
    for (const auto& ini : multi_initials) {
        if (base.rfind(ini, 0) == 0) { 
            p_initial = ini;
            break;
        }
    }
    if (p_initial.empty()) {
        std::string first_char = base.substr(0, 1);
        if (INITIAL_MAPPING.count(first_char)) {
            p_initial = first_char;
        }
    }

    parts.initial = p_initial;
    parts.final = base.substr(p_initial.length());
    
    return parts;
}

std::string ZHG2P::pinyin_to_ipa_convert(const std::string& pinyin) {
    auto parts = parse_pinyin(pinyin);
    std::vector<std::string> ipa_segments;

    // 1. Initial
    if (!parts.initial.empty() && INITIAL_MAPPING.count(parts.initial)) {
        ipa_segments.push_back(INITIAL_MAPPING.at(parts.initial)[0]);
    }

    // 2. Final
    std::vector<std::string> final_phonemes;
    bool handled = false;
    bool is_erhua = false;

    if (!parts.final.empty() && parts.final.back() == 'R') {
        is_erhua = true;
        parts.final.pop_back();
    }

    if ((parts.initial == "zh" || parts.initial == "ch" || parts.initial == "sh" || parts.initial == "r") 
        && FINAL_MAPPING_ZH_CH_SH_R.count(parts.final)) {
        final_phonemes = FINAL_MAPPING_ZH_CH_SH_R.at(parts.final);
        handled = true;
    } else if ((parts.initial == "z" || parts.initial == "c" || parts.initial == "s") 
        && FINAL_MAPPING_Z_C_S.count(parts.final)) {
        final_phonemes = FINAL_MAPPING_Z_C_S.at(parts.final);
        handled = true;
    }

    if (!handled && FINAL_MAPPING.count(parts.final)) {
        final_phonemes = FINAL_MAPPING.at(parts.final);
    }

    if (final_phonemes.empty() && !parts.final.empty()) {
        final_phonemes.push_back(parts.final); 
    }

    // 3. Apply Tone
    std::string tone_mark = (TONE_MAPPING.count(parts.tone)) ? TONE_MAPPING.at(parts.tone) : "";
    
    for (const auto& ph : final_phonemes) {
        std::string processed = ph;
        processed = replace_all(processed, "0", tone_mark);
        ipa_segments.push_back(processed);
    }

    if (is_erhua) {
        ipa_segments.push_back("ɚ"); // or proper IPA for rhoticity
    }

    return join(ipa_segments, "");
}

std::string ZHG2P::py2ipa(const std::string& py) {
    std::string ipa = pinyin_to_ipa_convert(py);
    return retone(ipa);
}

std::string ZHG2P::map_punctuation(std::string text) {
    // Note: using u8 string literals
    text = replace_all(text, u8"、", ", ");
    text = replace_all(text, u8"，", ", ");
    text = replace_all(text, u8"。", ". ");
    text = replace_all(text, u8"．", ". ");
    text = replace_all(text, u8"！", "! ");
    text = replace_all(text, u8"：", ": ");
    text = replace_all(text, u8"；", "; ");
    text = replace_all(text, u8"？", "? ");
    text = replace_all(text, u8"«", u8" “");
    text = replace_all(text, u8"»", u8"” ");
    text = replace_all(text, u8"《", u8" “");
    text = replace_all(text, u8"》", u8"” ");
    text = replace_all(text, u8"「", u8" “");
    text = replace_all(text, u8"」", u8"” ");
    text = replace_all(text, u8"【", u8" “");
    text = replace_all(text, u8"】", u8"” ");
    text = replace_all(text, u8"（", " (");
    text = replace_all(text, u8"）", ") ");
    
    size_t first = text.find_first_not_of(" \t\n\r");
    if (std::string::npos == first) return text;
    size_t last = text.find_last_not_of(" \t\n\r");
    return text.substr(first, (last - first + 1));
}

bool ZHG2P::is_chinese(const std::string& str) {
    // Simple heuristic check for CJK range in UTF-8
    for (unsigned char c : str) {
        if (c >= 0xE4 && c <= 0xE9) return true;
    }
    return false;
}

std::string ZHG2P::legacy_call(const std::string& text) {
    std::string result = "";
    
    auto words = processor->cut(text);
    for (const auto& pair : words) {
        std::string w = pair.first;
        if (is_chinese(w)) {
            auto pinyins = processor->word_to_pinyin(w);
            for (const auto& py : pinyins) {
                result += py2ipa(py);
            }
            // segment = ' '.join(word2ipa(w) for w in words)
            // So YES, space between words.
            result += " "; 
        } else {
            result += w; 
        }
    }
    
    // Trim trailing space if needed
    if (!result.empty() && result.back() == ' ') result.pop_back();

    // Remove \u032F (815)
    result = replace_all(result, "\u032F", "");
    return result;
}

std::pair<std::string, std::string> ZHG2P::operator()(const std::string& text) {
    if (text.empty()) return {"", ""};

    std::string processed_text = text;
    if (processor) {
        processed_text = processor->convert_numbers(processed_text);
    }
    processed_text = map_punctuation(processed_text);

    // Default to legacy_call for now as we don't have the 1.1 frontend ported
    if (frontend) {
        auto tokens = (*frontend)(processed_text);
        std::string result;
        bool last_was_eng = false;

        for (const auto& tk : tokens) {
             bool is_eng = (tk.tag == "eng");

             // Logic to add spaces around English words
             // 1. Before English word (if not at start)
             if (is_eng && !result.empty() && result.back() != ' ') {
                 result += " ";
             }
             // 2. After English word (if next is not punctuation)
             if (last_was_eng && !is_eng && tk.tag != "x" && !result.empty() && result.back() != ' ') {
                 result += " ";
             }

             if (tk.tag == "x" || tk.tag == "eng") {
                 for (const auto& p : tk.phonemes) {
                     std::string converted_part;
                     if (eng_g2p && tk.tag == "eng") {
                         converted_part = eng_g2p->convert(p);
                     }
                     
                     // Fallback for English words: letter by letter
                     if ((converted_part.empty() || converted_part == p) && tk.tag == "eng") {
                         // If conversion failed or returned same (meaning no dict entry found typically),
                         // try letter mapping for pure alpha strings
                         bool all_alpha = true;
                         for(char c : p) {
                             if(!isalpha(c)) { all_alpha = false; break; }
                         }
                         
                         if (all_alpha) {
                             converted_part = "";
                             for(char c : p) {
                                 if(LETTER_TO_IPA.count(c)) {
                                     converted_part += LETTER_TO_IPA.at(c);
                                 } else {
                                     converted_part += c;
                                 }
                             }
                         } else {
                             converted_part = p; // keep as is if not all alpha
                         }
                     } else if (converted_part.empty()) {
                         converted_part = p;
                     }

                     result += converted_part;
                 }
             } else {
                 // Split phonemes into pinyin syllables based on tone digits
                 std::string pinyin_acc;
                 for (const auto& p : tk.phonemes) {
                     pinyin_acc += p;
                     // Check if p contains a digit
                     bool has_digit = false;
                     for (char c : p) {
                         if (isdigit(static_cast<unsigned char>(c))) {
                             has_digit = true;
                             break;
                         }
                     }
                     
                     if (has_digit) {
                         result += py2ipa(pinyin_acc);
                         pinyin_acc.clear();
                     }
                 }
                 // Handle any trailing part
                 if (!pinyin_acc.empty()) {
                     result += py2ipa(pinyin_acc);
                 }
             }
             last_was_eng = is_eng;
        }
        return {result, ""};
    }
    return {legacy_call(processed_text), ""};
}
