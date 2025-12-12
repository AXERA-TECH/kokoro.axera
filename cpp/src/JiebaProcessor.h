#pragma once

#include "ZHG2P.h"
#include "PinyinFinder.h"
#include "cppjieba/Jieba.hpp"
#include "Utils.h"
#include <regex>

class JiebaProcessor : public TextProcessor {
public:
    JiebaProcessor(const std::string& dict_path, 
                   const std::string& hmm_path, 
                   const std::string& user_dict_path,
                   const std::string& idf_path, 
                   const std::string& stop_word_path,
                   const std::string& pinyin_char_path,
                   const std::string& pinyin_word_path) 
        : jieba(dict_path, hmm_path, user_dict_path, idf_path, stop_word_path) 
    {
        finder = std::make_shared<PinyinFinder>();
        if (!finder->init(pinyin_char_path, pinyin_word_path)) {
            std::cerr << "Failed to init PinyinFinder" << std::endl;
        }
    }

    std::vector<std::pair<std::string, std::string>> cut(const std::string& text) override {
        std::vector<std::pair<std::string, std::string>> result;
        std::vector<std::pair<std::string, std::string>> tag_words;
        
        // Use cppjieba Tagging
        jieba.Tag(text, tag_words);
        
        for (const auto& w : tag_words) {
            std::string word = w.first;
            std::string tag = w.second;
            
            // Ensure punctuation is 'x' (jieba might return 'w' for punct)
            if (tag == "w") tag = "x";
            
            // FIX: If tag is 'x' but contains Chinese characters, force it to a valid tag (e.g. 'n')
            // This prevents words like "我要" being tagged as 'x' and skipped by G2P.
            if (tag == "x") {
                bool has_cn = false;
                for (unsigned char c : word) {
                    if (c >= 0xE4 && c <= 0xE9) {
                        has_cn = true; 
                        break;
                    }
                }
                if (has_cn) tag = "n";
            }
            
            result.push_back({word, tag});
        }
        return result;
    }

    std::vector<std::string> word_to_pinyin(const std::string& word) override {
        std::vector<std::string> pinyins;
        if (finder) {
            finder->find_best_pinyin(word, pinyins);
        }
        return pinyins;
    }

    std::string convert_numbers(const std::string& text) override {
        // Regex to find numbers: integers, floats, and IP-like strings
        // Examples: 123, -123, 3.14, 192.168.0.1
        // Pattern: [-+]?\d+(?:\.\d+)*
        
        std::regex num_regex("[-+]?\\d+(?:\\.\\d+)*");
        std::string result;
        
        auto words_begin = std::sregex_iterator(text.begin(), text.end(), num_regex);
        auto words_end = std::sregex_iterator();
        
        size_t last_pos = 0;
        
        for (std::sregex_iterator i = words_begin; i != words_end; ++i) {
            std::smatch match = *i;
            std::string match_str = match.str();
            
            // Append text before the number
            result += text.substr(last_pos, match.position() - last_pos);
            
            // Convert number to Chinese
            result += BasicStringUtil::NumberToChinese(match_str);
            
            last_pos = match.position() + match.length();
        }
        
        // Append remaining text
        if (last_pos < text.length()) {
            result += text.substr(last_pos);
        }
        
        return result;
    }

private:
    cppjieba::Jieba jieba;
    std::shared_ptr<PinyinFinder> finder;
};
