#include "Tokenizer.h"
#include "JiebaProcessor.h"
#include "ZHG2P.h"
#include <iostream>
#include <vector>
#include "espeak-ng/speak_lib.h"
#include "utils/logger.hpp"

Tokenizer::Tokenizer(const std::string& lang_code, const TokenizerConfig& config, const std::map<std::string, int>& vocab) 
    :   lang_code_(lang_code),
        vocab_(vocab) {
    
    std::string d = config.dict_dir;
    if (!d.empty() && d.back() != '/') d += "/";
    
    std::string jieba_dict = d + config.jieba_dict;
    std::string hmm_model = d + config.hmm_model;
    std::string user_dict = d + config.user_dict;
    std::string idf_path = d + config.idf_path;
    std::string stop_word_path = d + config.stop_word_path;
    std::string pinyin_char = d + config.pinyin_char;
    std::string pinyin_phrase = d + config.pinyin_phrase;
    std::string cmu_dict = d + config.cmu_dict;

    try {
        if (lang_code_ == "z") {
            processor_ = std::make_shared<JiebaProcessor>(
                jieba_dict, hmm_model, user_dict, idf_path, stop_word_path, pinyin_char, pinyin_phrase
            );
            g2p_ = std::make_unique<ZHG2P>(processor_, "1.1", "<unk>", cmu_dict);
        } else {
            const char* data_path = "./third_party/espeak-ng/share/espeak-ng-data";
            espeak_Initialize(AUDIO_OUTPUT_RETRIEVAL, 0, data_path, 0);

            espeak_VOICE voice;
            memset(&voice, 0, sizeof(voice));
            
            char lang_code_char = lang_code[0];
            switch (lang_code_char) {
                case 'a': {
                    ALOGI("Load language: English");
                    voice.name = "English_(America)";
                    voice.languages = "en-us";
                    voice.gender = 2;           // 女性（1=男，2=女），设为0则不指定
                    break;
                }
                case 'j': {
                    ALOGI("Load language: Japanese");
                    voice.languages = "ja";
                    voice.gender = 2;           // 女性（1=男，2=女），设为0则不指定
                    voice.age = 0;              // 年龄，0表示不指定
                    voice.variant = 0;          // 变体
                    voice.name = NULL;          // 让系统自动选择匹配的语音
                    break;
                }
                default: {
                    ALOGW("Unknown lang_code: %c, fallback to English", lang_code_char);
                    voice.name = "English_(America)";
                    voice.languages = "en-us";
                    break;
                }
            }
            
            espeak_SetVoiceByProperties(&voice);
        }
        
    } catch (const std::exception& e) {
        ALOGE("Failed to initialize Tokenizer dependencies: %s", e.what());
    }
}

Tokenizer::~Tokenizer() {
    espeak_Terminate();
}

static std::vector<std::string> split_utf8(const std::string& str) {
    std::vector<std::string> chars;
    for (size_t i = 0; i < str.length();) {
        unsigned char c = static_cast<unsigned char>(str[i]);
        size_t char_len = 0;
        if (c < 0x80) char_len = 1;
        else if ((c & 0xE0) == 0xC0) char_len = 2;
        else if ((c & 0xF0) == 0xE0) char_len = 3;
        else if ((c & 0xF8) == 0xF0) char_len = 4;
        else char_len = 1; 

        if (i + char_len > str.length()) char_len = str.length() - i;
        
        chars.push_back(str.substr(i, char_len));
        i += char_len;
    }
    return chars;
}

std::vector<int> Tokenizer::tokenize(const std::string& phonemes) {
    std::vector<int> tokens;
    std::vector<std::string> chars = split_utf8(phonemes);
    
    for (const auto& c : chars) {
        if (vocab_.count(c)) {
            tokens.push_back(vocab_.at(c));
        }
    }
    return tokens;
}

std::string Tokenizer::phonemize(const std::string& text, bool norm) {
    if (lang_code_ == "z") {
        if (!g2p_) return text;
        auto result = (*g2p_)(text);
        return result.first; 
    } else {
        const char* textPtr = text.c_str();
        int phonememode = ('_' << 8) | 0x02;
        const char * phonemes = espeak_TextToPhonemes(
                reinterpret_cast<const void **>(&textPtr), espeakCHARS_UTF8, phonememode);
        return std::string(phonemes);
    }
}
