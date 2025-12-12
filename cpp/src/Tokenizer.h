#pragma once
#include <string>
#include <vector>
#include <map>
#include <memory>

class ZHG2P;
class JiebaProcessor;

struct TokenizerConfig {
    std::string dict_dir = "dict/";
    std::string jieba_dict = "jieba.dict.utf8";
    std::string hmm_model = "hmm_model.utf8";
    std::string user_dict = "user.dict.utf8";
    std::string idf_path = "idf.utf8";
    std::string stop_word_path = "stop_words.utf8";
    std::string pinyin_char = "pinyin.txt";
    std::string pinyin_phrase = "pinyin_phrase.txt";
    std::string cmu_dict = "cmudict-0.7b/cmudict.dict";
};

class Tokenizer {
public:
    Tokenizer(const std::string& lang_code = "z", const TokenizerConfig& config = {}, const std::map<std::string, int>& vocab = {});
    ~Tokenizer();
    
    std::vector<int> tokenize(const std::string& phonemes);
    std::string phonemize(const std::string& text, bool norm = true);

private:
    std::string lang_code_;
    std::map<std::string, int> vocab_;
    std::shared_ptr<JiebaProcessor> processor_;
    std::unique_ptr<ZHG2P> g2p_;
};
