#pragma once
#include <string>
#include <vector>
#include <unordered_map>
#include <memory>
#include <utility>
#include "Utils.h"
#include "ZHFrontend.h"
#include "EnG2P.h"

// ZHG2P class definition...

class ZHG2P {
public:
    // 构造函数注入文本处理器依赖
    ZHG2P(std::shared_ptr<TextProcessor> processor, const std::string& version = "1.1", const std::string& unk = "<unk>", const std::string& eng_dict_path = "");

    // 主调用接口: 返回 (IPA字符串, 附加信息/None)
    std::pair<std::string, std::string> operator()(const std::string& text);

    // 静态工具方法
    static std::string retone(std::string p);
    static std::string py2ipa(const std::string& py);
    static std::string map_punctuation(std::string text);
    
    // 核心逻辑
    std::string legacy_call(const std::string& text);

    struct PinyinParts {
        std::string initial;
        std::string final;
        int tone;
    };
    static PinyinParts parse_pinyin(const std::string& pinyin);

    bool is_chinese(const std::string& str);

private:
    std::string version;
    std::string unk;
    std::shared_ptr<TextProcessor> processor;
    std::unique_ptr<ZHFrontend> frontend;
    std::unique_ptr<EnG2P> eng_g2p;

    // IPA 转换相关的内部结构
    static std::string pinyin_to_ipa_convert(const std::string& pinyin);
    
};
