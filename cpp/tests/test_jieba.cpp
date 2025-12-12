/**************************************************************************************************
 *
 * Copyright (c) 2019-2025 Axera Semiconductor (Ningbo) Co., Ltd. All Rights Reserved.
 *
 * This source file is the property of Axera Semiconductor (Ningbo) Co., Ltd. and
 * may not be copied or distributed in any isomorphic form without the prior
 * written consent of Axera Semiconductor (Ningbo) Co., Ltd.
 *
 **************************************************************************************************/

#include "ZHG2P.h"
#include "PinyinFinder.h"
#include <iostream>
#include <fstream>

#include "JiebaProcessor.h"

int main() {
    #ifdef _WIN32
    system("chcp 65001");
    #endif

    // Configuration paths (Assumed relative to executable or work dir)
    const std::string DICT_DIR = "dict/";
    
    const std::string JIEBA_DICT      = DICT_DIR + "jieba.dict.utf8";
    const std::string HMM_MODEL       = DICT_DIR + "hmm_model.utf8";
    const std::string USER_DICT       = DICT_DIR + "user.dict.utf8";
    const std::string IDF_PATH        = DICT_DIR + "idf.utf8";
    const std::string STOP_WORD_PATH  = DICT_DIR + "stop_words.utf8";
    
    const std::string PINYIN_CHAR     = DICT_DIR + "pinyin.txt";
    const std::string PINYIN_PHRASE   = DICT_DIR + "pinyin_phrase.txt";
    const std::string CMU_DICT        = DICT_DIR + "cmudict-0.7b/cmudict.dict";

    std::cout << "Initializing JiebaProcessor..." << std::endl;
    
    // Check if files exist (simple check)
    std::ifstream check_jieba(JIEBA_DICT);
    if (!check_jieba.good()) {
        std::cerr << "Error: Jieba dict not found at " << JIEBA_DICT << std::endl;
        std::cerr << "Please make sure cppjieba dicts are in " << DICT_DIR << std::endl;
        return 1;
    }

    try {
        auto proc = std::make_shared<JiebaProcessor>(
            JIEBA_DICT,
            HMM_MODEL,
            USER_DICT,
            IDF_PATH,
            STOP_WORD_PATH,
            PINYIN_CHAR,
            PINYIN_PHRASE
        );
        
        ZHG2P g2p(proc, "1.1", "<unk>", CMU_DICT);
        
        std::cout << "Testing ZHG2P with Jieba" << std::endl;
        std::cout << "--------------------------------" << std::endl;

        // Test convert_numbers
        std::cout << "Debug Number Conversion:" << std::endl;
        std::vector<std::string> num_tests = {
            "123",
            "-5",
            "3.14",
            "123.456",
            "这里有500个苹果",
            "气温是-3.5度"
        };
        
        for (const auto& t : num_tests) {
            std::cout << "[" << t << "] -> " << proc->convert_numbers(t) << std::endl;
        }
        std::cout << "--------------------------------" << std::endl;

        // Test cases
        std::vector<std::string> tests = {
            // 基础与混合
            "中国",
            "你好中国",
            "Hello中国!",
            
            // 词义消歧与多音字
            "这个东西很便宜", 
            "我要去长安",
            "着火了",      // zhao2
            "看着",        // zhe
            "音乐",        // yue4
            "快乐",        // le4
            
            // 数字与符号
            "今天天气真不错",
            "我有123块钱",
            "今天气温-5度",
            "圆周率是3.14159",
            "IPV4地址是192.168.0.1",
            
            // 变调测试：三声连读
            "洗澡",       // xi3 zao3 -> xi2 zao3
            "管理",       // guan3 li3 -> guan2 li3
            "展览馆",     // zhan3 lan3 guan3 -> zhan2 lan2 guan3 (通常 333->223)
            
            // 变调测试：“一”
            "一",         // yi1 (单独)
            "第一",       // yi1 (序数)
            "一天",       // yi4 (一声前)
            "一年",       // yi4 (二声前)
            "一起",       // yi4 (三声前)
            "一个",       // yi2 (四声前)
            
            // 变调测试：“不”
            "不吃",       // bu4 (一声前)
            "不玩",       // bu4 (二声前)
            "不懂",       // bu4 (三声前)
            "不对",       // bu2 (四声前)
            
            // 儿化音
            "花儿",       // hua er -> huar
            "小院儿",     // yuan er -> yuanr
            "玩儿",       // wan er -> wanr
            
            // 轻声 (部分词典可能带轻声标记，或者前端处理)
            "爸爸",
            "妈妈",
            "看看",

            "去去去"
        };
        
        std::cout << "Debug Jieba Cut:" << std::endl;
        for (const auto& t : tests) {
            auto cuts = proc->cut(t);
            std::cout << "[" << t << "]: ";
            for (const auto& pair : cuts) {
                std::cout << pair.first << "/" << pair.second << " ";
            }
            std::cout << std::endl;
        }
        std::cout << "--------------------------------" << std::endl;
        
        for (const auto& t : tests) {
            auto res = g2p(t);
            std::cout << "Input: " << t << " -> Output: " << res.first << std::endl;
        }
        
    } catch (const std::exception& e) {
        std::cerr << "Exception: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}