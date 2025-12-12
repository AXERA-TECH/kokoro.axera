#include "ToneSandhi.h"
#include "Utils.h"
#include <iostream>
#include <algorithm>

// Helper to check if string ends with tone 4 '4'
static bool is_tone4(const std::string& final) {
    return !final.empty() && final.back() == '4';
}

// Helper to check if string ends with tone 3 '3'
static bool is_tone3(const std::string& final) {
    return !final.empty() && final.back() == '3';
}

// Change tone to '2'
static void to_tone2(std::string& final) {
    if (!final.empty() && isdigit(final.back())) {
        final.back() = '2';
    } else {
        final += '2';
    }
}

// Change tone to '5' (neutral)
static void to_tone5(std::string& final) {
    if (!final.empty() && isdigit(final.back())) {
        final.back() = '5';
    } else {
        final += '5';
    }
}

ToneSandhi::ToneSandhi() {
    punc = "、：，；。？！“”‘’':,;.?!" ;
    
    must_neural_tone_words = {
        "麻烦", "麻利", "鸳鸯", "高粱", "骨头", "骆驼", "马虎", "首饰", "馒头", "馄饨", "风筝",
        "难为", "队伍", "阔气", "闺女", "门道", "锄头", "铺盖", "铃铛", "铁匠", "钥匙", "里脊",
        "里头", "部分", "那么", "道士", "造化", "迷糊", "连累", "这么", "这个", "运气", "过去",
        "软和", "转悠", "踏实", "跳蚤", "跟头", "趔趄", "财主", "豆腐", "讲究", "记性", "记号",
        "认识", "规矩", "见识", "裁缝", "补丁", "衣裳", "衣服", "衙门", "街坊", "行李", "行当",
        "蛤蟆", "蘑菇", "薄荷", "葫芦", "葡萄", "萝卜", "荸荠", "苗条", "苗头", "苍蝇", "芝麻",
        "舒服", "舒坦", "舌头", "自在", "膏药", "脾气", "脑袋", "脊梁", "能耐", "胳膊", "胭脂",
        "胡萝", "胡琴", "胡同", "聪明", "耽误", "耽搁", "耷拉", "耳朵", "老爷", "老实", "老婆",
        "戏弄", "将军", "翻腾", "罗嗦", "罐头", "编辑", "结实", "红火", "累赘", "糨糊", "糊涂",
        "精神", "粮食", "簸箕", "篱笆", "算计", "算盘", "答应", "笤帚", "笑语", "笑话", "窟窿",
        "窝囊", "窗户", "稳当", "稀罕", "称呼", "秧歌", "秀气", "秀才", "福气", "祖宗", "砚台",
        "码头", "石榴", "石头", "石匠", "知识", "眼睛", "眯缝", "眨巴", "眉毛", "相声", "盘算",
        "白净", "痢疾", "痛快", "疟疾", "疙瘩", "疏忽", "畜生", "生意", "甘蔗", "琵琶", "琢磨",
        "琉璃", "玻璃", "玫瑰", "玄乎", "狐狸", "状元", "特务", "牲口", "牙碜", "牌楼", "爽快",
        "爱人", "热闹", "烧饼", "烟筒", "烂糊", "点心", "炊帚", "灯笼", "火候", "漂亮", "滑溜",
        "溜达", "温和", "清楚", "消息", "浪头", "活泼", "比方", "正经", "欺负", "模糊", "槟榔",
        "棺材", "棒槌", "棉花", "核桃", "栅栏", "柴火", "架势", "枕头", "枇杷", "机灵", "本事",
        "木头", "木匠", "朋友", "月饼", "月亮", "暖和", "明白", "时候", "新鲜", "故事", "收拾",
        "收成", "提防", "挖苦", "挑剔", "指甲", "指头", "拾掇", "拳头", "拨弄", "招牌", "招呼",
        "抬举", "护士", "折腾", "扫帚", "打量", "打算", "打扮", "打听", "打发", "扎实", "扁担",
        "戒指", "懒得", "意识", "意思", "悟性", "怪物", "思量", "怎么", "念头", "念叨", "别人",
        "快活", "忙活", "志气", "心思", "得罪", "张罗", "弟兄", "开通", "应酬", "庄稼", "干事",
        "帮手", "帐篷", "希罕", "师父", "师傅", "巴结", "巴掌", "差事", "工夫", "岁数", "屁股",
        "尾巴", "少爷", "小气", "小伙", "将就", "对头", "对付", "寡妇", "家伙", "客气", "实在",
        "官司", "学问", "字号", "嫁妆", "媳妇", "媒人", "婆家", "娘家", "委屈", "姑娘", "姐夫",
        "妯娌", "妥当", "妖精", "奴才", "女婿", "头发", "太阳", "大爷", "大方", "大意", "大夫",
        "多少", "多么", "外甥", "壮实", "地道", "地方", "在乎", "困难", "嘴巴", "嘱咐", "嘟囔",
        "嘀咕", "喜欢", "喇嘛", "喇叭", "商量", "唾沫", "哑巴", "哈欠", "哆嗦", "咳嗽", "和尚",
        "告诉", "告示", "含糊", "吓唬", "后头", "名字", "名堂", "合同", "吆喝", "叫唤", "口袋",
        "厚道", "厉害", "千斤", "包袱", "包涵", "匀称", "勤快", "动静", "动弹", "功夫", "力气",
        "前头", "刺猬", "刺激", "别扭", "利落", "利索", "利害", "分析", "出息", "凑合", "凉快",
        "冷战", "冤枉", "冒失", "养活", "关系", "先生", "兄弟", "便宜", "使唤", "佩服", "作坊",
        "体面", "位置", "似的", "伙计", "休息", "什么", "人家", "亲戚", "亲家", "交情", "云彩",
        "事情", "买卖", "主意", "丫头", "丧气", "两口", "东西", "东家", "世故", "不由", "下水",
        "下巴", "上头", "上司", "丈夫", "丈人", "一辈", "那个", "菩萨", "父亲", "母亲", "咕噜",
        "邋遢", "费用", "冤家", "甜头", "介绍", "荒唐", "大人", "泥鳅", "幸福", "熟悉", "计划",
        "扑腾", "蜡烛", "姥爷", "照顾", "喉咙", "吉他", "弄堂", "蚂蚱", "凤凰", "拖沓", "寒碜",
        "糟蹋", "倒腾", "报复", "逻辑", "盘缠", "喽啰", "牢骚", "咖喱", "扫把", "惦记"
    };

    must_not_neural_tone_words = {
        "男子", "女子", "分子", "原子", "量子", "莲子", "石子", "瓜子", "电子", "人人", "虎虎",
        "幺幺", "干嘛", "学子", "哈哈", "数数", "袅袅", "局地", "以下", "娃哈哈", "花花草草", "留得",
        "耕地", "想想", "熙熙", "攘攘", "卵子", "死死", "冉冉", "恳恳", "佼佼", "吵吵", "打打",
        "考考", "整整", "莘莘", "落地", "算子", "家家户户", "青青"
    };
}

void ToneSandhi::setPinyinProvider(PinyinProvider provider) {
    pinyin_provider = provider;
}

bool ToneSandhi::_all_tone_three(const std::vector<std::string>& finals) {
    for (const auto& f : finals) {
        if (!is_tone3(f)) return false;
    }
    return true;
}

std::vector<std::string> ToneSandhi::_split_word(const std::string& word) {
    std::u16string u16;
    BasicStringUtil::u8tou16(word.c_str(), word.size(), u16);
    
    std::vector<std::string> res;
    if (u16.length() <= 1) {
        res.push_back(word);
        return res;
    }
    
    size_t split_idx = 1;
    if (u16.length() >= 3) split_idx = 2;
    if (u16.length() >= 4) split_idx = 2;

    std::u16string s1 = u16.substr(0, split_idx);
    std::u16string s2 = u16.substr(split_idx);
    
    std::string u8_1, u8_2;
    BasicStringUtil::u16tou8(s1.data(), s1.size(), u8_1);
    BasicStringUtil::u16tou8(s2.data(), s2.size(), u8_2);
    
    res.push_back(u8_1);
    res.push_back(u8_2);
    return res;
}

std::vector<std::string> ToneSandhi::_bu_sandhi(const std::string& word, std::vector<std::string> finals) {
    std::string BU = "不";
    if (word.length() > BU.length() * 2) { 
         std::u16string u16;
         BasicStringUtil::u8tou16(word.c_str(), word.size(), u16);
         if (u16.length() == 3 && u16[1] == 0x4E0D) { // 不
             if (finals.size() > 1) to_tone5(finals[1]);
             return finals;
         }
    }
    
    std::u16string u16;
    BasicStringUtil::u8tou16(word.c_str(), word.size(), u16);
    for (size_t i = 0; i < u16.length(); ++i) {
        if (u16[i] == 0x4E0D && i + 1 < u16.length()) {
            if (finals.size() > i + 1 && is_tone4(finals[i+1])) {
                if (finals.size() > i) to_tone2(finals[i]);
            }
        }
    }
    return finals;
}

std::vector<std::string> ToneSandhi::_yi_sandhi(const std::string& word, std::vector<std::string> finals) {
    std::string YI = "一";
    if (word.find(YI) == std::string::npos) return finals;
    
    std::u16string u16;
    BasicStringUtil::u8tou16(word.c_str(), word.size(), u16);
    
    bool all_numeric = true;
    for (auto c : u16) {
        if (c != 0x4E00 && !(c >= '0' && c <= '9') && 
            c != 0x96F6 && c != 0x4E8C && c != 0x4E09 && c != 0x56DB && 
            c != 0x4E94 && c != 0x516D && c != 0x4E03 && c != 0x516B && c != 0x4E5D && c != 0x5341) {
            all_numeric = false; 
            break; 
        }
    }
    
    if (all_numeric) return finals;
    
    if (u16.length() == 3 && u16[1] == 0x4E00 && u16[0] == u16[2]) {
        if (finals.size() > 1) to_tone5(finals[1]);
    } else if (word.find("第一") == 0) { 
    } else {
        for (size_t i = 0; i < u16.length(); ++i) {
            if (u16[i] == 0x4E00 && i + 1 < u16.length()) {
                if (finals.size() > i + 1) {
                    if (is_tone4(finals[i+1]) || finals[i+1].back() == '5') { 
                         if (finals.size() > i) to_tone2(finals[i]);
                    } else {
                         std::string& f = finals[i];
                         if (isdigit(f.back())) f.back() = '4';
                         else f += '4';
                    }
                }
            }
        }
    }
    return finals;
}

std::vector<std::string> ToneSandhi::_neural_sandhi(const std::string& word, const std::string& pos, std::vector<std::string> finals) {
    if (must_not_neural_tone_words.count(word)) return finals;
    
    std::u16string u16;
    BasicStringUtil::u8tou16(word.c_str(), word.size(), u16);
    
    if (u16.length() > 1) {
         for (size_t j = 1; j < u16.length(); ++j) {
             if (u16[j] == u16[j-1]) {
                 if (pos.find('n') == 0 || pos.find('v') == 0 || pos.find('a') == 0) {
                     if (finals.size() > j) to_tone5(finals[j]);
                 }
             }
         }
    }

    if (u16.length() >= 1) {
        char16_t last = u16.back();
        std::u16string particles = u"吧呢啊呐噻嘛吖嗨呐哦哒滴哩哟喽啰耶喔诶";
        if (particles.find(last) != std::string::npos) {
            if (!finals.empty()) to_tone5(finals.back());
        }
        else if (last == 0x7684 || last == 0x5730 || last == 0x5F97) {
             if (!finals.empty()) to_tone5(finals.back());
        }
        else if (u16.length() == 1 && (last == 0x4E86 || last == 0x7740 || last == 0x8FC7)) {
             if (pos == "ul" || pos == "uz" || pos == "ug")
                if (!finals.empty()) to_tone5(finals.back());
        }
        else if (u16.length() > 1 && (last == 0x4EEC || last == 0x5B50) && (pos == "r" || pos == "n")) {
             if (!finals.empty()) to_tone5(finals.back());
        }
        else if (u16.length() > 1 && (last == 0x4E0A || last == 0x4E0B) && (pos == "s" || pos == "l" || pos == "f")) {
             if (!finals.empty()) to_tone5(finals.back());
        }
         else if (u16.length() > 1 && (last == 0x6765 || last == 0x53BB)) {
             char16_t prev = u16[u16.length()-2];
             std::u16string dirs = u"上下进出回过起开";
             if (dirs.find(prev) != std::string::npos) {
                 if (!finals.empty()) to_tone5(finals.back());
             }
         }
    }
    
    if (must_neural_tone_words.count(word)) {
        if (!finals.empty()) to_tone5(finals.back());
    }
    
    return finals;
}

std::vector<std::string> ToneSandhi::_three_sandhi(const std::string& word, std::vector<std::string> finals) {
    std::u16string u16;
    BasicStringUtil::u8tou16(word.c_str(), word.size(), u16);
    
    if (u16.length() == 2 && _all_tone_three(finals)) {
        if (finals.size() > 0) to_tone2(finals[0]);
    } else if (u16.length() == 3) {
        auto word_list = _split_word(word); 
        if (_all_tone_three(finals)) {
             std::u16string p1;
             BasicStringUtil::u8tou16(word_list[0].c_str(), word_list[0].size(), p1);
             if (p1.length() == 2) { 
                 if (finals.size() > 1) {
                     to_tone2(finals[0]);
                     to_tone2(finals[1]);
                 }
             } else if (p1.length() == 1) { 
                 if (finals.size() > 1) {
                     to_tone2(finals[1]);
                 }
             }
        } 
    }
    
    return finals;
}

std::vector<std::string> ToneSandhi::modified_tone(const std::string& word, const std::string& pos, 
                                       std::vector<std::string> finals) {
    finals = _bu_sandhi(word, finals);
    finals = _yi_sandhi(word, finals);
    finals = _neural_sandhi(word, pos, finals);
    finals = _three_sandhi(word, finals);
    return finals;
}

std::vector<std::pair<std::string, std::string>> ToneSandhi::_merge_bu(const std::vector<std::pair<std::string, std::string>>& seg) {
    std::vector<std::pair<std::string, std::string>> new_seg;
    std::string BU = "不";
    
    for (size_t i = 0; i < seg.size(); ++i) {
        std::string word = seg[i].first;
        std::string pos = seg[i].second;
        
        if (pos != "x" && pos != "eng") {
             std::string last_word = "";
             if (!new_seg.empty()) last_word = new_seg.back().first;
             
             if (last_word == BU) {
                 new_seg.back().first += word;
                 continue;
             }
        }
        new_seg.push_back({word, pos});
    }
    return new_seg;
}

std::vector<std::pair<std::string, std::string>> ToneSandhi::_merge_yi(const std::vector<std::pair<std::string, std::string>>& seg) {
    return seg;
}

std::vector<std::pair<std::string, std::string>> ToneSandhi::_merge_reduplication(const std::vector<std::pair<std::string, std::string>>& seg) {
    return seg;
}

std::vector<std::pair<std::string, std::string>> ToneSandhi::_merge_er(const std::vector<std::pair<std::string, std::string>>& seg) {
    std::vector<std::pair<std::string, std::string>> new_seg;
    for (size_t i = 0; i < seg.size(); ++i) {
        if (i > 0 && seg[i].first == "儿" && !new_seg.empty() && new_seg.back().second != "x" && new_seg.back().second != "eng") {
            new_seg.back().first += seg[i].first;
        } else {
            new_seg.push_back(seg[i]);
        }
    }
    return new_seg;
}

std::vector<std::pair<std::string, std::string>> ToneSandhi::pre_merge_for_modify(
        const std::vector<std::pair<std::string, std::string>>& seg) {
    auto res = _merge_bu(seg);
    res = _merge_yi(res);
    res = _merge_reduplication(res);
    res = _merge_er(res);
    return res;
}
