#include "PinyinFinder.h"

#include <regex>
#include <vector>
#include <locale>
#include <limits>
#include <cstring>
#include <cstdio>

PinyinFinder::PinyinFinder() {}

PinyinFinder::~PinyinFinder() {}

bool PinyinFinder::init(const std::string& singleCharacterDictPath,
          const std::string& wordsDictPath) {
  
  FILE* fp = fopen(singleCharacterDictPath.c_str(), "r");
  if (fp == NULL) {
    LOG_WARNING << "Failed to open file: " << singleCharacterDictPath << std::endl;
    return false;
  }
  char line[4096] = {0};
  int tn = 0;
  while (fgets(line, sizeof(line) - 1, fp)) {
    if (line[0] == '#') continue;
    size_t len = strlen(line);
    while (len > 0 && (line[len-1] == '\r' || line[len-1] == '\n')) {
        line[--len] = '\0';
    }
    if (len == 0) continue;

    // Format: U+XXXX: pinyin # comment
    if (strncmp(line, "U+", 2) != 0) continue;
    
    char* colon = strchr(line, ':');
    if (!colon) continue;
    
    // Parse Hex
    // U+ is at line[0], hex starts at line[2], length is colon - line - 2
    std::string hexStr(line + 2, colon - (line + 2));
    uint32_t unicode = 0;
    try {
        unicode = std::stoul(hexStr, nullptr, 16);
    } catch(...) { continue; }
    
    // Parse Pinyin
    char* pinyinStart = colon + 1;
    while (*pinyinStart == ' ' || *pinyinStart == '\t') pinyinStart++;
    
    char* hash = strchr(pinyinStart, '#');
    char* pinyinEnd = hash ? hash : (line + len);
    while (pinyinEnd > pinyinStart && (*(pinyinEnd-1) == ' ' || *(pinyinEnd-1) == '\t')) {
        pinyinEnd--;
    }
    
    if (pinyinEnd <= pinyinStart) continue;
    
    std::string pinyin(pinyinStart, pinyinEnd - pinyinStart);
    
    UnicodeStr ustr;
    ustr.append(1, static_cast<UnicodeCharT>(unicode));
    
    std::vector<std::string> ss;
    BasicStringUtil::SplitString(pinyin.c_str(), pinyin.size(), ',', &ss);
    if (!ss.empty()) {
        word_pinyin_dict_[ustr] = ss[0];
    }
    tn += 1;
  }
  LOG_INFO << "total pinyin character count: " << tn << std::endl;

  fclose(fp);
  fp = fopen(wordsDictPath.c_str(), "r");
  if (fp == NULL) {
    LOG_WARNING << "Failed to open file: " << wordsDictPath << std::endl;
    return false;
  }
  
  // Format: word: pinyin1 pinyin2
  int pc = 0;
  while (fgets(line, sizeof(line) - 1, fp)) {
    if (line[0] == '#') continue;
    
    // Strip comments
    char* comment = strchr(line, '#');
    if (comment) *comment = '\0';
    
    size_t len = strlen(line);
    while (len > 0 && (line[len-1] == '\r' || line[len-1] == '\n')) {
        line[--len] = '\0';
    }
    if (len == 0) continue;

    char* colon = strchr(line, ':');
    if (!colon) continue;
    
    // Parse Word
    char* wordEnd = colon;
    while (wordEnd > line && (*(wordEnd-1) == ' ' || *(wordEnd-1) == '\t')) {
        wordEnd--;
    }
    std::string word(line, wordEnd - line);
    if (word.empty()) continue;
    
    // Parse Pinyin
    char* pinyinStart = colon + 1;
    while (*pinyinStart == ' ' || *pinyinStart == '\t') pinyinStart++;
    
    char* pinyinEnd = line + len;
    while (pinyinEnd > pinyinStart && (*(pinyinEnd-1) == ' ' || *(pinyinEnd-1) == '\t')) {
        pinyinEnd--;
    }
    
    std::string pinyin(pinyinStart, pinyinEnd - pinyinStart);
    
    UnicodeStr ustr;
    BasicStringUtil::u8tou16(word.c_str(), word.size(), ustr);
    word_pinyin_dict_[ustr] = pinyin;

    pc += 1;
  }
  LOG_INFO << "total pinyin phrase count: " << pc << std::endl;
  return true;
}

void PinyinFinder::find_best_pinyin(const std::string& phrasestr, std::vector<std::string>& pinyins) {
  UnicodeStr phrase;
  BasicStringUtil::u8tou16(phrasestr.c_str(), phrasestr.size(), phrase);
  int n = phrase.size();
  if (n == 0) return;

  std::vector<std::vector<int>> dp(n, std::vector<int>(n, std::numeric_limits<int>::max()));
  std::vector<std::vector<int>> opts(n, std::vector<int>(n, -1));
  
  for (int length = 1; length <= n; ++length) {
    for (int i = 0; i <= n - length; ++i) {
      int j = i + length - 1;
      if (length == 1) {
        dp[i][j] = 1;
        opts[i][j] = j;
      } else {
        int maxtry = length;
        if (length > kMaxChars) {
          maxtry = kMaxChars;
        }
        for (int k = maxtry; k >= 1; k--) {
          int to = i + k - 1;
          UnicodeStr sub = phrase.substr(i, k);
          
          if (word_pinyin_dict_.find(sub) != word_pinyin_dict_.end()) {
            if (to == j) {
              dp[i][j] = k == 1 ? 1 : 0; // Preference for longer matches
              opts[i][j] = j;
            } else {
              // Cost calculation logic from original code
              // k == 1 means we took a single char
              int cost = (k == 1) ? (dp[to + 1][j] + 1) : dp[to + 1][j];
              
              if (dp[i][j] > cost) {
                dp[i][j] = cost;
                opts[i][j] = to;
              }
            }
          }
        }
      }
    }
  }

  // Construct the best pinyin string using dp and opts
  int i = 0;
  int j = n - 1;
  while (i <= j) {
    int opt = opts[i][j];
    if(opt == -1){
      opt = i;
    }
    UnicodeStr sub = phrase.substr(i, opt - i + 1);
    auto it = word_pinyin_dict_.find(sub);
    
    if(it == word_pinyin_dict_.end()){
      std::string tstr;
      BasicStringUtil::u16tou8(sub.data(), sub.size(), tstr);
      pinyins.emplace_back(tstr);
    } else {
      std::vector<std::string> tmps;
      BasicStringUtil::SplitString(it->second.c_str(), it->second.size(), ' ', &tmps);
      for(auto& str: tmps){
        pinyins.emplace_back(str);
      }
    }
    i = opt + 1;
  }
}
