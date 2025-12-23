// minimal_japanese_phoneme.cpp
#include "espeak-ng/speak_lib.h"
#include <iostream>
#include <cstring>

int main(int argc, char* argv[]) {
    if (argc != 2) {
        std::cout << "Usage: " << argv[0] << " \"Japanese text\"" << std::endl;
        return 1;
    }
    
    // 初始化espeak-ng
    const char* data_path = "./third_party/espeak-ng/share/espeak-ng-data";
    if (espeak_Initialize(AUDIO_OUTPUT_RETRIEVAL, 0, data_path, 0) <= 0) {
        std::cerr << "Failed to initialize espeak-ng" << std::endl;
        return 1;
    }
    
    // 设置日语语音
    // espeak_SetVoiceByName("ja");
    // 2. 声明并初始化语音属性结构体
    espeak_VOICE voice_spec;
    memset(&voice_spec, 0, sizeof(espeak_VOICE)); // 重要：先清零

    // 3. 设置日语语音属性
    voice_spec.languages = "ja";     // 日语语言代码
    voice_spec.gender = 2;           // 女性（1=男，2=女），设为0则不指定
    voice_spec.age = 0;              // 年龄，0表示不指定
    voice_spec.variant = 0;          // 变体
    voice_spec.name = NULL;          // 让系统自动选择匹配的语音

    // 4. 应用语音设置
    // 函数会返回匹配成功的语音指针，如果失败则返回NULL
    int ret = espeak_SetVoiceByProperties(&voice_spec);
    
    if (ret != EE_OK) {
        printf("无法找到匹配的日语语音！\n");
        // 可能需要检查你的 eSpeak 是否安装了日语语音包
        espeak_Terminate();
        return -1;
    } else {
        printf("成功设置日语语音。\n");
    }
    
    // 启用音素输出
    // espeak_SetParameter(espeakPHONEMES, 1, 0);
    
    const char* japanese_text = argv[1];
    std::cout << "Japanese text: " << japanese_text << std::endl;
    
    // 转换为音素
    int phonememode = ('_' << 8) | 0x02;

    const char* result = espeak_TextToPhonemes(
        reinterpret_cast<const void**>(&japanese_text),
        espeakCHARS_UTF8,
        phonememode
    );
    
    std::cerr << "Phoneme: " << result << std::endl;
    
    espeak_Terminate();
    return 0;
}