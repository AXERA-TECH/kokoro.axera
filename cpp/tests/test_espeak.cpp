#include "utils/cmdline.hpp"
#include "espeak-ng/speak_lib.h"

int main(int argc, char** argv) {
    cmdline::parser cmd;
    cmd.add<std::string>("language", 'l', "Language choice: a, j", true, "");
    cmd.add<std::string>("text", 't', "Text to be parsed", true, "");
    cmd.parse_check(argc, argv);

    // get app args, can be removed from user's app
    auto language = cmd.get<std::string>("language");
    auto text = cmd.get<std::string>("text");

    printf("language: %s\n", language.c_str());
    printf("text: %s\n", text.c_str());

    printf("Initializing espeak-ng...\n");
    espeak_Initialize(AUDIO_OUTPUT_SYNCHRONOUS, 0, nullptr, 0);

    // espeak_VOICE voice;
    // memset(&voice, 0, sizeof(voice));
    
    // char lang_code_char = language[0];
    // switch (lang_code_char) {
    //     case 'a': {
    //         voice.name = "English_(America)";
    //         voice.languages = "en-us";
    //         break;
    //     }
    //     case 'j': {
    //         voice.name = "Japanese";
    //         voice.languages = "ja";
    //         break;
    //     }
    //     default: {
    //         voice.name = "English_(America)";
    //         voice.languages = "en-us";
    //         break;
    //     }
    // }

    // voice.gender = 2;
    // espeak_SetVoiceByProperties(&voice);

    espeak_SetVoiceByName("ja");

    printf("Initialize done.\n");

    const char* textPtr = text.c_str();
    int phonememode = ('_' << 8) | 0x02;
    const char * phonemes = espeak_TextToPhonemes(
            reinterpret_cast<const void **>(&textPtr), espeakCHARS_UTF8, phonememode);

    printf("Phonemes: %s\n", phonemes);
    
    espeak_Terminate();
    return 0;
}