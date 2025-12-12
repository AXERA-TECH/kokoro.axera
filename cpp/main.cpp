#include <stdio.h>
#include <vector>
#include <fstream>
#include <ax_sys_api.h>
#include <ctime>
#include <sys/time.h>
#include <locale>

#include "utils/cmdline.hpp"
#include "AudioFile.h"
#include "utils/timer.hpp"
#include "utils/logger.hpp"
#include "Kokoro.h"


int main(int argc, char** argv) {
    // 设置locale为UTF-8
    std::setlocale(LC_ALL, "en_US.UTF-8");

    cmdline::parser cmd;
    cmd.add<std::string>("axmodel_dir", 0, "model path", false, "../models");
    cmd.add<std::string>("text", 't', "Text to be generated", false, "我想留在大家身边，从过去，一同迈向明天");
    cmd.add<std::string>("lang", 'l', "Language code, Only support a(American English) or z(Chinese) currently", false, "z");
    cmd.add<std::string>("voice_path", 0, "Binary voices store path", false, "./voices");
    cmd.add<std::string>("voice_name", 'v', "Speaker voice name, check possible choices from voices/", false, "zf_xiaoxiao");
    cmd.add<std::string>("output", 'o', "Output file path", false, "output.wav");
    cmd.add<float>("fade_out", 'f', "Fade out ratio between sentences", false, 0.0f);
    cmd.add<int>("max_len", 'm', "Max input token num, fixed by model, no need to change usually", false, 96);
    cmd.parse_check(argc, argv);

    // 0. get app args, can be removed from user's app
    auto axmodel_dir = cmd.get<std::string>("axmodel_dir");
    auto text = cmd.get<std::string>("text");
    auto lang = cmd.get<std::string>("lang");
    auto voice_path = cmd.get<std::string>("voice_path");
    auto voice_name = cmd.get<std::string>("voice_name");
    auto output = cmd.get<std::string>("output");
    auto fade_out = cmd.get<float>("fade_out");
    auto max_len = cmd.get<int>("max_len");

    ALOGI("Args:");
    ALOGI("axmodel_dir: %s", axmodel_dir.c_str());
    ALOGI("text: %s", text.c_str());
    ALOGI("lang: %s", lang.c_str());
    ALOGI("voice_path: %s", voice_path.c_str());
    ALOGI("voice_name: %s", voice_name.c_str());
    ALOGI("output: %s", output.c_str());
    ALOGI("fade_out: %.2f", fade_out);
    ALOGI("max_len: %d", max_len);

    const float SPEED = 1.0f;
    const float PAUSE = 0.0f;
    const int sample_rate = 24000;

    int ret = AX_SYS_Init();
    if (0 != ret) {
        fprintf(stderr, "AX_SYS_Init failed! ret = 0x%x\n", ret);
        return -1;
    }

#if defined(CHIP_AX650)
    AX_ENGINE_NPU_ATTR_T npu_attr;
    memset(&npu_attr, 0, sizeof(npu_attr));
    npu_attr.eHardMode = static_cast<AX_ENGINE_NPU_MODE_T>(0);
    ret = AX_ENGINE_Init(&npu_attr);
    if (0 != ret) {
        fprintf(stderr, "Init ax-engine failed{0x%8x}.\n", ret);
        return -1;
    }
#else
    AX_ENGINE_NPU_ATTR_T npu_attr;
    memset(&npu_attr, 0, sizeof(npu_attr));
    npu_attr.eHardMode = AX_ENGINE_VIRTUAL_NPU_DISABLE;
    ret = AX_ENGINE_Init(&npu_attr);
    if (0 != ret) {
        fprintf(stderr, "Init ax-engine failed{0x%8x}.\n", ret);
        return -1;
    }
#endif   

    Timer timer;

    timer.start();
    Kokoro kokoro;
    if (!kokoro.init(axmodel_dir, max_len, lang, voice_path, voice_name)) {
        ALOGE("Init kokoro failed!");
        return -1;
    }
    timer.stop();
    ALOGI("Init kokoro take %.4f seconds", timer.elapsed<std::chrono::seconds>());

    timer.start();
    std::vector<float> audio;
    if (!kokoro.tts(text, voice_name, SPEED, sample_rate, fade_out, PAUSE, audio)) {
        printf("run whisper failed!\n");
        return -1;
    }
    timer.stop();

    AudioFile<float> audio_file;
    std::vector<std::vector<float> > audio_samples{audio};
    audio_file.setAudioBuffer(audio_samples);
    audio_file.setSampleRate(sample_rate);
    if (!audio_file.save(output)) {
        ALOGE("Save audio file failed!\n");
        return -1;
    }

    ALOGI("Audio save to %s", output.c_str());

    float elapsed = timer.elapsed<std::chrono::seconds>();
    float duration = audio.size() * 1.f / sample_rate;
    ALOGI("RTF: %.4f, process_time: %.4f seconds, audio duration: %.2f seconds\n", elapsed / duration, elapsed, duration);
    return 0;
}
