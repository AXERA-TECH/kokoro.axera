#include <stdio.h>
#include <vector>
#include <fstream>
#include <ax_sys_api.h>
#include <ctime>
#include <sys/time.h>
#include <locale>

#include "utils/cmdline.hpp"
#include "KokoroHTTPServer.hpp"


// 服务端启动
int main(int argc, char** argv) {
    cmdline::parser cmd;
    cmd.add<int>("port", 'p', "Server port", false, 8080);
    cmd.add<std::string>("axmodel_dir", 0, "model path", false, "../models");
    cmd.add<std::string>("lang", 'l', "Language code, Only support a(American English) or z(Chinese) currently", false, "z");
    cmd.add<std::string>("voice_path", 0, "Binary voices store path", false, "./voices");
    cmd.add<std::string>("voice_name", 'v', "Speaker voice name, check possible choices from voices/", false, "zf_xiaoxiao");
    cmd.add<int>("max_len", 'm', "Max input token num, fixed by model, no need to change usually", false, 96);
    cmd.parse_check(argc, argv);

    // 0. get app args, can be removed from user's app
    auto port = cmd.get<int>("port");
    auto axmodel_dir = cmd.get<std::string>("axmodel_dir");
    auto lang = cmd.get<std::string>("lang");
    auto voice_path = cmd.get<std::string>("voice_path");
    auto voice_name = cmd.get<std::string>("voice_name");
    auto max_len = cmd.get<int>("max_len");

    ALOGI("Args:");
    ALOGI("port: %d", port);
    ALOGI("axmodel_dir: %s", axmodel_dir.c_str());
    ALOGI("lang: %s", lang.c_str());
    ALOGI("voice_path: %s", voice_path.c_str());
    ALOGI("voice_name: %s", voice_name.c_str());
    ALOGI("max_len: %d", max_len);

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

    KokoroHTTPServer server;
    
    ALOGI("Initializing TTS Server......");
    if (!server.init(axmodel_dir, max_len, lang, voice_path, voice_name, "dict/vocab.txt")) {
        std::cerr << "Failed to initialize server" << std::endl;
        return -1;
    }
    
    ALOGI("Start TTS Server at port %d", port);
    server.start(port);
    return 0;
}