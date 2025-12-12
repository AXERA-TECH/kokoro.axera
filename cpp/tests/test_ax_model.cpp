/**************************************************************************************************
 *
 * Copyright (c) 2019-2025 Axera Semiconductor (Ningbo) Co., Ltd. All Rights Reserved.
 *
 * This source file is the property of Axera Semiconductor (Ningbo) Co., Ltd. and
 * may not be copied or distributed in any isomorphic form without the prior
 * written consent of Axera Semiconductor (Ningbo) Co., Ltd.
 *
 **************************************************************************************************/

#include "ax_model_runner/ax_model_runner.hpp"
#include "utils/cmdline.hpp"
#include "utils/timer.hpp"

#include <ax_sys_api.h>
#include <ax_engine_api.h>

#include <stdio.h>

int main(int argc, char** argv) {
    cmdline::parser cmd;
    cmd.add<std::string>("model", 'm', "axmodel path", true, "");
    cmd.add<int>("repeat", 'r', "repeat times", false, 10);
    cmd.parse_check(argc, argv);

    // get app args, can be removed from user's app
    auto model_path = cmd.get<std::string>("model");
    auto repeat = cmd.get<int>("repeat");

    printf("model: %s\n", model_path.c_str());
    printf("repeat: %d\n", repeat);

    int ret = AX_SYS_Init();
    if (0 != ret) {
        fprintf(stderr, "AX_SYS_Init failed! ret = 0x%x\n", ret);
        return -1;
    }

    AX_ENGINE_NPU_ATTR_T npu_attr;
    memset(&npu_attr, 0, sizeof(npu_attr));
    npu_attr.eHardMode = static_cast<AX_ENGINE_NPU_MODE_T>(0);
    ret = AX_ENGINE_Init(&npu_attr);
    if (0 != ret) {
        fprintf(stderr, "Init ax-engine failed{0x%8x}.\n", ret);
        return -1;
    }

    // load model
    AxModelRunner model;
    if (0 != model.load_model(model_path.c_str())) {
        fprintf(stderr, "load model failed!\n");
        return -1;
    }

    int input_num = model.get_input_num();
    int output_num = model.get_output_num();

    printf("================ MODEL INFO ================\n");
    printf("Input num: %d\n", input_num);
    printf("Output num: %d\n", output_num);
    printf("\n");

    printf("======== Input ========\n");
    for (int i = 0; i < input_num; i++) {
        printf("[%d]:\n", i);
        printf("name: %s\n", model.get_input_name(i));
        printf("size: %d\n", model.get_input_size(i));
        printf("shape: ");
        auto shape = model.get_input_shape(i);
        for (int n = 0; n < shape.size(); n++) {
            printf("%d ", shape[n]);
        }
        printf("\n\n");
    }

    printf("======== Output ========\n");
    for (int i = 0; i < output_num; i++) {
        printf("[%d]:\n", i);
        printf("name: %s\n", model.get_output_name(i));
        printf("size: %d\n", model.get_output_size(i));
        printf("shape:\n\t");
        auto shape = model.get_output_shape(i);
        for (int n = 0; n < shape.size(); n++) {
            printf("%d ", shape[n]);
        }
        printf("\n\n");
    }

    Timer timer;

    std::vector<std::vector<char>> inputs, outputs;
    inputs.resize(input_num);
    outputs.resize(output_num);

    for (int i = 0; i < input_num; i++) {
        inputs[i].resize(model.get_input_size(i));
    }
    for (int i = 0; i < output_num; i++) {
        outputs[i].resize(model.get_output_size(i));
    }

    timer.start();
    for (int n = 0; n < repeat; n++) {
        for (int i = 0; i < input_num; i++) {
            model.set_input(i, inputs[i].data());
        }
        model.run();
        for (int i = 0; i < output_num; i++) {
            model.get_output(i, outputs[i].data());
        }
    }
    timer.stop();

    printf("average latency over %d times: %.4fms\n", repeat, timer.elapsed() / repeat);

    return 0;
}