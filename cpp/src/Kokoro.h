#pragma once

#include <string>
#include <vector>
#include <memory>
#include <map>
#include <optional>
#include <iostream>
#include <algorithm>
#include <cmath>
#include "ax_model_runner/ax_model_runner.hpp"
#include "onnxruntime_cxx_api.h"

// Forward declarations or placeholder for dependencies
class Tokenizer;
struct KoKoroConfig;

// Constants from config
const int MAX_PHONEME_LENGTH = 510; // max position embedding - 2
const int SAMPLE_RATE = 24000;      // Example value

// Preprocess parameters
const int FIXED_SEQ_LEN = 96;
const int N_FFT = 20;
const int HOP_LENGTH = 5;
const int DOUBLE_INPUT_THRESHOLD = 32;  // 输入长度小于此值时复制一倍,适配短文本
const int STYLE_DIM = 256;

const float DEFAULT_SPEED = 1.0;
const float DEFAULT_FADE_OUT = 0.05;
const float DEFAULT_PAUSE = 0.05;


// 定义结构体来存储句子信息
struct SentenceInfo {
    std::string sentence;
    std::vector<int> input_ids;  
    std::string phonemes;    
    int content_len;
    bool is_long;
    std::vector<SentenceInfo> sub_results;  // 长句的分割结果
    
    // 构造函数：短句
    SentenceInfo(const std::string& s, 
                 const std::vector<int>& ids,
                 const std::string& ph,
                 int len)
        : sentence(s), input_ids(ids), phonemes(ph), 
          content_len(len), is_long(false) {}
    
    // 构造函数：长句
    SentenceInfo(const std::string& s, 
                 const std::vector<SentenceInfo>& sub)
        : sentence(s), content_len(0), is_long(true), sub_results(sub) {}
};

// 合并组的结果
struct MergedGroup {
    bool is_long_split = false;
    std::vector<int> input_ids;
    std::string phonemes;
    std::vector<SentenceInfo> sub_results;
    
    // 构造函数：短句合并组
    MergedGroup(const std::vector<int>& ids, const std::string& ph)
        : is_long_split(false), input_ids(ids), phonemes(ph) {}
    
    // 构造函数：长句分割组
    MergedGroup(const std::vector<SentenceInfo>& sub)
        : is_long_split(true), sub_results(sub) {}
};


class Kokoro {
public:
    Kokoro();
    ~Kokoro();

    bool init(const std::string& model_path, 
        int max_seq_len = FIXED_SEQ_LEN, 
        const std::string& lang_code = "z",
        const std::string& voices_path = "./voices", 
        const std::string& voice_name = "af_heart",
        const std::string& vocab_path = "dict/vocab.txt");
    
    bool get_voice_style(const std::string& voices_path, const std::string& voice_name);

    bool tts(
        const std::string& text,
        const std::string& voice_name,
        float speed,
        int sample_rate,
        float fade_out,
        float pause_duration,
        std::vector<float>& generated_audio
    );

    bool inference(
        std::vector<int>& input_ids,
        const std::vector<float>& ref_s,
        float speed,
        float fade_out_duration,
        std::vector<float>& audio
    );

    bool run_batch_inference(std::vector<MergedGroup>& merged_group, const std::string& voice_name, 
                       float speed, float fade_out_duration, int sr,
                       std::vector<std::vector<float>>& audio_list
    );

private:
    // Internal methods
    bool inference_single_chunk(
        std::vector<int>& input_ids,
        const std::vector<float>& ref_s,
        int actual_len,
        float speed,
        std::vector<float>& audio,
        int& actual_content_frames,
        int& total_frames
    );

    void split_input_ids_semantic(std::vector<int>& input_ids, int fixed_seq_len, int& actual_len);

    void _trim_audio_by_content(std::vector<float>& audio, int actual_content_frames, int total_frames, int actual_len);

    void apply_fade_out(std::vector<float>& audio, int fade_samples);

    std::vector<std::string> _split_phonemes(const std::string& phonemes);

    void _prepare_input_ids(std::vector<int>& input_ids, int& actual_len, bool& is_doubled); 
    
    void _compute_external_preprocessing(const std::vector<int>& input_ids, int actual_len, std::vector<int>& input_lengths, std::vector<uint8_t>& text_mask);

    // 处理duration并对齐
    void _process_duration(const std::vector<float>& duration, int actual_len, float speed, std::vector<int>& pred_dur, int& total_frames);
    std::vector<float> _create_alignment_matrix(const std::vector<int>& pred_dur, int total_frames);

    void _compute_har_onnx(std::vector<float>& F0_pred, std::vector<float>& har);

    void _postprocess_x_to_audio(std::vector<float>& x, std::vector<float>& audio);

    void generate_input_ids_from_text(const std::string& text, std::vector<int>& input_ids, std::string& phonemes);
    
    std::vector<SentenceInfo> split_long_sentence(
        const std::string& sentence,
        const std::string& lang_code,
        int max_merge_len = 78
    );

    std::vector<MergedGroup> process_and_merge_sentences(
        const std::string& text,
        const std::string& lang_code,
        int max_merge_len = 96
    );

    std::vector<float> load_voice_embedding(int phoneme_len);

private:
    int max_seq_len_;
    std::string lang_code_;
    std::string voices_path_;
    std::string voice_name_;
    int voice_pack_size_;
    std::vector<float> voice_tensor_;

    AxModelRunner model1_, model2_, model3_;
    Ort::Env env_;
    Ort::Session model4_{nullptr};
    Ort::AllocatorWithDefaultOptions allocator_;
    
    // Placeholder for voices data: map from name to vector
    std::map<std::string, std::vector<float>> voices_;
    
    std::unique_ptr<Tokenizer> tokenizer_;

    // model outputs data
    std::vector<float> duration_, d_;
    std::vector<float> F0_pred_, N_pred_, asr_;
    std::vector<float> x_;

    // model outputs shape
    std::vector<int> duration_shape_, d_shape_;
    std::vector<int> F0_pred_shape_;
    std::vector<int> x_shape_;
};
