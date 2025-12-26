#include "Kokoro.h"
#include "Tokenizer.h"
#include <fstream>
#include <sstream>
#include <regex>
#include <numeric>
#include <cstring>
#include <algorithm>
#include <math.h>
#include <locale>
#include <codecvt>
#include "utils/logger.hpp"
#include "librosa/eigen3/Eigen/Dense"
#include "librosa/librosa.h"
#include "split_utils.hpp"

using namespace std;

typedef Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> DynMat;

typedef std::vector<std::vector<std::complex<float>>> FFT_RESULT;

// DEBUG
template <typename T>
static void save_file(const std::vector<T>& data, const char* filename) {
    FILE* fp = fopen(filename, "wb");
    fwrite(data.data(), sizeof(T), data.size(), fp);
    fclose(fp);
}

void save_fft_result(const FFT_RESULT& f, const char* filename) {
    FILE* fp = fopen(filename, "wb");
    for (size_t i = 0; i < f.size(); i++) {
        for (size_t j = 0; j < f[0].size(); j++) {
            float real = f[i][j].real();
            float imag = f[i][j].imag();
            fwrite(&real, sizeof(float), 1, fp);
            fwrite(&imag, sizeof(float), 1, fp);
        }
    }
    fclose(fp);
}

// Helper functions
static std::vector<float> sigmoid(const std::vector<float>& x) {
    std::vector<float> result(x.size());
    for (int i = 0; i < x.size(); i++) {
        result[i] = 1.0f / (1.0f + expf(-x[i]));
    }
    return result;
}

template <typename T>
std::vector<T> load_file(const char* filename) {
    // 打开文件（二进制模式）
    std::ifstream file(filename, std::ios::binary);
    
    // 获取文件大小
    file.seekg(0, std::ios::end);
    size_t file_size = file.tellg();
    file.seekg(0, std::ios::beg);
    
    // 计算float数量
    size_t num_floats = file_size / sizeof(T);
    
    // 创建向量并读取数据
    std::vector<T> result;
    result.resize(num_floats);
    file.read(reinterpret_cast<char*>(result.data()), file_size);
    
    file.close();
    return result;
}

template <typename T>
vector<size_t> argsort(const vector<T> &v, int len, bool reverse) {
    // initialize original index locations
    vector<size_t> idx(len);
    iota(idx.begin(), idx.end(), 0);

    // sort indexes based on comparing values in v
    // using std::stable_sort instead of std::sort
    // to avoid unnecessary index re-orderings
    // when v contains elements of equal values 
    if (!reverse)
        stable_sort(idx.begin(), idx.end(),
            [&v](size_t i1, size_t i2) {return v[i1] < v[i2];});
    else
        stable_sort(idx.begin(), idx.end(),
            [&v](size_t i1, size_t i2) {return v[i1] > v[i2];});

    return idx;
}

template <typename T>
vector<T> np_repeat(const vector<T> &v, const vector<int>& times) {
    vector<T> result;
    for (size_t i = 0; i < times.size(); i++) {
        for (int n = 0; n < times[i]; n++)
            result.push_back(v[i]);
    }
    return result;
}

template <typename T>
std::vector<T> linspace(T a, T b, size_t N) {
    T h = (b - a) / static_cast<T>(N-1);
    std::vector<T> xs(N);
    typename std::vector<T>::iterator x;
    T val;
    for (x = xs.begin(), val = a; x != xs.end(); ++x, val += h)
        *x = val;
    return xs;
}

/**
 * 清理文本：移除多余空格、控制字符等
 * @param text 输入文本
 * @return 清理后的文本
 */
string clean_text(const string& text) {
    if (text.empty()) return "";
    
    string result;
    result.reserve(text.length());
    
    // 第一步：处理空格和替换控制字符
    bool last_was_space = false;
    for (char c : text) {
        // 保留换行符、回车符、制表符
        if (c == '\n' || c == '\r' || c == '\t') {
            result.push_back(c);
            last_was_space = false;
        }
        // 处理普通空格
        else if (isspace(static_cast<unsigned char>(c))) {
            if (!last_was_space) {
                result.push_back(' ');
                last_was_space = true;
            }
        }
        // 保留可打印字符（ASCII >= 32）
        else if (static_cast<unsigned char>(c) >= 32) {
            result.push_back(c);
            last_was_space = false;
        }
        // 其他控制字符被忽略
    }
    
    // 第二步：去除首尾空格
    // 去除开头的空格
    size_t start = 0;
    while (start < result.length() && result[start] == ' ') {
        start++;
    }
    
    // 去除结尾的空格
    size_t end = result.length();
    while (end > start && result[end - 1] == ' ') {
        end--;
    }
    
    return result.substr(start, end - start);
}

/**
 * 拼接音频片段
 * @param segment_data_list 音频片段列表（每个片段是float向量）
 * @param sample_rate 采样率，默认24000
 * @param speed 语速，默认1.0
 * @param pause_duration 停顿时长（秒），默认0.5
 * @return 拼接后的音频数据（std::vector<float>）
 */
std::vector<float> audio_numpy_concat(
    const std::vector<std::vector<float>>& segment_data_list,
    int sample_rate = 24000,
    float speed = 1.0f,
    float pause_duration = 0.5f
) {
    // 如果输入为空，返回空向量
    if (segment_data_list.empty()) {
        return std::vector<float>();
    }
    
    // 计算停顿的样本数
    int pause_samples = 0;
    if (pause_duration > 0.0f && speed > 0.0f) {
        pause_samples = static_cast<int>((sample_rate * pause_duration) / speed);
        if (pause_samples < 0) {
            pause_samples = 0;
        }
    }
    
    // 首先计算总长度，预分配内存（提高性能）
    size_t total_length = 0;
    for (const auto& segment : segment_data_list) {
        total_length += segment.size();
    }
    
    // 添加停顿的长度（在片段之间）
    size_t num_pauses = 0;
    if (segment_data_list.size() > 1 && pause_samples > 0) {
        num_pauses = segment_data_list.size() - 1;
        total_length += pause_samples * num_pauses;
    }
    
    // 创建结果向量并预分配内存
    std::vector<float> result;
    result.reserve(total_length);
    
    // 拼接所有片段
    for (size_t i = 0; i < segment_data_list.size(); ++i) {
        // 添加当前音频片段
        const auto& current_segment = segment_data_list[i];
        result.insert(result.end(), 
                     current_segment.begin(), 
                     current_segment.end());
        
        // 如果不是最后一个片段，添加停顿
        if (i < segment_data_list.size() - 1 && pause_samples > 0) {
            result.insert(result.end(), pause_samples, 0.0f);
        }
    }
    
    return result;
}


Kokoro::Kokoro() 
{
    
}

Kokoro::~Kokoro() {
    // Resources cleaned up by wrappers
}

bool Kokoro::init(const std::string& model_path, int max_seq_len, const std::string& lang_code, const std::string& voices_path, const std::string& voice_name, const std::string& vocab_path) {
    max_seq_len_ = max_seq_len;
    voices_path_ = voices_path;
    voice_name_ = voice_name;
    lang_code_ = lang_code;

    env_ = Ort::Env(ORT_LOGGING_LEVEL_WARNING, "Kokoro");
    // Initialize session options
    Ort::SessionOptions session_options;
    session_options.SetIntraOpNumThreads(1);
    // session_options.SetLogSeverityLevel(ORT_LOGGING_LEVEL_VERBOSE);
    session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);

    // Load models
    std::string model1_path = model_path + "/kokoro_part1_96.axmodel";
    std::string model2_path = model_path + "/kokoro_part2_96.axmodel";
    std::string model3_path = model_path + "/kokoro_part3_96.axmodel";
    std::string model4_path = model_path + "/model4_har_sim.onnx";

    // Load voice
    if (!get_voice_style(voices_path, voice_name)) {
        ALOGE("Load voice failed!");
        return false;
    }

    int ret = model1_.load_model(model1_path.c_str());
    if (0 != ret) {
        ALOGE("load model1 %s failed! ret=0x%x", model1_path.c_str(), ret);
        return false;
    }

    ret = model2_.load_model(model2_path.c_str());
    if (0 != ret) {
        ALOGE("load model2 %s failed! ret=0x%x", model2_path.c_str(), ret);
        return false;
    }

    ret = model3_.load_model(model3_path.c_str());
    if (0 != ret) {
        ALOGE("load model3 %s failed! ret=0x%x", model3_path.c_str(), ret);
        return false;
    }

    model4_ = Ort::Session(env_, model4_path.c_str(), session_options);

    // Load vocab
    std::map<std::string, int> vocab;
    std::ifstream in(vocab_path);
    if (in.is_open()) {
        std::string line;
        while (std::getline(in, line)) {
            // Expected format: token<TAB>id
            size_t tab = line.find('\t');
            if (tab != std::string::npos) {
                std::string token = line.substr(0, tab);
                std::string id_str = line.substr(tab + 1);
                // Unescape token if needed (\n, \r, \t)
                size_t pos = 0;
                while((pos = token.find("\\n", pos)) != std::string::npos) { token.replace(pos, 2, "\n"); pos += 1; }
                pos = 0;
                while((pos = token.find("\\r", pos)) != std::string::npos) { token.replace(pos, 2, "\r"); pos += 1; }
                pos = 0;
                while((pos = token.find("\\t", pos)) != std::string::npos) { token.replace(pos, 2, "\t"); pos += 1; }
                
                try {
                    vocab[token] = std::stoi(id_str);
                } catch (...) {}
            }
        }
        ALOGI("Loaded %ld tokens from %s", vocab.size(), vocab_path.c_str());
    } else {
        ALOGE("Failed to open vocab file %s", vocab_path.c_str());
        return false;
    }

    // Initialize Tokenizer
    tokenizer_ = std::make_unique<Tokenizer>(lang_code_, TokenizerConfig{}, vocab);

    // Prepare model outputs
    duration_.resize(model1_.get_output_size(0) / sizeof(float));
    d_.resize(model1_.get_output_size(1) / sizeof(float));

    // F0_pred, N_pred, asr = outputs2
    F0_pred_.resize(model2_.get_output_size(0) / sizeof(float));
    N_pred_.resize(model2_.get_output_size(1) / sizeof(float));
    asr_.resize(model2_.get_output_size(2) / sizeof(float));

    x_.resize(model3_.get_output_size(0) / sizeof(float));

    duration_shape_ = model1_.get_output_shape(0);
    d_shape_ = model1_.get_output_shape(1);

    F0_pred_shape_ = model2_.get_output_shape(0);

    x_shape_ = model3_.get_output_shape(0);

    return true;
}

bool Kokoro::get_voice_style(const std::string& voices_path, const std::string& voice_name) {
    // 打开文件（二进制模式）
    std::string voice_bin_path = voices_path + "/" + voice_name + ".bin";
    std::ifstream file(voice_bin_path, std::ios::binary);
    if (!file.is_open()) {
        ALOGE("Open file %s failed!", voice_bin_path.c_str());
        return false;
    }
    
    // 获取文件大小
    file.seekg(0, std::ios::end);
    size_t file_size = file.tellg();
    file.seekg(0, std::ios::beg);
    
    // 计算float数量
    size_t num_floats = file_size / sizeof(float);

    voice_pack_size_ = num_floats / STYLE_DIM;
    
    // 创建向量并读取数据
    voice_tensor_.resize(num_floats);
    file.read(reinterpret_cast<char*>(voice_tensor_.data()), file_size);
    
    file.close();
    return true;
}

std::vector<std::string> Kokoro::_split_phonemes(const std::string& phonemes) {
    std::vector<std::string> batches;
    std::regex re("([.,!?;])");
    std::sregex_token_iterator it(phonemes.begin(), phonemes.end(), re, {-1, 0}); // -1 for non-match, 0 for match
    std::sregex_token_iterator end;

    std::string current_batch;
    
    for (; it != end; ++it) {
        std::string part = *it;
        // Removing leading/trailing whitespace
        part = std::regex_replace(part, std::regex("^\\s+|\\s+$"), "");
        
        if (part.empty()) continue;

        if (current_batch.length() + part.length() + 1 >= MAX_PHONEME_LENGTH) {
            batches.push_back(current_batch);
            current_batch = part;
        } else {
             if (std::string(".,!?;").find(part) != std::string::npos) {
                current_batch += part;
             } else {
                if (!current_batch.empty()) current_batch += " ";
                current_batch += part;
             }
        }
    }
    if (!current_batch.empty()) {
        batches.push_back(current_batch);
    }
    return batches;
}

void Kokoro::_prepare_input_ids(std::vector<int>& input_ids, int& actual_len, bool& is_doubled) {
    // 准备输入ID，对短输入进行复制处理
    is_doubled = false;
    int original_actual_len = actual_len;

    // printf("actual_len 3: %d\n", actual_len);
    if (actual_len <= DOUBLE_INPUT_THRESHOLD) {
        // printf("doubled!\n");
        is_doubled = true;
        // valid_content = input_ids[:, :actual_len]
        std::vector<int> valid_content(input_ids.begin(), input_ids.begin() + actual_len);
        // input_ids_doubled = np.concatenate([valid_content, valid_content], axis=1)
        std::vector<int> input_ids_doubled;
        // input_ids_doubled.reserve(2 * actual_len);
        input_ids_doubled.insert(input_ids_doubled.end(), valid_content.begin(), valid_content.end());
        input_ids_doubled.insert(input_ids_doubled.end(), valid_content.begin(), valid_content.end());
        
        // padding_len = self.max_seq_len_ - input_ids_doubled.shape[1]
        int padding_len = max_seq_len_ - 2 * actual_len;
        // printf("padding_len: %d\n", padding_len);
        if (padding_len > 0) {
            // input_ids = np.concatenate([input_ids_doubled, np.zeros((1, padding_len), dtype=input_ids.dtype)], axis=1)
            std::vector<int> padding(padding_len, 0);
            input_ids_doubled.insert(input_ids_doubled.end(), padding.begin(), padding.end());
        }
        else {
            // input_ids = input_ids_doubled[:, :self.max_seq_len_]
            input_ids_doubled.resize(max_seq_len_);
        }

        // save_file(input_ids_doubled, "input_ids2_1.bin");
            
        input_ids = input_ids_doubled;
        actual_len = std::min(original_actual_len * 2, max_seq_len_);
    }
}

void Kokoro::_compute_external_preprocessing(const std::vector<int>& input_ids, int actual_len, std::vector<int>& input_lengths, std::vector<uint8_t>& text_mask) {
    // 计算输入预处理：长度和mask
    // input_lengths = np.full((input_ids.shape[0],), actual_len, dtype=np.int64)
    input_lengths = std::vector<int>{actual_len};
    // text_mask = np.arange(self.max_seq_len_)[np.newaxis, :] >= input_lengths[:, np.newaxis]
    text_mask.resize(max_seq_len_);
    for (int i = 0; i < max_seq_len_; i++) {
        text_mask[i] = (i >= actual_len) ? 1 : 0;
    }
}

void Kokoro::_process_duration(const std::vector<float>& duration, int actual_len, float speed, std::vector<int>& pred_dur, int& total_frames) {
    // """处理duration预测，调整到固定帧数"""
    // duration_processed = 1.0 / (1.0 + np.exp(-duration))
    // duration_processed = duration_processed.sum(axis=-1) / speed
    // pred_dur_original = np.round(duration_processed).clip(min=1).astype(np.int64).squeeze()
    std::vector<int> pred_dur_original(actual_len, 0);
    std::vector<float> duration_processed = sigmoid(duration);
    for (int i = 0; i < actual_len; i++) {
        float sum = 0;

        // duration shape: [1, 96, 50]
        for (int n = 0; n < duration_shape_[2]; n++) {
            sum += duration_processed[i * duration_shape_[2] + n];
        }
        sum /= speed;

        pred_dur_original[i] = int(std::max(1.f, roundf(sum)));
    }

    // # 分离实际内容和padding
    // pred_dur_actual = pred_dur_original[:actual_len]
    // pred_dur_padding = np.zeros(self.max_seq_len_ - actual_len, dtype=np.int64)
    // pred_dur = np.concatenate([pred_dur_actual, pred_dur_padding])
    std::vector<int> pred_dur_padding(max_seq_len_ - actual_len, 0);
    pred_dur = pred_dur_original;
    pred_dur.insert(pred_dur.end(), pred_dur_padding.begin(), pred_dur_padding.end());

    
    // # 调整实际内容部分，只处理长度超出情况
    // fixed_total_frames = self.max_seq_len_ * 2
    // diff = fixed_total_frames - pred_dur[:actual_len].sum()
    
    // if diff < 0:
    //     # 减少帧数
    //     indices = np.argsort(pred_dur[:actual_len])[::-1]
    //     decreased = 0
    //     for idx in indices:
    //         if pred_dur[idx] > 1 and decreased < abs(diff):
    //             pred_dur[idx] -= 1
    //             decreased += 1
    //         if decreased >= abs(diff):
    //             break

    // 调整实际内容部分，只处理长度超出情况
    int fixed_total_frames = max_seq_len_ * 2;
    int actual_frames = std::accumulate(pred_dur.begin(), pred_dur.begin() + actual_len, 0);
    int diff = fixed_total_frames - actual_frames;

    if (diff < 0) {
        // 减少帧数
        auto indices = argsort(pred_dur, actual_len, true);
        int decreased = 0;
        for (auto idx : indices) {
            if (pred_dur[idx] > 1 && decreased < std::abs(diff)) {
                pred_dur[idx]--;
                decreased++;
            }
            if (decreased >= std::abs(diff))
                break;
        }
    }
    
    // # 将剩余帧数分配到padding部分
    // remaining_frames = fixed_total_frames - pred_dur[:actual_len].sum()
    // padding_len = self.max_seq_len_ - actual_len
    // if remaining_frames > 0 and padding_len > 0:
    //     frames_per_padding = remaining_frames // padding_len
    //     remainder = remaining_frames % padding_len
    //     pred_dur[actual_len:] = frames_per_padding
    //     if remainder > 0:
    //         pred_dur[actual_len:actual_len+remainder] += 1

    actual_frames = std::accumulate(pred_dur.begin(), pred_dur.begin() + actual_len, 0);
    int remaining_frames = fixed_total_frames - actual_frames;
    int padding_len = max_seq_len_ - actual_len;
    // printf("actual_len 4: %d\n ", actual_len);
    // printf("remaining_frames 4: %d\n", remaining_frames);
    // printf("padding_len 4: %d\n", padding_len);

    if (remaining_frames > 0 && padding_len > 0) {
        int frames_per_padding = remaining_frames / padding_len;
        int remainder = remaining_frames % padding_len;

        for (int i = actual_len; i < pred_dur.size(); i++)
            pred_dur[i] = frames_per_padding;

        if (remainder > 0) {
            for (int i = actual_len; i < actual_len + remainder; i++) 
                pred_dur[i] += 1;
        }
    }
    
    // total_frames = pred_dur.sum()
    total_frames = std::accumulate(pred_dur.begin(), pred_dur.end(), 0);
    // printf("total_frames: %d\n", total_frames);
}

std::vector<float> Kokoro::_create_alignment_matrix(const std::vector<int>& pred_dur, int total_frames) {
    // """创建对齐矩阵"""
    // indices = np.repeat(np.arange(self.max_seq_len_), pred_dur)
    // pred_aln_trg = np.zeros((self.max_seq_len_, total_frames), dtype=np.float32)
    // if len(indices) > 0:
    //     pred_aln_trg[indices, np.arange(total_frames)] = 1.0
    // return pred_aln_trg[np.newaxis, ...]

    std::vector<int> seq_range(max_seq_len_);
    std::iota(seq_range.begin(), seq_range.end(), 0);
    auto indices = np_repeat(seq_range, pred_dur);

    std::vector<float> pred_aln_trg(max_seq_len_ * total_frames);
    if (!indices.empty()) {
        int col = 0;
        for (auto i : indices) {
            pred_aln_trg[i * total_frames + col] = 1.0f;
            col++;
        }
    }

    return pred_aln_trg;
}

void Kokoro::_compute_har_onnx(std::vector<float>& F0_pred, std::vector<float>& har) {
    // Querying model inputs is possible but let's just assume one set for this translation or use a check.
    // For brevity, I'll use the older "tokens" set as default or try to match python logic if I can access names.
    int64_t input_shape[] = {F0_pred_shape_[0], F0_pred_shape_[1]};
    std::vector<const char*> input_names = {"F0_pred"};

    std::vector<Ort::Value> input_tensors;
    
    // Create tensors
    auto memory_info = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);
    
    input_tensors.push_back(Ort::Value::CreateTensor<float>(
        memory_info, F0_pred.data(), F0_pred.size(), input_shape, F0_pred_shape_.size()));

    // Check model output name usually
    // Or get it from session
    std::vector<const char*> output_names = {"har"};

    auto output_tensors = model4_.Run(
        Ort::RunOptions{nullptr},
        input_names.data(),
        input_tensors.data(),
        input_tensors.size(),
        output_names.data(),
        output_names.size()
    );

    auto& output_tensor = output_tensors.front();
            
    // 获取输出信息
    auto tensor_info = output_tensor.GetTensorTypeAndShapeInfo();
    size_t element_count = tensor_info.GetElementCount();
    auto output_shape = tensor_info.GetShape();
    
    float* output_data = output_tensor.GetTensorMutableData<float>();
    
    har.resize(element_count);
    std::memcpy(har.data(), output_data, element_count * sizeof(float));
}

void Kokoro::_postprocess_x_to_audio(std::vector<float>& x, std::vector<float>& audio) {
    // 将频谱转换为音频波形
    // spec_part = x[:, :self.N_FFT//2+1, :]
    // phase_part = x[:, self.N_FFT//2+1:, :]
    int half_n_fft = N_FFT / 2 + 1;
    int num_frames = x_shape_[2];
    std::vector<float> spec_part(half_n_fft * num_frames);
    std::vector<float> phase_part(half_n_fft * num_frames);
    std::vector<float> cos_part(half_n_fft * num_frames);
    spec_part.assign(x.begin(), x.begin() + half_n_fft * num_frames);
    phase_part.assign(x.begin() + half_n_fft * num_frames, x.end());
    
    // spec = np.exp(spec_part)
    // phase = np.sin(phase_part)
    
    // spec_torch = torch.from_numpy(spec).float()
    // phase_torch = torch.from_numpy(phase).float()
    // cos_part = torch.sqrt(1.0 - phase_torch.pow(2).clamp(0, 1))
    
    // real = spec_torch * cos_part
    // imag = spec_torch * phase_torch
    // complex_spec = torch.complex(real, imag)

    for (int i = 0; i < half_n_fft * num_frames; i++) {
        spec_part[i] = expf(spec_part[i]);
        phase_part[i] = sinf(phase_part[i]);
        cos_part[i] = sqrtf(1.f - std::max(0.f, std::min(powf(phase_part[i], 2), 1.0f)));
    }

    // save_file(spec_part, "spec.bin");
    // save_file(phase_part, "phase.bin");
    // save_file(cos_part, "cos_part.bin");

    FFT_RESULT complex_spec(half_n_fft, vector<complex<float>>(num_frames));
    for (int i = 0; i < half_n_fft; i++) {
        for (int n = 0; n < num_frames; n++) {
            float spec = spec_part[i * num_frames + n];

            float real_part = spec * cos_part[i * num_frames + n];
            float imag_part = spec * phase_part[i * num_frames + n];

            complex_spec[i][n] = std::complex<float>(real_part, imag_part);
        }
    }
    
    // save_fft_result(complex_spec, "complex_spec.bin");

    // audio = torch.istft(
    //     complex_spec, n_fft=self.N_FFT, hop_length=self.HOP_LENGTH,
    //     win_length=self.N_FFT, window=torch.hann_window(self.N_FFT),
    //     center=True, return_complex=False
    // )

    audio = librosa::Feature::istft(complex_spec, N_FFT, HOP_LENGTH, "hann", true, "reflect", false);
}

bool Kokoro::inference_single_chunk(
    std::vector<int>& input_ids,
    const std::vector<float>& ref_s,
    int actual_len,
    float speed,
    std::vector<float>& audio,
    int& actual_content_frames,
    int& total_frames
) {
    int ret = 0;
    // Prepare inputs
    bool is_doubled = false;
    // printf("actual_len 2: %d\n", actual_len);

    // save_file(input_ids, "input_ids.bin");

    int original_actual_len = actual_len;

    _prepare_input_ids(input_ids, actual_len, is_doubled);

    // save_file(input_ids, "input_ids2.bin");

    std::vector<int> input_lengths;
    std::vector<uint8_t> text_mask;
    _compute_external_preprocessing(input_ids, actual_len, input_lengths, text_mask);

    // save_file(input_ids, "input_ids3.bin");

    // outputs1 = self.session1.run(None, {'input_ids': input_ids.astype(np.int32), 'ref_s': ref_s, 'text_mask': text_mask.astype(np.uint8)})
    std::vector<void*> model1_inputs{(void*)input_ids.data(), (void*)ref_s.data(), (void*)text_mask.data()};
    std::vector<void*> model1_outputs{(void*)duration_.data(), (void*)d_.data()};

    // printf("run model 1\n");
    model1_.set_inputs(model1_inputs);
    ret = model1_.run();
    if (0 != ret) {
        ALOGE("Run model1 failed! ret=0x%x", ret);
        return false;
    }
    model1_.get_outputs(model1_outputs);

    // save_file(input_ids, "input_ids.bin");
    // save_file(ref_s, "ref_s.bin");
    // save_file(duration_, "duration.bin");

    // 处理duration并对齐
    std::vector<int> pred_dur;
    _process_duration(duration_, actual_len, speed, pred_dur, total_frames);
    auto pred_aln_trg = _create_alignment_matrix(pred_dur, total_frames);

    // save_file(pred_dur, "pred_dur.bin");
    // save_file(pred_aln_trg, "pred_aln_trg.bin");
    // save_file(d_, "d.bin");

    // Model2: 预测F0和ASR特征
    // d_transposed = np.transpose(d, (0, 2, 1))
    // en = d_transposed @ pred_aln_trg
    DynMat M_d = Eigen::Map<DynMat>(
        d_.data(), 
        d_shape_[1],  // 96
        d_shape_[2]   // 640
    );
    
    DynMat M_pred_aln_trg = Eigen::Map<DynMat>(
        pred_aln_trg.data(), 
        max_seq_len_,  // 96
        total_frames   // 192
    );

    DynMat M_en = M_d.transpose() * M_pred_aln_trg;
    std::vector<float> en(M_en.size());
    std::memcpy(en.data(), M_en.data(), M_en.size() * sizeof(float));

    // save_file(en, "en.bin");

    std::vector<float> text_mask_float;
    std::transform(text_mask.begin(), text_mask.end(),
                   std::back_inserter(text_mask_float),
                   [](uint8_t i) { return static_cast<float>(i); });

    // outputs2 = self.session2.run(None, {
    //     'en': en.astype(np.float32),
    //     'ref_s': ref_s,
    //     'input_ids': input_ids.astype(np.int32),
    //     'text_mask': text_mask.astype(np.float32),
    //     'pred_aln_trg': pred_aln_trg.astype(np.float32)
    // })
    // F0_pred, N_pred, asr = outputs2
    std::vector<void*> model2_inputs{
        (void*)en.data(), 
        (void*)ref_s.data(), 
        (void*)input_ids.data(),
        (void*)text_mask_float.data(), 
        (void*)pred_aln_trg.data()
    };
    std::vector<void*> model2_outputs{
        (void*)F0_pred_.data(),
        (void*)N_pred_.data(), 
        (void*)asr_.data()
    };

    // printf("run model 2\n");
    // printf("M_en.size(): %d\n", M_en.rows() * M_en.cols());
    // printf("ref_s.size(): %d\n", ref_s.size());
    // printf("input_ids.size(): %d\n", input_ids.size());
    // printf("text_mask_float.size(): %d\n", text_mask_float.size());
    // printf("pred_aln_trg.size(): %d\n", pred_aln_trg.size());
    model2_.set_inputs(model2_inputs);
    ret = model2_.run();
    if (0 != ret) {
        ALOGE("Run model2 failed! ret=0x%x", ret);
        return false;
    }
    model2_.get_outputs(model2_outputs);

    // save_file(F0_pred_, "F0_pred.bin");
    // save_file(N_pred_, "N_pred.bin");
    // save_file(asr_, "asr.bin");
    
    std::vector<float> har;
    _compute_har_onnx(F0_pred_, har);

    // har = load_file<float>("../har.bin");
    // save_file(har, "har.bin");

    // outputs3 = self.session3.run(None, {
    //     'asr': asr, 'F0_pred': F0_pred, 'N_pred': N_pred, 'ref_s': ref_s, 'har': har
    // })
    // x = outputs3[0]
    std::vector<void*> model3_inputs{
        (void*)asr_.data(), 
        (void*)F0_pred_.data(), 
        (void*)N_pred_.data(),
        (void*)ref_s.data(), 
        (void*)har.data()
    };

    // printf("run model 3\n");
    model3_.set_inputs(model3_inputs);
    ret = model3_.run();
    if (0 != ret) {
        ALOGE("Run model3 failed! ret=0x%x", ret);
        return false;
    }
    model3_.get_output(0, x_.data());

    // save_file(x_, "x.bin");

    // 转换为音频
    _postprocess_x_to_audio(x_, audio);
    
    if (is_doubled) {
        actual_content_frames = std::accumulate(pred_dur.begin(), pred_dur.begin() + original_actual_len, 0);   
        size_t audio_len = audio.size();
        audio.erase(audio.begin() + audio_len / 2, audio.end());
        total_frames = total_frames / 2;
    } else {
        actual_content_frames = std::accumulate(pred_dur.begin(), pred_dur.begin() + actual_len, 0);
    }

    return true;
}

bool Kokoro::inference(
    std::vector<int>& input_ids,
    const std::vector<float>& ref_s,
    float speed,
    float fade_out_duration,
    std::vector<float>& audio
) {
    int actual_len = input_ids.size();
    // 填充到固定长度
    int padding_len = max_seq_len_ - actual_len;
    if (padding_len > 0) {
        std::vector<int> padding(padding_len, 0);
        input_ids.insert(input_ids.end(), padding.begin(), padding.end());
    }

    int fade_samples = 0;
    if (fade_out_duration > 0) {
        fade_samples = int(SAMPLE_RATE * fade_out_duration);
    }

    // printf("actual_len 1: %d\n", actual_len);
    
    int actual_content_frames;
    int total_frames;
    if (!inference_single_chunk(input_ids, ref_s, actual_len, speed, audio, actual_content_frames, total_frames)) {
        return false;
    }

    _trim_audio_by_content(
        audio, actual_content_frames, total_frames, actual_len
    );

    if (fade_samples > 0)
        apply_fade_out(audio, fade_samples);

    return true;
}

bool Kokoro::run_batch_inference(std::vector<MergedGroup>& merged_group, const std::string& voice_name, 
                       float speed, float fade_out_duration, int sr,
                       std::vector<std::vector<float>>& audio_list
) {
    // 重新加载音色
    if (voice_name_ != voice_name) {
        if (!get_voice_style(voices_path_, voice_name)) {
            ALOGW("Load voice %s failed, fallback to original voice %s", voice_name.c_str(), voice_name_.c_str());
        }
    }
    
    // 批量推理
    for (auto& group : merged_group) {
        if (group.is_long_split) {
            // 长句分割：对每个子片段推理后拼接
            std::vector<float> combined_audio;
            for (auto& sub : group.sub_results) {
                int phoneme_len = sub.input_ids.size() - 2;
                auto ref_s = load_voice_embedding(phoneme_len);

                std::vector<float> audio;
                if (!inference(sub.input_ids, ref_s, speed, 0, audio)) {
                    return false;
                }
                combined_audio.insert(combined_audio.end(), audio.begin(), audio.end());
            }
            audio_list.push_back(combined_audio);
        } else {
            // 短句或合并句：直接推理
            // DEBUG
            // group.input_ids = load_file<int>("../input_ids.bin");
            // printf("group.input_ids.size() = %d\n", group.input_ids.size());
            int phoneme_len = group.input_ids.size() - 2;

             auto ref_s = load_voice_embedding(phoneme_len);
            std::vector<float> audio;
            if (!inference(group.input_ids, ref_s, speed, fade_out_duration, audio)) {
                return false;
            }
            audio_list.push_back(audio);
        }
    }

    return true;
}

void Kokoro::split_input_ids_semantic(std::vector<int>& input_ids, int max_seq_len_, int& actual_len) {
    // input_ids分割
    // content = input_ids[0, 1:-1]
    // chunk_with_special = np.concatenate([[0], content, [0]])
    actual_len = input_ids.size();

    // 填充到固定长度
    int padding_len = max_seq_len_ - actual_len;
    if (padding_len > 0) {
        std::vector<int> padding(padding_len, 0);
        input_ids.insert(input_ids.end(), padding.begin(), padding.end());
    }
}

void Kokoro::_trim_audio_by_content(std::vector<float>& audio, int actual_content_frames, int total_frames, int actual_len) {
    // 根据实际内容比例裁剪音频
    int padding_len = max_seq_len_ - actual_len;
    if (padding_len > 0) {
        float content_ratio = actual_content_frames * 1.0f / total_frames;
        int audio_len_to_keep = int(audio.size() * content_ratio);
        audio.resize(audio_len_to_keep);
    }
}

void Kokoro::apply_fade_out(std::vector<float>& audio, int fade_samples) {
    // 末尾淡出音频
    if (audio.size() <= fade_samples || fade_samples <= 0)
        return;

    std::vector<float> fade_out = linspace(1.0f, 0.0f, fade_samples);
    // audio_faded = audio.copy()
    // audio_faded[-fade_samples:] *= fade_out
    // return audio_faded
    for (int i = 0; i < fade_samples; i++) {
        audio[i - fade_samples + audio.size()] *= fade_out[i];
    }
}

bool Kokoro::tts(
    const std::string& text,
    const std::string& voice_name,
    float speed,
    int sample_rate,
    float fade_out,
    float pause_duration,
    std::vector<float>& generated_audio
) {
    // merged_groups = process_and_merge_sentences(
    //     args.text, args.lang, g2p, g2p_type, vocab, max_merge_len=args.max_len
    // )
    auto merged_groups = process_and_merge_sentences(text, lang_code_, max_seq_len_);

    // save_file(merged_groups[0].input_ids, "input_ids.bin");

    // audio_list = run_batch_inference(
    //     engine, merged_groups, args.voice, vocab, 
    //     speed=SPEED, fade_out_duration=args.fade_out
    // )
    std::vector<std::vector<float>> audio_list;
    if (!run_batch_inference(merged_groups, voice_name, speed, fade_out, sample_rate, audio_list)) {
        ALOGE("run_batch_inference failed!");
        return false;
    }

    generated_audio = audio_numpy_concat(audio_list, sample_rate, speed, pause_duration);

    return true;
}

void Kokoro::generate_input_ids_from_text(const std::string& text, std::vector<int>& input_ids, std::string& phonemes) {
    phonemes = tokenizer_->phonemize(text);
    input_ids = tokenizer_->tokenize(phonemes);
    input_ids.insert(input_ids.begin(), 0);
    input_ids.push_back(0);
}

// 迭代版本（避免递归栈溢出）
std::vector<SentenceInfo> Kokoro::split_long_sentence(
    const std::string& sentence,
    const std::string& lang_code,
    int max_merge_len
) {
    std::vector<SentenceInfo> result;
    
    // 使用栈来模拟递归
    struct Task {
        std::string sentence;
        int depth;
        
        Task(const std::string& s, int d) : sentence(s), depth(d) {}
    };
    
    std::vector<Task> stack;
    stack.emplace_back(sentence, 0);
    
    const int MAX_DEPTH = 20;
    
    while (!stack.empty()) {
        Task current = std::move(stack.back());
        stack.pop_back();
        
        if (current.depth > MAX_DEPTH) {
            continue; // 跳过超过最大深度的任务
        }
        
        try {
            std::vector<int> input_ids;
            std::string phonemes;
  
            generate_input_ids_from_text(current.sentence, input_ids, phonemes);
            
            // 计算内容长度
            int content_len = input_ids.size();
            
            if (content_len <= max_merge_len) {
                result.emplace_back(current.sentence, input_ids, phonemes, content_len);
                continue;
            }
            
            // 需要分割
            std::string first_half, second_half;
            
            if (lang_code == std::string("z") || lang_code == std::string("j")) {
                // 中文/日文分割
                size_t mid = current.sentence.length() / 2;
                first_half = current.sentence.substr(0, mid);
                second_half = current.sentence.substr(mid);
            } else {
                // 英文分割
                std::istringstream iss(current.sentence);
                std::vector<std::string> words;
                std::string word;
                
                while (iss >> word) {
                    words.push_back(word);
                }
                
                if (words.size() > 1) {
                    size_t mid_word = words.size() / 2;
                    
                    // 构建前半部分
                    std::ostringstream oss1;
                    for (size_t i = 0; i < mid_word; ++i) {
                        if (i > 0) oss1 << " ";
                        oss1 << words[i];
                    }
                    first_half = oss1.str();
                    
                    // 构建后半部分
                    std::ostringstream oss2;
                    for (size_t i = mid_word; i < words.size(); ++i) {
                        if (i > mid_word) oss2 << " ";
                        oss2 << words[i];
                    }
                    second_half = oss2.str();
                } else {
                    // 只有一个单词，按字符分割
                    size_t mid = current.sentence.length() / 2;
                    first_half = current.sentence.substr(0, mid);
                    second_half = current.sentence.substr(mid);
                }
            }
            
            // 将分割后的任务推入栈中（先处理后半部分，再处理前半部分）
            stack.emplace_back(second_half, current.depth + 1);
            stack.emplace_back(first_half, current.depth + 1);
            
        } catch (...) {
            // 异常处理
            continue;
        }
    }
    
    return result;
}

std::vector<MergedGroup> Kokoro::process_and_merge_sentences(
    const std::string& text,
    const std::string& lang_code,
    int max_merge_len
) {
    // 1. 清理文本
    std::string cleaned_text = clean_text(text);
    
    // 2. 分割句子
    std::vector<std::string> sentences = split_sentence(cleaned_text, lang_code);
    // printf("sentence num: %d\n", sentences.size());
    
    // 3. 为每个句子生成 input_ids
    std::vector<SentenceInfo> sentence_data;
    sentence_data.reserve(sentences.size());
    
    for (const auto& sentence : sentences) {
        try {
            std::vector<int> input_ids;
            std::string phonemes;

            // printf("sentence: %s\n", sentence.c_str());
            
            generate_input_ids_from_text(sentence, input_ids, phonemes);
            
            int content_len = static_cast<int>(input_ids.size());
            
            if (content_len <= max_merge_len) {
                // 短句
                sentence_data.emplace_back(sentence, input_ids, phonemes, content_len);
            } else {
                // 长句，需要分割
                std::vector<SentenceInfo> sub_results = split_long_sentence(
                    sentence, lang_code, max_merge_len);
                
                sentence_data.emplace_back(sentence, sub_results);
            }
            
        } catch (const std::exception& e) {
            std::cerr << "错误处理句子 '" << sentence << "': " << e.what() << std::endl;
        } catch (...) {
            std::cerr << "未知错误处理句子 '" << sentence << "'" << std::endl;
        }
    }
    
    // 4. 检查是否生成了任何数据
    if (sentence_data.empty()) {
        throw std::runtime_error("没有生成任何 input_ids");
    }
    
    // 5. 长句保持分割，短句合并
    std::vector<MergedGroup> merged_groups;
    merged_groups.reserve(sentence_data.size());  // 预分配
    
    size_t i = 0;
    while (i < sentence_data.size()) {
        const SentenceInfo& current = sentence_data[i];
        
        if (current.is_long) {
            // 长句：直接添加分割结果
            merged_groups.emplace_back(current.sub_results);
            ++i;
        } else {
            // 短句：尝试合并
            std::vector<std::string> merged_sentences;
            int total_len = 0;
            size_t j = i;
            
            // 尝试合并尽可能多的短句
            while (j < sentence_data.size() && !sentence_data[j].is_long) {
                int next_len = sentence_data[j].content_len;
                
                // 检查是否超过最大长度
                if (total_len + next_len <= max_merge_len) {
                    merged_sentences.push_back(sentence_data[j].sentence);
                    total_len += next_len;
                    ++j;
                } else {
                    break;
                }
            }
            
            // 如果第一个句子就超过长度，至少包含它
            if (j == i) {
                merged_sentences.push_back(sentence_data[i].sentence);
                ++j;
            }
            
            // 生成合并后的文本
            std::ostringstream oss;
            for (size_t k = 0; k < merged_sentences.size(); ++k) {
                if (k > 0) {
                    oss << " ";  // 用空格连接句子
                }
                oss << merged_sentences[k];
            }
            std::string merged_text = oss.str();
            
            // 重新生成合并后的 input_ids
            std::vector<int> merged_input_ids;
            std::string merged_phonemes;
            
            generate_input_ids_from_text(merged_text, merged_input_ids, merged_phonemes);
            
            merged_groups.emplace_back(merged_input_ids, merged_phonemes);
            
            i = j;  // 移动到下一组
        }
    }
    
    return merged_groups;    
}

std::vector<float> Kokoro::load_voice_embedding(int phoneme_len) {
    // if phoneme_len is not None and phoneme_len < pack.shape[0]:
    //     ref_s = pack[phoneme_len:phoneme_len+1]
    // else:
    //     idx = pack.shape[0] // 2
    //     ref_s = pack[idx:idx+1]
    std::vector<float> ref_s(STYLE_DIM);
    if (phoneme_len < voice_pack_size_) {
        ref_s.assign(voice_tensor_.begin() + phoneme_len * STYLE_DIM, voice_tensor_.begin() + (phoneme_len + 1) * STYLE_DIM);
    } else {
        int idx = voice_pack_size_ / 2;
        ref_s.assign(voice_tensor_.begin() + idx * STYLE_DIM, voice_tensor_.begin() + (idx + 1) * STYLE_DIM);
    }
    return ref_s;
}