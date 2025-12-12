#pragma once

#include "httplib.h"
#include "Kokoro.h"
#include "utils/logger.hpp"
#include "nlohmann/json.hpp"
#include <sstream>
#include <cstring>
#include <vector>
#include <algorithm>

// WAV文件头结构
#pragma pack(push, 1)
struct WavHeader {
    // RIFF块
    char riff[4] = {'R', 'I', 'F', 'F'};
    uint32_t file_size = 0;  // 文件总大小-8
    char wave[4] = {'W', 'A', 'V', 'E'};
    
    // fmt子块
    char fmt[4] = {'f', 'm', 't', ' '};
    uint32_t fmt_size = 16;          // fmt块大小
    uint16_t audio_format = 3;       // 3表示IEEE float格式，1表示PCM
    uint16_t num_channels = 1;       // 单声道
    uint32_t sample_rate = 24000;    // 采样率
    uint32_t byte_rate = 0;          // 每秒字节数 = sample_rate * num_channels * bits_per_sample/8
    uint16_t block_align = 0;        // 每个样本的字节数 = num_channels * bits_per_sample/8
    uint16_t bits_per_sample = 32;   // 32位浮点数
    
    // data子块
    char data[4] = {'d', 'a', 't', 'a'};
    uint32_t data_size = 0;          // 音频数据大小
};
#pragma pack(pop)

class KokoroHTTPServer {
private:
    httplib::Server m_srv;
    Kokoro m_model;
    std::string m_default_voice;
    std::string m_default_lang;

public:
    KokoroHTTPServer(const std::string& default_voice = "zf_xiaoxiao", 
                     const std::string& default_lang = "z")
        : m_default_voice(default_voice), m_default_lang(default_lang) {}

    ~KokoroHTTPServer() = default;

    bool init(const std::string& model_path, int max_seq_len = 96,
              const std::string& lang_code = "z",
              const std::string& voices_path = "./voices", 
              const std::string& voice_name = "zf_xiaoxiao",
              const std::string& vocab_path = "dict/vocab.txt") {
        
        ALOGI("Initializing Kokoro TTS model...");
        ALOGI("Model path: %s", model_path.c_str());
        ALOGI("Voices path: %s", voices_path.c_str());
        ALOGI("Default voice: %s", voice_name.c_str());
        
        if (!m_model.init(model_path, max_seq_len, lang_code, voices_path, voice_name, vocab_path)) {
            ALOGE("Kokoro init failed!");
            return false;
        }
        
        ALOGI("Kokoro model initialized successfully!");
        return true;
    }

    void start(int port = 8080) {
        _setup_routes();
        
        ALOGI("Starting TTS server at port %d", port);
        ALOGI("Endpoints:");
        ALOGI("  POST /tts          - Text-to-Speech (returns WAV audio)");
        ALOGI("  POST /tts_raw      - Text-to-Speech (returns raw PCM)");
        ALOGI("  GET  /health       - Health check");
        
        m_srv.listen("0.0.0.0", port);
    }

    void stop() {
        ALOGI("Stopping TTS server...");
        m_srv.stop();
    }

private:
    void _setup_routes() {
        // 主要端点：接收JSON文本，返回WAV音频
        m_srv.Post("/tts", [this](const httplib::Request& req, httplib::Response& res) {
            _handle_tts_request(req, res, true);  // true表示返回WAV
        });

        // 原始PCM端点
        m_srv.Post("/tts_raw", [this](const httplib::Request& req, httplib::Response& res) {
            _handle_tts_request(req, res, false); // false表示返回原始PCM
        });

        // 健康检查
        m_srv.Get("/health", [](const httplib::Request& req, httplib::Response& res) {
            _handle_health_request(req, res);
        });

        // 预检请求（CORS）
        m_srv.Options(R"(.*)", [](const httplib::Request& req, httplib::Response& res) {
            _handle_options_request(req, res);
        });

        // 错误处理
        m_srv.set_error_handler([](const httplib::Request& req, httplib::Response& res) {
            _handle_error(req, res);
        });
    }
    
    void _handle_tts_request(const httplib::Request& req, httplib::Response& res, bool return_wav) {
        // 设置CORS头
        setCORSHeaders(res);
        
        try {
            // 1. 检查Content-Type
            std::string content_type = req.has_header("Content-Type") ? 
                                       req.get_header_value("Content-Type") : "";
            
            if (content_type.find("application/json") == std::string::npos &&
                content_type.find("text/plain") == std::string::npos) {
                res.status = 400;
                _send_json_error(res, "Unsupported Content-Type. Use application/json or text/plain");
                return;
            }
            
            // 2. 解析请求
            nlohmann::json request_json;
            std::string text;
            std::string voice_name = m_default_voice;
            std::string lang_code = m_default_lang;
            float speed = 1.0f;
            int sample_rate = 24000;
            float fade_out = 0.05f;
            float pause_duration = 0.05f;
            
            if (content_type.find("application/json") != std::string::npos) {
                // JSON请求
                try {
                    request_json = nlohmann::json::parse(req.body);
                    
                    // 必需字段：sentence
                    if (!request_json.contains("sentence") || 
                        !request_json["sentence"].is_string() || 
                        request_json["sentence"].get<std::string>().empty()) {
                        res.status = 400;
                        _send_json_error(res, "Field 'sentence' is required and must be non-empty string");
                        return;
                    }
                    
                    text = request_json["sentence"].get<std::string>();
                    
                    // 可选字段
                    if (request_json.contains("voice_name") && request_json["voice_name"].is_string()) {
                        voice_name = request_json["voice_name"].get<std::string>();
                    }
                    
                    if (request_json.contains("lang_code") && request_json["lang_code"].is_string()) {
                        lang_code = request_json["lang_code"].get<std::string>();
                    }
                    
                    if (request_json.contains("speed") && request_json["speed"].is_number()) {
                        speed = request_json["speed"].get<float>();
                    }
                    
                    if (request_json.contains("sample_rate") && request_json["sample_rate"].is_number_integer()) {
                        sample_rate = request_json["sample_rate"].get<int>();
                        if (sample_rate != 24000 && sample_rate != 22050 && sample_rate != 16000) {
                            ALOGW("Unsupported sample rate %d, using 24000", sample_rate);
                            sample_rate = 24000;
                        }
                    }
                    
                    if (request_json.contains("fade_out") && request_json["fade_out"].is_number()) {
                        fade_out = request_json["fade_out"].get<float>();
                    }
                    
                    if (request_json.contains("pause_duration") && request_json["pause_duration"].is_number()) {
                        pause_duration = request_json["pause_duration"].get<float>();
                    }
                    
                } catch (const nlohmann::json::exception& e) {
                    res.status = 400;
                    _send_json_error(res, std::string("Invalid JSON: ") + e.what());
                    return;
                }
            } else {
                // 纯文本请求
                text = req.body;
                if (text.empty()) {
                    res.status = 400;
                    _send_json_error(res, "Text cannot be empty");
                    return;
                }
            }
            
            // 3. 验证参数
            if (text.length() > 5000) {  // 限制文本长度
                res.status = 400;
                _send_json_error(res, "Text too long (max 5000 characters)");
                return;
            }
            
            // ALOGI("TTS Request: text='%.50s%s', voice=%s, lang=%s, speed=%.2f, format=%s", 
            //       text.c_str(), text.length() > 50 ? "..." : "", 
            //       voice_name.c_str(), lang_code.c_str(), speed,
            //       return_wav ? "WAV" : "PCM");
            
            // 4. 运行TTS
            std::vector<float> audio_data;
            auto start_time = std::chrono::steady_clock::now();
            
            bool success = m_model.tts(text, voice_name, speed, 
                                      sample_rate, fade_out, pause_duration, audio_data);
            
            auto end_time = std::chrono::steady_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
            
            if (!success) {
                ALOGE("TTS failed for text: %.50s...", text.c_str());
                res.status = 500;
                _send_json_error(res, "TTS synthesis failed");
                return;
            }
            
            ALOGI("TTS Success: generated %zu samples (%.2f seconds), took %lld ms", 
                  audio_data.size(), audio_data.size() / (float)sample_rate, duration.count());
            
            // 5. 构建响应
            if (return_wav) {
                // 返回WAV格式
                std::string wav_data = _float_to_wav(audio_data, sample_rate);
                
                res.status = 200;
                res.set_header("Content-Type", "audio/wav");
                res.set_header("Content-Disposition", "attachment; filename=\"tts_output.wav\"");
                res.set_header("X-Audio-Samples", std::to_string(audio_data.size()));
                res.set_header("X-Audio-Sample-Rate", std::to_string(sample_rate));
                res.set_header("X-Audio-Duration", std::to_string(audio_data.size() / (float)sample_rate));
                res.set_header("X-Processing-Time", std::to_string(duration.count()));
                
                res.body = wav_data;
            } else {
                // 返回原始PCM格式
                res.status = 200;
                res.set_header("Content-Type", "audio/pcm");
                res.set_header("Content-Disposition", "attachment; filename=\"tts_output.pcm\"");
                res.set_header("X-Audio-Samples", std::to_string(audio_data.size()));
                res.set_header("X-Audio-Sample-Rate", std::to_string(sample_rate));
                res.set_header("X-Audio-Duration", std::to_string(audio_data.size() / (float)sample_rate));
                res.set_header("X-Processing-Time", std::to_string(duration.count()));
                
                // 将float数组转换为字节流
                res.body.resize(audio_data.size() * sizeof(float));
                std::memcpy(res.body.data(), audio_data.data(), res.body.size());
            }
            
        } catch (const std::exception& e) {
            ALOGE("Exception in TTS request: %s", e.what());
            res.status = 500;
            _send_json_error(res, std::string("Internal server error: ") + e.what());
        }
    }
    
    // 将float音频数据转换为WAV格式
    std::string _float_to_wav(const std::vector<float>& audio_data, int sample_rate) {
        // 创建WAV头
        WavHeader header;
        header.sample_rate = sample_rate;
        header.byte_rate = sample_rate * sizeof(float);  // 32位浮点数，单声道
        header.block_align = sizeof(float);
        header.data_size = audio_data.size() * sizeof(float);
        header.file_size = 36 + header.data_size;  // 36 = 总头部大小
        
        // 构建完整的WAV数据
        std::string wav_data;
        wav_data.reserve(sizeof(WavHeader) + header.data_size);
        
        // 写入WAV头
        wav_data.append(reinterpret_cast<const char*>(&header), sizeof(header));
        
        // 写入音频数据
        wav_data.append(reinterpret_cast<const char*>(audio_data.data()), header.data_size);
        
        return wav_data;
    }
    
    // 可选的：转换为16位PCM的WAV（更通用的格式）
    std::string _float_to_wav_pcm16(const std::vector<float>& audio_data, int sample_rate) {
        // 创建WAV头（16位PCM）
        WavHeader header;
        header.audio_format = 1;  // PCM格式
        header.bits_per_sample = 16;
        header.sample_rate = sample_rate;
        header.byte_rate = sample_rate * 2;  // 16位，单声道
        header.block_align = 2;
        
        // 将float转换为16位PCM
        std::vector<int16_t> pcm_data;
        pcm_data.reserve(audio_data.size());
        
        for (float sample : audio_data) {
            // 将float [-1.0, 1.0] 转换为int16_t [-32768, 32767]
            int16_t pcm_sample = static_cast<int16_t>(std::clamp(sample, -1.0f, 1.0f) * 32767.0f);
            pcm_data.push_back(pcm_sample);
        }
        
        header.data_size = pcm_data.size() * sizeof(int16_t);
        header.file_size = 36 + header.data_size;
        
        // 构建WAV数据
        std::string wav_data;
        wav_data.reserve(sizeof(WavHeader) + header.data_size);
        
        // 写入WAV头
        wav_data.append(reinterpret_cast<const char*>(&header), sizeof(header));
        
        // 写入音频数据
        wav_data.append(reinterpret_cast<const char*>(pcm_data.data()), header.data_size);
        
        return wav_data;
    }
    
    static void _handle_health_request(const httplib::Request& req, httplib::Response& res) {
        setCORSHeaders(res);
        
        nlohmann::json response;
        response["status"] = "healthy";
        response["service"] = "Kokoro TTS";
        response["timestamp"] = std::chrono::system_clock::to_time_t(std::chrono::system_clock::now());
        response["endpoints"] = {
            {"POST /tts", "Text-to-Speech (returns WAV)"},
            {"GET /health", "Health check"}
        };
        
        res.set_content(response.dump(2), "application/json");
    }
    
    static void _handle_options_request(const httplib::Request& req, httplib::Response& res) {
        setCORSHeaders(res);
        res.status = 200;
    }
    
    static void _handle_error(const httplib::Request& req, httplib::Response& res) {
        setCORSHeaders(res);
        
        nlohmann::json error;
        error["error"] = "Not Found";
        error["message"] = "The requested endpoint was not found";
        error["path"] = req.path;
        
        res.status = 404;
        res.set_content(error.dump(2), "application/json");
    }
    
    // 设置CORS头
    static void setCORSHeaders(httplib::Response& res) {
        res.set_header("Access-Control-Allow-Origin", "*");
        res.set_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS");
        res.set_header("Access-Control-Allow-Headers", "Content-Type, Authorization");
        res.set_header("Access-Control-Max-Age", "86400");
    }
    
    // 发送JSON格式的错误响应
    static void _send_json_error(httplib::Response& res, const std::string& message, int status = 400) {
        nlohmann::json error;
        error["error"] = true;
        error["message"] = message;
        error["status"] = status;
        
        res.status = status;
        res.set_content(error.dump(2), "application/json");
    }
};