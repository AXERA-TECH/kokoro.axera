# export.py 使用说明  

将 Kokoro 语音合成模型分割导出为多个ONNX子模型。

模型下载后放到checkpoints/文件夹下  
[kokoro-v1.0.pth](https://github.com/AXERA-TECH/kokoro.axera/releases/download/v1.0.0/kokoro-v1_0.pth)  

## 依赖环境  
```bash
创建虚拟环境 
conda create -n kokoro_export python=3.10
conda activate kokoro_export

安装依赖库
pip install -r requirements.txt
```

## 参数说明

| 参数名称           | 说明                         |
|-------------------|------------------------------|
| `--config_file`/`-c`   | 配置文件路径，默认 checkpoints/config.json |
| `--checkpoint_path`/`-p` | 模型权重路径，默认 checkpoints/kokoro-v1_0.pth |
| `--output_dir`/`-o`     | 导出 ONNX 文件保存目录，默认 onnx |
| `--use_real_sample`     | 是否使用真实样本（否则随机生成）|
| `--lang_code`/`-l`      | 语言，默认 'a'           |
| `--input_length`        | 指定输入长度，默认96         |
| `--text`                | 指定输入文本，输入长度对应文本音素长度       |
| `--voice`               | 指定音色，在checkpoints/voices文件夹下选择一种对应的语言(a开头英文，z开头中文)    |

## 使用方法

1. 模型配置和权重文件（默认在 checkpoints/ 目录下）。
2. 运行脚本导出 ONNX 子模型：

指定
```bash
python export.py -o onnx -l a --input_length 96 --voice checkpoints/voices/af_heart.pt
```

如需使用真实样本：

```bash
python export.py --use_real_sample --text "The sky above the port was the color of television, tuned to a dead channel." -l a -o onnx --voice checkpoints/voices/af_heart.pt
```

## onnx导出

- 导出的 ONNX 文件将保存在指定 output_dir 目录下。
- zip文件可作为量化数据，带sim后缀的onnx模型用于量化和推理

## 模型转换（onnx->axera）
```
model1,model2，model3 3个模型需要进行转换
以modle1为例：
pulsar2 build --config config/kokoro_1.json
```