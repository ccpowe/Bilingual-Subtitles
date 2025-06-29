# 音频转双语字幕工具

简洁高效的音频转字幕工具，支持音频转录和字幕翻译，一键生成双语字幕。

## 🔄 功能结构图

```mermaid
graph TD
    A["🎵 音频/视频文件<br/>(MP4, MP3, WAV, etc.)"] --> B["main.py<br/>主控制脚本"]
    
    B --> C{"选择模式"}
    C -->|full| D["完整工作流"]
    C -->|transcribe| E["仅转录"]
    C -->|translate| F["仅翻译"]
    
    D --> G["audio_to_srt.py<br/>音频转录模块"]
    E --> G
    F --> I["srt_translator_agent.py<br/>翻译模块"]
    
    G --> H["📝 SRT字幕文件<br/>(单语)"]
    H --> I
    I --> J["📝 双语SRT字幕<br/>(原文+译文)"]
    
    G -.-> K["🤖 Whisper模型<br/>(tiny/base/small/medium/large)"]
    I -.-> L["🧠 LLM模型<br/>(GPT-4o-mini/GPT-4/Gemini)"]
    
    M["⚙️ .env配置<br/>(API密钥/模型/代理)"] -.-> I
```

## 🚀 快速开始

### 1. 安装依赖
```bash
# 使用 uv（推荐）
uv add faster-whisper soundfile psutil langgraph langchain-openai

# 或使用 pip
pip install faster-whisper soundfile psutil langgraph langchain-openai
```

### 2. 环境配置
复制 `.env_example` 文件并配置你的 API 密钥：

```bash
cp .env_example .env
```

编辑 `.env` 文件：
```bash
OPENAI_API_KEY=你的API密钥
MODEL_NAME=gpt-4o-mini  # 或其他模型
MODEL_BASE_URL=https://your.proxy.com/v1  # 如果使用代理
```

### 3. 一键运行
```bash
# 完整流程：音频 → SRT字幕 → 双语字幕
uv run python main.py your_video.mp4 --mode full -l en -s 英文 -t 中文

# ⚠️ 长字幕提醒：如果字幕超过500条，建议使用 -b 15 或更大的批量大小
uv run python main.py long_video.mp4 --mode full -l en -s 英文 -t 中文 -b 15
```

## 📁 核心文件说明

### 1. `main.py` - 主控制脚本 🏆
**功能：** 统一的入口点，支持三种运行模式

**三种模式：**
- `full` - 完整工作流（音频 → SRT → 双语字幕）
- `transcribe` - 仅音频转录（音频 → SRT）
- `translate` - 仅字幕翻译（SRT → 双语字幕）

### 2. `audio_to_srt.py` - 音频转录模块
**功能：** 使用 Whisper 模型将音频转换为 SRT 字幕

### 3. `srt_translator_agent.py` - 翻译模块
**功能：** 基于 LangGraph 的智能翻译 Agent，将单语字幕转换为双语字幕

## 🔧 使用方法

### 方式1：使用主控制脚本（推荐）

#### 完整工作流
```bash
# 英文音频 → 中文双语字幕
uv run python main.py video.mp4 --mode full -l en -s 英文 -t 中文

# 中文音频 → 英文双语字幕  
uv run python main.py video.mp4 --mode full -l zh -s 中文 -t 英文
```

#### 仅音频转录
```bash
# 转录英文音频
uv run python main.py video.mp4 --mode transcribe -l en

# 转录中文音频
uv run python main.py video.mp4 --mode transcribe -l zh
```

#### 仅字幕翻译
```bash
# 翻译现有字幕文件
uv run python main.py existing.srt --mode translate -s 英文 -t 中文
```

### 方式2：单独运行各模块

#### 单独运行音频转录
```bash
# 基础转录
uv run python audio_to_srt.py video.mp4 -l en

# 指定模型和输出文件
uv run python audio_to_srt.py video.mp4 -l en -m base -o output.srt
```

#### 单独运行字幕翻译
```bash
# 基础翻译
uv run python srt_translator_agent.py input.srt -s 英文 -t 中文

# 指定批量大小和输出文件
uv run python srt_translator_agent.py input.srt -s 英文 -t 中文 -b 3 -o output.srt
```

## ⚙️ 参数说明

### main.py 参数
```bash
uv run python main.py INPUT [OPTIONS]

必需参数：
  INPUT                    输入文件（音频文件或SRT文件）

基础选项：
  --mode {full,transcribe,translate}  运行模式（默认：full）
  -l, --language {en,zh,ja,ko,fr,de}  音频语言（转录时需要）
  -s, --source-lang TEXT   源语言（翻译时需要，如：英文）
  -t, --target-lang TEXT   目标语言（翻译时需要，如：中文）

高级选项：
  -m, --whisper-model {tiny,base,small,medium,large-v3}  Whisper模型
  --llm-model TEXT         LLM模型（默认从环境变量读取）
  -b, --batch-size INT     翻译批量大小（默认：5）
  --compute-type {int8,float16,float32}  计算精度（默认：int8）
  --cpu-threads INT        CPU线程数（默认：4）

控制选项：
  -v, --verbose           详细输出
  -q, --quiet            安静模式
  --dry-run              验证配置但不执行
```

### audio_to_srt.py 参数
```bash
uv run python audio_to_srt.py INPUT [OPTIONS]

必需参数：
  INPUT                    音频文件路径

选项：
  -o, --output TEXT       输出SRT文件路径
  -m, --model TEXT        Whisper模型（默认：base）
  -l, --language TEXT     语言代码（如：en, zh）
  -d, --device TEXT       设备（默认：cpu）
  --compute-type TEXT     计算精度（默认：float32）
  --prompt TEXT           初始提示词
  --batch                 批量处理目录
```

### srt_translator_agent.py 参数
```bash
uv run python srt_translator_agent.py INPUT [OPTIONS]

必需参数：
  INPUT                    SRT文件路径

选项：
  -o, --output TEXT       输出文件路径
  -s, --source-lang TEXT  源语言（如：英文）
  -t, --target-lang TEXT  目标语言（如：中文）
  -m, --model TEXT        LLM模型
  -b, --batch-size INT    批量大小（默认：5）
  --api-key TEXT          API密钥
```

## 🎯 模型配置指南

### Whisper模型规模对比
| 模型 | 文件大小 | 内存需求 | 处理速度 | 转录质量 | 推荐场景 |
|------|----------|----------|----------|----------|----------|
| `tiny` | 39MB | ~1-2GB | 最快 | 基础 | 快速预览、低配置系统 |
| `base` | 74MB | ~2-3GB | 快速 | 良好 | **日常使用推荐** |
| `small` | 244MB | ~4-5GB | 中等 | 很好 | 高质量需求 |
| `medium` | 769MB | ~6-8GB | 较慢 | 优秀 | 专业用途 |
| `large-v3` | 1550MB | ~10-12GB | 慢 | 最好 | 最高质量需求 |

### 计算精度对比
| 精度 | 内存使用 | 质量影响 | 兼容性 | 推荐场景 |
|------|----------|----------|--------|----------|
| `float32` | 100% | 最好 | 标准 | 内存充足时 |
| `float16` | ~50% | 很好 | 需GPU支持 | GPU加速 |
| `int8` | ~25% | 良好 | **最佳兼容** | **集成显卡推荐** |

### 🚀 CUDA加速配置

#### 方法1：修改 audio_to_srt.py（单独运行时）
```bash
# 编辑 audio_to_srt.py，找到第35行左右的设备配置
# 将 device="cpu" 改为 device="cuda"
uv run python audio_to_srt.py video.mp4 -l en -d cuda --compute-type float16
```

#### 方法2：修改 main.py（推荐）
在 `main.py` 中找到 `transcribe_audio` 方法（约第100行），将：
```python
model = faster_whisper.WhisperModel(
    model_size_or_path=model_size,
    device="cpu",  # 改为 "cuda"
    compute_type=compute_type,
    cpu_threads=cpu_threads
)
```

**CUDA配置建议：**
- **GPU内存 >= 4GB**：使用 `device="cuda"` + `compute_type="float16"`
- **GPU内存 >= 8GB**：使用 `device="cuda"` + `compute_type="float32"`
- **GPU内存不足**：保持 `device="cpu"` + `compute_type="int8"`

#### 验证CUDA可用性
```python
# 在Python中测试
import torch
print(f"CUDA可用: {torch.cuda.is_available()}")
print(f"CUDA设备数: {torch.cuda.device_count()}")
if torch.cuda.is_available():
    print(f"GPU名称: {torch.cuda.get_device_name(0)}")
    print(f"GPU内存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f}GB")
```

### 性能优化建议

#### 低配置系统（内存 < 8GB）
```bash
# 使用最小模型和优化精度
uv run python main.py video.mp4 --mode full -l en -s 英文 -t 中文 \
  -m tiny --compute-type int8 --cpu-threads 2 -b 2
```

#### 中等配置系统（内存 8-16GB）
```bash
# 平衡质量和性能
uv run python main.py video.mp4 --mode full -l en -s 英文 -t 中文 \
  -m base --compute-type int8 --cpu-threads 4 -b 3
```

#### 高配置系统（内存 > 16GB + GPU）
```bash
# 最高质量配置（需要修改代码启用CUDA）
uv run python main.py video.mp4 --mode full -l en -s 英文 -t 中文 \
  -m large-v3 --compute-type float16 --cpu-threads 8 -b 5
```

#### 长字幕处理（> 500条字幕）
```bash
# 大批量配置，避免递归限制问题
uv run python main.py video.mp4 --mode full -l en -s 英文 -t 中文 \
  -m base --compute-type int8 -b 15

# 超长字幕配置（> 1000条）
uv run python main.py video.mp4 --mode full -l en -s 英文 -t 中文 \
  -m base --compute-type int8 -b 25
```

## 📊 输出文件

### 文件命名规则
- 转录输出：`原文件名_YYYYMMDD_HHMMSS.srt`（原始字幕）
- 翻译输出：`原文件名_translation_YYYYMMDD_HHMMSS.srt`（纯翻译字幕）
- 双语输出：`原文件名_bilingual_YYYYMMDD_HHMMSS.srt`（双语字幕）

### 输出目录
```
srt_file/                           # 自动创建的输出目录
├── video_20250629_172654.srt      # 原始字幕
├── video_translation_20250629_172720.srt  # 纯翻译字幕
└── video_bilingual_20250629_172720.srt    # 双语字幕
```

### 三种字幕格式说明

#### 原始字幕格式（转录输出）
```
1
00:00:01,000 --> 00:00:03,000
Hello, welcome to our presentation.

2
00:00:03,000 --> 00:00:06,000
Today we will discuss...
```

#### 纯翻译字幕格式（翻译输出）
```
1
00:00:01,000 --> 00:00:03,000
你好，欢迎观看我们的演示。

2
00:00:03,000 --> 00:00:06,000
今天我们将讨论...
```

#### 双语字幕格式（双语输出）
```
1
00:00:01,000 --> 00:00:03,000
Hello, welcome to our presentation.
你好，欢迎观看我们的演示。

2
00:00:03,000 --> 00:00:06,000
Today we will discuss...
今天我们将讨论...
```

## 🔍 常见使用场景

### 场景1：处理英文教学视频
```bash
# 一键生成中英双语字幕
uv run python main.py lecture.mp4 --mode full -l en -s 英文 -t 中文
```

### 场景2：处理中文音频
```bash
# 转录中文音频为字幕
uv run python main.py audio.mp3 --mode transcribe -l zh
```

### 场景3：翻译现有字幕
```bash
# 将现有英文字幕翻译为双语
uv run python main.py subtitles.srt --mode translate -s 英文 -t 中文
```

### 场景4：批量处理
```bash
# 批量处理多个文件
for file in *.mp4; do
    uv run python main.py "$file" --mode full -l en -s 英文 -t 中文
done
```

### 场景5：长字幕处理（重要）
```bash
# 中等长度字幕（500-1000条）
uv run python main.py long_lecture.mp4 --mode full -l en -s 英文 -t 中文 -b 15

# 超长字幕（>1000条）- 避免递归限制
uv run python main.py very_long_video.mp4 --mode full -l en -s 英文 -t 中文 -b 25

# 极长字幕处理 - 同时修改代码中的recursion_limit
uv run python main.py extremely_long.mp4 --mode full -l en -s 英文 -t 中文 -b 30
```

## ❓ 常见问题

### Q: 翻译失败，提示API密钥错误？
A: 检查 `.env` 文件中的 `OPENAI_API_KEY` 是否正确设置。

### Q: 转录速度很慢？
A: 尝试使用更小的模型（如 `-m tiny`）或调整计算精度（`--compute-type int8`）。

### Q: 翻译质量不满意？
A: 可以尝试：
- 使用更高级的模型（如 `--llm-model gpt-4`）
- 减少批量大小（`-b 2`）
- 检查源语言和目标语言设置

### Q: 翻译长字幕时出现递归限制错误？
A: **重要提醒**：当字幕条目很多时（如 > 500条），可能会超过LangGraph的递归限制。

**解决方案：**
1. **调大批量大小**：`-b 10` 或 `-b 20`（减少翻译批次数）
2. **手动调整代码**：在 `srt_translator_agent.py` 中修改递归限制
   ```python
   config={
       "recursion_limit": 2000,  # 从1000调高到2000或更高
       "configurable": {"thread_id": f"translation_{int(time.time())}"}
   }
   ```

**批次计算公式**：`总批次数 = 字幕条数 ÷ 批量大小`
- 例如：534条字幕 ÷ 5批量 = 107批次
- 建议：超过100批次时，使用 `-b 10` 或更大批量

### Q: 支持哪些音频格式？
A: 支持 MP3、WAV、FLAC、M4A、MP4、AVI、MOV、MKV 等常见格式。

### Q: 如何使用代理API？
A: 在 `.env` 文件中设置 `MODEL_BASE_URL` 为你的代理地址。

## 📋 环境要求

- Python 3.8+
- 建议内存：4GB+ （使用 `base` 模型）
- 支持 CPU 和 GPU 加速
- 网络连接（用于下载模型和API调用）

## 📂 项目结构

```
srt_translate/
├── main.py                 # 主控制脚本
├── audio_to_srt.py        # 音频转录模块
├── srt_translator_agent.py # 翻译模块
├── .env_example           # 环境变量示例
├── README.md              # 本文档
└── srt_file/              # 输出目录（自动创建）
```

## 🎯 设计理念

- **简洁易用**：三个核心文件，功能清晰
- **模块化**：可独立运行，也可组合使用
- **高效稳定**：针对小模型优化，降低硬件要求
- **用户友好**：详细的日志输出和错误提示