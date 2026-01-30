# DeepSeek-OCR-2 部署

DeepSeek-OCR-2 是 DeepSeek 于 2026-01-27 发布的 3B 参数视觉语言模型，专注于文档理解和 OCR。

## 环境要求

- Python 3.12
- CUDA 11.8+ (x86_64) 或 CUDA 12.x (ARM)
- GPU 显存: 约 8GB+ (BF16)

## 安装

### 0. 安装 Miniconda（如果没有 conda）

```bash
# 查看 CPU 架构
uname -m

# x86_64 执行:
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh

# aarch64 (ARM, 如 DGX Spark) 执行:
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-aarch64.sh
bash Miniconda3-latest-Linux-aarch64.sh

# 安装完成后
source ~/.bashrc
```

### 1. 接受 conda 服务条款（新版 conda 要求）

```bash
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r
```

### 2. 创建 conda 环境

```bash
conda create -n deepseek-ocr2 python=3.12 -y
conda activate deepseek-ocr2

# 隔离用户目录的全局包，避免依赖冲突
export PYTHONNOUSERSITE=1
```

### 3. 安装 PyTorch

```bash
# x86_64 + CUDA 11.8:
pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 \
    --index-url https://download.pytorch.org/whl/cu118

# ARM (aarch64) + CUDA 12.4 (Hopper 及更早架构):
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

# ARM (aarch64) + Blackwell 架构 (如 DGX Spark GB10):
# Blackwell (sm_121) 太新，需要 PyTorch nightly + CUDA 12.8
pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu128
```

### 4. 安装依赖

```bash
pip install -r requirements.txt

# PDF 支持（可选）
pip install pdf2image
sudo apt install poppler-utils  # pdf2image 需要
```

### 5. （可选）安装 flash-attn 加速

> **注意**：flash-attn 需要 CUDA 版本与 PyTorch 编译版本匹配，ARM 架构可能编译困难。
> 不装也能跑，只是推理速度稍慢。

```bash
export CUDA_HOME=/usr/local/cuda
pip install flash-attn==2.7.3 --no-build-isolation
```

如果安装成功，修改 `infer.py` 中的 `_attn_implementation="eager"` 为 `"flash_attention_2"` 可获得加速。

## 使用

```bash
# 图片 OCR
python infer.py your_document.jpg --mode markdown

# PDF OCR（自动逐页转换）
python infer.py your_document.pdf --mode markdown

# 纯 OCR（只提取文字，不保留格式）
python infer.py your_document.jpg --mode ocr

# 指定输出目录
python infer.py your_document.pdf -o ./results
```

## 两种模式

| 模式 | 用途 | 提示词 |
|------|------|--------|
| markdown | 文档转换，保留表格/格式 | `<image>\n<\|grounding\|>Convert the document to markdown.` |
| ocr | 纯文字提取 | `<image>\nFree OCR.` |

## 参考

- GitHub: https://github.com/deepseek-ai/DeepSeek-OCR-2
- HuggingFace: https://huggingface.co/deepseek-ai/DeepSeek-OCR-2
