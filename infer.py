#!/usr/bin/env python3
"""
DeepSeek-OCR-2 推理脚本
用法: python infer.py <image_path> [--output <output_dir>] [--mode ocr|markdown]
"""

import argparse
import os
import torch
from transformers import AutoModel, AutoTokenizer


def load_model(device: str = "cuda:0"):
    """加载 DeepSeek-OCR-2 模型"""
    model_name = "deepseek-ai/DeepSeek-OCR-2"

    print(f"加载模型: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModel.from_pretrained(
        model_name,
        _attn_implementation="eager",  # flash_attention_2 需要编译，eager 兼容性更好
        trust_remote_code=True,
        use_safetensors=True,
    )
    model = model.eval().to(device).to(torch.bfloat16)
    print("模型加载完成")

    return model, tokenizer


def infer(
    model,
    tokenizer,
    image_path: str,
    output_dir: str = "./output",
    mode: str = "markdown",
):
    """
    执行 OCR 推理

    Args:
        model: 加载的模型
        tokenizer: 分词器
        image_path: 图片路径
        output_dir: 输出目录
        mode: "markdown" 或 "ocr"
    """
    if mode == "markdown":
        prompt = "<image>\n<|grounding|>Convert the document to markdown. "
    else:
        prompt = "<image>\nFree OCR. "

    os.makedirs(output_dir, exist_ok=True)

    result = model.infer(
        tokenizer,
        prompt=prompt,
        image_file=image_path,
        output_path=output_dir,
        base_size=1024,
        image_size=768,
        crop_mode=True,
        save_results=True,
    )

    return result


def main():
    parser = argparse.ArgumentParser(description="DeepSeek-OCR-2 推理")
    parser.add_argument("image", help="输入图片路径")
    parser.add_argument("--output", "-o", default="./output", help="输出目录")
    parser.add_argument(
        "--mode", "-m",
        choices=["ocr", "markdown"],
        default="markdown",
        help="模式: ocr(纯文字) 或 markdown(保留格式)"
    )
    parser.add_argument("--device", "-d", default="cuda:0", help="设备")

    args = parser.parse_args()

    if not os.path.exists(args.image):
        print(f"错误: 图片不存在 {args.image}")
        return 1

    model, tokenizer = load_model(args.device)
    result = infer(model, tokenizer, args.image, args.output, args.mode)

    print("\n=== 识别结果 ===")
    print(result)

    return 0


if __name__ == "__main__":
    exit(main())
