#!/usr/bin/env python3
"""
DeepSeek-OCR-2 推理脚本
用法: python infer.py <image_or_pdf_path> [--output <output_dir>] [--mode ocr|markdown]
"""

import argparse
import os
import tempfile
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


def pdf_to_images(pdf_path: str, output_dir: str):
    """将 PDF 转换为图片列表"""
    from pdf2image import convert_from_path

    images = convert_from_path(pdf_path, dpi=200)
    image_paths = []

    for i, image in enumerate(images):
        image_path = os.path.join(output_dir, f"page_{i+1:03d}.png")
        image.save(image_path, "PNG")
        image_paths.append(image_path)

    return image_paths


def infer_single(
    model,
    tokenizer,
    image_path: str,
    output_dir: str = "./output",
    mode: str = "markdown",
):
    """对单张图片执行 OCR 推理"""
    if mode == "markdown":
        prompt = "<image>\n<|grounding|>Convert the document to markdown. "
    else:
        prompt = "<image>\nFree OCR. "

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


def infer(
    model,
    tokenizer,
    file_path: str,
    output_dir: str = "./output",
    mode: str = "markdown",
):
    """执行 OCR 推理，支持图片和 PDF"""
    os.makedirs(output_dir, exist_ok=True)

    # 检查是否为 PDF
    if file_path.lower().endswith(".pdf"):
        print(f"检测到 PDF 文件，正在转换为图片...")
        image_paths = pdf_to_images(file_path, output_dir)
        print(f"共 {len(image_paths)} 页")

        results = []
        for i, image_path in enumerate(image_paths):
            print(f"处理第 {i+1}/{len(image_paths)} 页...")
            result = infer_single(model, tokenizer, image_path, output_dir, mode)
            results.append(f"## Page {i+1}\n\n{result}")

        return "\n\n---\n\n".join(results)
    else:
        return infer_single(model, tokenizer, file_path, output_dir, mode)


def main():
    parser = argparse.ArgumentParser(description="DeepSeek-OCR-2 推理")
    parser.add_argument("image", help="输入图片或 PDF 路径")
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
        print(f"错误: 文件不存在 {args.image}")
        return 1

    model, tokenizer = load_model(args.device)
    result = infer(model, tokenizer, args.image, args.output, args.mode)

    # 保存合并后的结果
    output_file = os.path.join(args.output, "full_result.md")
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(result)
    print(f"\n结果已保存到: {output_file}")

    print("\n=== 识别结果 ===")
    print(result)

    return 0


if __name__ == "__main__":
    exit(main())
