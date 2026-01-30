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
    page_output_dir: str,
    mode: str = "markdown",
):
    """对单张图片执行 OCR 推理"""
    if mode == "markdown":
        prompt = "<image>\n<|grounding|>Convert the document to markdown. "
    else:
        prompt = "<image>\nFree OCR. "

    os.makedirs(page_output_dir, exist_ok=True)

    model.infer(
        tokenizer,
        prompt=prompt,
        image_file=image_path,
        output_path=page_output_dir,
        base_size=1024,
        image_size=768,
        crop_mode=True,
        save_results=True,
    )

    # 读取模型保存的结果
    result_file = os.path.join(page_output_dir, "result.mmd")
    if os.path.exists(result_file):
        with open(result_file, "r", encoding="utf-8") as f:
            return f.read()
    return ""


def infer(
    model,
    tokenizer,
    file_path: str,
    output_dir: str = "./output",
    mode: str = "markdown",
):
    """执行 OCR 推理，支持图片和 PDF"""
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, "full_result.md")

    # 检查是否为 PDF
    if file_path.lower().endswith(".pdf"):
        print(f"检测到 PDF 文件，正在转换为图片...")
        image_paths = pdf_to_images(file_path, output_dir)
        print(f"共 {len(image_paths)} 页")

        # 清空输出文件
        with open(output_file, "w", encoding="utf-8") as f:
            f.write("")

        results = []
        for i, image_path in enumerate(image_paths):
            print(f"处理第 {i+1}/{len(image_paths)} 页...")
            # 每页用独立子目录，避免 result.mmd 被覆盖
            page_output_dir = os.path.join(output_dir, f"page_{i+1:03d}")
            result = infer_single(model, tokenizer, image_path, page_output_dir, mode)
            page_content = f"## Page {i+1}\n\n{result}\n\n---\n\n"

            # 逐页追加到文件
            with open(output_file, "a", encoding="utf-8") as f:
                f.write(page_content)

            results.append(page_content)

        return "".join(results)
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

    output_file = os.path.join(args.output, "full_result.md")
    print(f"\n结果已保存到: {output_file}")

    print("\n=== 识别结果 ===")
    print(result)

    return 0


if __name__ == "__main__":
    exit(main())
