#!/usr/bin/env python3
"""
DeepSeek-OCR-2 API 服务
启动: python server.py [--host 0.0.0.0] [--port 8000]
"""

import argparse
import base64
import io
import os
import tempfile
import uuid
from contextlib import asynccontextmanager

import torch
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from transformers import AutoModel, AutoTokenizer

# 全局模型实例
model = None
tokenizer = None


def load_model(device: str = "cuda:0"):
    """加载 DeepSeek-OCR-2 模型"""
    model_name = "deepseek-ai/DeepSeek-OCR-2"

    print(f"加载模型: {model_name}")
    tok = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    mdl = AutoModel.from_pretrained(
        model_name,
        _attn_implementation="eager",
        trust_remote_code=True,
        use_safetensors=True,
    )
    mdl = mdl.eval().to(device).to(torch.bfloat16)
    print("模型加载完成")

    return mdl, tok


@asynccontextmanager
async def lifespan(app: FastAPI):
    """启动时加载模型"""
    global model, tokenizer
    device = os.environ.get("CUDA_DEVICE", "cuda:0")
    model, tokenizer = load_model(device)
    yield


app = FastAPI(
    title="DeepSeek-OCR-2 API",
    description="OCR 服务，支持图片和 PDF",
    lifespan=lifespan,
)


def do_ocr(image_path: str, mode: str = "markdown") -> str:
    """执行 OCR 推理"""
    if mode == "markdown":
        prompt = "<image>\n<|grounding|>Convert the document to markdown. "
    else:
        prompt = "<image>\nFree OCR. "

    with tempfile.TemporaryDirectory() as tmp_dir:
        model.infer(
            tokenizer,
            prompt=prompt,
            image_file=image_path,
            output_path=tmp_dir,
            base_size=1024,
            image_size=768,
            crop_mode=True,
            save_results=True,
        )

        result_file = os.path.join(tmp_dir, "result.mmd")
        if os.path.exists(result_file):
            with open(result_file, "r", encoding="utf-8") as f:
                return f.read()
    return ""


def pdf_to_images(pdf_path: str, output_dir: str) -> list[str]:
    """将 PDF 转换为图片列表"""
    from pdf2image import convert_from_path

    images = convert_from_path(pdf_path, dpi=200)
    image_paths = []

    for i, image in enumerate(images):
        image_path = os.path.join(output_dir, f"page_{i+1:03d}.png")
        image.save(image_path, "PNG")
        image_paths.append(image_path)

    return image_paths


class OCRRequest(BaseModel):
    """Base64 方式提交的请求"""
    image_base64: str
    mode: str = "markdown"  # markdown 或 ocr
    filename: str = "image.png"  # 用于判断是否为 PDF


class OCRResponse(BaseModel):
    """OCR 响应"""
    text: str
    pages: int = 1


@app.post("/ocr", response_model=OCRResponse)
async def ocr_base64(request: OCRRequest):
    """
    Base64 方式提交图片/PDF 进行 OCR

    - image_base64: 图片或 PDF 的 base64 编码
    - mode: "markdown"(保留格式) 或 "ocr"(纯文字)
    - filename: 文件名，用于判断类型（以 .pdf 结尾则按 PDF 处理）
    """
    try:
        file_bytes = base64.b64decode(request.image_base64)
    except Exception:
        raise HTTPException(status_code=400, detail="无效的 base64 编码")

    with tempfile.TemporaryDirectory() as tmp_dir:
        # 保存文件
        ext = ".pdf" if request.filename.lower().endswith(".pdf") else ".png"
        file_path = os.path.join(tmp_dir, f"input{ext}")
        with open(file_path, "wb") as f:
            f.write(file_bytes)

        # PDF 处理
        if ext == ".pdf":
            image_paths = pdf_to_images(file_path, tmp_dir)
            results = []
            for i, img_path in enumerate(image_paths):
                result = do_ocr(img_path, request.mode)
                results.append(f"## Page {i+1}\n\n{result}")
            return OCRResponse(text="\n\n---\n\n".join(results), pages=len(image_paths))
        else:
            result = do_ocr(file_path, request.mode)
            return OCRResponse(text=result, pages=1)


@app.post("/ocr/upload", response_model=OCRResponse)
async def ocr_upload(
    file: UploadFile = File(...),
    mode: str = Form(default="markdown"),
):
    """
    文件上传方式进行 OCR

    - file: 图片或 PDF 文件
    - mode: "markdown"(保留格式) 或 "ocr"(纯文字)
    """
    with tempfile.TemporaryDirectory() as tmp_dir:
        # 保存上传文件
        ext = os.path.splitext(file.filename or "")[1].lower() or ".png"
        file_path = os.path.join(tmp_dir, f"input{ext}")
        content = await file.read()
        with open(file_path, "wb") as f:
            f.write(content)

        # PDF 处理
        if ext == ".pdf":
            image_paths = pdf_to_images(file_path, tmp_dir)
            results = []
            for i, img_path in enumerate(image_paths):
                result = do_ocr(img_path, mode)
                results.append(f"## Page {i+1}\n\n{result}")
            return OCRResponse(text="\n\n---\n\n".join(results), pages=len(image_paths))
        else:
            result = do_ocr(file_path, mode)
            return OCRResponse(text=result, pages=1)


@app.get("/health")
async def health():
    """健康检查"""
    return {"status": "ok", "model_loaded": model is not None}


def main():
    import uvicorn

    parser = argparse.ArgumentParser(description="DeepSeek-OCR-2 API 服务")
    parser.add_argument("--host", default="0.0.0.0", help="监听地址")
    parser.add_argument("--port", type=int, default=8000, help="监听端口")
    parser.add_argument("--device", default="cuda:0", help="CUDA 设备")

    args = parser.parse_args()

    os.environ["CUDA_DEVICE"] = args.device

    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
