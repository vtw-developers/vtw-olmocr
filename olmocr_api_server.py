from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
import tempfile
import shutil
import asyncio
import uvicorn
import os
from pypdf import PdfReader  # PDF 페이지 수 확인용

# olmocr 파이프라인 import
from olmocr.pipeline import process_page, PageResult

app = FastAPI()

# args 기본값 예시 (실제 환경에 맞게 조정 필요)
class Args:
    max_page_retries = 16
    target_longest_image_dim = 2048
    target_anchor_text_len = 12000
    model_max_context = 32768
    apply_filter = False
    max_page_error_rate = 0.2
    workers = 16
    model = os.environ.get("OLMOCR_MODEL", "allenai/olmOCR-7B-0225-preview")
    sglang_server_url = os.environ.get("SGLANG_SERVER_URL", "http://sglang:30000")

@app.post("/ocr/pdf")
async def ocr_pdf(file: UploadFile = File(...), page_num: int = 1):
    # 업로드된 파일을 임시 파일로 저장
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = tmp.name

    args = Args()
    worker_id = 0
    # process_page에서 sglang 서버 주소를 동적으로 사용하도록 monkey patch
    import olmocr.pipeline as pipeline
    # SGLANG_SERVER_URL에서 host와 port를 모두 추출하여 pipeline에 할당
    sglang_url = args.sglang_server_url
    if sglang_url.startswith("http://"):
        sglang_url = sglang_url[len("http://"):]
    elif sglang_url.startswith("https://"):
        sglang_url = sglang_url[len("https://"):]
    host, port = sglang_url.split(":")
    pipeline.SGLANG_SERVER_HOST = host
    pipeline.SGLANG_SERVER_PORT = int(port)
    try:
        # olmocr의 process_page 호출 (첫 페이지만 예시)
        page_result: PageResult = await process_page(args, worker_id, tmp_path, tmp_path, page_num)
        if page_result and page_result.response:
            return JSONResponse({
                "text": page_result.response.natural_text,
                "page_num": page_num,
                "is_fallback": page_result.is_fallback
            })
        else:
            raise HTTPException(status_code=500, detail="No response from OCR pipeline.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        os.remove(tmp_path)

@app.post("/ocr/pdf/all")
async def ocr_pdf_all(file: UploadFile = File(...)):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = tmp.name

    args = Args()
    worker_id = 0
    import olmocr.pipeline as pipeline
    sglang_url = args.sglang_server_url
    if sglang_url.startswith("http://"):
        sglang_url = sglang_url[len("http://"):]
    elif sglang_url.startswith("https://"):
        sglang_url = sglang_url[len("https://"):]
    host, port = sglang_url.split(":")
    pipeline.SGLANG_SERVER_HOST = host
    pipeline.SGLANG_SERVER_PORT = int(port)

    try:
        # PDF 전체 페이지 수 확인
        reader = PdfReader(tmp_path)
        num_pages = len(reader.pages)
        results = []
        for page_num in range(1, num_pages + 1):
            page_result: PageResult = await process_page(args, worker_id, tmp_path, tmp_path, page_num)
            if page_result and page_result.response:
                results.append({
                    "text": page_result.response.natural_text,
                    "page_num": page_num,
                    "is_fallback": page_result.is_fallback
                })
            else:
                results.append({
                    "text": None,
                    "page_num": page_num,
                    "is_fallback": True
                })
        return JSONResponse({"results": results})
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        os.remove(tmp_path)

if __name__ == "__main__":
    uvicorn.run("olmocr_api_server:app", host="0.0.0.0", port=8000) 