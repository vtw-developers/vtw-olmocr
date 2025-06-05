FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04

WORKDIR /app

# 필수 패키지 설치
RUN apt-get update && apt-get install -y \
    python3 python3-pip git && \
    rm -rf /var/lib/apt/lists/*

# requirements.txt 복사 및 설치
COPY requirements.txt ./
RUN pip3 install --upgrade pip && pip3 install -r requirements.txt

# app.py 복사
COPY app.py ./

# 모델 경로 마운트 예정

EXPOSE 30001

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "30001"] 