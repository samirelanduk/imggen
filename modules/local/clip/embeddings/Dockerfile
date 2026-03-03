FROM python:3.11-slim
RUN apt-get update && apt-get install -y procps && rm -rf /var/lib/apt/lists/*
RUN pip install safetensors==0.7.0 numpy==2.4.2 packaging==26.0 \
    torch==2.10.0 --extra-index-url https://download.pytorch.org/whl/cpu