FROM python:3.11-slim
RUN apt-get update && apt-get install -y procps && rm -rf /var/lib/apt/lists/*
RUN pip install safetensors==0.7.0 torch==2.10.0 packaging
