FROM python:3.11-slim
RUN apt-get update && apt-get install -y procps && rm -rf /var/lib/apt/lists/*
RUN pip install packaging==26.0 \
    torch==2.10.0 --extra-index-url https://download.pytorch.org/whl/cpu
