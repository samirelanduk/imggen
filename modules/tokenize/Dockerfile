FROM python:3.11-slim
RUN apt-get update && apt-get install -y procps && rm -rf /var/lib/apt/lists/*
RUN pip install transformers==5.2.0