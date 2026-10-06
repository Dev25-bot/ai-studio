FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 HF_HOME=/cache/hf GRADIO_SERVER_NAME=0.0.0.0
WORKDIR /app

# CPU-only torch keeps the image small; use an nvidia/cuda base + default torch wheel for GPUs.
COPY requirements.txt .
RUN pip install --index-url https://download.pytorch.org/whl/cpu torch \
 && pip install -r requirements.txt

COPY . .
EXPOSE 7860
VOLUME ["/cache/hf", "/app/outputs"]
CMD ["python", "app.py", "--host", "0.0.0.0"]
