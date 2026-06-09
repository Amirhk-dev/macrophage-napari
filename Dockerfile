FROM python:3.11-slim

WORKDIR /app

# No napari / Qt — only what the API needs
RUN pip install --no-cache-dir \
    "fastapi[standard]" \
    onnxruntime \
    numpy \
    tifffile \
    python-multipart

COPY napari_macrophage/models/best.onnx models/best.onnx
COPY api/ api/

ENV MODEL_PATH=/app/models/best.onnx

EXPOSE 8000

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
