FROM python:3.11-slim

# System dependencies:
#   libgl1 + libglib2.0-0  — required by opencv-python-headless at runtime
#   libimage-exiftool-perl — metadata scrubbing (exiftool)
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 \
        libglib2.0-0 \
        libimage-exiftool-perl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies.
# torch is CPU-only — the default PyPI wheel includes CUDA and is ~800 MB larger.
COPY requirements.txt .
RUN pip install --no-cache-dir \
        --extra-index-url https://download.pytorch.org/whl/cpu \
        -r requirements.txt

# Copy source.
COPY . .

# Pre-download GFPGAN weights into the image so users don't wait on first run.
RUN python scripts/download_models.py

# Pre-download InsightFace buffalo_l detection weights into the image.
# This avoids a ~300 MB download on first container start.
RUN python - <<'EOF'
from insightface.app import FaceAnalysis
app = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"])
app.prepare(ctx_id=0, det_size=(640, 640))
EOF

# models/ is mounted at runtime (contains inswapper_128.onnx).
# input/ and output/ are mounted for image I/O.
VOLUME ["/app/models", "/app/input", "/app/output"]

EXPOSE 7860

CMD ["python", "ui/app.py"]
