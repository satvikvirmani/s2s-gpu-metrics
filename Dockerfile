FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

# ---------------- System deps ----------------
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    python3-dev \
    ffmpeg \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

# ---------------- Python setup ----------------
RUN ln -s /usr/bin/python3 /usr/bin/python
RUN python -m pip install --upgrade pip

# ---------------- App ----------------
WORKDIR /app

COPY requirements.txt .

# Install PyTorch GPU build explicitly
RUN pip install torch torchvision \
    --index-url https://download.pytorch.org/whl/cu121

RUN pip install --no-cache-dir -r requirements.txt

COPY main.py metrics.py ./

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]