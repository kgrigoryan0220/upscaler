# Real-ESRGAN API Server Dockerfile
FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV CUDA_VISIBLE_DEVICES=0

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    python3-venv \
    wget \
    curl \
    git \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .

# Install PyTorch with CUDA support first (largest package)
# Используем совместимые версии PyTorch и torchvision
# Устанавливаем numpy ПЕРЕД torch, так как torch может требовать его
RUN pip3 install --no-cache-dir --upgrade pip && \
    pip3 install --no-cache-dir numpy && \
    pip3 install --no-cache-dir torch==2.0.1 torchvision==0.15.2 --index-url https://download.pytorch.org/whl/cu118

# Install Python dependencies (install basicsr separately to avoid space issues)
RUN pip3 install --no-cache-dir opencv-python Pillow tqdm && \
    pip3 install --no-cache-dir basicsr>=1.4.2 && \
    pip3 install --no-cache-dir facexlib>=0.2.5 gfpgan>=1.3.5 && \
    pip3 install --no-cache-dir fastapi>=0.104.0 uvicorn[standard]>=0.24.0 python-multipart>=0.0.6

# Copy application files
COPY . .

# Install Real-ESRGAN package
RUN python3 setup.py develop

# Create weights directory
RUN mkdir -p weights

# Expose API port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Run the API server
# Use uvicorn with single worker for GPU (multiple workers compete for GPU memory)
CMD ["python3", "-m", "uvicorn", "api_server:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]

