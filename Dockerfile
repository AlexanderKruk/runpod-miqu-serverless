FROM nvidia/cuda:12.2.2-devel-ubuntu22.04

# System dependencies with essential build tools
RUN apt-get update && \
    apt-get install -y \
    python3 python3-pip \
    build-essential cmake ninja-build \
    libopenblas-dev libclang-14-dev

# Configure Python and pip
RUN ln -s /usr/bin/python3 /usr/bin/python && \
    python -m pip install --upgrade pip setuptools wheel

# Install requirements with explicit CUDA configuration
COPY requirements.txt .
RUN CMAKE_ARGS="-DLLAMA_CUBLAS=ON -DCMAKE_CUDA_ARCHITECTURES=86" \
    FORCE_CMAKE=1 \
    pip install --no-cache-dir -v -r requirements.txt

# Application setup
COPY handler.py .
COPY start.sh /start.sh
RUN chmod +x /start.sh

# Environment configuration
ENV MODEL_PATH=/runpod-volume/Midnight-Miqu-70B-v1.5.Q4_K_M.gguf
ENV N_CTX=4096
ENV N_THREADS=8
ENV N_GPU_LAYERS=83
ENV GGML_CUBLAS=1

CMD ["/start.sh"]