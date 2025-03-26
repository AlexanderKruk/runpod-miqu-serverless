FROM nvidia/cuda:12.2.2-devel-ubuntu22.04

# System dependencies
RUN apt-get update && \
    apt-get install -y \
    python3 python3-pip \
    build-essential cmake ninja-build \
    libopenblas-dev clang

# Configure Python environment
RUN ln -s /usr/bin/python3 /usr/bin/python && \
    python -m pip install --upgrade pip setuptools wheel

# Set CUDA paths
ENV PATH="/usr/local/cuda/bin:${PATH}"
ENV LD_LIBRARY_PATH="/usr/local/cuda/lib64:${LD_LIBRARY_PATH}"

# Install requirements in separate steps
COPY requirements.txt .

# First install runpod without build variables
RUN pip install --no-cache-dir runpod>=0.9.0

# Then install llama.cpp with CUDA flags
RUN CMAKE_ARGS="-DLLAMA_CUBLAS=ON -DCMAKE_CUDA_ARCHITECTURES=86" \
    FORCE_CMAKE=1 \
    pip install --no-cache-dir llama-cpp-python[server]==0.2.23

# Copy application files
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