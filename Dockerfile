FROM nvidia/cuda:12.2.2-devel-ubuntu22.04

# System dependencies
RUN apt-get update && \
    apt-get install -y python3 python3-pip build-essential cmake ninja-build

# Set Python symlink
RUN ln -s /usr/bin/python3 /usr/bin/python

# Install pip first
RUN python -m pip install --upgrade pip

# Set build arguments before requirements installation
ENV CMAKE_ARGS="-DLLAMA_CUBLAS=ON -DCMAKE_CUDA_ARCHITECTURES=all-major"
ENV FORCE_CMAKE=1
ENV LLAMA_CUBLAS=1

# Copy requirements before installation for caching
COPY requirements.txt .
RUN pip install -r requirements.txt --verbose

# Application files
COPY handler.py .
COPY start.sh /start.sh

# Environment variables
ENV MODEL_PATH=/runpod-volume/Midnight-Miqu-70B-v1.5.Q4_K_M.gguf
ENV N_CTX=4096
ENV N_THREADS=8
ENV N_GPU_LAYERS=83
ENV GGML_CUBLAS=1

# Permissions
RUN chmod +x /start.sh

CMD ["/start.sh"]