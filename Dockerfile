FROM nvidia/cuda:12.2.2-devel-ubuntu22.04

# Install system dependencies
RUN apt-get update && \
    apt-get install -y python3 python3-pip build-essential cmake

# Configure Python
RUN ln -s /usr/bin/python3 /usr/bin/python

# Install CUDA-compatible llama-cpp-python
COPY requirements.txt .
RUN CMAKE_ARGS="-DLLAMA_CUBLAS=on" FORCE_CMAKE=1 \
    python -m pip install --upgrade pip && \
    pip install -r requirements.txt

# Copy application files
COPY handler.py .
COPY start.sh /start.sh

# Environment variables
ENV MODEL_PATH=/runpod-volume/Midnight-Miqu-70B-v1.5.Q4_K_M.gguf
ENV N_CTX=4096
ENV N_THREADS=8
ENV N_GPU_LAYERS=83
ENV LLAMA_CUBLAS=1

# Set permissions
RUN chmod +x /start.sh

CMD ["/start.sh"]