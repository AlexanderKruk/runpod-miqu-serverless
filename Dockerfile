FROM nvidia/cuda:12.2.2-devel-ubuntu22.04

RUN apt-get update && \
    apt-get install -y \
    python3 python3-pip \
    build-essential cmake ninja-build \
    libopenblas-dev clang

RUN ln -s /usr/bin/python3 /usr/bin/python && \
    python -m pip install --upgrade pip setuptools wheel

ENV PATH="/usr/local/cuda/bin:${PATH}"
ENV LD_LIBRARY_PATH="/usr/local/cuda/lib64:${LD_LIBRARY_PATH}"

COPY requirements.txt .
RUN pip install --no-cache-dir runpod>=0.9.0
RUN CMAKE_ARGS="-DLLAMA_CUBLAS=ON -DCMAKE_CUDA_ARCHITECTURES=86" \
    FORCE_CMAKE=1 \
    pip install --no-cache-dir llama-cpp-python[server]==0.2.23

COPY handler.py .
COPY start.sh /start.sh
RUN chmod +x /start.sh

ENV MODEL_PATH=/runpod-volume/Midnight-Miqu-70B-v1.5.Q4_K_M.gguf
ENV N_CTX=2048
ENV N_GPU_LAYERS=79
ENV N_BATCH=1024
ENV GPU_SPLIT=100
ENV MAIN_GPU=0
ENV GGML_CUBLAS=1

CMD ["/start.sh"]