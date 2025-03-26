# Use the official NVIDIA CUDA image
FROM nvidia/cuda:12.1.1-devel-ubuntu22.04

WORKDIR /app
ENV DEBIAN_FRONTEND=noninteractive

# Install Python3, Pip, Git, build tools, cmake AND Upgrade Pip
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 \
    python3-pip \
    git \
    build-essential \
    cmake \
 && python3 -m pip install --upgrade pip \
 && rm -rf /var/lib/apt/lists/*

# --- Set Environment Variables for CUDA ---
# Explicitly point to the CUDA installation within the base image
ENV CUDA_HOME=/usr/local/cuda
ENV PATH=/usr/local/cuda/bin:${PATH}
ENV LD_LIBRARY_PATH=/usr/local/cuda/lib64:${LD_LIBRARY_PATH}
# Also set CMAKE_ARGS for building from source (fallback)
ENV CMAKE_ARGS="-DLLAMA_CUBLAS=on"
ENV FORCE_CMAKE=1
# --- End Environment Variables ---

# Arguments for version and CUDA compatibility
ARG LLAMA_CPP_PYTHON_VERSION=0.2.79
ARG CUDA_VERSION=cu121

# --- Install llama-cpp-python ---
RUN python3 -m pip uninstall llama-cpp-python -y || echo "llama-cpp-python not previously installed"
RUN python3 -m pip install wheel
# Add --verbose to pip install for more detailed logs during this step
RUN python3 -m pip install llama-cpp-python==${LLAMA_CPP_PYTHON_VERSION} \
    --force-reinstall --no-cache-dir --verbose \
    --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/${CUDA_VERSION}

# --- Build-Time Check ---
RUN echo "Verifying llama-cpp-python install during build..." && \
    python3 -c "import llama_cpp; info = llama_cpp.llama_info(); print(info); assert info.get('ggml_build_cublas', False) or info.get('ggml_build_cuda', False), 'ERROR: Build-time check FAILED - llama-cpp-python installed without CUDA/cuBLAS support!'" || \
    (echo "Build-time check failed, exiting." && exit 1)

# --- Install RunPod SDK ---
RUN python3 -m pip install runpod --no-cache-dir

# --- Copy Handler Code ---
COPY handler.py .

# --- Set Default Command: Execute handler.py directly ---
CMD ["python3", "-u", "handler.py"]