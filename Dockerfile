FROM runpod/base:0.4.0-cuda11.8.0

# Install system dependencies
RUN apt-get update && \
    apt-get install -y build-essential cmake

# Install Python dependencies
COPY requirements.txt .
RUN pip install -r requirements.txt

# Copy application files
COPY handler.py .
COPY start.sh /start.sh

# Set environment variables
ENV MODEL_PATH=/runpod-volume/Midnight-Miqu-70B-v1.5.Q4_K_M.gguf
ENV N_CTX 4096
ENV N_THREADS 8
ENV N_GPU_LAYERS 83

# Set permissions
RUN chmod +x /start.sh

CMD ["/start.sh"]