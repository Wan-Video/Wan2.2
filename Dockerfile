# Use a PyTorch image that includes CUDA and development tools (for nvcc)
FROM pytorch/pytorch:2.2.2-cuda11.8-cudnn8-devel

# Set environment variables for CUDA
ENV CUDA_HOME=/usr/local/cuda
ENV PATH=$CUDA_HOME/bin:$PATH
ENV LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    build-essential \
    ninja-build \
    libaio-dev \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Clone repository
RUN git clone https://github.com/Wan-Video/Wan2.2.git .

# Upgrade pip and setuptools to latest
RUN pip install --upgrade pip setuptools wheel

# Install requirements except flash_attn
RUN grep -v "flash_attn" requirements.txt > requirements_temp.txt && \
    pip install --no-cache-dir -r requirements_temp.txt

RUN pip install --upgrade typing-extensions
# Install flash-attn (MUST be after CUDA is set)
RUN pip install flash-attn --no-build-isolation

# Install decord and librosa
RUN pip install decord librosa

# Set work directory (optional)
WORKDIR /workspace

# Default to interactive shell
CMD ["/bin/bash"]
