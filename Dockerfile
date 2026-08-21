# Wan2.2 - Video Generation
FROM nvidia/cuda:12.1.0-runtime-ubuntu22.04

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3-pip \
    git \
    wget \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt . 2>/dev/null || echo "requirements.txt not found"

# Install PyTorch
RUN pip install --no-cache-dir torch torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/cu121

# Install other dependencies
RUN pip install --no-cache-dir gradio transformers diffusers Pillow numpy pydantic \
    requests huggingface-hub tqdm opencv-python einops omegaconf

# Copy application files
COPY *.py ./
COPY *.md ./

# Create directories
RUN mkdir -p ./models ./input_images ./output_videos

# Expose port
EXPOSE 7860

# Default command: start Gradio app
CMD ["python3", "gradio_app_enhanced.py"]
