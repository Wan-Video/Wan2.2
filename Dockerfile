# --- Base image (pre-built ComfyUI + CUDA + Torch) ---
FROM wlsdml1114/multitalk-base:1.7 AS runtime

# --- Install required utilities ---
RUN pip install -U "huggingface_hub[hf_transfer]" runpod websocket-client onnxruntime-gpu==1.22

# --- Working directory ---
WORKDIR /

# --- Clone ComfyUI core and essential nodes ---
RUN git clone --depth=1 https://github.com/comfyanonymous/ComfyUI.git && \
    cd /ComfyUI && pip install -r requirements.txt && \
    rm -rf .git

RUN cd /ComfyUI/custom_nodes && \
    git clone --depth=1 https://github.com/Comfy-Org/ComfyUI-Manager.git && \
    cd ComfyUI-Manager && pip install -r requirements.txt && \
    cd .. && rm -rf ComfyUI-Manager/.git

RUN cd /ComfyUI/custom_nodes && \
    git clone --depth=1 https://github.com/kijai/ComfyUI-WanVideoWrapper && \
    cd ComfyUI-WanVideoWrapper && pip install -r requirements.txt && \
    cd .. && rm -rf ComfyUI-WanVideoWrapper/.git

RUN cd /ComfyUI/custom_nodes && \
    git clone --depth=1 https://github.com/kijai/ComfyUI-KJNodes && \
    cd ComfyUI-KJNodes && pip install -r requirements.txt && \
    cd .. && rm -rf ComfyUI-KJNodes/.git

RUN cd /ComfyUI/custom_nodes && \
    git clone --depth=1 https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite && \
    cd ComfyUI-VideoHelperSuite && pip install -r requirements.txt && \
    cd .. && rm -rf ComfyUI-VideoHelperSuite/.git

RUN cd /ComfyUI/custom_nodes && \
    git clone --depth=1 https://github.com/kijai/ComfyUI-WanAnimatePreprocess && \
    cd ComfyUI-WanAnimatePreprocess && pip install -r requirements.txt && \
    cd .. && rm -rf ComfyUI-WanAnimatePreprocess/.git

# --- Extra helper nodes ---
RUN cd /ComfyUI/custom_nodes && \
    git clone --depth=1 https://github.com/kijai/ComfyUI-segment-anything-2 && rm -rf ComfyUI-segment-anything-2/.git && \
    git clone --depth=1 https://github.com/eddyhhlure1Eddy/IntelligentVRAMNode && rm -rf IntelligentVRAMNode/.git && \
    git clone --depth=1 https://github.com/eddyhhlure1Eddy/auto_wan2.2animate_freamtowindow_server && rm -rf auto_wan2.2animate_freamtowindow_server/.git && \
    git clone --depth=1 https://github.com/eddyhhlure1Eddy/ComfyUI-AdaptiveWindowSize && \
    cd ComfyUI-AdaptiveWindowSize/ComfyUI-AdaptiveWindowSize && mv * ../ && \
    cd .. && rm -rf ComfyUI-AdaptiveWindowSize/.git

# --- Preload required models (these stay small, the rest load dynamically from HuggingFace) ---
RUN mkdir -p /ComfyUI/models/{vae,clip_vision,text_encoders,diffusion_models,loras,detection}

RUN wget -q https://huggingface.co/Kijai/WanVideo_comfy/resolve/main/Wan2_1_VAE_bf16.safetensors -O /ComfyUI/models/vae/Wan2_1_VAE_bf16.safetensors
RUN wget -q https://huggingface.co/Comfy-Org/Wan_2.1_ComfyUI_repackaged/resolve/main/split_files/clip_vision/clip_vision_h.safetensors -O /ComfyUI/models/clip_vision/clip_vision_h.safetensors
RUN wget -q https://huggingface.co/Kijai/WanVideo_comfy/resolve/main/umt5-xxl-enc-bf16.safetensors -O /ComfyUI/models/text_encoders/umt5-xxl-enc-bf16.safetensors
RUN wget -q https://huggingface.co/Kijai/WanVideo_comfy_fp8_scaled/resolve/main/Wan22Animate/Wan2_2-Animate-14B_fp8_e4m3fn_scaled_KJ.safetensors -O /ComfyUI/models/diffusion_models/Wan2_2-Animate-14B_fp8_e4m3fn_scaled_KJ.safetensors

RUN wget -q https://huggingface.co/eddy1111111/lightx2v_it2v_adaptive_fusionv_1.safetensors/resolve/main/lightx2v_elite_it2v_animate_face.safetensors -O /ComfyUI/models/loras/lightx2v_elite_it2v_animate_face.safetensors
RUN wget -q https://huggingface.co/eddy1111111/lightx2v_it2v_adaptive_fusionv_1.safetensors/resolve/main/WAN22_MoCap_fullbodyCOPY_ED.safetensors -O /ComfyUI/models/loras/WAN22_MoCap_fullbodyCOPY_ED.safetensors
RUN wget -q https://huggingface.co/eddy1111111/lightx2v_it2v_adaptive_fusionv_1.safetensors/resolve/main/FullDynamic_Ultimate_Fusion_Elite.safetensors -O /ComfyUI/models/loras/FullDynamic_Ultimate_Fusion_Elite.safetensors
RUN wget -q https://huggingface.co/eddy1111111/lightx2v_it2v_adaptive_fusionv_1.safetensors/resolve/main/Wan2.2-Fun-A14B-InP-Fusion-Elite.safetensors -O /ComfyUI/models/loras/Wan2.2-Fun-A14B-InP-Fusion-Elite.safetensors

RUN wget -q https://huggingface.co/Wan-AI/Wan2.2-Animate-14B/resolve/main/process_checkpoint/det/yolov10m.onnx -O /ComfyUI/models/detection/yolov10m.onnx
RUN wget -q https://huggingface.co/Kijai/vitpose_comfy/resolve/main/onnx/vitpose_h_wholebody_model.onnx -O /ComfyUI/models/detection/vitpose_h_wholebody_model.onnx
RUN wget -q https://huggingface.co/Kijai/vitpose_comfy/resolve/main/onnx/vitpose_h_wholebody_data.bin -O /ComfyUI/models/detection/vitpose_h_wholebody_data.bin

# --- Copy your repo files and entrypoint script ---
COPY . /
RUN chmod +x /entrypoint.sh

CMD ["/entrypoint.sh"]
