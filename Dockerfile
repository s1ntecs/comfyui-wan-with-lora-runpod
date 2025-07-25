# Используем базовый образ RunPod с Python 3.12 и CUDA 12.1
# FROM runpod/pytorch:2.4.0-py3.12-cuda12.4.1-devel-ubuntu22.04
FROM runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04
# runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04
# Работаем от корня проекта
WORKDIR /

# Обновляем пакеты и устанавливаем системную зависимость ffmpeg
# RUN apt-get update && \
#     apt upgrade -y && \
#     apt-get install -y ffmpeg && \
#     rm -rf /var/lib/apt/lists/*

RUN rm -f /etc/apt/sources.list.d/cuda-ubuntu*.list
RUN apt-get update && \
    apt-get install -y --no-install-recommends ffmpeg && \
    rm -rf /var/lib/apt/lists/
# Копируем все файлы из текущей директории в контейнер

# Устанавливаем Python-зависимости из списка
RUN pip install \
    torch \
    torchvision \
    torchaudio \
    torchsde \
    einops \
    "transformers>=4.39.3" \
    "tokenizers>=0.13.3" \
    sentencepiece \
    "safetensors>=0.3.0" \
    aiohttp \
    "accelerate>=1.1.1" \
    pyyaml \
    Pillow \
    scipy \
    tqdm \
    psutil \
    spandrel \
    soundfile \
    "kornia>=0.7.1" \
    "websocket-client==1.6.3" \
    "diffusers>=0.31.0" \
    av \
    "comfyui-frontend-package==1.11.8" \
    dill \
    webcolors \
    "albumentations==1.4.3" \
    cmake \
    imageio \
    joblib \
    matplotlib \
    pilgram \
    scikit-learn \
    rembg \
    numba \
    pandas \
    numexpr \
    insightface \
    onnx \
    "segment-anything" \
    piexif \
    "ultralytics!=8.0.177" \
    timm \
    importlib_metadata \
    "opencv-python-headless>=4.0.1.24" \
    filelock \
    numpy \
    scikit-image \
    python-dateutil \
    mediapipe \
    svglib \
    fvcore \
    yapf \
    omegaconf \
    ftfy \
    addict \
    yacs \
    "trimesh[easy]" \
    librosa \
    "color-matcher" \
    facexlib \
    "open-clip-torch>=2.24.0" \
    "pytorch-lightning>=2.2.1" \
    "huggingface_hub[hf-transfer]" \
    iopath \
    runpod

# Скачиваем и делаем исполняемым pget (аналог секции run в спецификации)
RUN curl -o /usr/local/bin/pget -L "https://github.com/replicate/pget/releases/latest/download/pget_$(uname -s)_$(uname -m)" && \
    chmod +x /usr/local/bin/pget

COPY install_custom_nodes.py download_checkpoints.py prepare_comfy.py custom_nodes.json /

RUN python3 install_custom_nodes.py
RUN python3 download_checkpoints.py
RUN python3 prepare_comfy.py

COPY . /

COPY --chmod=755 start_standalone.sh /start.sh

# Start the container
ENTRYPOINT /start.sh