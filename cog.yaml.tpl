build:
  gpu: true
  cuda: "12.1"
  system_packages:
    - ffmpeg
  python_version: "3.12"
  python_packages:
    - torch
    - torchvision
    - torchaudio
    - torchsde
    - einops
    - transformers>=4.39.3
    - tokenizers>=0.13.3
    - sentencepiece
    - safetensors>=0.3.0
    - aiohttp
    - accelerate>=1.1.1
    - pyyaml
    - Pillow
    - scipy
    - tqdm
    - psutil
    - spandrel
    - soundfile
    - kornia>=0.7.1
    - websocket-client==1.6.3
    - diffusers>=0.31.0
    - av
    - comfyui-frontend-package==1.11.8

    # ComfyUI-AdvancedLivePortrait
    - dill

    # Inspire
    - webcolors

    # fix for pydantic issues in cog
    # https://github.com/replicate/cog/issues/1623
    - albumentations==1.4.3

    # was-node-suite-comfyui
    # https://github.com/WASasquatch/was-node-suite-comfyui/blob/main/requirements.txt
    - cmake
    - imageio
    - joblib
    - matplotlib
    - pilgram
    - scikit-learn
    - rembg

    # ComfyUI_essentials
    - numba

    # ComfyUI_FizzNodes
    - pandas
    - numexpr

    # comfyui-reactor-node
    - insightface
    - onnx

    # ComfyUI-Impact-Pack
    - segment-anything
    - piexif

    # ComfyUI-Impact-Subpack
    - ultralytics!=8.0.177

    # comfyui_segment_anything
    - timm

    # comfyui_controlnet_aux
    # https://github.com/Fannovel16/comfyui_controlnet_aux/blob/main/requirements.txt
    - importlib_metadata
    - opencv-python-headless>=4.0.1.24
    - filelock
    - numpy
    - scikit-image
    - python-dateutil
    - mediapipe
    - svglib
    - fvcore
    - yapf
    - omegaconf
    - ftfy
    - addict
    - yacs
    - trimesh[easy]

    # ComfyUI-KJNodes
    - librosa
    - color-matcher

    # PuLID
    - facexlib

    # SUPIR
    - open-clip-torch>=2.24.0
    - pytorch-lightning>=2.2.1

    # For train.py and custom loras
    - huggingface_hub[hf-transfer]

    # ComfyUI-segment-anything-2
    - iopath
  run:
    - curl -o /usr/local/bin/pget -L "https://github.com/replicate/pget/releases/latest/download/pget_$(uname -s)_$(uname -m)" && chmod +x /usr/local/bin/pget
    - pip install onnxruntime-gpu --extra-index-url https://aiinfra.pkgs.visualstudio.com/PublicPackages/_packaging/onnxruntime-cuda-12/pypi/simple/
predict: "predict.py:$PREDICTOR"
