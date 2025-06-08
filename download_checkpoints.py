#!/usr/bin/env python3
import os
from huggingface_hub import hf_hub_download

from  styles import STYLE_URLS_UNIQUE

# Репозиторий на Hugging Face
REPO_ID = "Comfy-Org/Wan_2.1_ComfyUI_repackaged"

# wan2.1_i2v_480p_14B_bf16
# Список файлов для скачивания: (путь внутри репозитория, локальная папка для сохранения)
FILES_TO_DOWNLOAD = [
    # Уже существующий файл из примера
    ("split_files/diffusion_models/wan2.1_i2v_480p_14B_fp16.safetensors", "ComfyUI/models/diffusion_models"),
    ("split_files/diffusion_models/wan2.1_i2v_480p_14B_bf16.safetensors", "ComfyUI/models/diffusion_models"),
    # Новые файлы для ComfyUI
    ("split_files/text_encoders/umt5_xxl_fp16.safetensors",   "ComfyUI/models/text_encoders"),
    ("split_files/vae/wan_2.1_vae.safetensors",               "ComfyUI/models/vae"),
    ("split_files/clip_vision/clip_vision_h.safetensors",     "ComfyUI/models/clip_vision"),
]

def download_wan_files():
    """
    Скачивает из репозитория на Hugging Face все файлы из списка FILES_TO_DOWNLOAD
    и сохраняет их в соответствующие папки внутри проекта ComfyUI.
    """
    for remote_path, local_subdir in FILES_TO_DOWNLOAD:
        # Создаем локальную папку, если её ещё нет
        os.makedirs(local_subdir, exist_ok=True)

        try:
            # Скачиваем файл
            local_path = hf_hub_download(
                repo_id=REPO_ID,
                filename=remote_path,
                local_dir=local_subdir,
                local_dir_use_symlinks=False
            )
            print(f"Скачан файл: {remote_path} → {local_path}")
        except Exception as e:
            print(f"Ошибка при скачивании {remote_path}: {e}")


def download_loras():
    downloaded_paths = []
    for key, lora_url in STYLE_URLS_UNIQUE.items():
        try:
            parts = lora_url.split("/")
            repo_id = "/".join(parts[3:5])
            filename = parts[-1]
            print(f"Downloading LoRA: key={key}, repo_id={repo_id}, filename={filename}")
            local_path = hf_hub_download(
                repo_id=repo_id,
                filename=filename,
                local_dir="./ComfyUI/models/loras"
            )
            downloaded_paths.append(local_path)
        except Exception as e:
            print(f"Error downloading LoRA '{key}':", str(e))
            # либо продолжаем (continue), либо прерываем полностью, в зависимости от задачи
            raise RuntimeError(f"Failed to download LoRA '{key}': {str(e)}")
    return downloaded_paths


if __name__ == "__main__":
    download_wan_files()
    download_loras()