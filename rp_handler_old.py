import os
import base64
import time
import traceback
import uuid
import torch
import tempfile
import requests
import numpy as np
from dataclasses import dataclass

from diffusers import AutoencoderKLWan, WanImageToVideoPipeline
from diffusers.utils import export_to_video, load_image
from transformers import CLIPVisionModel
from huggingface_hub import hf_hub_download
from comfyui import ComfyUI

import runpod
from runpod.serverless.utils.rp_validator import validate
from runpod.serverless.utils.rp_download import file
from runpod.serverless.modules.rp_logger import RunPodLogger
from cog_model_helpers import seed as seed_helper

from styles import STYLE_URLS, STYLE_NAMES  # ваши словари


# -------------------------------------------------------------
#  Схема входных данных для валидации
# -------------------------------------------------------------



device = "cuda" if torch.cuda.is_available() else "cpu"
model_id = "Wan-AI/Wan2.1-I2V-14B-480P-Diffusers"
cache = "./hf_cache"
MODEL_FRAME_RATE = 16

# Будем хранить текущий активный стиль и путь
CURRENT_LORA_NAME = "./loras/wan_SmNmRmC.safetensors"

class Predictor():
    def setup(self):
        """ 
        Загружаем CLIPVisionModel, VAE и сам WanImageToVideoPipeline. 
        Вызывается один раз перед первым predict.
        """
        try:
            self.image_encoder = CLIPVisionModel.from_pretrained(
                model_id,
                subfolder="image_encoder",
                cache_dir=cache,
                torch_dtype=torch.float32
            )
            self.vae = AutoencoderKLWan.from_pretrained(
                model_id, subfolder="vae",
                cache_dir=cache,
                torch_dtype=torch.float32
            )
            self.pipe = WanImageToVideoPipeline.from_pretrained(
                model_id,
                vae=self.vae,
                cache_dir=cache,
                image_encoder=self.image_encoder,
                torch_dtype=torch.bfloat16
            ).to(device)
            self.pipe.enable_model_cpu_offload()

            self.vae_scale_factor_spatial  = self.pipe.vae_scale_factor_spatial
            self.vae_scale_factor_temporal = self.pipe.vae_scale_factor_temporal
            
            # Загрузим LoRA по умолчанию, если он есть
            self.pipe.load_lora_weights(CURRENT_LORA_NAME, multiplier=1.0)
            print(f"Model loaded. VAE scales: spatial={self.vae_scale_factor_spatial}, temporal={self.vae_scale_factor_temporal}")
        except Exception as e:
            print("Error loading pipeline:", str(e))
            raise RuntimeError(f"Failed to load pipeline: {str(e)}")

    def _get_local_lora_path(self, lora_style: str) -> str:
        """
        Сформировать локальный путь к файлу LoRA по ключу стиля:
          - ищем в STYLE_NAMES имя файла,
          - проверяем, лежит ли он в ./loras/,
          - если нет — возвращаем None.
        """
        if not lora_style:
            return None

        filename = STYLE_NAMES.get(lora_style)
        if filename is None:
            return None

        # считаем, что локальные лоры лежат в папке ./loras/
        local_path = os.path.join("./loras", filename)
        if os.path.isfile(local_path):
            return local_path
        return None

    def _download_lora_if_needed(self, lora_style: str) -> str:
        """
        Если у нас уже есть локально — вернём путь.
        Если нет и есть ссылка в STYLE_URLS — скачиваем в ./loras/ и вернём путь.
        """
        # 1) Узнаём файл по ключу
        filename = STYLE_NAMES.get(lora_style)
        if filename is None:
            raise RuntimeError(f"Unknown LORA style: {lora_style}")

        target_dir = "./loras"
        os.makedirs(target_dir, exist_ok=True)
        local_path = os.path.join(target_dir, filename)

        # Если файл уже скачан — сразу возвращаем
        if os.path.isfile(local_path):
            return local_path

        # Иначе — скачиваем по URL
        url = STYLE_URLS.get(lora_style)
        if url is None:
            raise RuntimeError(f"No URL found for LORA style: {lora_style}")

        print(f"Downloading LoRA '{lora_style}' from {url} into {local_path} ...")
        # Если ссылка ведёт на HF «blob»-вид, нужно чуть подправить URL, чтобы его можно было прям скачать raw
        if "huggingface.co" in url and "/blob/" in url:
            url = url.replace("/blob/", "/resolve/")
        resp = requests.get(url)
        if resp.status_code != 200:
            raise RuntimeError(f"Failed to download LORA from {url}: HTTP {resp.status_code}")

        with open(local_path, "wb") as f:
            f.write(resp.content)

        print(f"Successfully saved LoRA to {local_path}")
        return local_path

    def load_lora(self, lora_style: str, lora_strength: float = 1.0):
        """
        Верхнеуровневая функция «установки» LoRA:
          - Проверяем, меняется ли стиль (global CURRENT_LORA_STYLE).
          - Если нет, то выходим (уже загружен нужный).
          - Если да, то вызываем pipe.unload_lora_weights(), скачиваем/загружаем новую.
        """
        global CURRENT_LORA_NAME

        # Если стиль не передан — ничего не делаем
        if not lora_style:
            return

        # Скачиваем (или берём локальный) нужный файл
        local_path = self._download_lora_if_needed(lora_style)

         # Если стиль не поменялся — нет смысла перезагрузки
        if CURRENT_LORA_NAME == local_path:
            return

        try:
            self.pipe.unload_lora_weights()
        except Exception:
            # возможно, раньше pipe был без LoRA, пропускаем
            pass
        # Устанавливаем через diffusers
        print(f"Loading LoRA weights from local_path = {local_path} (style={lora_style}, strength={lora_strength})")
        self.pipe.load_lora_weights(local_path, multiplier=lora_strength)
        print("LoRA applied.")

        # Обновляем глобальное состояние
        CURRENT_LORA_NAME = local_path

    def predict(
        self,
        prompt: str,
        negative_prompt: str | None = None,
        image_url: str | None = None,
        aspect_ratio: str = "16:9",
        frames: int = 81,
        model: str | None = None,
        lora_style: str | None = None,
        lora_strength_model: float = 1.0,
        lora_strength_clip: float = 1.0,
        fast_mode: str = "Balanced",
        sample_shift: float = 8.0,
        sample_guide_scale: float = 5.0,
        sample_steps: int = 30,
        seed: int | None = None,
        resolution: str = "480p",
    ) -> str:
        """
        Запускаем генерацию видео и возвращаем Base64.
        """

        # 4) Загружаем изображение
        try:
            input_image = load_image(str(image_url))
        except Exception as e:
            raise RuntimeError(f"Failed to load input image: {str(e)}")

        seed = seed_helper.generate(seed)

        lora_filename = None
        inferred_model_type = None
        elif lora_url:
            if m := re.match(
                r"^(?:https?://replicate.com/)?([^/]+)/([^/]+)/?$", lora_url
            ):
                owner, model_name = m.groups()
                lora_filename, inferred_model_type = download_replicate_weights(
                    f"https://replicate.com/{owner}/{model_name}/_weights",
                    COMFYUI_LORAS_DIR,
                )
            elif lora_url.startswith("https://replicate.delivery"):
                lora_filename, inferred_model_type = download_replicate_weights(
                    lora_url, COMFYUI_LORAS_DIR
                )

            if inferred_model_type and inferred_model_type != model:
                print(
                    f"Warning: Model type mismatch between requested model ({model}) and inferred model type ({inferred_model_type}). Using {inferred_model_type}."
                )
                model = inferred_model_type

        if resolution == "720p" and model == "1.3b":
            print("Warning: 720p is not supported for 1.3b, using 480p instead")
            resolution = "480p"

        with open(api_json_file, "r") as file:
            workflow = json.loads(file.read())

        self.update_workflow(
            workflow,
            prompt=prompt,
            negative_prompt=negative_prompt,
            seed=seed,
            fast_mode=fast_mode,
            sample_shift=sample_shift,
            sample_guide_scale=sample_guide_scale,
            sample_steps=sample_steps,
            model=model,
            frames=frames,
            aspect_ratio=aspect_ratio,
            lora_filename=lora_filename,
            lora_url=lora_url,
            lora_strength_model=lora_strength_model,
            lora_strength_clip=lora_strength_clip,
            resolution=resolution,
            image_filename=image_filename,
        )

        wf = self.comfyUI.load_workflow(workflow)
        self.comfyUI.connect()
        self.comfyUI.run_workflow(wf)

        return self.comfyUI.get_files(OUTPUT_DIR, file_extensions=["mp4"])



# -------------------------------------------------------------
#  RunPod Handler
# -------------------------------------------------------------
logger = RunPodLogger()

if torch.cuda.is_available():
    print("Current CUDA device:", torch.cuda.current_device())
    print("Device name:", torch.cuda.get_device_name(torch.cuda.current_device()))
    predictor = Predictor()
    predictor.setup()


def handler(job):
    global predictor
    try:
        payload = job.get("input", {})
        if predictor is None:
            predictor = Predictor()
            predictor.setup()

        # Скачиваем входное изображение
        image_url = payload["image_url"]
        image_obj = file(image_url)
        image_path = image_obj["file_path"]

        prompt              = payload["prompt"]
        negative_prompt     = payload.get("negative_prompt", "")
        lora_style          = payload.get("lora_style", None)
        lora_strength       = payload.get("lora_strength", 1.0)
        duration            = payload.get("duration", 3.0)
        fps                 = payload.get("fps", 16)
        guidance_scale      = payload.get("guidance_scale", 5.0)
        resize_mode         = payload.get("resize_mode", "auto")
        seed                = payload.get("seed", None)
        aspect_ratio       = payload.get("aspect_ratio", "auto")
        frames = payload.get("frames", None)
        model = payload.get("model", "wan2.1_i2v_480p_14B_fp16")
        resolution = payload.get("resolution", "480p")
        lora_strength_clip = payload.get("lora_strength_clip", 1.0)
        fast_mode = payload.get("fast_mode", False)
        sample_shift = payload.get("sample_shift", 0.0)
        sample_guide_scale = payload.get("sample_guide_scale", 1.0)
        sample_steps = payload.get("sample_steps", 1)

        video_b64 = predictor.predict(
            image=image_path,
            prompt=prompt,
            negative_prompt=negative_prompt,
            lora_style=lora_style,
            lora_strength_model=lora_strength,
            duration=duration,
            fps=fps,
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps,
            resize_mode=resize_mode,
            seed=seed
        )

        return {"video_base64": video_b64}

    except Exception as e:
        logger.error(f"An exception was raised: {e}")
        return {
            "error": str(e),
            "output": traceback.format_exc(),
            "refresh_worker": True
        }


if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})
