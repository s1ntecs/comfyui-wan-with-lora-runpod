import json
import os
import base64
from pathlib import Path
import mimetypes
import traceback
import torch

from comfyui import ComfyUI

import runpod
from runpod.serverless.utils.rp_download import file
from runpod.serverless.modules.rp_logger import RunPodLogger
import random

from styles import STYLE_URLS, STYLE_NAMES  # ваши словари

OUTPUT_DIR = "/results"
INPUT_DIR = "/job_files"
COMFYUI_TEMP_OUTPUT_DIR = "ComfyUI/temp"
COMFYUI_LORAS_DIR = "ComfyUI/models/loras"
ALL_DIRECTORIES = [OUTPUT_DIR, INPUT_DIR, COMFYUI_TEMP_OUTPUT_DIR]

mimetypes.add_type("image/webp", ".webp")
api_json_file = "workflow.json"

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
        self.comfyUI = ComfyUI("127.0.0.1:8188")
        self.comfyUI.start_server(OUTPUT_DIR, INPUT_DIR)

        os.makedirs("ComfyUI/models/loras", exist_ok=True)

        with open(api_json_file, "r") as file:
            workflow = json.loads(file.read()) 
        self.comfyUI.handle_weights(
            workflow,
            weights_to_download=[
                "wan_2.1_vae.safetensors",
                "umt5_xxl_fp16.safetensors",
                "clip_vision_h.safetensors",
            ],
        )

    def get_width_and_height(self, resolution: str, aspect_ratio: str):
        sizes = {
            "480p": {
                "16:9": (832, 480),
                "9:16": (480, 832),
                "1:1": (644, 644),
            },
            "720p": {
                "16:9": (1280, 720),
                "9:16": (720, 1280),
                "1:1": (980, 980),
            },
        }
        return sizes[resolution][aspect_ratio]

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

    def filename_with_extension(self, input_file, prefix):
        extension = os.path.splitext(input_file.name)[1]
        return f"{prefix}{extension}"

    def update_workflow(self, workflow, **kwargs):
        is_image_to_video = kwargs["image_filename"] is not None
        model = "14b-i2v-480p"  # kwargs['model']
        # "14b-i2v-480p"
        workflow["37"]["inputs"]["unet_name"] = "wan2.1_i2v_480p_14B_bf16.safetensors"

        positive_prompt = workflow["6"]["inputs"]
        positive_prompt["text"] = kwargs["prompt"]

        negative_prompt = workflow["7"]["inputs"]
        neg = kwargs.get("negative_prompt") or ""
        negative_prompt["text"] = f"nsfw{', ' + neg if neg else ''}"

        sampler = workflow["3"]["inputs"]
        sampler["seed"] = kwargs["seed"]
        sampler["cfg"] = kwargs["sample_guide_scale"]
        sampler["steps"] = kwargs["sample_steps"]

        shift = workflow["48"]["inputs"]
        shift["shift"] = kwargs["sample_shift"]

        if is_image_to_video:
            del workflow["40"]
            wan_i2v_latent = workflow["58"]["inputs"]
            wan_i2v_latent["length"] = kwargs["frames"]

            image_loader = workflow["55"]["inputs"]
            image_loader["image"] = kwargs["image_filename"]

            image_resizer = workflow["56"]["inputs"]
            if kwargs["resolution"] == "720p":
                image_resizer["target_size"] = 1008
            else:
                image_resizer["target_size"] = 644

        else:
            del workflow["55"]
            del workflow["56"]
            del workflow["57"]
            del workflow["58"]
            del workflow["59"]
            del workflow["60"]
            width, height = self.get_width_and_height(
                kwargs["resolution"], kwargs["aspect_ratio"]
            )
            empty_latent_video = workflow["40"]["inputs"]
            empty_latent_video["length"] = kwargs["frames"]
            empty_latent_video["width"] = width
            empty_latent_video["height"] = height

            sampler["model"] = ["48", 0]
            sampler["positive"] = ["6", 0]
            sampler["negative"] = ["7", 0]
            sampler["latent_image"] = ["40", 0]

        thresholds = {
            "14b": {
                "Balanced": 0.15,
                "Fast": 0.2,
                "coefficients": "14B",
            },
            "14b-i2v-480p": {
                "Balanced": 0.19,
                "Fast": 0.26,
                "coefficients": "i2v_480",
            },
            "14b-i2v-720p": {
                "Balanced": 0.2,
                "Fast": 0.3,
                "coefficients": "i2v_720",
            },
            "1.3b": {
                "Balanced": 0.07,
                "Fast": 0.08,
                "coefficients": "1.3B",
            },
        }

        fast_mode = kwargs["fast_mode"]
        if fast_mode == "Off":
            # Turn off tea cache
            del workflow["54"]
            workflow["49"]["inputs"]["model"] = ["37", 0]
        else:
            tea_cache = workflow["54"]["inputs"]
            tea_cache["coefficients"] = thresholds[model]["coefficients"]
            tea_cache["rel_l1_thresh"] = thresholds[model][fast_mode]

        if kwargs["lora_filename"]:
            lora_loader = workflow["49"]["inputs"]
            if kwargs["lora_filename"]:
                lora_loader["lora_name"] = kwargs["lora_filename"]
            elif kwargs["lora_url"]:
                lora_loader["lora_name"] = kwargs["lora_url"]

            lora_loader["strength_model"] = kwargs["lora_strength_model"]
            lora_loader["strength_clip"] = kwargs["lora_strength_clip"]
        else:
            del workflow["49"]  # delete lora loader node
            positive_prompt["clip"] = ["38", 0]
            shift["model"] = ["37", 0] if fast_mode == "Off" else ["54", 0]

    def generate(
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
    ) -> str:
        self.comfyUI.cleanup(ALL_DIRECTORIES)
        
        image_obj = file(image_url)
        image_path = Path(image_obj["file_path"])
        
        if seed is None:
            seed = random.randint(0, 2**32 - 1)

        # Правильно — передаём Path-объект, у него есть .name
        image_filename = self.filename_with_extension(image_path, "image")

        lora_filename = STYLE_NAMES.get(lora_style)
        # lora_path = f"{COMFYUI_LORAS_DIR}/{lora_filename}" if lora_filename else None
        # inferred_model_type = "14b"
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
            lora_strength_model=lora_strength_model,
            lora_strength_clip=lora_strength_clip,
            resolution="480p",
            image_filename=image_filename
        )

        wf = self.comfyUI.load_workflow(workflow)
        self.comfyUI.connect()
        self.comfyUI.run_workflow(wf)

        files =  self.comfyUI.get_files(OUTPUT_DIR, file_extensions=["mp4"])
        with open(files[0], "rb") as f:
            video_bytes = f.read()
        video_b64 = base64.b64encode(video_bytes).decode("utf-8")
        return video_b64
# -------------------------------------------------------------
#  RunPod Handler
# -------------------------------------------------------------
logger = RunPodLogger()

predictor = None

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
        

        prompt = payload["prompt"]
        negative_prompt = payload.get("negative_prompt", "")
        lora_style = payload.get("lora_style", None)
        aspect_ratio = payload.get("aspect_ratio", "16:9")
        frames = payload.get("frames", 81)
        model = payload.get("model", "14b-i2v-480p")
        resolution = payload.get("resolution", "480p")
        lora_strength_clip = payload.get("lora_strength_clip", 1.0)
        fast_mode = payload.get("fast_mode", "Balanced")
        num_inference_steps = payload.get("num_inference_steps", 28)
        guidance_scale = payload.get("guidance_scale", 5.0)
        sample_shift = payload.get("sample_shift", 8.0)
        lora_strength = payload.get("lora_strength", 1.0)
        seed = payload.get("seed", None)

        video_b64 = predictor.generate(
            prompt=prompt,
            negative_prompt=negative_prompt,
            image_url=image_url,
            aspect_ratio=aspect_ratio,
            frames=frames,
            model=model,
            lora_style=lora_style,
            lora_strength_model=lora_strength,
            lora_strength_clip=lora_strength_clip,
            fast_mode=fast_mode,
            sample_shift=sample_shift,
            sample_guide_scale=guidance_scale,
            sample_steps=num_inference_steps,
            seed=seed,
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
