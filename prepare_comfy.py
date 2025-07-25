#!/usr/bin/env python3
import os, json, shutil
from pathlib import Path
from comfyui import ComfyUI

SPLIT_DIR = Path("ComfyUI/models/split_files/diffusion_models")
TARGET_DIR = Path("ComfyUI/models/diffusion_models")


with open("workflow.json") as f:
    WORKFLOW = json.load(f)


def mirror_split_files():
    """Копирует или линкует файлы UNet из split_files → diffusion_models."""
    if not SPLIT_DIR.exists():
        return
    TARGET_DIR.mkdir(parents=True, exist_ok=True)
    for src in SPLIT_DIR.iterdir():
        if src.is_file():
            dst = TARGET_DIR / src.name
            if not dst.exists():          # не дублируем
                try:
                    os.link(src, dst)     # жёсткая ссылка — ноль копий
                except OSError:
                    shutil.copy2(src, dst)


def preload_weights():
    c = ComfyUI("127.0.0.1:8188")         # адрес не важен, соединяться не будем
    c.load_workflow(WORKFLOW, check_inputs=False)  # качаем всё указанное в JSON
    mirror_split_files()                  # и «выравниваем» пути


if __name__ == "__main__":
    preload_weights()