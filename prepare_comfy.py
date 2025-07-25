#!/usr/bin/env python3
from comfyui import ComfyUI


with open("workflow.json", "r") as file:
    WORKFLOW_JSON = file.read()


def download_comfy():
    """Если у тебя в helpers.comfyui есть метод предзагрузки — вызываем корректно."""
    comfy = ComfyUI("127.0.0.1:8188")
    # Заметка: у тебя было `comfyUI.download_pre_start_models` без вызова.
    if hasattr(comfy, "load_workflow"):
        comfy.load_workflow(WORKFLOW_JSON)
    else:
        # либо просто заглушка
        pass


if __name__ == "__main__":
    download_comfy()
