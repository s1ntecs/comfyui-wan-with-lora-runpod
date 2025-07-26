from pathlib import Path
import os, shutil


ROOT = Path("ComfyUI/models")
SUBDIRS = ["diffusion_models", "vae", "text_encoders", "clip_vision", "loras"]


def flatten():
    for sub in SUBDIRS:
        top = ROOT / sub
        if not top.exists():
            continue

        # ищем файлы, у которых в пути встречается /split_files/
        for f in top.rglob("*"):
            if f.is_file() and "/split_files/" in f.as_posix():
                dst = top / f.name                      # место, куда хотим положить
                if dst.exists():
                    continue                           # уже разложили раньше

                dst.parent.mkdir(parents=True, exist_ok=True)
                try:
                    os.link(f, dst)                    # 0-байтовая жёсткая ссылка
                    print("🔗", f.relative_to(ROOT), "→", dst.relative_to(ROOT))
                except OSError:
                    shutil.copy2(f, dst)
                    print("📄", f.relative_to(ROOT), "→", dst.relative_to(ROOT))


if __name__ == "__main__":
    flatten()