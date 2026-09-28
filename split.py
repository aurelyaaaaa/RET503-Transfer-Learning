import random
import shutil
from pathlib import Path

SEED = 42
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
RAW = Path("dataset_raw")
OUT = Path("dataset")
IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def main():
    random.seed(SEED)
    if not RAW.exists():
        raise FileNotFoundError("dataset_raw/ belum ada. Jalankan download_and_prepare.py terlebih dahulu.")
    if OUT.exists():
        shutil.rmtree(OUT)

    for cls_dir in sorted([p for p in RAW.iterdir() if p.is_dir()]):
        images = [p for p in cls_dir.iterdir() if p.suffix.lower() in IMG_EXT]
        random.shuffle(images)
        n = len(images)
        n_train = int(n * TRAIN_RATIO)
        n_val = int(n * VAL_RATIO)
        splits = {
            "train": images[:n_train],
            "val": images[n_train:n_train+n_val],
            "test": images[n_train+n_val:],
        }
        for split_name, items in splits.items():
            dst = OUT / split_name / cls_dir.name
            dst.mkdir(parents=True, exist_ok=True)
            for img in items:
                shutil.copy2(img, dst / img.name)
        print(cls_dir.name, {k: len(v) for k, v in splits.items()})
    print("\nPembagian dataset selesai.")


if __name__ == "__main__":
    main()
