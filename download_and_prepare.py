import shutil
from pathlib import Path
from datetime import date

import pandas as pd

N_PER_CLASS = 50
CLASSES = ["bolt", "nut", "locatingpin", "washer"]
OUT_DIR = Path("dataset_raw")


def find_class_dirs(root: Path):
    found = {}
    for p in root.rglob("*"):
        if p.is_dir():
            name = p.name.lower().replace(" ", "").replace("_", "")
            for cls in CLASSES:
                if name == cls.replace("_", ""):
                    found[cls] = p
    return found


def main():
    OUT_DIR.mkdir(exist_ok=True)

    try:
        import kagglehub
        print("Mengunduh dataset dari Kaggle...")
        root = Path(kagglehub.dataset_download(
            "manikantanrnair/images-of-mechanical-parts-boltnut-washerpin"
        ))
        print("Dataset berada di:", root)
    except Exception as e:
        print("Download otomatis gagal:", e)
        print("Silakan download manual dataset Kaggle, lalu letakkan folder kelas di dataset_raw/.")
        return

    class_dirs = find_class_dirs(root)
    rows = []

    for cls in CLASSES:
        src = class_dirs.get(cls)
        if src is None:
            print(f"[PERINGATAN] folder kelas {cls} tidak ditemukan.")
            continue

        images = sorted([
            p for p in src.iterdir()
            if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
        ])

        if len(images) < N_PER_CLASS:
            raise RuntimeError(f"Kelas {cls} hanya memiliki {len(images)} gambar.")

        dst_cls = OUT_DIR / cls
        dst_cls.mkdir(parents=True, exist_ok=True)

        for i, src_img in enumerate(images[:N_PER_CLASS], start=1):
            dst_name = f"{cls}_{i:03d}{src_img.suffix.lower()}"
            dst = dst_cls / dst_name
            shutil.copy2(src_img, dst)
            rows.append({
                "nama_file": dst_name,
                "kelas": cls,
                "tanggal": date.today().isoformat(),
                "kondisi_cahaya": "sumber_dataset",
            })

        print(f"{cls}: {N_PER_CLASS} gambar")

    pd.DataFrame(rows).to_csv("metadata.csv", index=False)
    print(f"\nSelesai. Total gambar: {len(rows)}")
    print("metadata.csv dibuat.")


if __name__ == "__main__":
    main()
