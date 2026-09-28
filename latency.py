import time
from pathlib import Path

import torch
from torchvision import models, transforms
from PIL import Image

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMG_SIZE = 224

tfm = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


def first_image():
    for p in Path("dataset/test").rglob("*"):
        if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}:
            return p
    raise FileNotFoundError("Tidak ada gambar di dataset/test.")


def measure(model, x, n=50):
    model.eval().to(DEVICE)
    x = x.to(DEVICE)
    with torch.no_grad():
        for _ in range(10):
            _ = model(x)
        if DEVICE.type == "cuda":
            torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(n):
            _ = model(x)
        if DEVICE.type == "cuda":
            torch.cuda.synchronize()
    total = time.perf_counter() - start
    ms = total / n * 1000
    fps = 1000 / ms
    return ms, fps


def main():
    img_path = first_image()
    x = tfm(Image.open(img_path).convert("RGB")).unsqueeze(0)
    resnet = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    mobile = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.IMAGENET1K_V1)
    r_ms, r_fps = measure(resnet, x)
    m_ms, m_fps = measure(mobile, x)
    print(f"Device: {DEVICE}")
    print(f"ResNet-18         : {r_ms:.2f} ms/frame | {r_fps:.2f} FPS")
    print(f"MobileNetV3-Small : {m_ms:.2f} ms/frame | {m_fps:.2f} FPS")


if __name__ == "__main__":
    main()
