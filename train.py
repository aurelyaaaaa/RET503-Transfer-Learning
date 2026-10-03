import argparse
import json
import time
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from sklearn.metrics import ConfusionMatrixDisplay, classification_report, confusion_matrix
from torch import nn
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms

SEED = 42
EPOCHS = 10
BATCH_SIZE = 16
NUM_CLASSES = 4

torch.manual_seed(SEED)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATA = Path("dataset")
RESULTS = Path("results")

train_tfms = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

eval_tfms = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


def make_model(mode):
    if mode in {"feature", "partial"}:
        model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    else:
        model = models.resnet18(weights=None)

    if mode == "feature":
        for p in model.parameters():
            p.requires_grad = False
        model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)
    elif mode == "partial":
        for p in model.parameters():
            p.requires_grad = False
        for p in model.layer4.parameters():
            p.requires_grad = True
        model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)
    else:
        model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)
    return model


def loaders():
    train_ds = datasets.ImageFolder(DATA / "train", transform=train_tfms)
    val_ds = datasets.ImageFolder(DATA / "val", transform=eval_tfms)
    test_ds = datasets.ImageFolder(DATA / "test", transform=eval_tfms)

    # num_workers=0 lebih aman untuk Windows dan tetap cukup untuk dataset kecil.
    train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_dl = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    test_dl = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    return train_ds, val_ds, test_ds, train_dl, val_dl, test_dl


def run_epoch(model, loader, criterion, optimizer=None):
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    correct = 0
    total = 0

    for x, y in loader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        if training:
            optimizer.zero_grad()
        with torch.set_grad_enabled(training):
            logits = model(x)
            loss = criterion(logits, y)
            if training:
                loss.backward()
                optimizer.step()
        total_loss += loss.item() * x.size(0)
        correct += (logits.argmax(1) == y).sum().item()
        total += y.size(0)

    return total_loss / total, correct / total


def predict(model, loader):
    model.eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for x, y in loader:
            logits = model(x.to(DEVICE))
            y_true.extend(y.numpy().tolist())
            y_pred.extend(logits.argmax(1).cpu().numpy().tolist())
    return y_true, y_pred


def main(mode):
    RESULTS.mkdir(exist_ok=True)
    out = RESULTS / mode
    out.mkdir(exist_ok=True)

    train_ds, val_ds, test_ds, train_dl, val_dl, test_dl = loaders()
    model = make_model(mode).to(DEVICE)

    if mode == "partial":
        optimizer = Adam([
            {"params": model.layer4.parameters(), "lr": 1e-4},
            {"params": model.fc.parameters(), "lr": 1e-3},
        ])
    else:
        optimizer = Adam(
            [p for p in model.parameters() if p.requires_grad], lr=1e-3
        )

    scheduler = CosineAnnealingLR(optimizer, T_max=EPOCHS)
    criterion = nn.CrossEntropyLoss()
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    best_acc = -1.0
    best_state = None
    epoch_90 = None
    start = time.perf_counter()

    print(f"Mode: {mode}")
    print(f"Device: {DEVICE}")
    print(f"Classes: {train_ds.classes}")

    for epoch in range(1, EPOCHS + 1):
        tr_loss, tr_acc = run_epoch(model, train_dl, criterion, optimizer)
        va_loss, va_acc = run_epoch(model, val_dl, criterion)
        scheduler.step()

        history["train_loss"].append(tr_loss)
        history["train_acc"].append(tr_acc)
        history["val_loss"].append(va_loss)
        history["val_acc"].append(va_acc)

        if va_acc > best_acc:
            best_acc = va_acc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        if epoch_90 is None and va_acc >= 0.90:
            epoch_90 = epoch

        print(
            f"[{mode}] Epoch {epoch:02d}/{EPOCHS} "
            f"train_acc={tr_acc:.3f} val_acc={va_acc:.3f}"
        )

    train_time = time.perf_counter() - start

    if best_state is not None:
        model.load_state_dict(best_state)
        model.to(DEVICE)

    test_loss, test_acc = run_epoch(model, test_dl, criterion)
    y_true, y_pred = predict(model, test_dl)

    torch.save(model.state_dict(), out / "best_model.pth")

    report = classification_report(
        y_true, y_pred, target_names=test_ds.classes, output_dict=True, zero_division=0
    )
    with open(out / "classification_report.json", "w") as f:
        json.dump(report, f, indent=2)

    with open(out / "metrics.json", "w") as f:
        json.dump({
            "mode": mode,
            "device": str(DEVICE),
            "best_val_accuracy": best_acc,
            "test_accuracy": test_acc,
            "test_loss": test_loss,
            "training_time_seconds": train_time,
            "epoch_val_accuracy_ge_90": epoch_90,
            "classes": train_ds.classes,
        }, f, indent=2)

    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=test_ds.classes)
    disp.plot(values_format="d")
    plt.title(f"Confusion Matrix - {mode}")
    plt.tight_layout()
    plt.savefig(out / "confusion_matrix.png", dpi=150)
    plt.close()

    epochs = range(1, EPOCHS + 1)
    plt.figure(figsize=(7, 4))
    plt.plot(epochs, history["train_acc"], label="Train accuracy")
    plt.plot(epochs, history["val_acc"], label="Validation accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title(f"Accuracy per Epoch - {mode}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out / "accuracy_curve.png", dpi=150)
    plt.close()

    print(f"\n{mode} selesai.")
    print(f"Best val accuracy : {best_acc:.4f}")
    print(f"Test accuracy     : {test_acc:.4f}")
    print(f"Training time     : {train_time:.1f} detik")
    print(f"Epoch >= 90%      : {epoch_90}")
    print(f"Hasil disimpan di : {out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["feature", "partial", "scratch"], required=True)
    args = parser.parse_args()
    main(args.mode)
