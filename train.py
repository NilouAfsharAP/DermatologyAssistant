from __future__ import annotations

import argparse
import math
import random
import time
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import models, transforms
from PIL import Image


# Must match the app's class order (inference.py)
APP_CLASSES: List[str] = [
    "Actinic Keratoses",
    "Basal Cell Carcinoma",
    "Benign Keratosis-like Lesions",
    "Dermatofibroma",
    "Melanocytic Nevi",
    "Melanoma",
    "Vascular Lesions",
]

HAM_DX_TO_APP: Dict[str, str] = {
    "akiec": "Actinic Keratoses",
    "bcc": "Basal Cell Carcinoma",
    "bkl": "Benign Keratosis-like Lesions",
    "df": "Dermatofibroma",
    "nv": "Melanocytic Nevi",
    "mel": "Melanoma",
    "vasc": "Vascular Lesions",
}

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ----------------------------
# Preprocessing (OpenCV)
# ----------------------------
def hair_removal_bgr(img_bgr: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (17, 17))
    blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)
    _, mask = cv2.threshold(blackhat, 10, 255, cv2.THRESH_BINARY)
    return cv2.inpaint(img_bgr, mask, 1, cv2.INPAINT_TELEA)


def clahe_rgb(img_bgr: np.ndarray) -> np.ndarray:
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe_op = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l2 = clahe_op.apply(l)
    lab2 = cv2.merge((l2, a, b))
    return cv2.cvtColor(lab2, cv2.COLOR_LAB2RGB)


def preprocess_to_pil(
    path: Path,
    *,
    apply_hair: bool,
    apply_clahe: bool,
) -> Image.Image:
    img_bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img_bgr is None:
        raise RuntimeError(f"Failed to read image: {path}")

    if apply_hair:
        img_bgr = hair_removal_bgr(img_bgr)

    if apply_clahe:
        rgb = clahe_rgb(img_bgr)
    else:
        rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    return Image.fromarray(rgb)


# ----------------------------
# Datasets
# ----------------------------
class PathDataset(Dataset):
    def __init__(
        self,
        items: List[Tuple[Path, int]],
        transform: transforms.Compose,
        *,
        p_hair: float,
        p_clahe: float,
    ):
        self.items = items
        self.transform = transform
        self.p_hair = float(p_hair)
        self.p_clahe = float(p_clahe)

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int):
        path, y = self.items[idx]
        apply_hair = (random.random() < self.p_hair)
        apply_clahe = (random.random() < self.p_clahe)
        img = preprocess_to_pil(path, apply_hair=apply_hair, apply_clahe=apply_clahe)
        x = self.transform(img)
        return x, y


def load_imagefolder_items(root: Path, class_names: List[str]) -> List[Tuple[Path, int]]:
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    class_to_idx = {c: i for i, c in enumerate(class_names)}
    items: List[Tuple[Path, int]] = []

    for cname in class_names:
        cdir = root / cname
        if not cdir.is_dir():
            raise FileNotFoundError(
                f"Missing class folder: {cdir}\n"
                f"Folder names must exactly match:\n{class_names}"
            )
        for p in cdir.rglob("*"):
            if p.is_file() and p.suffix.lower() in exts:
                items.append((p, class_to_idx[cname]))

    if not items:
        raise RuntimeError(f"No images found under: {root}")
    return items


def load_csv_items(
    csv_path: Path,
    images_dir: Path,
    class_names: List[str],
    *,
    img_col: str,
    label_col: str,
    ext: str,
    map_ham_dx: bool,
) -> List[Tuple[Path, int]]:
    df = pd.read_csv(csv_path)
    if img_col not in df.columns or label_col not in df.columns:
        raise ValueError(f"CSV missing columns. Need '{img_col}' and '{label_col}'.")

    class_to_idx = {c: i for i, c in enumerate(class_names)}

    items: List[Tuple[Path, int]] = []
    for _, row in df.iterrows():
        image_id = str(row[img_col]).strip()
        raw_label = str(row[label_col]).strip()

        cname = HAM_DX_TO_APP.get(raw_label, raw_label) if map_ham_dx else raw_label
        if cname not in class_to_idx:
            continue

        p = Path(image_id)
        if p.suffix:
            img_path = images_dir / p
        else:
            img_path = images_dir / f"{image_id}{ext}"

        if img_path.is_file():
            items.append((img_path, class_to_idx[cname]))

    if not items:
        raise RuntimeError("No usable items found from CSV. Check paths/columns.")
    return items


# ----------------------------
# Model
# ----------------------------
def build_model(num_classes: int, pretrained: bool) -> nn.Module:
    if pretrained:
        model = models.efficientnet_v2_s(weights=models.EfficientNet_V2_S_Weights.DEFAULT)
    else:
        model = models.efficientnet_v2_s(weights=None)

    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Sequential(
        nn.Dropout(p=0.4, inplace=True),
        nn.Linear(in_features, num_classes),
    )
    return model


# ----------------------------
# Metrics / helpers
# ----------------------------
@torch.no_grad()
def acc_topk(logits: torch.Tensor, y: torch.Tensor, k: int) -> float:
    k = min(k, logits.size(1))
    pred = torch.topk(logits, k=k, dim=1).indices
    correct = pred.eq(y.view(-1, 1)).any(dim=1).float().sum().item()
    return correct / y.size(0)


def class_weights_from_targets(targets: List[int], num_classes: int) -> torch.Tensor:
    counts = np.bincount(np.array(targets), minlength=num_classes).astype(np.float32)
    counts = np.maximum(counts, 1.0)
    weights = counts.sum() / (num_classes * counts)
    return torch.tensor(weights, dtype=torch.float32)


def make_weighted_sampler(targets: List[int], num_classes: int) -> WeightedRandomSampler:
    cw = class_weights_from_targets(targets, num_classes).numpy()
    sample_weights = [float(cw[t]) for t in targets]
    return WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)


# ----------------------------
# Train / eval
# ----------------------------
def train_one_epoch(model, loader, criterion, optimizer, device, scaler, amp):
    model.train()
    loss_sum = 0.0
    acc1_sum = 0.0
    acc3_sum = 0.0
    n = 0

    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = torch.as_tensor(y, dtype=torch.long, device=device)

        optimizer.zero_grad(set_to_none=True)

        if amp and scaler is not None:
            with torch.cuda.amp.autocast():
                logits = model(x)
                loss = criterion(logits, y)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

        bs = x.size(0)
        loss_sum += loss.item() * bs
        acc1_sum += acc_topk(logits.detach(), y, 1) * bs
        acc3_sum += acc_topk(logits.detach(), y, 3) * bs
        n += bs

    return loss_sum / n, acc1_sum / n, acc3_sum / n


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    loss_sum = 0.0
    acc1_sum = 0.0
    acc3_sum = 0.0
    n = 0

    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = torch.as_tensor(y, dtype=torch.long, device=device)

        logits = model(x)
        loss = criterion(logits, y)

        bs = x.size(0)
        loss_sum += loss.item() * bs
        acc1_sum += acc_topk(logits, y, 1) * bs
        acc3_sum += acc_topk(logits, y, 3) * bs
        n += bs

    return loss_sum / n, acc1_sum / n, acc3_sum / n


def main():
    ap = argparse.ArgumentParser(description="Train DermAssist classifier (EfficientNet-V2-S).")

    ap.add_argument("--mode", choices=["imagefolder", "csv"], default="imagefolder")
    ap.add_argument("--data-root", type=str, default="data",
                    help="ImageFolder root with train/ and val/ subfolders.")
    ap.add_argument("--csv-path", type=str, default="")
    ap.add_argument("--images-dir", type=str, default="")
    ap.add_argument("--img-col", type=str, default="image_id")
    ap.add_argument("--label-col", type=str, default="dx")
    ap.add_argument("--ext", type=str, default=".jpg")
    ap.add_argument("--map-ham-dx", action="store_true")

    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--img-size", type=int, default=224)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--pretrained", action="store_true")

    # preprocessing probabilities (robustness)
    ap.add_argument("--p-hair", type=float, default=1.0)
    ap.add_argument("--p-clahe", type=float, default=1.0)

    # imbalance handling
    ap.add_argument("--use-weighted-sampler", action="store_true")
    ap.add_argument("--use-class-weights", action="store_true")
    ap.add_argument("--label-smoothing", type=float, default=0.0)

    # runtime
    ap.add_argument("--amp", action="store_true")
    ap.add_argument("--patience", type=int, default=7)
    ap.add_argument("--out", type=str, default="models/skin_classifier.pth")
    ap.add_argument("--save-metadata", action="store_true")

    args = ap.parse_args()
    seed_everything(args.seed)
    torch.backends.cudnn.benchmark = True

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp = bool(args.amp and device.type == "cuda")
    scaler = torch.cuda.amp.GradScaler() if amp else None

    class_names = APP_CLASSES
    num_classes = len(class_names)

    # Transforms
    train_tf = transforms.Compose([
        transforms.RandomResizedCrop(args.img_size, scale=(0.75, 1.0), ratio=(0.9, 1.1)),
        transforms.RandomHorizontalFlip(0.5),
        transforms.RandomVerticalFlip(0.2),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.15),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])

    val_tf = transforms.Compose([
        transforms.Resize(int(args.img_size * 1.14)),
        transforms.CenterCrop(args.img_size),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])

    # Build datasets
    if args.mode == "imagefolder":
        root = Path(args.data_root)
        train_root = root / "train"
        val_root = root / "val"
        if not train_root.is_dir() or not val_root.is_dir():
            raise SystemExit("ImageFolder mode requires: data-root/train and data-root/val folders.")

        train_items = load_imagefolder_items(train_root, class_names)
        val_items = load_imagefolder_items(val_root, class_names)

    else:
        if not args.csv_path or not args.images_dir:
            raise SystemExit("CSV mode requires --csv-path and --images-dir")
        train_items = load_csv_items(Path(args.csv_path), Path(args.images_dir), class_names,
                                     img_col=args.img_col, label_col=args.label_col,
                                     ext=args.ext, map_ham_dx=args.map_ham_dx)
        random.shuffle(train_items)
        n_val = max(1, int(0.15 * len(train_items)))
        val_items = train_items[:n_val]
        train_items = train_items[n_val:]

    train_targets = [y for _, y in train_items]

    train_ds = PathDataset(train_items, train_tf, p_hair=args.p_hair, p_clahe=args.p_clahe)
    val_ds = PathDataset(val_items, val_tf, p_hair=1.0, p_clahe=1.0)

    # Sampling / loading
    if args.use_weighted_sampler:
        sampler = make_weighted_sampler(train_targets, num_classes)
        shuffle = False
    else:
        sampler = None
        shuffle = True

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=shuffle,
        sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=max(1, args.num_workers // 2),
        pin_memory=(device.type == "cuda"),
    )

    # Model / loss / optimizer / scheduler
    model = build_model(num_classes=num_classes, pretrained=args.pretrained).to(device)

    if args.use_class_weights:
        cw = class_weights_from_targets(train_targets, num_classes).to(device)
    else:
        cw = None
    criterion = nn.CrossEntropyLoss(weight=cw, label_smoothing=float(args.label_smoothing))

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.lr * 0.05)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    best_acc = -1.0
    best_epoch = -1
    no_improve = 0

    print(f"Device: {device} | AMP: {amp}")
    print(f"Train: {len(train_ds)} | Val: {len(val_ds)}")
    print(f"Saving best checkpoint to: {out_path}")

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()

        tr_loss, tr_acc1, tr_acc3 = train_one_epoch(model, train_loader, criterion, optimizer, device, scaler, amp)
        va_loss, va_acc1, va_acc3 = evaluate(model, val_loader, criterion, device)
        scheduler.step()

        dt = time.time() - t0
        lr_now = optimizer.param_groups[0]["lr"]

        print(
            f"Epoch {epoch:02d}/{args.epochs} | lr={lr_now:.2e} | "
            f"train loss={tr_loss:.4f} acc1={tr_acc1*100:.2f}% acc3={tr_acc3*100:.2f}% | "
            f"val loss={va_loss:.4f} acc1={va_acc1*100:.2f}% acc3={va_acc3*100:.2f}% | "
            f"time={dt:.1f}s"
        )

        if va_acc1 > best_acc:
            best_acc = va_acc1
            best_epoch = epoch
            no_improve = 0

            ckpt = {
                "state_dict": model.state_dict(),
                "class_names": class_names,
                "arch": "efficientnet_v2_s",
                "img_size": args.img_size,
                "imagenet_mean": list(IMAGENET_MEAN),
                "imagenet_std": list(IMAGENET_STD),
                "best_val_acc1": float(best_acc),
                "best_epoch": int(best_epoch),
                "pretrained": bool(args.pretrained),
                "p_hair": float(args.p_hair),
                "p_clahe": float(args.p_clahe),
            }
            torch.save(ckpt, str(out_path))
            print(f"✅ Saved best model (val acc1={best_acc*100:.2f}%)")
        else:
            no_improve += 1
            if no_improve >= args.patience:
                print(
                    f"⏹ Early stopping: no val acc improvement for {args.patience} epochs. "
                    f"Best {best_acc*100:.2f}% @ epoch {best_epoch}."
                )
                break

    if args.save_metadata:
        meta_path = out_path.with_name(out_path.stem + "_metadata.txt")
        meta_path.write_text(
            f"arch=efficientnet_v2_s\n"
            f"classes={class_names}\n"
            f"best_val_acc1={best_acc}\n"
            f"best_epoch={best_epoch}\n",
            encoding="utf-8",
        )
        print(f"🧾 Saved metadata: {meta_path}")


if __name__ == "__main__":
    main()
