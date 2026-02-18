from __future__ import annotations

import argparse
import os
import shutil
from collections import Counter
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split
from tqdm import tqdm


APP_CLASSES = [
    "Actinic Keratoses",
    "Basal Cell Carcinoma",
    "Benign Keratosis-like Lesions",
    "Dermatofibroma",
    "Melanocytic Nevi",
    "Melanoma",
    "Vascular Lesions",
]

HAM_DX_TO_APP = {
    "akiec": "Actinic Keratoses",
    "bcc": "Basal Cell Carcinoma",
    "bkl": "Benign Keratosis-like Lesions",
    "df": "Dermatofibroma",
    "nv": "Melanocytic Nevi",
    "mel": "Melanoma",
    "vasc": "Vascular Lesions",
}


def build_image_index(images_root: Path) -> dict[str, Path]:
    """Map image stem -> file path for fast lookup."""
    exts = {".jpg", ".jpeg", ".png"}
    index: dict[str, Path] = {}
    for p in images_root.rglob("*"):
        if p.is_file() and p.suffix.lower() in exts:
            index[p.stem] = p
    return index


def majority_label(labels: list[str]) -> str:
    return Counter(labels).most_common(1)[0][0]


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def safe_copy(src: Path, dst: Path, mode: str) -> None:
    ensure_dir(dst.parent)
    if dst.exists():
        return

    if mode == "copy":
        shutil.copy2(src, dst)
        return

    if mode == "hardlink":
        try:
            os.link(src, dst)
            return
        except Exception:
            shutil.copy2(src, dst)
            return

    if mode == "symlink":
        try:
            os.symlink(src, dst)
            return
        except Exception:
            shutil.copy2(src, dst)
            return

    raise ValueError(f"Unknown mode: {mode}")


def main():
    ap = argparse.ArgumentParser(description="Prepare HAM10000 into ImageFolder train/val split (lesion_id grouped).")
    ap.add_argument("--meta", required=True, help="Path to HAM10000_metadata.csv")
    ap.add_argument("--images", required=True, help="Folder containing all images (e.g., ham10000/images)")
    ap.add_argument("--out", default="data", help="Output root folder (default: data)")
    ap.add_argument("--val-ratio", type=float, default=0.15, help="Validation ratio (default: 0.15)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--mode", choices=["copy", "hardlink", "symlink"], default="copy",
                    help="How to write files into output (copy safest; hardlink saves disk).")
    args = ap.parse_args()

    meta_path = Path(args.meta)
    images_root = Path(args.images)
    out_root = Path(args.out)

    if not meta_path.is_file():
        raise FileNotFoundError(meta_path)
    if not images_root.is_dir():
        raise FileNotFoundError(images_root)

    df = pd.read_csv(meta_path)
    required = {"image_id", "lesion_id", "dx"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Metadata missing columns: {missing}. Found: {list(df.columns)}")

    # Map dx codes to app classes; filter unknown
    df["class_name"] = df["dx"].map(HAM_DX_TO_APP)
    df = df.dropna(subset=["class_name"]).reset_index(drop=True)

    # Index images
    idx = build_image_index(images_root)

    # Keep only rows where image exists
    df = df[df["image_id"].astype(str).apply(lambda s: str(s) in idx)].reset_index(drop=True)
    if len(df) == 0:
        raise RuntimeError("No images matched metadata. Check your --images path and extraction.")

    # Group split by lesion_id (prevents leakage)
    lesion_groups = df.groupby("lesion_id")["dx"].apply(list).reset_index()
    lesion_groups["group_dx"] = lesion_groups["dx"].apply(majority_label)
    lesion_groups["group_class"] = lesion_groups["group_dx"].map(HAM_DX_TO_APP)

    lesions = lesion_groups["lesion_id"].tolist()
    strat = lesion_groups["group_class"].tolist()

    train_lesions, val_lesions = train_test_split(
        lesions,
        test_size=float(args.val_ratio),
        random_state=int(args.seed),
        stratify=strat,
    )

    train_set = set(train_lesions)
    val_set = set(val_lesions)

    # Prepare output dirs
    train_root = out_root / "train"
    val_root = out_root / "val"
    for c in APP_CLASSES:
        ensure_dir(train_root / c)
        ensure_dir(val_root / c)

    n_train = 0
    n_val = 0

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Writing ImageFolder"):
        image_id = str(row["image_id"])
        lesion_id = row["lesion_id"]
        cls = row["class_name"]

        src = idx[image_id]
        dst_dir = train_root / cls if lesion_id in train_set else val_root / cls
        dst = dst_dir / src.name

        safe_copy(src, dst, args.mode)

        if lesion_id in train_set:
            n_train += 1
        else:
            n_val += 1

    print("✅ Done.")
    print(f"Train images: {n_train}")
    print(f"Val images:   {n_val}")
    print(f"Output: {out_root.resolve()}")
    print("Classes:", APP_CLASSES)


if __name__ == "__main__":
    main()
