from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torchvision import models
from pytorch_grad_cam import GradCAMPlusPlus
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image


DEFAULT_CLASSES = [
    "Actinic Keratoses",
    "Basal Cell Carcinoma",
    "Benign Keratosis-like Lesions",
    "Dermatofibroma",
    "Melanocytic Nevi",
    "Melanoma",
    "Vascular Lesions",
]


class DermatologyAI:
    """EfficientNet-V2-S classifier + Grad-CAM++ explainability."""

    def __init__(self, weights_path: str | Path | None = None):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Model architecture must match training
        self.model = models.efficientnet_v2_s(weights=None)
        num_ftrs = self.model.classifier[1].in_features
        self.model.classifier[1] = nn.Sequential(
            nn.Dropout(p=0.4, inplace=True),
            nn.Linear(num_ftrs, 7),
        )

        self.classes = list(DEFAULT_CLASSES)

        # Load weights (supports plain state_dict or checkpoint dict)
        weights_path = Path(weights_path) if weights_path else None
        if weights_path and weights_path.is_file():
            try:
                state = torch.load(str(weights_path), map_location=self.device)

                # If checkpoint dict, optionally pull class_names
                if isinstance(state, dict) and "class_names" in state and isinstance(state["class_names"], (list, tuple)):
                    if len(state["class_names"]) == 7:
                        self.classes = list(state["class_names"])

                if isinstance(state, dict) and "state_dict" in state:
                    sd = state["state_dict"]
                elif isinstance(state, dict) and "model" in state:
                    sd = state["model"]
                else:
                    sd = state  # assume plain state_dict

                incompatible = self.model.load_state_dict(sd, strict=False)
                missing = getattr(incompatible, "missing_keys", [])
                unexpected = getattr(incompatible, "unexpected_keys", [])
                if missing or unexpected:
                    print(f"⚠️  Weights loaded with mismatches. Missing={len(missing)} Unexpected={len(unexpected)}")
                else:
                    print("✅  Model weights loaded successfully.")
            except Exception:
                print(f"⚠️  Failed to load weights. Using random initialization.")
        else:
            print("ℹ️ No valid weights found. App will run in demo mode (random predictions).")

        self.model.to(self.device).eval()

        # Cache Grad-CAM instance (faster than recreating per call)
        self._target_layers = [self.model.features[-1]]
        self._cam = GradCAMPlusPlus(model=self.model, target_layers=self._target_layers)

    def analyze(self, tensor, rgb_norm, *, use_tta: bool = True, topk: int = 3):
        """Run inference and produce a Grad-CAM++ heatmap.

        Args:
            tensor: Preprocessed input tensor (1, 3, H, W).
            rgb_norm: Normalized RGB image in [0, 1] used for visualization.
            use_tta: If True, averages predictions across 3 simple augmentations.
            topk: Number of top predictions to return for the UI.

        Returns:
            top1_label: Best class label.
            top1_conf_percent: Best class probability in percent.
            topk_list: List of (class_label, prob_percent), length=min(topk, num_classes).
            heatmap_uint8: RGB heatmap image uint8.
        """
        img_v1 = tensor.to(self.device)

        with torch.no_grad():
            if use_tta:
                img_v2 = torch.flip(img_v1, [3])  # horizontal
                img_v3 = torch.flip(img_v1, [2])  # vertical
                logits = (self.model(img_v1) + self.model(img_v2) + self.model(img_v3)) / 3.0
            else:
                logits = self.model(img_v1)

            probs = torch.softmax(logits, dim=1)[0]
            k = max(1, min(int(topk), probs.numel()))
            topk_probs, topk_idxs = torch.topk(probs, k=k)

            class_idx = int(topk_idxs[0].item())
            confidence = float(topk_probs[0].item()) * 100.0

        topk_list = [
            (self.classes[int(i)], float(p) * 100.0)
            for p, i in zip(topk_probs.detach().cpu().numpy(), topk_idxs.detach().cpu().numpy())
        ]

        # Grad-CAM++ (on predicted class)
        targets = [ClassifierOutputTarget(class_idx)]
        mask = self._cam(input_tensor=img_v1, targets=targets)[0, :]

        heatmap = show_cam_on_image(rgb_norm, mask, use_rgb=True)
        heatmap_uint8 = (heatmap * 255).astype("uint8")

        return self.classes[class_idx], confidence, topk_list, heatmap_uint8
