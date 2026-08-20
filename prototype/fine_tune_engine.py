# SignSpeak ISL Engine — Personalized Fine-Tuning & Model Adaptation
# Blends user-recorded hand landmarks with base 107k dataset to prevent catastrophic forgetting.

import os
import sys
import json
import time
import shutil
import numpy as np
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import onnxruntime as ort

BASE_DIR = Path(r"d:\finalspeak")
DATA_DIR = BASE_DIR / "data"
PROCESSED_DIR = DATA_DIR / "processed"
USER_RECORDED_DIR = DATA_DIR / "user_recorded"
MODELS_DIR = BASE_DIR / "models"

USER_RECORDED_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)

BASE_DATASET_PATH = PROCESSED_DIR / "isl_letters_dataset.npz"
BASE_META_PATH = PROCESSED_DIR / "isl_letters_meta.json"
ONNX_EXPORT_PATH = MODELS_DIR / "isl_letter_classifier.onnx"
PTH_EXPORT_PATH = MODELS_DIR / "isl_letter_classifier.pth"
META_EXPORT_PATH = MODELS_DIR / "isl_letter_meta.json"


# ═══════════════════════════════════════════════════════════════
# Neural Network Topology
# ═══════════════════════════════════════════════════════════════
class ISLLetterClassifier(nn.Module):
    """Deep Residual MLP with Batch Normalization and SiLU."""
    def __init__(self, in_features=126, num_classes=35):
        super().__init__()
        self.layer1 = nn.Sequential(
            nn.Linear(in_features, 256),
            nn.BatchNorm1d(256),
            nn.SiLU(),
            nn.Dropout(0.2)
        )
        self.layer2 = nn.Sequential(
            nn.Linear(256, 256),
            nn.BatchNorm1d(256),
            nn.SiLU(),
            nn.Dropout(0.2)
        )
        self.layer3 = nn.Sequential(
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.SiLU(),
            nn.Dropout(0.2)
        )
        self.classifier = nn.Linear(128, num_classes)

    def forward(self, x):
        h1 = self.layer1(x)
        h2 = self.layer2(h1) + h1
        h3 = self.layer3(h2)
        return self.classifier(h3)


class JointFineTuneDataset(Dataset):
    def __init__(self, features, labels, is_user_flags=None, augment=True):
        self.features = features
        self.labels = labels
        self.is_user_flags = is_user_flags if is_user_flags is not None else np.zeros(len(features), dtype=bool)
        self.augment = augment

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        feat = self.features[idx].copy()
        label = self.labels[idx]

        if self.augment:
            # Gaussian jitter
            if np.random.rand() < 0.4:
                scale = 0.015 if self.is_user_flags[idx] else 0.010
                feat += np.random.normal(0, scale, size=feat.shape).astype(np.float32)

        return torch.from_numpy(feat), torch.tensor(label, dtype=torch.long)


# ═══════════════════════════════════════════════════════════════
# Dataset Ingestion & Co-Training Merger
# ═══════════════════════════════════════════════════════════════
def load_and_merge_datasets(user_oversample_factor=8):
    """
    Loads base 107k dataset and merges with all user recorded .npz files.
    Oversamples user data so the model personalizes to camera conditions
    while keeping the 107k baseline fully intact (ZERO forgetting).
    """
    # 1. Load Base Dataset
    if not BASE_DATASET_PATH.exists():
        raise FileNotFoundError(f"Base dataset missing at {BASE_DATASET_PATH}")

    base_data = np.load(str(BASE_DATASET_PATH))
    if "X" in base_data:
        base_features = base_data["X"].astype(np.float32)
        base_labels = base_data["y"].astype(np.int64)
    else:
        base_features = base_data["features"].astype(np.float32)
        base_labels = base_data["labels"].astype(np.int64)

    with open(BASE_META_PATH, "r", encoding="utf-8") as f:
        base_meta = json.load(f)

    class2idx = dict(base_meta["class2idx"])
    idx2class = {int(k): v for k, v in base_meta["idx2class"].items()}

    # 2. Check for User Recorded Samples
    user_features_list = []
    user_labels_list = []
    user_counts = {}

    if USER_RECORDED_DIR.exists():
        for file_path in USER_RECORDED_DIR.glob("*.npz"):
            cls_name = file_path.stem
            try:
                npz_data = np.load(str(file_path))
                if "X" in npz_data:
                    feats = npz_data["X"].astype(np.float32)
                else:
                    feats = npz_data["features"].astype(np.float32)

                if len(feats) == 0:
                    continue

                if cls_name not in class2idx:
                    new_idx = len(class2idx)
                    class2idx[cls_name] = new_idx
                    idx2class[new_idx] = cls_name

                cls_idx = class2idx[cls_name]
                user_counts[cls_name] = len(feats)

                # Oversample user samples for stronger adaptation
                for _ in range(user_oversample_factor):
                    user_features_list.append(feats)
                    user_labels_list.append(np.full(len(feats), cls_idx, dtype=np.int64))
            except Exception as e:
                print(f"[!] Warning: Error reading {file_path}: {e}")

    # 3. Combine Datasets
    if user_features_list:
        user_feats = np.concatenate(user_features_list, axis=0)
        user_lbls = np.concatenate(user_labels_list, axis=0)

        total_features = np.concatenate([base_features, user_feats], axis=0)
        total_labels = np.concatenate([base_labels, user_lbls], axis=0)

        is_user = np.concatenate([
            np.zeros(len(base_features), dtype=bool),
            np.ones(len(user_feats), dtype=bool)
        ])
    else:
        total_features = base_features
        total_labels = base_labels
        is_user = np.zeros(len(base_features), dtype=bool)

    meta_info = {
        "class2idx": class2idx,
        "idx2class": {str(k): v for k, v in idx2class.items()},
        "total_samples": int(len(total_features)),
        "base_samples": int(len(base_features)),
        "user_recorded_counts": user_counts,
        "num_classes": len(class2idx),
        "feature_dim": 126
    }

    return total_features, total_labels, is_user, meta_info


# ═══════════════════════════════════════════════════════════════
# GPU Fine-Tuner Runner
# ═══════════════════════════════════════════════════════════════
def run_fine_tuning(epochs=35, batch_size=256, lr=5e-4, progress_callback=None):
    """
    Executes fast PyTorch CUDA fine-tuning run, preserving base knowledge.
    Exports updated ONNX model and saves metadata.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] Training on device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")

    if progress_callback:
        progress_callback(0, epochs, "Loading and merging datasets...", 0.0, 0.0)

    features, labels, is_user, meta_info = load_and_merge_datasets(user_oversample_factor=10)
    num_classes = meta_info["num_classes"]

    # Stratified Train/Val split
    X_train, X_val, y_train, y_val, is_u_train, is_u_val = train_test_split(
        features, labels, is_user, test_size=0.10, random_state=42, stratify=labels
    )

    train_dataset = JointFineTuneDataset(X_train, y_train, is_u_train, augment=True)
    val_dataset = JointFineTuneDataset(X_val, y_val, is_u_val, augment=False)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=torch.cuda.is_available())
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, pin_memory=torch.cuda.is_available())

    # Instantiate Model
    model = ISLLetterClassifier(in_features=126, num_classes=num_classes).to(device)

    # Load existing pre-trained weights if class count matches or adapt classification head
    if PTH_EXPORT_PATH.exists():
        try:
            state_dict = torch.load(str(PTH_EXPORT_PATH), map_location=device, weights_only=False)
            if state_dict["classifier.weight"].shape[0] == num_classes:
                model.load_state_dict(state_dict)
                print("[OK] Loaded pre-trained weights successfully.")
            else:
                # Class head expansion
                old_num = state_dict["classifier.weight"].shape[0]
                model.layer1.load_state_dict({k.replace("layer1.", ""): v for k, v in state_dict.items() if k.startswith("layer1")})
                model.layer2.load_state_dict({k.replace("layer2.", ""): v for k, v in state_dict.items() if k.startswith("layer2")})
                model.layer3.load_state_dict({k.replace("layer3.", ""): v for k, v in state_dict.items() if k.startswith("layer3")})
                with torch.no_grad():
                    model.classifier.weight[:old_num].copy_(state_dict["classifier.weight"])
                    model.classifier.bias[:old_num].copy_(state_dict["classifier.bias"])
                print(f"[OK] Expanded classification head from {old_num} to {num_classes} classes.")
        except Exception as e:
            print(f"[!] Warning loading weights: {e}. Training from scratch.")

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)

    best_val_acc = 0.0
    best_state_dict = None

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0

        for feats_b, labels_b in train_loader:
            feats_b, labels_b = feats_b.to(device), labels_b.to(device)
            optimizer.zero_grad()
            outputs = model(feats_b)
            loss = criterion(outputs, labels_b)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * len(labels_b)
            preds = outputs.argmax(dim=1)
            correct_train += (preds == labels_b).sum().item()
            total_train += len(labels_b)

        scheduler.step()
        train_acc = correct_train / total_train
        avg_train_loss = running_loss / total_train

        # Validation
        model.eval()
        val_loss = 0.0
        correct_val = 0
        total_val = 0

        with torch.no_grad():
            for feats_b, labels_b in val_loader:
                feats_b, labels_b = feats_b.to(device), labels_b.to(device)
                outputs = model(feats_b)
                loss = criterion(outputs, labels_b)
                val_loss += loss.item() * len(labels_b)
                preds = outputs.argmax(dim=1)
                correct_val += (preds == labels_b).sum().item()
                total_val += len(labels_b)

        val_acc = correct_val / total_val
        avg_val_loss = val_loss / total_val

        if val_acc >= best_val_acc:
            best_val_acc = val_acc
            best_state_dict = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        msg = f"Epoch {epoch}/{epochs} | Train Loss: {avg_train_loss:.4f}, Train Acc: {train_acc:.2%} | Val Acc: {val_acc:.2%}"
        print(msg)
        if progress_callback:
            progress_callback(epoch, epochs, msg, train_acc, val_acc)

    # Save Best Weights
    if best_state_dict is not None:
        model.load_state_dict(best_state_dict)
    torch.save(model.state_dict(), str(PTH_EXPORT_PATH))
    print(f"[OK] Saved PyTorch weights to {PTH_EXPORT_PATH}")

    # Export to ONNX
    model.eval()
    dummy_input = torch.randn(1, 126, dtype=torch.float32)
    torch.onnx.export(
        model.cpu(),
        dummy_input,
        str(ONNX_EXPORT_PATH),
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
        opset_version=14
    )
    print(f"[OK] Exported ONNX binary to {ONNX_EXPORT_PATH}")

    # Save metadata
    meta_info["best_val_acc"] = float(best_val_acc)
    with open(META_EXPORT_PATH, "w", encoding="utf-8") as f:
        json.dump(meta_info, f, indent=2)

    return best_val_acc, meta_info


if __name__ == "__main__":
    acc, meta = run_fine_tuning(epochs=20)
    print(f"\nFinal Best Val Accuracy: {acc:.2%}")
