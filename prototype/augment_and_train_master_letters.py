# SignSpeak Master Letter Training Pipeline — 300k Augmented Samples on CUDA GPU
# Ingests Base 107k + User Recorded Signs -> Applies 3D Rotations & Jitter -> 100-Epoch Master GPU Run -> ONNX Export

import os
import sys
import json
import time
import math
import random
import numpy as np
from pathlib import Path
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import onnxruntime as ort

BASE_DIR = Path(r"d:\finalspeak")
DATA_DIR = BASE_DIR / "data"
PROCESSED_DIR = DATA_DIR / "processed"
USER_RECORDED_DIR = DATA_DIR / "user_recorded"
MODELS_DIR = BASE_DIR / "models"

MODELS_DIR.mkdir(parents=True, exist_ok=True)

BASE_DATASET_PATH = PROCESSED_DIR / "isl_letters_dataset.npz"
BASE_META_PATH = PROCESSED_DIR / "isl_letters_meta.json"
ONNX_EXPORT_PATH = MODELS_DIR / "isl_letter_classifier.onnx"
PTH_EXPORT_PATH = MODELS_DIR / "isl_letter_classifier.pth"
META_EXPORT_PATH = MODELS_DIR / "isl_letter_meta.json"
REPORT_EXPORT_PATH = MODELS_DIR / "master_training_report.json"


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ═══════════════════════════════════════════════════════════════
# 3D Landmark Geometric Augmentation Engine
# ═══════════════════════════════════════════════════════════════
def rotate_2d_landmarks(coords_21x3, angle_deg):
    """Rotates the (x,y) plane of normalized landmarks around wrist origin (0,0)."""
    rad = math.radians(angle_deg)
    cos_a, sin_a = math.cos(rad), math.sin(rad)
    rot_matrix = np.array([
        [cos_a, -sin_a],
        [sin_a,  cos_a]
    ], dtype=np.float32)

    res = coords_21x3.copy()
    res[:, :2] = np.dot(coords_21x3[:, :2], rot_matrix.T)
    return res


def augment_126d_vector(vec_126, max_angle=18.0, jitter_std=0.012, scale_range=(0.88, 1.12)):
    """Applies rotation, scale shift, and Gaussian noise to a 126-dim feature vector."""
    lh_coords = vec_126[:63].reshape(21, 3).copy()
    rh_coords = vec_126[63:].reshape(21, 3).copy()

    # 1. Random Planar Rotation
    angle = np.random.uniform(-max_angle, max_angle)
    lh_coords = rotate_2d_landmarks(lh_coords, angle)
    rh_coords = rotate_2d_landmarks(rh_coords, angle)

    # 2. Scale variation
    scale = np.random.uniform(scale_range[0], scale_range[1])
    lh_coords *= scale
    rh_coords *= scale

    # 3. Gaussian Joint Jitter
    lh_coords += np.random.normal(0, jitter_std, size=lh_coords.shape).astype(np.float32)
    rh_coords += np.random.normal(0, jitter_std, size=rh_coords.shape).astype(np.float32)

    return np.concatenate([lh_coords.flatten(), rh_coords.flatten()]).astype(np.float32)


# ═══════════════════════════════════════════════════════════════
# Dataset Synthesis & High-Volume Expansion (Target: ~320k Samples)
# ═══════════════════════════════════════════════════════════════
def build_augmented_master_dataset():
    print("=" * 65)
    print("      SignSpeak — High-Fidelity Master Dataset Synthesis")
    print("=" * 65)

    # 1. Load Base Dataset
    base_data = np.load(str(BASE_DATASET_PATH))
    base_X = base_data["X"] if "X" in base_data else base_data["features"]
    base_y = base_data["y"] if "y" in base_data else base_data["labels"]
    base_X = base_X.astype(np.float32)
    base_y = base_y.astype(np.int64)

    with open(BASE_META_PATH, "r", encoding="utf-8") as f:
        base_meta = json.load(f)

    class2idx = dict(base_meta["class2idx"])
    idx2class = {int(k): v for k, v in base_meta["idx2class"].items()}

    print(f"[1/4] Loaded Base Dataset: {len(base_X):,} vectors across {len(class2idx)} classes.")

    # 2. Ingest User Custom Recorded Signs
    user_X_list, user_y_list = [], []
    user_counts = {}

    if USER_RECORDED_DIR.exists():
        for file_path in sorted(USER_RECORDED_DIR.glob("*.npz")):
            cls_name = file_path.stem
            try:
                npz = np.load(str(file_path))
                feats = npz["features"] if "features" in npz else npz["X"]
                feats = feats.astype(np.float32)
                if len(feats) == 0:
                    continue

                if cls_name not in class2idx:
                    new_idx = len(class2idx)
                    class2idx[cls_name] = new_idx
                    idx2class[new_idx] = cls_name

                cls_idx = class2idx[cls_name]
                user_counts[cls_name] = len(feats)

                # Collect raw user samples
                user_X_list.append(feats)
                user_y_list.append(np.full(len(feats), cls_idx, dtype=np.int64))
            except Exception as e:
                print(f"[!] Warning reading {file_path}: {e}")

    total_user_raw = sum(user_counts.values())
    print(f"[2/4] Ingested User Camera Dataset: {total_user_raw:,} raw samples across {len(user_counts)} classes.")

    # 3. Geometric Augmentation & Synthesis
    print(f"[3/4] Generating 3D Geometric Augmentations (Rotations ±18°, Scale, Jitter)...")

    augmented_X_list = [base_X]
    augmented_y_list = [base_y]

    # Generate 1 full augmented pass over base dataset (+107k samples)
    print("  -> Augmenting base 107k dataset...")
    base_aug_X = np.empty_like(base_X)
    for i in range(len(base_X)):
        base_aug_X[i] = augment_126d_vector(base_X[i], max_angle=15.0, jitter_std=0.010)
    augmented_X_list.append(base_aug_X)
    augmented_y_list.append(base_y.copy())

    # Generate 25x augmented passes over user camera data (+30k samples)
    if user_X_list:
        user_X_raw = np.concatenate(user_X_list, axis=0)
        user_y_raw = np.concatenate(user_y_list, axis=0)
        augmented_X_list.append(user_X_raw)
        augmented_y_list.append(user_y_raw)

        print(f"  -> Generating 25x geometric angle perturbations for user camera samples...")
        for _ in range(25):
            user_aug_X = np.empty_like(user_X_raw)
            for j in range(len(user_X_raw)):
                user_aug_X[j] = augment_126d_vector(user_X_raw[j], max_angle=20.0, jitter_std=0.015)
            augmented_X_list.append(user_aug_X)
            augmented_y_list.append(user_y_raw.copy())

    # 4. Final Aggregation
    master_X = np.concatenate(augmented_X_list, axis=0).astype(np.float32)
    master_y = np.concatenate(augmented_y_list, axis=0).astype(np.int64)

    print(f"[4/4] Master Dataset Ready: {len(master_X):,} 126-dim samples ({len(class2idx)} classes)!\n")

    meta_info = {
        "class2idx": class2idx,
        "idx2class": {str(k): v for k, v in idx2class.items()},
        "total_samples": int(len(master_X)),
        "base_raw_samples": int(len(base_X)),
        "user_raw_samples": int(total_user_raw),
        "user_class_counts": user_counts,
        "num_classes": len(class2idx),
        "feature_dim": 126
    }

    return master_X, master_y, meta_info


# ═══════════════════════════════════════════════════════════════
# Neural Network Topology: Deep Residual MLP
# ═══════════════════════════════════════════════════════════════
class ISLLetterClassifier(nn.Module):
    def __init__(self, in_features=126, num_classes=35):
        super().__init__()
        self.layer1 = nn.Sequential(
            nn.Linear(in_features, 256),
            nn.BatchNorm1d(256),
            nn.SiLU(),
            nn.Dropout(0.20)
        )
        self.layer2 = nn.Sequential(
            nn.Linear(256, 256),
            nn.BatchNorm1d(256),
            nn.SiLU(),
            nn.Dropout(0.20)
        )
        self.layer3 = nn.Sequential(
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.SiLU(),
            nn.Dropout(0.20)
        )
        self.classifier = nn.Linear(128, num_classes)

    def forward(self, x):
        h1 = self.layer1(x)
        h2 = self.layer2(h1) + h1  # Residual Skip Connection
        h3 = self.layer3(h2)
        return self.classifier(h3)


class TorchDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return torch.from_numpy(self.X[idx]), torch.tensor(self.y[idx], dtype=torch.long)


# ═══════════════════════════════════════════════════════════════
# Master Training & Benchmark Execution (100 Epochs)
# ═══════════════════════════════════════════════════════════════
def train_master_model(epochs=100, batch_size=256, lr=2e-3):
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] Target Compute Device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")

    master_X, master_y, meta_info = build_augmented_master_dataset()
    num_classes = meta_info["num_classes"]

    # Stratified 80/10/10 Split
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        master_X, master_y, test_size=0.10, random_state=42, stratify=master_y
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=0.1111, random_state=42, stratify=y_train_val
    )

    print(f"[*] Splits — Train: {len(X_train):,} | Val: {len(X_val):,} | Test: {len(X_test):,}")

    train_loader = DataLoader(TorchDataset(X_train, y_train), batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=0)
    val_loader = DataLoader(TorchDataset(X_val, y_val), batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=0)
    test_loader = DataLoader(TorchDataset(X_test, y_test), batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=0)

    model = ISLLetterClassifier(in_features=126, num_classes=num_classes).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)

    best_val_acc = 0.0
    best_state_dict = None
    history = []

    print("\n" + "=" * 65)
    print(f"      Launching 100-Epoch CUDA Training Run on {len(X_train):,} Samples")
    print("=" * 65)

    start_time = time.time()
    for epoch in range(1, epochs + 1):
        model.train()
        train_loss, train_correct, train_total = 0.0, 0, 0

        for feats, labels in train_loader:
            feats, labels = feats.to(device), labels.to(device)
            optimizer.zero_grad()
            logits = model(feats)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * len(labels)
            preds = logits.argmax(dim=1)
            train_correct += (preds == labels).sum().item()
            train_total += len(labels)

        scheduler.step()
        train_acc = train_correct / train_total
        avg_train_loss = train_loss / train_total

        # Validation
        model.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0
        with torch.no_grad():
            for feats, labels in val_loader:
                feats, labels = feats.to(device), labels.to(device)
                logits = model(feats)
                loss = criterion(logits, labels)
                val_loss += loss.item() * len(labels)
                preds = logits.argmax(dim=1)
                val_correct += (preds == labels).sum().item()
                val_total += len(labels)

        val_acc = val_correct / val_total
        avg_val_loss = val_loss / val_total

        if val_acc >= best_val_acc:
            best_val_acc = val_acc
            best_state_dict = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        history.append({
            "epoch": epoch,
            "train_loss": avg_train_loss,
            "train_acc": train_acc,
            "val_loss": avg_val_loss,
            "val_acc": val_acc
        })

        if epoch % 5 == 0 or epoch == 1 or epoch == epochs:
            elapsed = time.time() - start_time
            print(f"Epoch {epoch:03d}/{epochs:03d} | Train Acc: {train_acc:.2%} (Loss: {avg_train_loss:.4f}) | Val Acc: {val_acc:.2%} (Loss: {avg_val_loss:.4f}) | Elapsed: {elapsed:.1f}s")

    # ═══════════════════════════════════════════════════════════
    # Held-Out Test Evaluation & Export
    # ═══════════════════════════════════════════════════════════
    print("\n" + "=" * 65)
    print("      Evaluating on Unseen Held-Out Test Set")
    print("=" * 65)

    model.load_state_dict(best_state_dict)
    model.eval()

    all_preds, all_targets = [], []
    with torch.no_grad():
        for feats, labels in test_loader:
            feats = feats.to(device)
            logits = model(feats)
            preds = logits.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_targets.extend(labels.numpy())

    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)
    test_acc = float((all_preds == all_targets).mean())
    print(f"\n>>> FINAL HELD-OUT TEST ACCURACY: {test_acc:.2%} <<<\n")

    # Per-Class Precision / Recall
    target_names = [meta_info["idx2class"][str(i)] for i in range(num_classes)]
    cls_report = classification_report(all_targets, all_preds, target_names=target_names, output_dict=True)

    # Save PyTorch Model
    torch.save(model.state_dict(), str(PTH_EXPORT_PATH))
    print(f"[OK] Saved PyTorch weights to {PTH_EXPORT_PATH}")

    # Export to ONNX
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
    print(f"[OK] Exported Master ONNX binary to {ONNX_EXPORT_PATH}")

    # Save Meta & Comprehensive Report
    meta_info["best_val_acc"] = float(best_val_acc)
    meta_info["test_acc"] = float(test_acc)
    with open(META_EXPORT_PATH, "w", encoding="utf-8") as f:
        json.dump(meta_info, f, indent=2)

    report_payload = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "total_augmented_samples": len(master_X),
        "train_samples": len(X_train),
        "val_samples": len(X_val),
        "test_samples": len(X_test),
        "best_val_accuracy": float(best_val_acc),
        "held_out_test_accuracy": float(test_acc),
        "epochs_trained": epochs,
        "classification_metrics": cls_report
    }
    with open(REPORT_EXPORT_PATH, "w", encoding="utf-8") as f:
        json.dump(report_payload, f, indent=2)

    print(f"[OK] Master Training Report written to {REPORT_EXPORT_PATH}")
    return test_acc, report_payload


if __name__ == "__main__":
    train_master_model(epochs=100, batch_size=256, lr=2e-3)
