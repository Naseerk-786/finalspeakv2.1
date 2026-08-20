# SignSpeak ISL Letter Pipeline — Step 2: Model Training & ONNX Export
# Trains a high-precision hand landmark classifier for ISL letters and exports to ONNX.

import os
import sys
import json
import time
import random
import numpy as np
from pathlib import Path
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import onnxruntime as ort

# ═══════════════════════════════════════════════════════════════
# Random Seed & Reproducibility
# ═══════════════════════════════════════════════════════════════
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

# ═══════════════════════════════════════════════════════════════
# PyTorch Dataset
# ═══════════════════════════════════════════════════════════════
class ISLLetterDataset(Dataset):
    def __init__(self, features, labels, augment=False):
        self.features = features
        self.labels = labels
        self.augment = augment

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        feat = self.features[idx].copy()
        label = self.labels[idx]

        if self.augment:
            # Random gaussian landmark jitter
            if np.random.rand() < 0.3:
                feat += np.random.normal(0, 0.01, size=feat.shape).astype(np.float32)

        return torch.from_numpy(feat), torch.tensor(label, dtype=torch.long)

# ═══════════════════════════════════════════════════════════════
# High-Accuracy Hand Landmark MLP Architecture
# ═══════════════════════════════════════════════════════════════
class ISLLetterClassifier(nn.Module):
    """Deep Multi-Layer Perceptron with Residual Connections for Hand Landmarks."""

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
        # x shape: [Batch, 126]
        h1 = self.layer1(x)
        h2 = self.layer2(h1) + h1  # Residual skip connection!
        h3 = self.layer3(h2)
        logits = self.classifier(h3)
        return logits

# ═══════════════════════════════════════════════════════════════
# Training Script
# ═══════════════════════════════════════════════════════════════
def main():
    print("=========================================================")
    print("     SignSpeak - ISL Letter Model Training (PyTorch -> ONNX)")
    print("=========================================================\n")

    set_seed(42)
    BASE_DIR = Path(r"d:\finalspeak")
    DATA_DIR = BASE_DIR / "data"
    PROCESSED_DIR = DATA_DIR / "processed"
    MODELS_DIR = BASE_DIR / "models"
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    dataset_path = PROCESSED_DIR / "isl_letters_dataset.npz"
    meta_path = PROCESSED_DIR / "isl_letters_meta.json"

    if not dataset_path.exists():
        print(f"[!] Dataset file not found at {dataset_path}. Please run dataset harvester first.")
        return

    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    num_classes = len(meta["class2idx"])
    idx2class = meta["idx2class"]

    data = np.load(dataset_path)
    X = data["X"]
    y = data["y"]

    print(f"[OK] Loaded Dataset: {X.shape[0]} samples, {X.shape[1]}-dim features across {num_classes} classes.")

    # Stratified Train/Val/Test Split (80% / 10% / 10%)
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp)

    print(f"[OK] Split Sizes -> Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")

    batch_size = 64
    train_ds = ISLLetterDataset(X_train, y_train, augment=True)
    val_ds = ISLLetterDataset(X_val, y_val, augment=False)
    test_ds = ISLLetterDataset(X_test, y_test, augment=False)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, pin_memory=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[OK] Target Training Device: {device}")

    model = ISLLetterClassifier(in_features=X.shape[1], num_classes=num_classes).to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
    epochs = 200
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)

    best_val_acc = 0.0
    best_pth_path = MODELS_DIR / "isl_letter_classifier.pth"

    print("\n--- Starting Training ---")
    start_time = time.time()
    for epoch in range(1, epochs + 1):
        model.train()
        train_loss, train_correct = 0.0, 0
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * inputs.size(0)
            preds = outputs.argmax(dim=1)
            train_correct += (preds == targets).sum().item()

        scheduler.step()

        train_loss /= len(train_ds)
        train_acc = (train_correct / len(train_ds)) * 100.0

        # Validation
        model.eval()
        val_loss, val_correct = 0.0, 0
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)

                val_loss += loss.item() * inputs.size(0)
                preds = outputs.argmax(dim=1)
                val_correct += (preds == targets).sum().item()

        val_loss /= len(val_ds)
        val_acc = (val_correct / len(val_ds)) * 100.0

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), best_pth_path)

        if epoch % 5 == 0 or epoch == epochs:
            print(f"Epoch {epoch:02d}/{epochs:02d} | Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}% || Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%", flush=True)

    training_time = time.time() - start_time
    print(f"\n[OK] Training completed in {training_time:.2f} seconds!")
    print(f"[OK] Best Validation Accuracy: {best_val_acc:.2f}%")

    # Evaluation on Test Set
    print("\n--- Evaluating Best Model on Test Set ---")
    model.load_state_dict(torch.load(best_pth_path, weights_only=True))
    model.eval()
    test_correct = 0
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            preds = outputs.argmax(dim=1)
            test_correct += (preds == targets).sum().item()

    test_acc = (test_correct / len(test_ds)) * 100.0
    print(f"[RESULTS] Test Set Top-1 Accuracy: {test_acc:.2f}%")

    # Export to ONNX
    print("\n--- Exporting Model to ONNX Binary ---")
    onnx_path = MODELS_DIR / "isl_letter_classifier.onnx"
    dummy_input = torch.randn(1, X.shape[1], device=device)

    torch.onnx.export(
        model,
        dummy_input,
        onnx_path,
        export_params=True,
        opset_version=14,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}}
    )
    print(f"[OK] Saved ONNX Model: {onnx_path}")

    # Copy metadata class mapping to models dir
    models_meta_path = MODELS_DIR / "isl_letter_meta.json"
    with open(models_meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    print(f"[OK] Saved Class Mapping Meta: {models_meta_path}")

    print("\n=========================================================")
    print("      ISL LETTER MODEL TRAINING & ONNX EXPORT SUCCESS!   ")
    print("=========================================================")

if __name__ == "__main__":
    main()
