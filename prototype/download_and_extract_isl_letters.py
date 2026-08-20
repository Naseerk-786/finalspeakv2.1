# SignSpeak ISL & Alphabet Pipeline — Multi-Core Landmark Harvester across ISL + ASL Datasets
# Multi-Process Parallelized Landmark Extractor across 12 CPU Workers.

import os
import sys
import json
import shutil
import zipfile
import urllib.request
import numpy as np
from pathlib import Path
import cv2
import mediapipe as mp
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
import kagglehub

# ═══════════════════════════════════════════════════════════════
# Paths & Setup
# ═══════════════════════════════════════════════════════════════
BASE_DIR = Path(r"d:\finalspeak")
DATA_DIR = BASE_DIR / "data"
RAW_DIR = DATA_DIR / "raw" / "isl_alphabets"
PROCESSED_DIR = DATA_DIR / "processed"

RAW_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

VALID_CLASSES = [chr(c) for c in range(ord('A'), ord('Z')+1)] + [str(i) for i in range(1, 10)]
CLASS2IDX = {cls_name: i for i, cls_name in enumerate(VALID_CLASSES)}
IDX2CLASS = {i: cls_name for i, cls_name in enumerate(VALID_CLASSES)}


# ═══════════════════════════════════════════════════════════════
# Worker Function for Multi-Processing (Top-level scope)
# ═══════════════════════════════════════════════════════════════
def process_single_image(args):
    img_path_str, label_idx = args
    img_path = Path(img_path_str)
    img = cv2.imread(str(img_path))
    if img is None or img.size == 0:
        return None

    mp_hands = mp.solutions.hands
    with mp_hands.Hands(static_image_mode=True, max_num_hands=2, min_detection_confidence=0.4) as hands:
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = hands.process(img_rgb)

        if not results.multi_hand_landmarks:
            return None

        lh_feats = np.zeros((21, 3), dtype=np.float32)
        rh_feats = np.zeros((21, 3), dtype=np.float32)
        has_lh, has_rh = False, False

        for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
            label = handedness.classification[0].label
            coords = np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark], dtype=np.float32)

            wrist = coords[0]
            hand_span = np.linalg.norm(coords[9] - wrist) + 1e-6
            norm_coords = (coords - wrist) / hand_span

            if label == "Left":
                lh_feats = norm_coords
                has_lh = True
            else:
                rh_feats = norm_coords
                has_rh = True

        if has_lh and not has_rh:
            rh_feats = lh_feats.copy()
        elif has_rh and not has_lh:
            lh_feats = rh_feats.copy()

        feature_vector = np.concatenate([lh_feats.flatten(), rh_feats.flatten()])
        return feature_vector, label_idx


# ═══════════════════════════════════════════════════════════════
# Downloaders
# ═══════════════════════════════════════════════════════════════
def download_kaggle_dataset():
    print("\n--- [1/3] Checking Kaggle ISL Dataset (prathumarikeri/indian-sign-language-isl) ---")
    try:
        path = kagglehub.dataset_download("prathumarikeri/indian-sign-language-isl")
        print(f"[OK] Kaggle ISL Dataset Path: {path}")
        return Path(path)
    except Exception as e:
        print(f"[!] Warning: Kaggle download failed ({e}).")
        return None

def download_asl_dataset():
    print("\n--- [2/3] Checking Kaggle ASL Dataset (grassknoted/asl-alphabet) ---")
    try:
        path = kagglehub.dataset_download("grassknoted/asl-alphabet")
        print(f"[OK] Kaggle ASL Dataset Path: {path}")
        return Path(path)
    except Exception as e:
        print(f"[!] Warning: ASL download failed ({e}).")
        return None

def download_github_dataset():
    print("\n--- [3/3] Checking GitHub ISL Dataset (ayeshatasnim-h/Indian-Sign-Language-dataset) ---")
    dest_zip = RAW_DIR / "isl_github_dataset.zip"
    url = "https://github.com/ayeshatasnim-h/Indian-Sign-Language-dataset/archive/refs/heads/main.zip"
    
    if not dest_zip.exists():
        print(f"Downloading from GitHub: {url} ...")
        try:
            urllib.request.urlretrieve(url, dest_zip)
            print(f"[OK] Downloaded {dest_zip.name}")
        except Exception as e:
            print(f"[!] Warning: GitHub download failed: {e}")
            return None

    extract_dir = RAW_DIR / "github_extracted"
    if not extract_dir.exists() and dest_zip.exists():
        print("Extracting GitHub zip dataset...")
        with zipfile.ZipFile(dest_zip, 'r') as zip_ref:
            zip_ref.extractall(extract_dir)
        print(f"[OK] Extracted to {extract_dir}")
    
    return extract_dir


# ═══════════════════════════════════════════════════════════════
# Data Harvester & Multi-Core Landmark Extractor
# ═══════════════════════════════════════════════════════════════
def main():
    print("=========================================================")
    print("   SignSpeak - Multi-Core Combined Alphabet Harvester")
    print("=========================================================\n")

    raw_paths = []
    
    kaggle_path = download_kaggle_dataset()
    if kaggle_path and kaggle_path.exists():
        raw_paths.append(kaggle_path)

    asl_path = download_asl_dataset()
    if asl_path and asl_path.exists():
        raw_paths.append(asl_path)

    github_path = download_github_dataset()
    if github_path and github_path.exists():
        raw_paths.append(github_path)

    print("\n--- Scanning for Image Files ---")
    work_tasks = []
    supported_exts = {".jpg", ".jpeg", ".png", ".bmp"}

    for root_path in raw_paths:
        for p in root_path.rglob("*"):
            if p.is_file() and p.suffix.lower() in supported_exts:
                parent_name = p.parent.name.upper().strip()
                stem_name = p.stem.upper().strip()

                matched_class = None
                for cls_name in VALID_CLASSES:
                    if parent_name == cls_name or parent_name.startswith(cls_name + "_") or parent_name.endswith("_" + cls_name):
                        matched_class = cls_name
                        break
                    elif stem_name.startswith(cls_name + "_") or stem_name.startswith(cls_name):
                        matched_class = cls_name

                if matched_class in CLASS2IDX:
                    work_tasks.append((str(p), CLASS2IDX[matched_class]))

    print(f"[OK] Queued {len(work_tasks)} images across {len(VALID_CLASSES)} classes for multi-core extraction.")

    if not work_tasks:
        print("[!] Error: No candidate images found.")
        return

    num_workers = min(12, os.cpu_count() or 4)
    print(f"\n--- Extracting Landmarks using {num_workers} Parallel CPU Workers ---")

    all_features = []
    all_labels = []
    valid_count = 0
    discarded_count = 0

    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = [executor.submit(process_single_image, task) for task in work_tasks]
        for future in tqdm(as_completed(futures), total=len(work_tasks), desc="MediaPipe Extraction"):
            res = future.result()
            if res is not None:
                feat, label_idx = res
                all_features.append(feat)
                all_labels.append(label_idx)
                valid_count += 1
            else:
                discarded_count += 1

    print(f"\n[OK] Processing Complete!")
    print(f"   - Valid Hand Samples Retained: {valid_count}")
    print(f"   - Invalid/No-Hand Images Discarded: {discarded_count}")

    if valid_count == 0:
        print("[!] Error: No valid hand landmarks extracted!")
        return

    X = np.array(all_features, dtype=np.float32)
    y = np.array(all_labels, dtype=np.int64)

    output_npz = PROCESSED_DIR / "isl_letters_dataset.npz"
    np.savez_compressed(output_npz, X=X, y=y)
    print(f"[OK] Saved Compressed Landmark Dataset: {output_npz} ({output_npz.stat().st_size / 1024 / 1024:.2f} MB)")

    mapping_meta = {
        "class2idx": CLASS2IDX,
        "idx2class": IDX2CLASS,
        "total_samples": valid_count,
        "feature_dim": X.shape[1]
    }
    output_meta = PROCESSED_DIR / "isl_letters_meta.json"
    with open(output_meta, "w", encoding="utf-8") as f:
        json.dump(mapping_meta, f, indent=2)
    print(f"[OK] Saved Metadata Mapping: {output_meta}")

    github_zip = RAW_DIR / "isl_github_dataset.zip"
    if github_zip.exists():
        github_zip.unlink()

    print("\n=========================================================")
    print("     COMBINED LANDMARK EXTRACTION COMPLETE               ")
    print("=========================================================")

if __name__ == "__main__":
    main()
