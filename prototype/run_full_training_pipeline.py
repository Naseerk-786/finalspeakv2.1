# SignSpeak — Automated Master Pipeline Orchestrator
# Waits for multi-core landmark harvester task to complete, then trains the MLP model,
# evaluates accuracy, and exports ONNX binaries.

import os
import sys
import time
import subprocess
from pathlib import Path

BASE_DIR = Path(r"d:\finalspeak")
PROCESSED_DIR = BASE_DIR / "data" / "processed"
DATASET_NPZ = PROCESSED_DIR / "isl_letters_dataset.npz"

def main():
    print("=========================================================")
    print("  SignSpeak - Master Training Orchestrator (No Shortcuts)")
    print("=========================================================\n")

    print("Checking landmark dataset isl_letters_dataset.npz...")
    
    while not DATASET_NPZ.exists():
        time.sleep(2)

    prev_size = -1
    while True:
        curr_size = DATASET_NPZ.stat().st_size
        if curr_size > 0 and curr_size == prev_size:
            break
        prev_size = curr_size
        time.sleep(2)

    print(f"\n[OK] Found completed landmark dataset: {DATASET_NPZ} ({DATASET_NPZ.stat().st_size / 1024 / 1024:.2f} MB)")
    print("\n--- [Step 2/2] Launching 200-Epoch Model Training & ONNX Export ---")

    train_script = BASE_DIR / "prototype" / "train_letter_model.py"
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"

    res = subprocess.run([sys.executable, str(train_script)], env=env, check=True)

    print("\n=========================================================")
    print("   MASTER PIPELINE COMPLETE! MODEL & ONNX READY.         ")
    print("=========================================================")

if __name__ == "__main__":
    main()
