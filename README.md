# SignSpeak Universal — Indian Sign Language Studio (v2.1)

An ultra-low latency, 100% offline Indian Sign Language (ISL) fingerspelling and alphabet recognition application with integrated neural speech synthesis.

---

## 🌟 Key Highlights
* **Zero Cloud Dependency:** 100% offline, privacy-first, $0.00/month runtime cost.
* **Sub-2ms Inference:** Lightweight Deep Residual MLP (556 KB ONNX) running on CPU/GPU.
* **99.84%–99.96% Accuracy:** Trained on 107k+ 3D landmark samples + personalized 33-class user recordings with 3D geometric augmentation.
* **Neural Speech Synthesis:** Built-in Piper TTS (`en_US-lessac-medium.onnx`) for natural spoken voice output.
* **Asynchronous 4-Thread Engine:** Dedicated threads for Video Capture, ONNX Inference, PyQt6 UI, and Piper Audio.

---

## 🚀 Quickstart

### 1. Requirements
* Python 3.10+
* PyTorch (with CUDA support for training)
* OpenCV, MediaPipe, ONNX Runtime, PyQt6, Piper TTS

### 2. Run the Main Application
```powershell
python prototype/part_3_letters.py
```

### 3. Record Personal Signs
```powershell
python prototype/sign_recorder_studio.py
```

### 4. GPU Fine-Tuning & Co-Training
```powershell
python prototype/fine_tune_engine.py
```

---

## 📁 Repository Structure
* `prototype/` — Core Python application, sign recorder studio, and training pipelines.
* `models/` — Exported ONNX model binaries, PyTorch checkpoints, and class mappings.
* `data/user_recorded/` — User-recorded 3D landmark arrays for personalized fine-tuning.
* `PROTOTYPE_EVOLUTION_TRACKER.md` — Living log of architecture evolution and milestones.
