# SignSpeak Studio — Indian Sign Language Assistive Workspace (v3.1)

An ultra-low latency, camera-first Indian Sign Language (ISL) fingerspelling and alphabet communication application featuring real-time neural speech synthesis, multilingual Indian regional voice output, AI grammar polishing, and a bidirectional Deaf ↔ Hearing conversational loop.

---

## 🌟 Key Highlights
* **Camera-First Assistive Workspace:** Spacious 60% hero video surface with aspect-ratio preservation, subtle status overlays, and accessible 48px touch targets.
* **Sub-2ms Neural Inference:** Ultra-lightweight Deep Residual MLP (`556 KB` ONNX binary) running on CPU/GPU in `<1.8ms` latency.
* **99.84%–99.96% Accuracy:** Trained on 107k+ 3D landmark samples blended with personalized user sign recordings and 3D geometric augmentations (246k samples).
* **0.8s Steady-Hold Dwell Capture:** Continuous hands-free letter recognition with audio confirmation ticks and Spacebar word commitment.
* **Gboard-Style AI Autocomplete:** Dual-tier predictive autocomplete (Groq Cloud LLM + instant <0.1ms offline dictionary) with 1-touch keys (`1`, `2`, `3`).
* **1-Click AI Sign Grammar Polish (`Ctrl + P`):** Converts telegraphic sign glosses (*"ME WATER DRINK WANT"*) into fluent natural sentences (*"I want to drink water."*) with non-destructive revert (`Ctrl + Z`).
* **8 Indian Regional Languages Output (`Ctrl + T` / `Enter`):** 1-click translation and native vocalization in **Hindi, Telugu, Tamil, Marathi, Kannada, Bengali, Gujarati, and English**.
* **Two-Way Deaf ↔ Hearing Loop (`Ctrl + M` / `F2`):** Non-blocking live microphone listener with Whisper AI transcription in <200ms, incoming subtitles, live ISL visual fingerspelling badge generator, and 1-click session dialogue transcript exporter.
* **Zero Cloud Lock-In:** Core vision, inference, and Piper neural voice run 100% offline with zero cloud runtime cost ($0.00/month).

---

## 🚀 Quickstart

### 1. Requirements & Installation
```powershell
pip install -r requirements.txt  # Or install PyQt6, opencv-python, mediapipe, onnxruntime, piper-tts, sounddevice
```

### 2. Launch Main Assistive Studio
```powershell
python prototype/part_3_letters.py
```

### 3. Record Personalized Signs
```powershell
python prototype/sign_recorder_studio.py
```

### 4. GPU Fine-Tuning & Co-Training
```powershell
python prototype/fine_tune_engine.py
```

---

## ⌨️ Primary Keyboard Shortcuts

| Shortcut | Action |
| :--- | :--- |
| **Hold Sign (0.8s)** | Auto-captures recognized letter into current word |
| **Keys `1` / `2` / `3` & Numpad** | Selects AI autocomplete suggestion chip |
| **Spacebar** | Commits active word into full spoken sentence |
| **Backspace** | Deletes last letter (or restores previous word for editing) |
| **Ctrl + P / Ctrl + Z** | 1-Click AI Grammar Polish / Revert to raw signs |
| **Ctrl + T** | 1-Click translate to selected Indian language |
| **Enter** | Vocalizes sentence in selected regional voice |
| **Ctrl + M / F2** | Toggle live microphone for hearing partner voice |
| **Escape** | Resets word and sentence buffers |
| **F1** | Opens interactive keyboard shortcuts reference modal |

---

## 📁 Repository Structure
* `prototype/part_3_letters.py` — Main SignSpeak Studio desktop application.
* `prototype/sign_recorder_studio.py` — Interactive sign recording studio.
* `prototype/fine_tune_engine.py` — GPU fine-tuning engine.
* `models/` — Exported ONNX model binaries (`556 KB`), PyTorch checkpoints, and Piper voice models.
* `data/user_recorded/` — User-recorded 3D landmark arrays for personalized fine-tuning.
* `PROTOTYPE_EVOLUTION_TRACKER.md` — Living master documentation of architecture evolution and milestones.

