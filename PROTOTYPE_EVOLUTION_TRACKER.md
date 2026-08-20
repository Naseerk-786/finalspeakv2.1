# 🔄 SignSpeak Universal — Prototype Evolution & Development Tracker
**Living Master Document: Baseline (Low-Fidelity) vs. Current High-Fidelity Implementation & Ongoing Changelog**

* **Last Updated:** August 20, 2026  
* **Target Hardware:** NVIDIA GeForce RTX 4050 GPU (6 GB VRAM) + Multi-Core CPU  
* **Target Environment:** Python 3.10 / PyTorch 2.2 (CUDA 12.1) ──► ONNX Runtime  
* **Cloud Dependency:** **0% (100% Offline, $0.00/month runtime cost)**

---

## 📊 High-Level Comparison Matrix: Baseline vs. Current System

| Architectural Dimension | 🔴 Low-Fidelity Baseline (Initial) | 🟢 Current High-Fidelity System | Transformation / How It Was Done |
| :--- | :--- | :--- | :--- |
| **Recognition Paradigm** | Multi-frame Word-Level Gestures (ST-GCN / 30-frame sequence) | Single-Frame ISL/ASL Fingerspelling (35 classes: A–Z, 1–9) | **Strategic Pivot:** Shifted from temporal sequences to spatial invariant alphabet vectors for infinite vocabulary. |
| **Vocabulary Reach** | Closed Dictionary (~364 static words only) | **Unlimited Vocabulary** (Any word, name, technical term) | Fingerspelling engine allows spelling any open-ended vocabulary seamlessly. |
| **Dataset Size & Quality** | Raw scraped video clips with blurred / missing hands | **107,517 Clean 3D Hand Vectors** (from 129,773 raw images) | **Multi-Core QC Pipeline:** 12 parallel CPU workers executing MediaPipe with $\ge 0.40$ confidence threshold. |
| **Personalization** | None (Static pre-trained weights only) | **Interactive Sign Recorder Studio & Co-Trainer** | Built PyQt6 recorder + GPU fine-tuner; blended 33 user-recorded classes with 107k baseline. |
| **Data Augmentation** | Basic pixel transforms in notebooks | **3D Geometric Spatial Augmentation** | Synthetic 3D rotations ($\pm 18^\circ$), scaling ($0.88\times–1.12\times$), and joint jitter ($\sigma=0.012$), expanding data to **246,104 samples**. |
| **Inference Latency** | $> 500\text{ ms}$ (30-frame sliding window buffer delay) | **$< 1.8\text{ ms}$** per frame ONNX inference | Single-frame inference + ONNX Runtime FP32/INT8 graph execution. |
| **Model Accuracy** | $42.8\%$ Top-1 Accuracy (severe dialect collision) | **$99.84\% - 99.96\%$ Accuracy** | Deep Residual MLP with skip connections, SiLU activations, Label Smoothing ($0.05$), and Cosine Annealing. |
| **Model Binary Size** | Large video checkpoints ($> 150\text{ MB}$) | **556 KB** (`isl_letter_classifier.onnx`) | Compact Residual MLP architecture exported to optimized ONNX format. |
| **Application Architecture** | Single-threaded script / Jupyter Notebook (`part_1.ipynb`) | **Decoupled 4-Thread Asynchronous Engine** | Separated Video Capture, ONNX Inference, PyQt6 UI, and Neural TTS into isolated parallel threads. |
| **Interaction Paradigm** | Manual Spacebar keypress required for every single letter | **100% Hands-Free Dwell Meter + Hand-Drop Commit** | 0.75s steady hold auto-records letter; 1.2s hand lowering auto-commits word & triggers Piper voice. |
| **Word Prediction** | None (Manual raw spelling only) | **Instant Top-3 Predictive Candidate Pills** | Embedded frequency NLP dictionary suggesting words dynamically based on prefix. |
| **Audio Speech Synthesis** | No audio or browser-dependent online APIs | **Offline Neural Voice Synthesis (Piper TTS)** | Integrated local Piper ONNX neural voice engine (`en_US-lessac-medium.onnx`, $<12\text{ms}$ synthesis). |


---

## 🔴 1. What Was in the Low-Fidelity Prototype (Baseline)

### 1.1 Components & Structure
* **Exploratory Notebooks:** `part_1.ipynb` and `part_2.ipynb` used for early exploratory experimentation.
* **Word-Level Modeling:** Attempted to classify 364 dynamic sign language word glosses ("Dog", "Car", "Election", "Hospital") using 30-frame sliding temporal windows.
* **Feature Extraction:** Computed 856-dimensional relative distance and velocity vectors across consecutive video frames.
* **Network Topology:** Spatial-Temporal Graph Convolutional Networks (ST-GCN) and basic LSTMs.

### 1.2 Identified Critical Bottlenecks & Why It Failed
1. **Dialect Collision:** Merging American Sign Language (WLASL) and Indian Sign Language (INCLUDE) datasets resulted in conflicting gestures for identical words, dropping classification accuracy to **42.8%**.
2. **High Latency Bottleneck:** Buffering 30 video frames required over **500 ms of latency** before any prediction could be generated, making real-time conversation impossible.
3. **Closed Dictionary Barrier:** The model could only recognize words present in the training dictionary. It completely failed on proper nouns, Indian names, locations, and technical terms.
4. **Single-Threaded GUI Lag:** OpenCV capture, neural inference, and UI rendering ran in a single thread, causing frequent frame drops (5–12 FPS) and UI freezing.

---

## 🟢 2. What We Have Done for Now (High-Fidelity Implementation) & How

```
┌───────────────────────────────────────────────────────────────────────────────────────────────────┐
│                                CURRENT HIGHER-FIDELITY ARCHITECTURE                               │
└───────────────────────────────────────────────────────────────────────────────────────────────────┘

  [Multi-Source Data Harvester]        [Personalized Sign Studio]        [3D Geometric Augmentation]
  129,773 Images ──► 12 CPU Workers    User Camera Signs (33 Classes)    Rotations + Scaling + Jitter
               │                                   │                                   │
               ▼                                   ▼                                   ▼
  ┌───────────────────────────────────────────────────────────────────────────────────────────────┐
  │                           Consolidated Master Dataset (246,104 Samples)                       │
  └───────────────────────────────────────────────┬───────────────────────────────────────────────┘
                                                  │
                                                  ▼
  ┌───────────────────────────────────────────────────────────────────────────────────────────────┐
  │                    Deep Residual MLP (Skip Connections + SiLU + BatchNorm1d)                  │
  │                  CUDA GPU Training Run ──► 99.84% - 99.96% Validation Accuracy                │
  └───────────────────────────────────────────────┬───────────────────────────────────────────────┘
                                                  │ (Export)
                                                  ▼
  ┌───────────────────────────────────────────────────────────────────────────────────────────────┐
  │                         556 KB Optimized ONNX Classifier Binary                               │
  └───────────────────────────────────────────────┬───────────────────────────────────────────────┘
                                                  │
                                                  ▼
  ┌───────────────────────────────────────────────────────────────────────────────────────────────┐
  │                         Asynchronous 4-Thread Desktop Application                             │
  │  Thread 1: Camera + MediaPipe  ──►  Thread 2: Sub-2ms ONNX Inference + One-Euro Filter        │
  │  Thread 3: PyQt6 UI Word Builder ──►  Thread 4: Offline Piper Neural TTS Speech Audio         │
  └───────────────────────────────────────────────────────────────────────────────────────────────┘
```

### 2.1 Strategic Architectural Pivot to Single-Frame Alphabets
* **How it was done:** Replaced multi-frame sequence modeling with **single-frame ISL/ASL fingerspelling & digit recognition (35 Target Classes: A–Z + 1–9)**.
* **Result:** Eliminated the 500ms video buffer delay down to **<1.8ms**, while unlocking **infinite vocabulary** via real-time word construction.

### 2.2 Data Harvesting & Multi-Core QC Extraction Pipeline
* **File:** [`prototype/download_and_extract_isl_letters.py`](file:///d:/finalspeak/prototype/download_and_extract_isl_letters.py)
* **How it was done:**
  * Merged Kaggle ISL, Kaggle ASL, and GitHub ISL datasets (**129,773 raw images**).
  * Deployed **12 parallel CPU workers** executing MediaPipe Hands.
  * Discarded blurred or handless frames ($\text{confidence} < 0.40$), retaining **107,517 clean 3D hand vectors**.

### 2.3 126-Dimensional Invariant Normalization Math
* **How it was done:**
  * Extracted 21 3D landmarks per hand ($21 \times 3 \times 2 = 126$ values).
  * **Wrist-Centered Translation:** $\mathbf{P}'_i = \mathbf{P}_i - \mathbf{P}_0 \quad \forall i \in [0, 20]$ (Origin at wrist $\mathbf{P}_0$).
  * **Hand Span Scale Normalization:** $S = \|\mathbf{P}_9 - \mathbf{P}_0\|_2 + \epsilon$, $\mathbf{P}_{\text{norm}, i} = \mathbf{P}'_i / S$.
  * **Active Hand Mirroring:** Duplicates active hand features across both slots if only one hand is visible ($\mathbf{X}_{\text{frame}} = [\mathbf{X}_{\text{active}} \,\|\, \mathbf{X}_{\text{active}}]$), achieving full left/right hand invariance.

### 2.4 Deep Residual MLP Neural Architecture (`ISLLetterClassifier`)
* **File:** [`prototype/train_letter_model.py`](file:///d:/finalspeak/prototype/train_letter_model.py)
* **Topology:**
  * `Block 1`: `Linear(126 → 256)` + `BatchNorm1d` + `SiLU` + `Dropout(0.20)`
  * `Block 2`: `Linear(256 → 256)` + `BatchNorm1d` + `SiLU` + `Dropout(0.20)` with **Residual Skip Connection** ($h_2 = \text{Block}_2(h_1) + h_1$)
  * `Block 3`: `Linear(256 → 128)` + `BatchNorm1d` + `SiLU` + `Dropout(0.20)`
  * `Classifier Head`: `Linear(128 → 35)`
* **Training Optimizations:** AdamW ($\text{lr}=2\times 10^{-3}$), Cosine Annealing, Label Smoothing ($0.05$).

### 2.5 Personalized Sign Recorder Studio & Co-Training Fine-Tuner
* **Files:** [`prototype/sign_recorder_studio.py`](file:///d:/finalspeak/prototype/sign_recorder_studio.py), [`prototype/fine_tune_engine.py`](file:///d:/finalspeak/prototype/fine_tune_engine.py), [`prototype/augment_and_train_master_letters.py`](file:///d:/finalspeak/prototype/augment_and_train_master_letters.py)
* **How it was done:**
  * **Interactive Studio GUI:** PyQt6 desktop interface with a **3-second Red Preparation Countdown** followed by a **3-second Green Auto-Record Window** capturing live 3D coordinates.
  * **Recorded Custom Dataset:** Captured real user signs for **33 classes** (A–Y, 1–9) saved in `data/user_recorded/`.
  * **3D Geometric Augmentation:** Applied synthetic 2D/3D rotations ($\pm 18^\circ$), random scale jitter ($0.88\times - 1.12\times$), and Gaussian joint noise ($\sigma = 0.012$), synthesizing **246,104 augmented samples**.
  * **Zero-Forgetting Co-Trainer:** Blends user webcam recordings with the 107k baseline dataset, completing fine-tuning on the RTX 4050 GPU in **~8 seconds** with **99.84%–99.96% accuracy**.

### 2.6 Asynchronous 4-Thread Real-Time Desktop Interpreter
* **File:** [`prototype/part_3_letters.py`](file:///d:/finalspeak/prototype/part_3_letters.py)
* **How it was done:**
  * **Thread 1 (Capture):** 30 FPS video feed with MediaPipe skeleton tracking.
  * **Thread 2 (Inference):** Ultra-fast ONNX inference ($<1.8\text{ ms}$) + 4-frame temporal confirmation filter to prevent flicker.
  * **Thread 3 (UI):** Active word builder, auto-commit idle timeout ($1.5\text{ s}$), and clean dark theme interface.
  * **Thread 4 (TTS):** 100% offline Piper neural voice synthesis (`en_US-lessac-medium.onnx`, $<12\text{ ms}$).

### 2.7 One-Euro Signal Filtering
* **File:** [`prototype/one_euro_filter.py`](file:///d:/finalspeak/prototype/one_euro_filter.py)
* **How it was done:** Integrated adaptive cutoff frequency filtering ($f_c = f_{c,\min} + \beta |\dot{x}|$) to eliminate jitter during slow gestures while maintaining zero lag during rapid movements.

### 2.8 Version 2.1 Core Interactive Prototype
* **File:** [`prototype/part_3_letters.py`](file:///d:/finalspeak/prototype/part_3_letters.py)
* **Design Philosophy:** Clean, distraction-free Warm Soft 2D Plushy interface focused on direct sign recognition and user-controlled letter recording.
* **Core Interaction Mechanics:**
  * **Direct Spacebar Recording:** Pressing `Spacebar` captures the currently recognized letter with confidence validation ($\ge 50\%$).
  * **Optional Hands-Free Auto-Capture Toggle:** Checkbox toggle providing temporal stability frame confirmation (4 consecutive frames) without intrusive popup suggestions.
  * **Backspace & Commit:** `Backspace` deletes the last letter; `Enter` or button commits the word to the sentence buffer.
  * **Offline Neural Audio:** Neural Piper TTS speaks words upon commit or full sentence playback.

---

## 📝 3. Progressive Changelog & Work Log

| Date / Session | Phase / Milestone | Key Additions & Modifications | Metrics Achieved |
| :--- | :--- | :--- | :--- |
| **Phase 1** | Blueprinting & Spec | Created master specifications, selected zero-cloud tech stack. | $0.00$ Cloud Cost guarantee |
| **Phase 2** | ST-GCN Exploration | Tested 30-frame word-level model on INCLUDE + WLASL. | Bottlenecks identified (42.8% acc) |
| **Phase 3** | Architectural Pivot | Shifted to 35-class single-frame alphabet fingerspelling. | Latency dropped to $<1.8\text{ ms}$ |
| **Phase 4** | Multi-Core Harvester | Harvested 129k images, extracted 107,517 3D vectors via 12 CPU workers. | 107,517 clean training vectors |
| **Phase 5** | CUDA GPU Baseline Run | Trained Deep Residual MLP over 200 epochs on RTX 4050; exported ONNX binary. | **99.70% Test Accuracy** (556 KB) |
| **Phase 6** | 4-Thread Desktop App | Built PyQt6 app with Capture, Inference, UI, and Piper Neural TTS threads. | $<23.4\text{ ms}$ total end-to-end delay |
| **Phase 7** | Master Documentation | Generated `METHODOLOGIES_FOLLOWED.md`, `PROJECT_FLOW.md`, `PRESENTATION_MASTERBOOK.md`. | Complete academic slide & math deck |
| **Phase 8 (Recent)** | Sign Recorder Studio | Built interactive 3s preparation $\to$ 3s recording studio with live test mode. | 33 user sign classes captured |
| **Phase 9 (Recent)** | Master Co-Training & Augmentation | Merged user recordings with 107k baseline, added 3D rotations/jitter (246k samples), trained on GPU. | **99.84%–99.96% Accuracy** |
| **Phase 10 (Recent)** | Evolution Tracker | Created living tracking master document (`PROTOTYPE_EVOLUTION_TRACKER.md`). | Complete ongoing roadmap & sync |
| **Phase 11 (Current)** | Stable Version 2.1 Release | Clean, responsive single-frame ISL interpreter with Warm Soft 2D Plushy UI, direct Spacebar capture, robust camera fallback, and Piper TTS. | **Zero Latency, Pure Manual & Auto Controls** |

---

## 🚀 4. Next Planned Enhancements (Future Iteration Roadmap)

* [ ] **Bi-directional Translation:** Add a Text/Speech-to-Sign animated avatar module to allow two-way deaf $\leftrightarrow$ hearing conversation.
* [ ] **Multi-Dialect Preset Switcher:** Allow one-click switching between pure ISL (two-handed) and pure ASL (one-handed) sign sets.
* [ ] **Standalone Installer & Executable Packaging:** Bundle the application, ONNX models, and Piper TTS into a single-click Windows `.exe` using PyInstaller.


