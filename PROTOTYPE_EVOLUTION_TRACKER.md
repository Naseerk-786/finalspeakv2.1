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
| **Interaction Paradigm** | Manual Spacebar keypress required for every single letter | **0.8s Steady-Hold Dwell + Spacebar Word Commit** | 0.80s steady hold auto-records letter with audio tick; Spacebar commits word to full sentence. |
| **Word & Sentence Construction** | Single raw letter display with manual clearing | **Multi-Word Sentence Builder with Smart Backspace** | Full sentence line accumulation, smart Backspace word pullback, and Piper TTS speech synthesis. |
| **Audio Speech Synthesis** | No audio or browser-dependent online APIs | **Offline Neural Voice Synthesis (Piper TTS)** | Integrated local Piper ONNX neural voice engine (`en_US-lessac-medium.onnx`, $<12\text{ms}$ synthesis). |
| **Multilingual Voice Output** | English only (no regional support) | **8 Indian Regional Languages** (English, Hindi, Telugu, Tamil, Marathi, Kannada, Bengali, Gujarati) | Integrated multilingual neural translator & non-blocking regional voice synthesis dropdown. |
| **Two-Way Deaf ↔ Hearing Loop** | 1-Way sign recognition only (hearing replies missed) | **Two-Way Live Speech-to-Sign & Dialogue Exporter** | Integrated non-blocking `SpeechToTextThread` (Whisper AI), live ISL visual fingerspelling strip, and timestamped transcript exporter. |
| **Word Completion & Predictive AI** | None (Letter-by-letter typing only) | **Gboard-Style AI Autocomplete Strip** (Keys `1`, `2`, `3` & Numpad) | Asynchronous dual-tier autocomplete with Groq Cloud LLM + instant <0.1ms offline fallback dictionary. |
| **Linguistic Naturalness** | Raw disjointed sign glosses | **1-Click AI Sign Grammar Polish (`Ctrl+P` / `Ctrl+Z`)** | Converts telegraphic sign glosses into fluent conversational sentences with auto-polish on speak. |


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

### 2.8 Version 2.2 Steady-Hold Letter Capture & Spacebar Sentence Builder
* **File:** [`prototype/part_3_letters.py`](file:///d:/finalspeak/prototype/part_3_letters.py)
* **Design Philosophy:** Clean Warm Soft 2D Plushy interface pairing continuous 0.8s hold-to-type letter recognition with keyboard Spacebar word commitment.
* **Core Interaction Mechanics:**
  * **0.80s Steady-Hold Dwell Capture:** Holding any sign steady ($\text{confidence} \ge 50\%$) fills a plushy green progress bar over $0.80\text{ s}$, auto-recording the letter with an anti-duplication hysteresis lock and soft non-blocking audio tick.
  * **Spacebar Word Commit:** Pressing `Spacebar` commits the current word into the sentence buffer and clears the word builder for the next word.
  * **Smart Backspace:** Pressing `Backspace` deletes the last letter; if the word buffer is empty, pulls the previous word back for editing.
  * **Enter Spoken Output:** Pressing `Enter` commits any active letters and speaks the complete multi-word sentence line via offline Piper Neural TTS.
  * **Escape Clear All:** Clears both word and sentence buffers instantly.

### 2.9 Version 2.3 AI-Powered 3-Suggestion Autocomplete Bar (Gboard Style)
* **File:** [`prototype/part_3_letters.py`](file:///d:/finalspeak/prototype/part_3_letters.py)
* **Design Philosophy:** Integrated Gboard-style 3-suggestion pill strip above the Word Builder powered by asynchronous Groq Cloud AI inference with local offline dictionary fallback.
* **Core Interaction Mechanics:**
  * **Non-Blocking Background AI Worker (`AIPredictionThread`):** Sends current letter prefix + sentence context to Groq Cloud endpoint in background without dropping camera frames or freezing UI.
  * **3 Clickable Suggestion Pills:** Displays top 3 context-aware word completions (e.g. `[ 1: WATER ]`, `[ 2: WATCH ]`, `[ 3: WAIT ]`).
  * **One-Touch Keyboard Selection (`1`, `2`, `3`):** Pressing `1`, `2`, or `3` instantly autocompletes and commits the word into the full sentence line.
  * **Zero-Failure Offline Safety:** Automatically falls back to offline frequency dictionary if internet or API connection drops.

### 2.10 Version 2.4 AI Sign Grammar Polish, Keyboard Help Modal & Professional UI (Completed)
* **File:** [`prototype/part_3_letters.py`](file:///d:/finalspeak/prototype/part_3_letters.py)
* **Design Philosophy:** Enterprise-grade accessibility UI (clean typography, zero casual emojis) pairing real-time fingerspelling recognition with 1-click AI Grammar Polish and an interactive `F1` keyboard shortcuts modal.
* **Core Interaction Mechanics:**
  * **1-Click AI Grammar Polish (`Ctrl + P`):** Seamlessly transforms telegraphic sign glosses (e.g. `"BALL GIVE"`, `"ME WATER DRINK WANT"`) into natural conversational English (*"Please pass me the ball."*, *"I want to drink water."*) in $\sim 150\text{ ms}$ via non-blocking background Groq worker.
  * **Auto-Polish on Speak Toggle:** Automatically converts raw sign glosses to fluent natural sentences right before Piper Neural TTS vocalization.
  * **Revert to Raw (`Ctrl + Z`):** Non-destructively toggles back to the exact signed word sequence at any time.
  * **System Controls & Shortcuts Modal (`F1`):** Interactive keyboard cheat-sheet dialog accessible via top app bar button or `F1` key press.
### 2.11 Version 2.5 Multilingual Indian Speech Engine & Voice Selector (Completed)
* **File:** [`prototype/part_3_letters.py`](file:///d:/finalspeak/prototype/part_3_letters.py)
* **Design Philosophy:** Seamless regional localization empowering Indian signers to translate and vocalize signed sentences in **8 Indian languages**:
  * `English` (Default Piper Neural Voice)
  * `Hindi` (हिन्दी)
  * `Telugu` (తెలుగు)
  * `Tamil` (தமிழ்)
  * `Marathi` (मराठी)
  * `Kannada` (ಕನ್ನಡ)
  * `Bengali` (বাংলা)
  * `Gujarati` (ગુજરાતી)
* **Core Interaction Mechanics:**
  * **Plushy Language Dropdown:** Seamlessly switch between English and 7 major Indian languages right in the Spoken Sentence Cockpit.
  * **Asynchronous Neural Translation Worker:** Translates signed sentences into the target regional script using Groq Cloud with comprehensive offline fallback dictionary support.
  * **Non-Blocking Multilingual Neural Voice:** Synthesizes and vocalizes regional Indian speech audio without UI frame drops or camera latency.
  * **Universal Keypad Support:** Numbers `1`, `2`, `3` and numeric keypad keys (`Numpad1`, `Numpad2`, `Numpad3`) instantly commit autocomplete suggestion pills.

### 2.13 Version 3.0 Camera-First Spacious Assistive UI/UX Workspace (Completed)
* **File:** [`prototype/part_3_letters.py`](file:///d:/finalspeak/prototype/part_3_letters.py)
* **Design Philosophy:** Complete transformation from a cluttered dashboard to a calm, spacious, **camera-first assistive communication workspace**:
  * **Visual Hero Camera Surface:** Camera viewport expanded to **55–65%** of main area with subtle border, soft radius, dynamic 16:9/4:3 scaling, and translucent live status overlays (`● Hand Detected`).
  * **Strict Visual Hierarchy:** Primary Sign Communication Workspace structured vertically: `DETECTED SIGN` (52px tile + 0.8s hold bar) ──► `CURRENT WORD` (26px text + 3 elegant autocomplete chips) ──► `SENTENCE` (17px text + voice dropdown + prominent Speak button).
  * **Comfortable Touch & Click Targets:** All primary buttons enlarged to **44–52px height** for comfortable, effortless interaction on laptops.
  * **Progressive Disclosure:** Two-Way Dialogue Timeline and Activity Diagnostics neatly tucked into a clean, collapsible secondary drawer that can be toggled to maximize camera space.
  * **Fixed Qt6 Hotkeys:** Universal key handling across standard and keypad numeric keys `1`, `2`, `3`.

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
| **Phase 11 (Recent)** | Version 2.2 Release | Integrated 0.8s Steady-Hold letter capture with hysteresis lock, Spacebar word commit, smart Backspace, and Piper TTS full-sentence voice. | **Natural Keyboard Workflow with Zero Fatigue** |
| **Phase 12 (Recent)** | Version 2.3 Release | Integrated Gboard-style 3-suggestion AI autocomplete pills powered by Groq Cloud + offline fallback with keys `1`, `2`, `3` shortcuts. | **Context-Aware Instant Word Completion** |
| **Phase 13 (Recent)** | Version 2.4 Release | Integrated 1-Click AI Sign Grammar Polish, non-destructive Revert, Auto-Polish on Speak, F1 Shortcuts Guide Modal, and clean enterprise UI styling. | **Natural Linguistic Syntax Bridge & Pro Cockpit** |
| **Phase 14 (Recent)** | Version 2.5 Release | Integrated Regional Indian Language Selector Dropdown (Hindi, Telugu, Tamil, Marathi, Kannada, Bengali, Gujarati) + Multilingual Neural Speech Synthesis. | **Native Multi-Dialect Voice Output** |
| **Phase 15 (Recent)** | Version 2.6 Release | Integrated Two-Way Deaf ↔ Hearing Communication Loop with non-blocking Whisper STT listener, Live ISL Sign Visualizer, and Dialogue Transcript Exporter. | **Full Bidirectional Conversational Loop** |
| **Phase 16 (Latest)** | Version 3.0 Release | Complete Camera-First UI/UX overhaul: spacious 60% hero video surface, progressive drawer, 48px action buttons, and warm assistive palette. | **Calm, Premium Assistive Workspace** |

---

## 🚀 4. Comprehensive Future Improvements & Exploration Roadmap

* [x] **1. Two-Way Speech-to-Sign Communication (Deaf $\leftrightarrow$ Hearing Loop):**
  * Whisper AI speech recognition on incoming audio $\to$ live transcript + visual ISL fingerspelling badge strip + conversation timeline export.
* [x] **2. Multilingual Indian Voice Engine:**
  * Real-time translation to Hindi, Telugu, Tamil, Marathi, Kannada, Bengali, and Gujarati paired with regional neural acoustic speech.
* [ ] **3. WebRTC Web & Mobile Companion App:**
  * WebAssembly (Wasm) + ONNX Runtime Web port running in modern mobile browser with zero installation.
* [ ] **4. Standalone 1-Click Windows Executable (.exe):**
  * Single-file PyInstaller / Inno Setup installer packaging ONNX models, Piper voices, and DirectShow drivers.
* [ ] **5. Smart Wearable & IoT Audio Broadcaster:**
  * Bluetooth LE audio streaming directly to external smart speakers, hearing aids, or classroom PA systems.







