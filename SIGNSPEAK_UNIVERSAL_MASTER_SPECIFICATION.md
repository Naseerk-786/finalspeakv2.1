# SignSpeak Universal — Ultimate Master Engineering & Product Specification
## Unified Product (PRD), Technical (TRD), System (SRD), ML Pipeline & Execution Manual

**Version:** 4.0 (Single Master Source of Truth)  
**Date:** July 2026  
**Classification:** Definitive Project Specification & Operational Blueprint  
**Target Hardware:** NVIDIA GeForce RTX 4050 (6 GB GDDR6 VRAM) + Intel CPU | 400 GB SSD  
**Target Operating Environment:** Python 3.10 (Desktop Prototype) ──► Mobile Port (Phase 4+)  
**Runtime Cost Guarantee:** $0.00 / Month (100% On-Device, Offline, Zero Cloud API Billing)  

---

> **Note on Phase 0 Prototype:**  
> For the dedicated Phase 0 Prototype architecture, hardware allocation strategy (58 GB drive), and optimal hybrid ST-GCN + Self-Attention methodology, see [SIGNSPEAK_PROTOTYPE_MASTER_SPECIFICATION.md](file:///d:/finalspeak/SIGNSPEAK_PROTOTYPE_MASTER_SPECIFICATION.md).

> **Purpose of this Master Document:**  
> This file is the single, absolute, uncompromised source of truth for the entire **SignSpeak Universal** lifecycle. It merges all product requirements (PRD), technical specifications (TRD), system architectures (SRD), dataset allocation strategies, risk matrices, mathematical formulations, and step-by-step developer execution roadmaps into **one unified master document**. No guesswork is required by AI assistants or human developers. Every technology, library, mathematical equation, feature index, directory path, thread queue, and code module is explicitly detailed.

---

# SECTION 1: PRODUCT REQUIREMENTS DOCUMENT (PRD)

## 1.1 Executive Summary & Vision
Sign languages are complete, natural languages with unique vocabularies, syntactic rules, and spatial grammars. Over 70 million Deaf individuals globally communicate in regional sign languages (e.g., ASL, ISL, DGS, BSL). Sign languages are **not universal** — ASL and ISL are structurally distinct languages. 

**SignSpeak Universal** is a real-time, privacy-first, on-device machine translation platform. It captures continuous sign language video via a standard webcam, extracts landmark coordinates, decodes continuous signing into sign glosses, translates gloss syntax into fluent spoken English, and synthesizes speech output — all locally, in real time (< 500 ms latency), with zero cloud API dependencies and zero recurring costs.

## 1.2 Target Personas & Real-World Use Cases

### Persona A: Deaf / Hard-of-Hearing Communicator (Primary)
*   **Profile:** Uses ASL or ISL as their primary language.
*   **Pain Point:** Faces severe communication barriers during spontaneous, real-world interactions (e.g., at retail stores, doctor appointments, government offices, or academic settings) with non-signers.
*   **Use Case:** The user positions their laptop camera. As they sign naturally, the application translates their signing into fluent English text on screen and reads it aloud via an offline text-to-speech engine in real time.

### Persona B: Hearing Learner / Educator / Partner (Secondary)
*   **Profile:** Hearing individuals learning sign language or communicating with Deaf family members/colleagues.
*   **Pain Point:** Struggles to interpret fast continuous signing or verify sign production correctness.
*   **Use Case:** Operates the software as an educational or interactive relay tool. The visual overlay displays recognized glosses, confidence scores, bounding boxes, and landmark skeletons for real-time validation.

## 1.3 Phased Feature Scope Matrix

| Module / Requirement | Phase 0 (Proof of Concept) | Phase 1 (Core CSLR Engine) | Phase 2 (Full Local Pipeline) | Phase 3 (Streaming Engine) | Phase 4+ (Expansion) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Input Source** | 30 FPS Live Webcam | 30 FPS Live Webcam / Video | 30 FPS Live Webcam Stream | Streaming Ring Buffer | Mobile Camera / Desktop |
| **Landmark Extractor** | MediaPipe Holistic | MediaPipe Holistic | MediaPipe Holistic | MediaPipe Holistic | MediaPipe Mobile / Custom |
| **Target Vocabulary** | 100 ASL + 263 ISL (Isolated) | 1,000 Glosses (CSLR) | 2,000 ASL + 263 ISL | Full Corpus | 300+ Sign Languages |
| **Vision Backbone** | 1D CNN + Bi-GRU | Transformer Encoder | Transformer Encoder | Transformer Encoder | Universal Multi-Lingual |
| **Sequence Decoder** | Dense Softmax | CTC Beam Search | CTC Beam Search | Streaming CTC Window | Direct Pose-to-Text |
| **Grammar Translator** | Direct Mapping | Gloss Token Output | Quantized T5-small | Quantized T5-small | Gloss-Free NMT |
| **Speech Output** | System Audio | N/A (Glosses Only) | Offline Piper TTS | Low-latency Piper TTS | Multilingual TTS |
| **User Interface** | Terminal / OpenCV GUI | Lightweight Tkinter | Full PyQt6 Dashboard | Desktop Overlay Window | Android Native App |

## 1.4 Non-Functional Requirements & Constraints

1.  **100% Privacy & Local Processing:** No video frames, landmark coordinate arrays, text transcripts, or telemetry are transmitted over the network. All inference occurs on-device.
2.  **$0.00 Runtime API Billing Guarantee:** The software must operate without invoking cloud APIs (e.g., OpenAI, Gemini, Claude, Google Cloud Vision). All vision encoders, CTC decoders, T5 translators, and TTS engines run locally.
3.  **Latency Boundary:** Time elapsed from the completion of a continuous gesture to spoken audio output must be $< 500\text{ ms}$.
4.  **Hardware Horizon (Student Constraints):**
    *   *Training Setup:* NVIDIA GeForce RTX 4050 Laptop GPU (6 GB VRAM) + Google Colab (Free/Pro) for pre-training.
    *   *Inference Setup:* Standard laptop CPU/GPU.
    *   *Storage Budget:* Maximum **119 GB** allocated for raw datasets on a **400 GB SSD**.
5.  **Quantized Memory Budget:** Total runtime footprint of all active INT8 ONNX models must remain $\le 280\text{ MB}$.

## 1.5 Benchmark KPIs

*   **Isolated Sign Recognition (ISLR):** $> 90\%$ Top-5 accuracy on WLASL-100 and INCLUDE validation sets.
*   **Continuous Sign Recognition (CSLR):** Word Error Rate (WER) $< 25\%$ on PHOENIX-2014T test set:
    $$\text{WER} = \frac{S + D + I}{N}$$
*   **Translation Quality (SLT):** BLEU-4 $> 20.0$, ROUGE-L $> 45.0$ on gloss-to-English translation benchmarks.
*   **Inference Frame Rate:** $\ge 30\text{ FPS}$ continuous video processing speed.

---

# SECTION 2: TECHNICAL REQUIREMENTS DOCUMENT (TRD)

## 2.1 Shortlisted Model Architectures & Evaluation Matrix

From the 71-paper audit in `SignSpeak_Projects_Reorganized.xlsx`, three optimal baseline architectures are selected:

```
┌────────────────────────────────┐     ┌────────────────────────────────┐     ┌────────────────────────────────┐
│ Baseline 1: Landmark + Bi-GRU │ ──► │ Baseline 2: Transformer + CTC  │ ──► │ Baseline 3: Gloss-Free SLT     │
│ (Phase 0 Proof of Concept)     │     │ (Phase 1-2 Core Production)    │     │ (Phase 3+ Advanced Scaling)    │
└────────────────────────────────┘     └────────────────────────────────┘     └────────────────────────────────┘
```

### Baseline 1: Landmark + 1D Temporal CNN + Bi-GRU (Phase 0 Baseline)
*   **Target Task:** Isolated sign classification (WLASL-100 & INCLUDE 263 ISL).
*   **Architecture:** MediaPipe Holistic → 30-frame window → 1D Temporal CNN → Bi-directional GRU (128 units) → Dense Softmax.
*   **Model Size:** ~3.5 MB (FP32) / ~1.0 MB (INT8).
*   **Role:** Verifies OpenCV capture, MediaPipe extraction, ML inference, and audio output in Week 1.

### Baseline 2: Landmark + Transformer Encoder + CTC (Phase 1–2 Core System)
*   **Target Task:** Continuous Sign Language Recognition (CSLR) & Translation (SLT).
*   **Architecture:**
    1.  *Vision Encoder:* 6-layer Multi-Head Self-Attention Transformer Encoder ($d_{\text{model}}=512$, 8 heads, $d_{\text{ff}}=2048$, dropout $=0.1$).
    2.  *Decoder:* CTC Beam Search Decoder (Beam Width $=8$).
    3.  *Translator:* Quantized INT8 HuggingFace `t5-small` sequence-to-sequence model.
    4.  *TTS Engine:* Offline Piper TTS engine.
*   **Model Size:** ~30 MB (Encoder) + ~0.5 MB (CTC Head) + ~65 MB (T5-small INT8) + ~50 MB (Piper TTS) = **~145.5 MB Total**.
*   **Role:** Production-grade engine providing SOTA accuracy-efficiency ratio and handling unaligned continuous signing.

### Baseline 3: Online Gloss-Free Pose-to-Text Transformer (Phase 3+ Endgame)
*   **Target Task:** Direct Pose-to-Text translation without gloss intermediate dependencies (inspired by AutoSign & GFSLT-VLP).
*   **Role:** Scales the system to unannotated sign language video sources across 300+ languages in Phase 4+.

## 2.2 Complete 6-Stage ML Pipeline Architecture

```
┌──────────────────┐     ┌───────────────────────┐     ┌────────────────────────┐
│ 640x480 RGB Video│ ──► │ MediaPipe Holistic    │ ──► │ Feature Normalization  │
│ Stream (30 FPS)  │     │ Landmark Extractor    │     │ (Origin, Scale, Vel)   │
└──────────────────┘     └───────────────────────┘     └────────────────────────┘
                                                                   │
                                                                   ▼
┌──────────────────┐     ┌───────────────────────┐     ┌────────────────────────┐
│ Piper Audio TTS  │ ◄── │ Seq2Seq T5 Translator │ ◄── │ Transformer Encoder    │
│ (Offline Playback│     │ (Gloss ──► English)   │     │ + CTC Beam Search      │
└──────────────────┘     └───────────────────────┘     └────────────────────────┘
```

### Stage 1: Video Capture & Dynamic Window Buffer
*   Input Stream: $640 \times 480$ RGB at 30 FPS.
*   Window Buffer: Dynamic buffer size $T \in [16, 128]$ frames with stride $s = 4$ frames for continuous streaming evaluation.

### Stage 2: Multi-Modal Landmark Selection (135 Key Points)
MediaPipe Holistic extracts 543 3D coordinates per frame, downsampled to **135 key landmarks**:
1.  **Left Hand:** 21 landmarks $(x, y, z) = 63$ features.
2.  **Right Hand:** 21 landmarks $(x, y, z) = 63$ features.
3.  **Upper Body Pose:** 23 landmarks $(x, y, z, \text{visibility}) = 92$ features (shoulders, elbows, wrists, nose, eyes).
4.  **Key Face Mesh:** 70 expressive facial points $(x, y, z) = 210$ features (eyebrows, mouth shape, eye gaze).

### Stage 3: Normalization & Velocity Formulations
1.  **Origin Centering:** Midpoint of left and right shoulders set to origin $(0, 0, 0)$:
    $$\vec{P}_{\text{norm}} = \vec{P} - \frac{\vec{P}_{\text{L\_shoulder}} + \vec{P}_{\text{R\_shoulder}}}{2}$$
2.  **Scale Invariance:** Distance between shoulders normalized to $1.0$:
    $$S = \|\vec{P}_{\text{L\_shoulder}} - \vec{P}_{\text{R\_shoulder}}\|_2 \implies \vec{P}_{\text{final}} = \frac{\vec{P}_{\text{norm}}}{S}$$
3.  **Velocity Calculation:** Temporal displacement across adjacent frames:
    $$\vec{V}_t = \vec{P}_{\text{final}, t} - \vec{P}_{\text{final}, t-1}$$
4.  **Final Vector Dimension:** $d_{\text{feat}} = 428 \text{ (Position)} + 428 \text{ (Velocity)} = \mathbf{856 \text{ float32 values per frame}}$.

### Stage 4: Transformer Vision Encoder & CTC Decoding
*   Linear projection maps $\mathbb{R}^{856} \to \mathbb{R}^{512}$.
*   Sinusoidal positional encodings are added to sequence tensor $[B, T, 512]$.
*   CTC Loss Formulation:
    $$\mathcal{L}_{\text{CTC}} = -\ln P(Y \mid X) = -\ln \sum_{\pi \in \mathcal{B}^{-1}(Y)} P(\pi \mid X)$$
    where $\mathcal{B}$ collapses consecutive identical tokens and removes blank tokens $\epsilon$.

### Stage 5: Gloss-to-Text NLP Translator
*   Quantized `t5-small` model converts raw gloss syntax (e.g., `"STORE MY MOTHER GO-TO EVERY-WEEK"`) into fluent natural English (`"My mother goes to the store every week."`).

### Stage 6: Offline Text-to-Speech (Piper TTS)
*   Piper TTS engine synthesizes audio using local voice model `en_US-lessac-medium` (ONNX format, ~50 MB).

---

# SECTION 3: SYSTEM REQUIREMENTS DOCUMENT (SRD)

## 3.1 Software Stack & Dependency Specifications

| Component | Library / Tool | Version | License / Cost |
| :--- | :--- | :--- | :--- |
| **Operating System** | Windows 11 / Linux (Ubuntu 22.04+) | 64-bit | User System |
| **Python Runtime** | Python | 3.10.x / 3.11.x | Open Source |
| **Deep Learning** | PyTorch / PyTorch Lightning | 2.2.0+ (CUDA 12.1) | PyTorch License (Free) |
| **Landmark Extractor** | MediaPipe Holistic | 0.10.x | Apache 2.0 (Free) |
| **Computer Vision** | OpenCV (`opencv-python`) | 4.9.0 | Apache 2.0 (Free) |
| **Inference Runtime** | ONNX Runtime GPU / CPU | 1.17.x | MIT License (Free) |
| **NLP Transformer** | HuggingFace `transformers` | 4.38.x | Apache 2.0 (Free) |
| **TTS Engine** | Piper TTS | 1.2.0 | MIT License (Free) |
| **Desktop UI** | PyQt6 | 6.6.x | GPL / Free |

## 3.2 4-Thread Asynchronous Concurrency Model

To ensure smooth 30 FPS video preview without GUI stuttering or model processing lag, execution is divided across **4 asynchronous threads**:

```
┌─────────────────────────────────────────────────────────────┐
│ Thread 1: Main GUI & Video Capture Loop (30 FPS)             │
│ - OpenCV frame grab + CLAHE contrast enhancement            │
│ - MediaPipe Holistic landmark extraction                    │
│ - Render webcam preview + skeleton overlay                  │
└──────────────────────────────┬──────────────────────────────┘
                               │ (Emits 135-dim landmark tensors to Queue 1)
                               ▼
┌─────────────────────────────────────────────────────────────┐
│ Thread 2: ML Recognition Engine Thread                      │
│ - Maintains 128-frame sliding ring buffer                   │
│ - Runs Vision Transformer Encoder + CTC Beam Search         │
│ - Emits decoded Gloss token stream                          │
└──────────────────────────────┬──────────────────────────────┘
                               │ (Emits Gloss Tokens to Queue 2)
                               ▼
┌─────────────────────────────────────────────────────────────┐
│ Thread 3: NLP Gloss-to-Text Translation Thread              │
│ - Accepts Gloss tokens (e.g. "STORE MY MOTHER GO-TO")        │
│ - Runs quantized INT8 T5-small model                        │
│ - Emits formatted English string                            │
└──────────────────────────────┬──────────────────────────────┘
                               │ (Emits English String to Queue 3)
                               ▼
┌─────────────────────────────────────────────────────────────┐
│ Thread 4: Offline Audio Synthesis Thread                    │
│ - Receives English string                                   │
│ - Invokes Piper TTS engine                                  │
│ - Streams audio output through speaker system               │
└─────────────────────────────────────────────────────────────┘
```

## 3.3 Directory Layout & Module Structure

```
d:\finalspeak\
├── data\                                 # Dataset storage (119 GB allocated)
│   ├── raw\                              # Raw dataset downloads
│   │   ├── include\                      # INCLUDE ISL Dataset (~2 GB)
│   │   ├── phoenix14t\                   # PHOENIX-2014T DGS Dataset (~12 GB)
│   │   └── wlasl\                        # WLASL Dataset (~75 GB)
│   ├── processed\                        # Pre-extracted .npy landmark arrays
│   └── annotations\                      # Gloss-text CSV mappings
├── models\                               # Exported ONNX binaries
│   ├── baseline_bigru.onnx               # Phase 0 model (~3.5 MB)
│   ├── encoder_cslr.onnx                 # Phase 1 Vision Transformer (~30 MB)
│   ├── translator_t5.onnx                # Phase 2 Quantized T5-small (~65 MB)
│   └── piper_en.onnx         # Piper TTS voice model (~50 MB)
├── src\                                  # Python source code
│   ├── capture\                          # Camera capture & MediaPipe extractor
│   │   ├── camera.py
│   │   └── extractor.py
│   ├── models\                           # PyTorch neural architectures
│   │   ├── bigru.py
│   │   ├── transformer.py
│   │   └── ctc_decoder.py
│   ├── nlp\                              # T5 Gloss translator module
│   │   └── translator.py
│   ├── audio\                            # Piper TTS audio engine
│   │   └── tts_engine.py
│   ├── ui\                               # PyQt6 Graphical User Interface
│   │   ├── main_window.py
│   │   └── components.py
│   └── utils\                            # Normalization & metric scripts
│       ├── normalization.py
│       └── metrics.py
├── train.py                              # PyTorch Lightning training runner
├── export.py                             # ONNX quantization script
├── app.py                                # Desktop application launcher
└── requirements.txt                      # Explicit Python dependencies
```

---

# SECTION 4: DATASET STRATEGY & STORAGE ALLOCATION

## 4.1 Datasets Inventory & Selection Rationale

| Tier | Dataset | Sign Language | Format | Disk Size | Role in Project |
| :---: | :--- | :---: | :---: | :---: | :--- |
| **Tier 1** | **INCLUDE (IIT Madras)** | ISL | ISLR (263 signs) | ~2 GB | Primary ISL isolated sign baseline. |
| **Tier 1** | **PHOENIX-2014T** | DGS | CSLR + SLT | ~12 GB | Gold standard continuous CSLR benchmark. |
| **Tier 1** | **WLASL (Subset 2000)** | ASL | ISLR (2000 signs) | ~75 GB | Primary ASL word recognition corpus. |
| **Tier 2** | **MS-ASL** | ASL | ISLR (1000 signs) | ~30 GB | Signer diversity augmentation. |
| **Tier 3** | **How2Sign** | ASL | CSLR + SLT | ~200 GB | Optional continuous corpus (deferred to Phase 3). |

**Total Storage Allocation:** $2 \text{ GB} + 12 \text{ GB} + 75 \text{ GB} + 30 \text{ GB} = \mathbf{119 \text{ GB}}$, leaving **~281 GB of free disk buffer** on your 400 GB SSD.

## 4.2 Offline Landmark Extraction (The Landmark Grind)
Raw MP4 video files are processed once offline:
1. `extractor.py` passes frames through MediaPipe Holistic.
2. Extracts 135 key coordinates $\times 3 = 405$ float32 values per frame.
3. Saves binary `.npy` arrays using `np.savez_compressed()`.
4. **Data Compression Ratio:** Reduces 100 GB of video into **~3.5 GB** of compact arrays, accelerating GPU loader throughput by 400%.

---

# SECTION 5: EMBEDDED SELF-OPTIMIZING RISK MITIGATION MATRIX

| Technical Risk | Cause | Automatic Software Mitigation |
| :--- | :--- | :--- |
| **Motion Blur Landmark Loss** | Fast hand movement causing tracking failure | **Cubic Spline Interpolation:** Fills coordinate gaps automatically; applies 3-frame Gaussian temporal smoothing filter. |
| **CUDA Out-Of-Memory (OOM)** | Large sequence batch sizes on RTX 4050 (6 GB VRAM) | **PyTorch AMP & Accumulation:** Automatic Mixed Precision (FP16) + Gradient Accumulation $= 4$. VRAM usage $< 4.2 \text{ GB}$. |
| **CTC Loss Divergence** | Random initialization of CTC projection layer | **Encoder Pre-Training Warmup:** Pre-train encoder on isolated signs first; apply Cosine Annealing learning rate schedule with max grad norm clipping $= 1.0$. |
| **Video Processing Lag** | Sequential execution in single Python thread | **4-Thread Async Queue:** Decouples video capture, ML inference, NLP translation, and audio playback into independent threads. |
| **Cloud Dependency Risk** | Third-party API changes or billing charges | **Zero Cloud Hard Lock:** 100% of models (MediaPipe, PyTorch/ONNX, T5-small, Piper TTS) run locally on device. |

---

# SECTION 6: PHASED IMPLEMENTATION ROADMAP & BLUEPRINT

```
===================================================================================
                       SIGNSPEAK UNIVERSAL ROADMAP BROCHURE
===================================================================================

  PHASE 0: PROOF OF CONCEPT (Weeks 1-2) ────► Baseline 1 Prototype
  ---------------------------------------------------------------------------------
  • Install PyTorch CUDA 12.1, OpenCV, MediaPipe Holistic, PyQt6.
  • Download INCLUDE (~2 GB) + WLASL-100 subset.
  • Extract 201-dim landmark vectors to compressed .npy caches.
  • Train Baseline 1 (1D CNN + Bi-GRU) on RTX 4050 GPU (< 30 min).
  • Deliverable: Live webcam script recognizing 100 ASL / 263 ISL words with audio.

  PHASE 1: CORE CSLR ENGINE (Weeks 3-6) ────► Baseline 2 Engine
  ---------------------------------------------------------------------------------
  • Download PHOENIX-2014T (~12 GB) and full WLASL-2000 (~75 GB).
  • Build 6-Layer Transformer Encoder with positional embeddings.
  • Implement PyTorch CTC Loss + CTC Beam Search Decoder.
  • Train Transformer Encoder on RTX 4050 with FP16 Mixed Precision.
  • Deliverable: Continuous sentence recognition yielding Gloss token streams.

  PHASE 2: FULL TRANSLATION & TTS (Weeks 7-10) ──► Full Local Pipeline
  ---------------------------------------------------------------------------------
  • Fine-tune HuggingFace T5-small on Gloss-to-English translation pairs.
  • Integrate Piper TTS engine for local offline voice synthesis.
  • Export PyTorch Vision Transformer & T5 model to ONNX INT8 format.
  • Build PyQt6 Desktop Interface displaying camera feed, glosses, text, audio.
  • Deliverable: Fully offline desktop application (SignSpeak Universal v1.0).

  PHASE 3: STREAMING & POLISH (Weeks 11-14) ──► Optimizations
  ---------------------------------------------------------------------------------
  • Implement 4-Thread Async Engine (Camera, Vision ML, NLP, Audio).
  • Implement dynamic sliding window ring buffer (16-128 frames).
  • Measure & verify WER < 25%, BLEU-4 > 20.0, Latency < 500 ms.
  • Optional: Download How2Sign (~200 GB) for additional ASL fine-tuning.
  • Deliverable: Optimized, production-ready desktop release.

  PHASE 4+: FUTURE EXTENSIONS (Month 4+) ──► Advanced Scaling
  ---------------------------------------------------------------------------------
  • Upgrade to Baseline 3 (Online Gloss-Free Pose-to-Text Transformer).
  • Port ONNX INT8 models to Mobile (TFLite / CoreML) for Android app release.
  • Scale to 300+ Sign Languages via LoRA adapters & Prototypical Networks.
  • Deliverable: SignSpeak Universal Mobile & Multilingual Edition.
===================================================================================
```

---

# SECTION 7: STEP-BY-STEP DEVELOPER EXECUTION MANUAL & CODE BLUEPRINTS

### Step 1: Environment Initialization Commands
```powershell
# Navigate to workspace
cd d:\finalspeak

# Create Python 3.10 virtual environment
python -m venv venv
.\venv\Scripts\activate

# Install PyTorch with CUDA 12.1 support
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Install project dependencies
pip install mediapipe opencv-python onnxruntime-gpu transformers pytorch-lightning pyttsx3 pyqt6 pandas numpy ctcdecode
```

### Step 2: Key Landmark Extractor Blueprint (`src/capture/extractor.py`)
```python
import cv2
import mediapipe as mp
import numpy as np

class HolisticExtractor:
    def __init__(self):
        self.mp_holistic = mp.solutions.holistic
        self.holistic = self.mp_holistic.Holistic(
            static_image_mode=False,
            model_complexity=1,
            smooth_landmarks=True
        )

    def extract_frame(self, frame_bgr):
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        results = self.holistic.process(frame_rgb)
        
        # Left Hand (21 x 3 = 63)
        lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(63)
        
        # Right Hand (21 x 3 = 63)
        rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(63)
        
        # Upper Body Pose (23 x 4 = 92)
        pose = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark[:23]]).flatten() if results.pose_landmarks else np.zeros(92)
        
        # Key Face Mesh (70 x 3 = 210)
        face = np.array([[res.x, res.y, res.z] for res in results.face_landmarks.landmark[:70]]).flatten() if results.face_landmarks else np.zeros(210)
        
        # Concatenate 135 key landmarks = 428 values
        return np.concatenate([lh, rh, pose, face])
```

### Step 3: PyTorch Transformer Vision Encoder Blueprint (`src/models/transformer.py`)
```python
import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=500):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]

class SignTransformerEncoder(nn.Module):
    def __init__(self, in_features=856, d_model=512, num_classes=1000, num_layers=6):
        super().__init__()
        self.input_proj = nn.Linear(in_features, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=8, dim_feedforward=2048, dropout=0.1, activation='gelu', batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.ctc_head = nn.Linear(d_model, num_classes + 1) # +1 for CTC blank token

    def forward(self, x):
        # x shape: [Batch, Time_Steps, 856]
        x = self.input_proj(x)
        x = self.pos_encoder(x)
        features = self.transformer(x)
        logits = self.ctc_head(features)
        return logits
```

---

*End of Master Specification Document — SignSpeak Universal Version 4.0*  
*This single file represents the complete, unified source of truth for the project.*
