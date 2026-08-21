# SIGNSPEAK UNIVERSAL: A REAL-TIME ASSISTIVE COMMUNICATION COCKPIT FOR INDIAN SIGN LANGUAGE
## Mini Project Mock-2 Comprehensive Technical Report

---

**Academic Year:** 2025 – 2026  
**Course / Degree:** Master of Technology / Bachelor of Technology  
**Milestone:** Mini Project Mock 2 Comprehensive Evaluation  
**Department:** Department of Computer Science and Engineering  
**Institution:** Ramrao Adik Institute of Technology (Under the ambit of D. Y. Patil Deemed to be University), Nerul, Navi Mumbai  

**Project Authors & Investigators:**
* **Khaja Naseeruddin M** — Roll No: `24MT7021`
* **Khan Amaan** — Roll No: `24MT7022`

**Project Guide & Supervisor:**
* **Dr. Pallavi Vasant Sapkale**

---

## 📜 Academic Certificate & Declaration

This is to certify that the technical report entitled **"SignSpeak Universal: A Real-Time Assistive Communication Cockpit for Indian Sign Language"** is a bonafide record of the engineering research, mathematical formulation, model training, architectural iteration, and software implementation carried out by **Khaja Naseeruddin M (24MT7021)** and **Khan Amaan (24MT7022)** in partial fulfillment of the requirements for the evaluation of **Mini Project Mock 2** under the supervision and guidance of **Dr. Pallavi Vasant Sapkale**.

The content presented in this report has not been submitted elsewhere for the award of any other degree or diploma.

---

## 🕊️ Acknowledgements

We express our deepest and most sincere gratitude to our project guide, **Dr. Pallavi Vasant Sapkale**, whose continuous encouragement, technical insight, and high standards guided us through every phase of this project. Her emphasis on building practical, human-centric assistive technology rather than purely theoretical models challenged us to address latency, dialect collisions, and user fatigue head-on.

We also extend our heartfelt appreciation to the Principal, Head of the Department, and the faculty members of the Department of Computer Science and Engineering for providing the laboratory infrastructure and computational environment necessary to conduct our experiments. We thank the open-source machine learning and accessibility communities whose foundational tools made this research possible. Finally, we thank our families and peers for their continuous moral support throughout this journey.

---

## 📋 Executive Summary / Abstract

Over 18 million individuals in India live with severe hearing and speech impairments, relying on Indian Sign Language (ISL) as their primary means of expression. Despite rapid advances in machine learning, most existing automated sign language recognition systems remain confined to academic papers due to three compounding flaws: high latency, closed-dictionary constraints, and a one-way communication architecture that assumes only the Deaf individual needs to communicate.

This report documents our engineering journey in conceiving, prototyping, diagnosing, pivoting, and deploying **SignSpeak Universal**—a real-time, camera-first assistive communication workspace. We begin by detailing our initial **Low-Fidelity Baseline Prototype**, which attempted to recognize 364 dynamic word-level glosses using Spatial-Temporal Graph Convolutional Networks (ST-GCN) over 30-frame sliding windows. We analyze why this baseline suffered from severe dialect collisions when combining international datasets (resulting in a dismal 42.8% accuracy), unbearable temporal buffering lag (>500ms), and single-threaded GUI freezing.

We then present our **Strategic Architectural Pivot** to a single-frame 35-class ISL/ASL fingerspelling and digit paradigm (A–Z, 1–9) with a 126-dimensional scale- and translation-invariant landmark geometry. To eliminate user variation, we built a multi-core harvesting pipeline that filtered 129,773 raw images into 107,517 clean 3D hand vectors, paired with a custom **Sign Recorder Studio** and 3D geometric augmentation pipeline (246,104 samples). Our Deep Residual MLP achieves **99.84%–99.96% accuracy** and is exported into an ultra-compact **556 KB ONNX runtime binary** executing in **<1.8ms**.

Finally, we trace how this engine evolved into a complete **Two-Way Assistive Communication Workspace** running across five asynchronous threads. The system features a continuous 0.8s steady-hold dwell stabilizer with hysteresis locking, a Gboard-style AI predictive autocomplete bar, 1-click AI sign grammar polishing, non-destructive syntax reversion, an 8-language regional Indian voice engine (Hindi, Telugu, Tamil, Marathi, Kannada, Bengali, Gujarati, English), and a reverse speech-to-sign loop powered by Whisper AI. We conclude by detailing our standalone compilation strategy and providing comprehensive empirical benchmarks verifying real-time performance on standard consumer laptops at zero cloud cost.

---

# TABLE OF CONTENTS

1. **Chapter 1: Introduction, Problem Formulation & Socio-Technical Context**
   * 1.1 Starting from Zero: The Human Motivation Behind SignSpeak
   * 1.2 Linguistic Characteristics of Indian Sign Language (ISL)
   * 1.3 Our Core Engineering Commitments
2. **Chapter 2: Phase 1 — The Low-Fidelity Prototype & Baseline Failure Analysis**
   * 2.1 Our Initial Hypothesis: The 364-Word Dynamic Gesture Model
   * 2.2 Feature Engineering: 856-Dimensional Spatio-Temporal Vectors
   * 2.3 The Failure Post-Mortem: Dialect Collision, Buffer Latency, and Closed Vocabularies
   * 2.4 Single-Threaded Desktop Lag (Frame Drops to 5–12 FPS)
3. **Chapter 3: Phase 2 — The Strategic Architectural Pivot & Coordinate Geometry**
   * 3.1 The Breakthrough: Pivoting to Spatial Fingerspelling
   * 3.2 Mathematical Formulation of 126-Dimensional Invariant Features
   * 3.3 Active Hand Mirroring for Left/Right Hand Invariance
   * 3.4 Multi-Core Harvesting and Automated Quality Control Filtering
4. **Chapter 4: Phase 3 & 4 — Deep Residual MLP, Sign Studio & 3D Augmentations**
   * 4.1 Deep Residual MLP Topology (`ISLLetterClassifier`)
   * 4.2 Loss Formulation with Label Smoothing
   * 4.3 200-Epoch GPU Training Run and ONNX Export (<1.8ms)
   * 4.4 The Inter-User Anatomical Variance Challenge
   * 4.5 Building Our Own Sign Recorder Studio GUI (33 Classes)
   * 4.6 3D Geometric Spatial Augmentation Mathematics (246k Samples)
   * 4.7 8-Second GPU Co-Training and Zero-Forgetting Fine-Tuning (99.96% Acc)
5. **Chapter 5: Phase 5 & 6 — Multi-Threading, Dwell Stabilizer & AI Linguistic Bridge**
   * 5.1 Five-Thread Asynchronous Decoupled Engine Architecture
   * 5.2 The 0.8s Steady-Hold Dwell Stabilizer with Hysteresis Lock
   * 5.3 One-Euro Signal Filtering for Webcam Tremor Elimination
   * 5.4 Ergonomic Keyboard Mechanics
   * 5.5 Gboard-Style AI Predictive Autocomplete Strip
   * 5.6 1-Click AI Sign Grammar Polish (`Ctrl + P`) and Revert (`Ctrl + Z`)
   * 5.7 Multilingual Indian Regional Speech Engine (8 Supported Languages)
6. **Chapter 6: Phase 7 & 8 — Two-Way Loop, UI/UX Overhaul, Evaluation & Standalone Deployment**
   * 6.1 The Breakthrough: Closing the Two-Way Conversational Loop
   * 6.2 Camera-First Spacious Assistive UI/UX Overhaul
   * 6.3 Quantitative Evaluation and Latency Benchmarks (23.4ms Total)
   * 6.4 Standalone Windows Executable (.exe) Compilation Strategy
   * 6.5 Conclusion and Future Horizons
7. **References**

---

# CHAPTER 1: INTRODUCTION AND PROBLEM FORMULATION

```
┌──────────────────────────────────────────────────────────────────────────────────────────────────┐
│                                   THE ACCESSIBILITY PROBLEM                                      │
├──────────────────────────────────────────────────────────────────┬───────────────────────────────┤
│                     THE DEAF SIGNER'S REALITY                    │    THE HEARING WORLD'S WALL   │
├──────────────────────────────────────────────────────────────────┼───────────────────────────────┤
│ • 18+ Million Deaf citizens in India (ISL as primary language)   │ • <0.01% Hearing population   │
│ • Rich visual-spatial grammar with non-linear syntax             │   understands sign language   │
│ • Daily communication barriers in hospitals, banks, and schools  │ • Certified interpreters are  │
│ • Sign glosses are telegraphic and lack spoken function words     │   scarce and expensive        │
└──────────────────────────────────────────────────────────────────┴───────────────────────────────┘
```

### 1.1 Starting from Zero: The Human Motivation Behind SignSpeak
Every engineering journey begins with a spark of human reality. When we first began conceptualizing this project, we asked ourselves a fundamental question: *In an era where artificial intelligence can generate photorealistic videos and compose music, why are over 18 million Deaf individuals in India still unable to have a simple, independent conversation at a clinic, a bank counter, or a grocery store?*

In India, Indian Sign Language (ISL) is the primary language, identity, and medium of expression for millions of citizens. Yet, fewer than 0.01% of the hearing population can understand even the most basic sign gesture. When a Deaf person visits a doctor, opens a bank account, or attends a lecture, they are almost always forced to rely on handwritten notes, awkward gestures, or family chaperones. Certified human interpreters are scarce, prohibitively expensive, and geographically restricted to major tier-one metropolitan hubs.

We set out with an ambitious goal: to build **SignSpeak Universal**—a lightweight, camera-first assistive communication application that turns any standard laptop into a real-time, bidirectional communication cockpit for Indian Sign Language without requiring specialized hardware or paid cloud subscriptions.

```
+--------------------------------------------------------------------------------------------------+
| FIG 1: SignSpeak Universal End-to-End Development Journey Timeline                               |
| Phase 1: Inception -> Phase 2: Low-Fi ST-GCN (42.8%) -> Phase 3: Strategic Pivot (35-Class)     |
| -> Phase 4: Harvester & Studio (246k) -> Phase 5: High-Fi MLP (99.96%) -> Phase 6: Multi-Thread |
| -> Phase 7: Two-Way Whisper Loop & Camera-First Workspace                                        |
+--------------------------------------------------------------------------------------------------+
```

### 1.2 Linguistic Nuances of Indian Sign Language
Early in our research, we realized that sign language is not simply spoken language translated into gestures. ISL is a rich, natural, visual-spatial language with its own distinct phonology and grammar:
1. **Topicalized SOV Syntax:** ISL predominantly structures ideas using Subject-Object-Verb (SOV) or Time-Subject-Object-Verb (TSOV) word orders, placing the main topic at the very beginning of the expression.
2. **Omission of Functional Copulas and Articles:** ISL signs omit auxiliary verbs (such as *is, are, was*), articles (*a, an, the*), and prepositions. For example, the conversational sentence *"Please give me a glass of water"* is physically signed as a sequence of disjointed semantic glosses:
   $$\text{[ ME ]} \longrightarrow \text{[ WATER ]} \longrightarrow \text{[ DRINK ]} \longrightarrow \text{[ WANT ]}$$
3. **The Essential Role of Fingerspelling:** While common words have dedicated dynamic gestures, open-ended vocabulary—such as Indian personal names (*Naseer, Amaan*), medication names (*Paracetamol*), technical terms, and regional towns—relies entirely on fingerspelling through standardized static and dynamic alphabet hand shapes.

### 1.3 Our Core Engineering Commitments
To ensure this project was not just another theoretical academic exercise, we committed to five strict engineering principles:
* **Infinite Vocabulary Reach:** The system must not be locked into a static dictionary of 200 or 300 pre-trained words. It must enable users to spell and construct any word in the English and Indian lexicon.
* **Sub-50ms Real-Time Latency:** The end-to-end delay from physical hand motion to speech output must remain under 50 milliseconds to preserve natural conversational rhythm.
* **Zero Cloud Lock-In ($0.00 Runtime Cost):** All computer vision, neural inference, and speech synthesis must run 100% locally on consumer laptops, protecting user privacy and eliminating cloud fees.
* **Natural Linguistic Syntax Bridge:** The system must automatically convert telegraphic sign glosses into fluent, grammatically natural spoken sentences.
* **True Two-Way Conversational Parity:** The system must not only vocalize what the deaf user signs, but also transcribe what the hearing partner says, displaying clear visual sign badges back to the signer.

---

# CHAPTER 2: PHASE 1 — THE LOW-FIDELITY PROTOTYPE & BASELINE FAILURE ANALYSIS

### 2.1 Our Initial Hypothesis: The 364-Word Dynamic Gesture Model
When we started our experimentation in early exploratory Jupyter notebooks (`part_1.ipynb` and `part_2.ipynb`), we followed the mainstream trend in academic literature: dynamic word-level sign recognition. Our initial plan was to build a system capable of recognizing 364 continuous sign language word glosses (such as *"Hospital", "Doctor", "Car", "Election", "Telephone"*).

To assemble a sufficiently large video dataset, we merged two prominent public repositories:
* **INCLUDE Dataset:** An Indian Sign Language video corpus recorded across Indian institutions.
* **WLASL Dataset:** A large-scale American Sign Language benchmark containing diverse video recordings.

Using MediaPipe Holistic, we extracted skeletal landmarks across the body, face, and hands. We computed relative joint distances and first-order velocity deltas across consecutive video frames, constructing an **856-dimensional feature vector per frame**. We fed 30-frame temporal sliding window tensors ($\mathbf{X} \in \mathbb{R}^{30 \times 856}$) into a Spatial-Temporal Graph Convolutional Network (ST-GCN).

### 2.2 The Failure Post-Mortem: What Went Wrong
When we deployed our trained ST-GCN model on our webcam, our initial excitement turned into a humbling realization. The prototype failed across four critical dimensions:

1. **Dialect Collision (Accuracy Collapsed to 42.8%):** Indian Sign Language (ISL) and American Sign Language (ASL) have fundamentally different gestural vocabularies. For instance, the sign for *"Hospital"* in ISL involves a two-handed cross gesture near the shoulder, whereas in ASL it requires drawing an *"H"* handshape downward across the opposite arm. By merging both datasets into one 364-class taxonomy, the neural network was forced to map conflicting spatio-temporal trajectories to the same label, causing top-1 accuracy to collapse to **42.8%**.
2. **The 30-Frame Buffering Lag (>1,000 ms Latency):** A dynamic sequence classifier requiring 30 video frames at 30 FPS cannot output a single prediction until all 30 frames are captured. This introduced a mandatory **1,000ms physical gesture buffering delay** plus 80–120ms of neural inference time. In live tests, signers felt an unnatural lag between their hands moving and text appearing on the screen.
3. **The Closed-Dictionary Barrier:** The word-level model could only recognize the exact 364 words in its training set. When we tried to sign our names (*"Amaan"* or *"Naseer"*) or a simple medical term like *"Paracetamol"*, the model had no way to process the input, outputting random high-loss predictions.
4. **Single-Threaded Desktop Lag (Frame Drops to 5–12 FPS):** Our initial prototype executed OpenCV video capture, MediaPipe landmark extraction, ST-GCN inference, and UI rendering in a single sequential execution loop. The heavy matrix operations of the graph neural network starved the camera feed of CPU cycles, causing severe video stuttering from 30 FPS down to **5–12 FPS**.

| Dimension | Low-Fidelity Baseline (Phase 1) | Identified Bottleneck |
| :--- | :--- | :--- |
| **Recognition Paradigm** | 30-Frame Word Sequences | 1,000ms buffer lag before prediction |
| **Vocabulary Reach** | Closed 364-Word Dictionary | Complete failure on names and medical terms |
| **Dataset Integration** | Merged ISL + ASL Videos | Severe dialect clashes (42.8% accuracy) |
| **System Architecture** | Single-Threaded Script | Severe GUI freezing (5–12 FPS) |

---

# CHAPTER 3: PHASE 2 — THE STRATEGIC ARCHITECTURAL PIVOT & COORDINATE GEOMETRY

### 3.1 The Breakthrough: Pivoting to Spatial Fingerspelling
After analyzing the failure of our word-level baseline, we had our defining insight:

> *Rather than forcing a neural network to memorize a fixed dictionary of dynamic words across a 30-frame temporal buffer, we pivoted to a high-speed, single-frame 35-class alphabet and digit fingerspelling paradigm (classes `A` through `Z` and digits `1` through `9`).*

This single architectural decision solved all three core bottlenecks:
1. **Sub-2ms Inference Latency:** Evaluating a single static frame eliminated the 30-frame video buffer, reducing inference latency from >500ms down to **<1.8 ms**.
2. **Infinite Vocabulary Reach:** Users can spell any word, proper noun, Indian name, or medical term character-by-character, which a downstream software layer can assemble into full sentences.
3. **Dialect Consistency:** Standard ISL/ASL fingerspelling hand postures exhibit consistent spatial geometry, eliminating the cross-regional dialect clashes that collapsed baseline accuracy.

### 3.2 Mathematical Formulation of 126-Dimensional Invariant Features
Raw pixel coordinates $(x_i, y_i, z_i)$ extracted by MediaPipe vary significantly depending on camera resolution, hand size, and distance from the lens. To guarantee generalization across diverse webcams, we derived a two-stage geometric normalization pipeline:

Let $\mathbf{P}_i = (x_i, y_i, z_i) \in \mathbb{R}^3$ denote the 3D spatial coordinates of the $i$-th hand keypoint for $i \in \{0, 1, \dots, 20\}$, where $\mathbf{P}_0$ represents the wrist landmark.

#### Stage 1: Wrist-Centered Origin Translation
All 21 keypoints are translated relative to the wrist origin:
$$\mathbf{P}'_i = \mathbf{P}_i - \mathbf{P}_0 \quad \forall i \in \{0, 1, \dots, 20\}$$

#### Stage 2: Hand Span Euclidean Scale Normalization
We compute the Euclidean distance between the wrist ($\mathbf{P}_0$) and the middle finger Metacarpophalangeal (MCP) joint ($\mathbf{P}_9$), defining the characteristic hand scale $S$:
$$S = \|\mathbf{P}_9 - \mathbf{P}_0\|_2 + \epsilon$$
$$\mathbf{P}_{\text{norm}, i} = \frac{\mathbf{P}'_i}{S} \quad \forall i \in \{0, 1, \dots, 20\}$$

Concatenating the 21 normalized 3D points for a single hand yields a 63-dimensional feature vector:
$$\mathbf{X}_{\text{hand}} = \left[ \mathbf{P}_{\text{norm}, 0}^T, \mathbf{P}_{\text{norm}, 1}^T, \dots, \mathbf{P}_{\text{norm}, 20}^T \right]^T \in \mathbb{R}^{63}$$

### 3.3 Active Hand Mirroring for Left/Right Hand Invariance
We extract vectors for both Left ($\mathbf{X}_{\text{LH}} \in \mathbb{R}^{63}$) and Right ($\mathbf{X}_{\text{RH}} \in \mathbb{R}^{63}$) hands, forming a combined 126-dimensional feature vector:
$$\mathbf{X}_{\text{frame}} = \begin{bmatrix} \mathbf{X}_{\text{LH}} \\ \mathbf{X}_{\text{RH}} \end{bmatrix} \in \mathbb{R}^{126}$$

When only one hand is visible in the frame, the active hand's 63-dimensional normalized vector is duplicated across both slots:
$$\mathbf{X}_{\text{frame}} = \begin{bmatrix} \mathbf{X}_{\text{active}} \\ \mathbf{X}_{\text{active}} \end{bmatrix}$$
This mathematical symmetry guarantees that the neural network receives an identical feature representation regardless of whether the user signs with their left or right hand.

### 3.4 Multi-Core Dataset Harvesting and Automated QC Filtering
We developed an automated multi-threaded harvesting pipeline (`download_and_extract_isl_letters.py`) that harvested 129,773 raw sign alphabet images across public repositories. Deploying **12 parallel CPU worker processes** running MediaPipe Hands with an automated quality threshold ($\ge 0.40$ confidence), the pipeline discarded 22,256 defective images, outputting **107,517 clean, validated 126-dimensional hand landmark vectors**.

---

# CHAPTER 4: PHASE 3 & 4 — DEEP RESIDUAL MLP, SIGN STUDIO & 3D AUGMENTATIONS

### 4.1 Deep Residual MLP Topology (`ISLLetterClassifier`)
We designed a specialized **Deep Residual Multi-Layer Perceptron (`ISLLetterClassifier`)**:
1. **Input Projection Block:** `Linear(126, 256)` + `BatchNorm1d(256)` + `SiLU` + `Dropout(0.20)`.
2. **Residual Bottleneck Block:** `Linear(256, 256)` with persistent skip connection ($\mathbf{h}_2 = \text{Block}_2(\mathbf{h}_1) + \mathbf{h}_1$).
3. **Compression Block:** `Linear(256, 128)` + `BatchNorm1d(128)` + `SiLU` + `Dropout(0.20)`.
4. **Classification Head:** `Linear(128, 35)` outputting 35 unnormalized class logits.

### 4.2 GPU Baseline Training & ONNX Export
Trained over 200 epochs on an **NVIDIA GeForce RTX 4050 GPU (6 GB VRAM, CUDA 12.1)** using **AdamW** and **Cosine Annealing**:
* **Training Accuracy:** **99.89%**
* **Validation Accuracy:** **99.67%**
* **Held-Out Test Accuracy:** **99.70%** (10,720 / 10,752 Correct)
* **Exported Binary Size:** **556 KB** (`isl_letter_classifier.onnx`, executing in **<1.8 ms** per frame).

### 4.3 Building Our Own Sign Recorder Studio GUI (33 Classes)
When we tested the model on our own hands, subtle differences in finger length ratios and resting hand postures caused minor recognition drops on letters like `R`, `U`, and `V`.

We built a dedicated desktop GUI (`sign_recorder_studio.py`) with a structured protocol:
* **3-Second Red Preparation Window:** Gives time to position the hand.
* **3-Second Green Recording Window:** Automatically captures 90 continuous 126-dimensional landmark vectors at 30 FPS.
* We recorded our own signs across **33 classes** (A–Y, 1–9) saved under `data/user_recorded/`.

### 4.4 3D Geometric Augmentation & 8-Second GPU Fine-Tuner
Our augmentation engine (`augment_and_train_master_letters.py`) applies:
1. **3D Synthetic Euler Rotations:** $\theta_x, \theta_y, \theta_z \sim \mathcal{U}(-18^\circ, +18^\circ)$.
2. **Isotropic Scale Jitter:** $s \sim \mathcal{U}(0.88, 1.12)$.
3. **Gaussian Joint Jitter:** $\mathbf{\delta} \sim \mathcal{N}(0, \sigma^2)$ with $\sigma = 0.012$.

This expanded the dataset to **246,104 augmented samples**. Our GPU co-trainer (`fine_tune_engine.py`) blends 20% user recordings with 80% baseline vectors, completing in **~8 seconds** and reaching **99.84%–99.96% validation accuracy** with zero catastrophic forgetting.

---

# CHAPTER 5: PHASE 5 & 6 — MULTI-THREADING, DWELL STABILIZER & AI LINGUISTIC BRIDGE

### 5.1 Five-Thread Asynchronous Decoupled Engine Architecture
To ensure zero camera lag or UI stuttering, SignSpeak runs across **five isolated parallel threads**:
* **Thread 1 (`CaptureThread`):** DirectShow camera capture at 30 FPS + MediaPipe 3D landmark extraction.
* **Thread 2 (`InferenceThread`):** Sub-2ms ONNX inference + 4-frame temporal confirmation filter.
* **Thread 3 (`SignSpeakApp`):** PyQt6 main UI event loop, dwell calculation, autocomplete chips, and keyboard routing.
* **Thread 4 (`TTSThread`):** Non-blocking Piper local neural voice (`en_US-lessac-medium.onnx`, <12ms) + regional Indian voice audio.
* **Thread 5 (`SpeechToTextThread`):** 16kHz background microphone audio streaming via `sounddevice` with Whisper AI transcription.

### 5.2 The 0.8s Steady-Hold Dwell Stabilizer with Hysteresis Lock
* **0.80s Hold Threshold:** Holding a recognizable sign steady fills a progress bar smoothly over 0.80s.
* **Audio Feedback:** Upon reaching 100%, the letter is captured with a soft 35ms audio tick (1250 Hz).
* **Anti-Duplication Hysteresis Lock:** Prevents accidental letter stuttering until the sign changes or the hand is lowered.

### 5.3 One-Euro Signal Filtering & Ergonomic Keyboard Controls
* **Adaptive One-Euro Filter:** Eliminates webcam coordinate jitter during stationary gestures ($f_c = 1.0\text{ Hz}$) with zero lag during transitions.
* **Spacebar Word Commit:** Commits active word into the sentence buffer.
* **Smart Backspace:** Deletes the last letter, or pulls the previous word back for editing if the builder is empty.
* **Escape Reset:** Clears word and sentence buffers instantly.

### 5.4 Gboard-Style AI Autocomplete & 1-Click Grammar Polish
* **Gboard AI Autocomplete:** Top 3 context-aware word completions (`1  HELLO`, `2  HELP`, `3  HEAR`) selectable via keys `1`, `2`, or `3`.
* **1-Click AI Grammar Polish (`Ctrl + P`):** Converts telegraphic sign glosses (*"ME WATER DRINK WANT"*) into fluent English (*"I want to drink water."*) in ~150ms.
* **Non-Destructive Revert (`Ctrl + Z`):** Restores raw signed word glosses without data loss.

### 5.5 Multilingual Indian Regional Speech Engine (8 Languages)
* **Supported Languages:** English, Hindi (हिन्दी), Telugu (తెలుగు), Tamil (தமிழ்), Marathi (मराठी), Kannada (ಕನ್ನಡ), Bengali (বাংলা), and Gujarati (ગુજરાતી).
* **1-Click Translate (`Ctrl + T`):** Translates signed text into regional script.
* **Dynamic Speak Trigger (`Enter`):** Button automatically adapts (`[ 🔊 Speak in Hindi  Enter ]`) and vocalizes sentence in one click.

---

# CHAPTER 6: PHASE 7 & 8 — TWO-WAY LOOP, UI/UX OVERHAUL, EVALUATION & STANDALONE DEPLOYMENT

### 6.1 The Two-Way Conversational Loop
SignSpeak completes the full conversational circle:
* **Forward (Signer $\to$ Hearing Partner):** Fingerspelling $\to$ Dwell capture $\to$ AI autocomplete $\to$ Grammar polish $\to$ Neural voice.
* **Reverse (Hearing Partner $\to$ Signer):**
  * Hearing partner speaks into microphone (`Ctrl + M` or `F2`).
  * `SpeechToTextThread` streams audio to Groq Whisper AI, returning transcripts in **<200ms**.
  * **Incoming Subtitles Card:** Displays spoken words in high-contrast subtitle bubble.
  * **Live ISL Fingerspelling Badges:** Renders visual sign badges (`[ H ] [ E ] [ L ] [ P ]`).
  * **Dialogue Timeline & Exporter:** Turn-by-turn chat history with 1-click export to timestamped `.txt` files or clipboard.

### 6.2 Camera-First Spacious Assistive UI/UX Overhaul
* **60% Hero Video Surface:** Camera centered with aspect-ratio preservation and status overlays.
* **Accessible 44–52px Touch Targets:** Large, comfortable click targets designed for laptop screens.
* **Progressive Disclosure Drawer:** Collapsible bottom drawer housing dialogue history and diagnostics.

### 6.3 Quantitative Evaluation and Latency Benchmarks
* **Classification Accuracy:** **99.96%** (Macro F1-Score: **99.86%**).
* **End-to-End Latency:** Measured at **23.4 ms** total lag (well below the 50ms human perception ceiling).

| Processing Stage | Measured Time | % of Total Lag |
| :--- | :--- | :--- |
| 1. Video Capture & Acquisition | 8.2 ms | 35.0% |
| 2. MediaPipe 3D Landmark Extraction | 4.1 ms | 17.5% |
| 3. Coordinate Normalization | 0.1 ms | 0.4% |
| 4. ONNX Residual MLP Forward Pass | 1.4 ms | 6.0% |
| 5. One-Euro Adaptive Filter | 0.1 ms | 0.4% |
| 6. PyQt6 UI Render & Dwell Step | 1.2 ms | 5.1% |
| 7. Piper Neural Voice Synthesis | 8.3 ms | 35.5% |
| **Total System Latency** | **23.4 ms** | **100.0%** |

### 6.4 Standalone Executable (.exe) Compilation Strategy
Packaged into a self-contained Windows executable via PyInstaller:
```powershell
pyinstaller --noconfirm --onedir --windowed `
  --add-data "models;models" `
  --add-data "data/user_recorded;data/user_recorded" `
  --hidden-import "piper" `
  --hidden-import "sounddevice" `
  --hidden-import "PyQt6" `
  prototype/part_3_letters.py
```

### 6.5 Conclusion & Future Horizons
SignSpeak Universal demonstrates that high-accuracy, low-latency assistive technology can run locally on standard laptops at zero cost. Future horizons include porting to WebAssembly (Wasm) for mobile web browsers, developing Android/iOS apps, and streaming synthesized voice directly to Bluetooth smart hearing aids.

---

# REFERENCES

1. G. Casiez, N. Roussel, and D. Vogel, "1€ filter: A simple speed-based low-pass filter for noisy input in HCI," in *Proceedings of the SIGCHI Conference on Human Factors in Computing Systems (CHI '12)*, pp. 2527–2530, 2012.
2. C. Lugaresi, J. Tang, H. Nash, C. McClanahan, E. Uboweja, M. Hays, F. Zhang, C.-L. Chang, M. G. Yong, J. Lee, et al., "MediaPipe: A framework for building perception pipelines," *arXiv preprint arXiv:1906.08172*, 2019.
3. D. Li, C. Rodriguez, X. Yu, and H. Li, "Word-level deep sign language recognition from video: A new large-scale dataset and methods comparison," in *Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision (WACV '20)*, pp. 1459–1469, 2020.
4. A. Sridhar, R. G. Ganesan, P. Kumar, and M. Khapra, "INCLUDE: A large scale dataset for Indian Sign Language recognition," in *Proceedings of the 28th ACM International Conference on Multimedia (MM '20)*, pp. 1366–1375, 2020.
5. S. Yan, Y. Xiong, and D. Lin, "Spatial temporal graph convolutional networks for skeleton-based action recognition," in *Proceedings of the AAAI Conference on Artificial Intelligence (AAAI '18)*, vol. 32, no. 1, pp. 7444–7452, 2018.
6. K. He, X. Zhang, S. Ren, and J. Sun, "Deep residual learning for image recognition," in *Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR '16)*, pp. 770–778, 2016.
7. S. Elfwing, E. Uchibe, and K. Doya, "Sigmoid-weighted linear units for neural network function approximation in reinforcement learning," *Neural Networks*, vol. 107, pp. 3–11, 2018.
8. R. Müller, S. Kornblith, and G. E. Hinton, "When does label smoothing help?" in *Advances in Neural Information Processing Systems (NeurIPS '19)*, vol. 32, pp. 4694–4703, 2019.
9. I. Loshchilov and F. Hutter, "Decoupled weight decay regularization," in *International Conference on Learning Representations (ICLR '19)*, 2019.
10. A. Radford, J. W. Kim, T. Xu, G. Brockman, C. McLeavey, and I. Sutskever, "Robust speech recognition via large-scale weak supervision," in *International Conference on Machine Learning (ICML '23)*, pp. 28492–28518, 2023.
