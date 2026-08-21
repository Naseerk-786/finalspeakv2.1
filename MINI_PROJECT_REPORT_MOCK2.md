# SIGNSPEAK UNIVERSAL: A REAL-TIME ASSISTIVE COMMUNICATION COCKPIT FOR INDIAN SIGN LANGUAGE
## Mini Project Mock-2 Comprehensive Technical Report

---

**Academic Year:** 2025 – 2026  
**Milestone:** Mini Project Mock 2 Evaluation  
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

---

## 🕊️ Acknowledgements

We express our deepest and most sincere gratitude to our project guide, **Dr. Pallavi Vasant Sapkale**, whose continuous encouragement, technical insight, and high standards guided us through every phase of this project. Her emphasis on building practical, human-centric assistive technology rather than purely theoretical models challenged us to address latency, dialect collisions, and user fatigue head-on.

We also extend our heartfelt appreciation to the Principal, Head of the Department, and the faculty members of the Department of Computer Science and Engineering for providing the laboratory infrastructure and computational environment necessary to conduct our experiments. We thank our families and peers for their continuous moral support throughout this journey.

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
   * 3.2 35-Class Spatial Taxonomy Matrix (26 Letters + 9 Digits)
   * 3.3 Mathematical Formulation of 126-Dimensional Invariant Features
   * 3.4 Active Hand Mirroring for Left/Right Hand Invariance
   * 3.5 Multi-Core Harvesting and Automated Quality Control Filtering
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
   * 5.2 Real-World Live Desktop Recognition Performance
   * 5.3 The 0.8s Steady-Hold Dwell Stabilizer with Hysteresis Lock
   * 5.4 One-Euro Signal Filtering for Webcam Tremor Elimination
   * 5.5 Ergonomic Keyboard Mechanics
   * 5.6 Gboard-Style AI Predictive Autocomplete Strip
   * 5.7 1-Click AI Sign Grammar Polish (`Ctrl + P`) and Revert (`Ctrl + Z`)
   * 5.8 Multilingual Indian Regional Speech Engine (8 Supported Languages)
6. **Chapter 6: Phase 7 & 8 — Two-Way Loop, UI/UX Overhaul, Evaluation & Standalone Deployment**
   * 6.1 The Breakthrough: Closing the Two-Way Conversational Loop
   * 6.2 Camera-First Spacious Assistive UI/UX Overhaul
   * 6.3 Quantitative Evaluation and Latency Benchmarks (23.4ms Total)
   * 6.4 Standalone Windows Executable (.exe) Compilation Strategy
   * 6.5 Conclusion and Future Horizons
7. **References**

---

# CHAPTER 1: INTRODUCTION AND PROBLEM FORMULATION

### 1.1 Starting from Zero: The Human Motivation Behind SignSpeak
Every engineering journey begins with a spark of human reality. When we first began conceptualizing this project, we asked ourselves a fundamental question: *In an era where artificial intelligence can generate photorealistic videos and compose music, why are over 18 million Deaf individuals in India still unable to have a simple, independent conversation at a clinic, a bank counter, or a grocery store?*

In India, Indian Sign Language (ISL) is the primary language, identity, and medium of expression for millions of citizens. Yet, fewer than 0.01% of the hearing population can understand even the most basic sign gesture. When a Deaf person visits a doctor, opens a bank account, or attends a lecture, they are almost always forced to rely on handwritten notes, awkward gestures, or family chaperones. Certified human interpreters are scarce, prohibitively expensive, and geographically restricted to major tier-one metropolitan hubs.

We set out with an ambitious goal: to build **SignSpeak Universal**—a lightweight, camera-first assistive communication application that turns any standard laptop into a real-time, bidirectional communication cockpit for Indian Sign Language without requiring specialized hardware or paid cloud subscriptions.

```
┌──────────────────────────────────────────────────────────────────────────────────────────────────┐
│ FIGURE 1: SignSpeak Universal End-to-End Development Journey Timeline                            │
│ Phase 1: Inception -> Phase 2: Low-Fi ST-GCN (42.8%) -> Phase 3: Strategic Pivot (35-Class)     │
│ -> Phase 4: Harvester & Studio (246k) -> Phase 5: High-Fi MLP (99.96%) -> Phase 6: Multi-Thread │
│ -> Phase 7: Two-Way Whisper Loop & Camera-First Workspace                                        │
└──────────────────────────────────────────────────────────────────────────────────────────────────┘
```

### 1.2 Linguistic Nuances of Indian Sign Language
Early in our research, we realized that sign language is not simply spoken language translated into gestures. ISL is a rich, natural, visual-spatial language with its own distinct phonology and grammar:
1. **Topicalized SOV Syntax:** ISL predominantly structures ideas using Subject-Object-Verb (SOV) or Time-Subject-Object-Verb (TSOV) word orders, placing the main topic at the very beginning of the expression.
2. **Omission of Functional Copulas and Articles:** ISL signs omit auxiliary verbs (such as *is, are, was*), articles (*a, an, the*), and prepositions. For example, the conversational sentence *"Please give me a glass of water"* is physically signed as a sequence of disjointed semantic glosses:
   $$\text{[ ME ]} \longrightarrow \text{[ WATER ]} \longrightarrow \text{[ DRINK ]} \longrightarrow \text{[ WANT ]}$$
3. **The Essential Role of Fingerspelling:** While common words have dedicated dynamic gestures, open-ended vocabulary—such as Indian personal names (*Naseer, Amaan*), medication names (*Paracetamol*), technical terms, and regional towns—relies entirely on fingerspelling through standardized static and dynamic alphabet hand shapes.

### 1.3 Our Core Engineering Commitments
* **Infinite Vocabulary Reach:** The system must enable users to construct any word in the lexicon.
* **Sub-50ms Real-Time Latency:** End-to-end delay must remain well under 50ms.
* **Zero Cloud Lock-In ($0.00 Runtime Cost):** All CV, neural inference, and speech synthesis run 100% locally.
* **Natural Linguistic Syntax Bridge:** Converts telegraphic sign glosses into fluent spoken sentences.
* **True Two-Way Conversational Parity:** Complete bidirectional loop with Whisper AI audio transcription and visual ISL badges.

---

# CHAPTER 2: PHASE 1 — THE LOW-FIDELITY PROTOTYPE & BASELINE FAILURE ANALYSIS

### 2.1 Our Initial Hypothesis: The 364-Word Dynamic Gesture Model
In exploratory notebooks (`part_1.ipynb` and `part_2.ipynb`), we built an ST-GCN model targeting 364 dynamic word glosses by merging the **INCLUDE Dataset** (ISL) and **WLASL Dataset** (ASL), extracting 856-dimensional spatio-temporal features over 30-frame temporal sliding windows.

```
┌──────────────────────────────────────────────────────────────────────────────────────────────────┐
│ FIGURE 2: Empirical Comparison: Low-Fi ST-GCN Baseline vs. High-Fi SignSpeak Universal           │
│ • Classification Accuracy: 42.8% (Baseline) vs. 99.96% (SignSpeak Residual MLP)                  │
│ • Inference Latency: 500.0 ms (Baseline) vs. 1.8 ms (SignSpeak ONNX)                             │
│ • Binary Footprint: 150.0 MB (Baseline) vs. 0.556 MB (SignSpeak ONNX)                            │
│ • GUI Framerate: 8 FPS stutter (Baseline) vs. 30 FPS smooth (SignSpeak 5-Thread Decoupled)       │
└──────────────────────────────────────────────────────────────────────────────────────────────────┘
```

### 2.2 The Failure Post-Mortem: What Went Wrong
1. **Dialect Collision (Accuracy Collapsed to 42.8%):** Conflicting spatio-temporal trajectories across ISL and ASL collapsed recognition accuracy.
2. **30-Frame Buffering Lag (>1,000 ms Delay):** Mandatory 1,000ms gesture buffering delay destroyed real-time responsiveness.
3. **Closed-Dictionary Barrier:** Complete inability to recognize proper nouns, Indian names, or medical terms.
4. **Single-Threaded OpenCV Freezing:** CPU starvation caused framerate to collapse from 30 FPS down to 5–12 FPS.

---

# CHAPTER 3: PHASE 2 — THE STRATEGIC ARCHITECTURAL PIVOT & COORDINATE GEOMETRY

### 3.1 The Breakthrough: Pivoting to Spatial Fingerspelling
We pivoted from dynamic word-sequence modeling to single-frame 35-class fingerspelling and digits (`A–Z`, `1–9`).

```
┌──────────────────────────────────────────────────────────────────────────────────────────────────┐
│ FIGURE 8: SignSpeak 35-Class Spatial Taxonomy Grid                                               │
│ [ A ] [ B ] [ C ] [ D ] [ E ] [ F ] [ G ]                                                        │
│ [ H ] [ I ] [ J ] [ K ] [ L ] [ M ] [ N ]                                                        │
│ [ O ] [ P ] [ Q ] [ R ] [ S ] [ T ] [ U ]                                                        │
│ [ V ] [ W ] [ X ] [ Y ] [ Z ]                                                                    │
│ [ 1 ] [ 2 ] [ 3 ] [ 4 ] [ 5 ] [ 6 ] [ 7 ] [ 8 ] [ 9 ]                                           │
└──────────────────────────────────────────────────────────────────────────────────────────────────┘
```

### 3.2 Mathematical Formulation of 126-Dimensional Invariant Features
```
┌──────────────────────────────────────────────────────────────────────────────────────────────────┐
│ FIGURE 3: 126-Dimensional Coordinate Invariance Geometry                                         │
│ • Origin Translation: P'_i = P_i - P_0 (Wrist Origin [0,0,0])                                    │
│ • Euclidean Hand Span Scale: S = ||P_9 - P_0||_2 + eps                                           │
│ • Normalized Keypoints: P_norm,i = P'_i / S                                                      │
│ • Active Hand Mirroring: X_frame = [X_active ; X_active] in R^126                                │
└──────────────────────────────────────────────────────────────────────────────────────────────────┘
```

### 3.3 Multi-Core Harvesting (107k Vectors from 129k Images)
Our 12-core CPU harvesting pipeline (`download_and_extract_isl_letters.py`) filtered 129,773 images with automated confidence thresholding ($\ge 0.40$), yielding **107,517 clean 126-dimensional vectors**.

---

# CHAPTER 4: PHASE 3 & 4 — DEEP RESIDUAL MLP, SIGN STUDIO & 3D AUGMENTATIONS

### 4.1 Deep Residual MLP Topology (`ISLLetterClassifier`)
* `Linear(126, 256)` $\to$ `BatchNorm1d` $\to$ `SiLU` $\to$ `Dropout(0.20)`
* `Linear(256, 256)` $\to$ Residual Skip Connection ($\mathbf{h}_2 = \text{Block}_2(\mathbf{h}_1) + \mathbf{h}_1$)
* `Linear(256, 128)` $\to$ `Linear(128, 35)` Logit Classification Head
* Trained with Label Smoothing ($\alpha = 0.05$) under AdamW and Cosine Annealing over 200 epochs.

```
┌──────────────────────────────────────────────────────────────────────────────────────────────────┐
│ FIGURE 4: CUDA GPU Training Curves on NVIDIA RTX 4050                                            │
│ • Training Accuracy: 99.89% | Validation Accuracy: 99.67% | Held-Out Test Accuracy: 99.70%      │
│ • Exported ONNX Runtime Binary: 556 KB (isl_letter_classifier.onnx, <1.8 ms inference)          │
└──────────────────────────────────────────────────────────────────────────────────────────────────┘
```

```
┌──────────────────────────────────────────────────────────────────────────────────────────────────┐
│ FIGURE / SNAPSHOT: Live Sign Recording Studio & RTX 4050 GPU Co-Trainer                          │
│ [Live Photo from Lab]: Demonstrating real-time sign capture with 3s green recording window       │
│ and GPU fine-tuner executing 35 epochs in ~8 seconds (validation accuracy: 99.92%).               │
└──────────────────────────────────────────────────────────────────────────────────────────────────┘
```

### 4.2 3D Geometric Augmentation (246k Samples) & 8-Second Fine-Tuning
Applying 3D synthetic Euler rotations ($\pm 18^\circ$), scale jitter ($0.88\times-1.12\times$), and Gaussian joint noise synthesized **246,104 samples**. The GPU co-trainer fine-tunes on RTX 4050 in **~8 seconds** to achieve **99.84%–99.96% accuracy**.

---

# CHAPTER 5: PHASE 5 & 6 — MULTI-THREADING, DWELL STABILIZER & AI LINGUISTIC BRIDGE

### 5.1 5-Thread Concurrency Architecture & Live Desktop Recognition
```
┌──────────────────────────────────────────────────────────────────────────────────────────────────┐
│ FIGURE 5: 5-Thread Asynchronous Decoupled Engine Architecture                                    │
│ [Thread 1: DirectShow Capture] -> [Thread 2: ONNX Inference] -> [Thread 3: PyQt6 UI Workspace]   │
│                                                                    |                             │
│       [Thread 5: Whisper STT] <------------------------------------+---> [Thread 4: Piper TTS]   │
└──────────────────────────────────────────────────────────────────────────────────────────────────┘
```

```
┌──────────────────────────────────────────────────────────────────────────────────────────────────┐
│ FIGURE / SNAPSHOT: Live Sign Recognition Session                                                 │
│ [Live Photo from Lab]: Student signing letter 'T' (93% confidence) with active Word Builder      │
│ constructing 'CU' smoothly at 30 FPS without frame drops.                                        │
└──────────────────────────────────────────────────────────────────────────────────────────────────┘
```

### 5.2 0.8s Steady-Hold Dwell Stabilizer with Hysteresis Lock
```
┌──────────────────────────────────────────────────────────────────────────────────────────────────┐
│ FIGURE 9: Dwell Stabilizer & Hysteresis Lock State Machine                                       │
│ [IDLE (<50%)] -> Hand Detected -> [DWELLING 0.8s Bar] -> Dwell 100% -> [CAPTURE (1250Hz Tick)]  │
│                                                                               |                  │
│ [RELEASE/RESET] <----------- Sign Changed or Hand Lowered <----------- [HYSTERESIS LOCK]         │
└──────────────────────────────────────────────────────────────────────────────────────────────────┘
```

### 5.3 AI Linguistic Bridge & Multilingual Regional Voice Engine
```
┌──────────────────────────────────────────────────────────────────────────────────────────────────┐
│ FIGURE 10: Multilingual Indian Regional Voice Pipeline                                           │
│ 1. Raw Sign Gloss: "ME WATER WANT" (Telegraphic)                                                 │
│ 2. 1-Click Polish (Ctrl+P): "I want to drink water." (Fluent English)                            │
│ 3. Multilingual Engine (Ctrl+T / Enter): Hindi (मुझे पानी चाहिए।), Telugu, Tamil, Marathi, etc.   │
└──────────────────────────────────────────────────────────────────────────────────────────────────┘
```

---

# CHAPTER 6: PHASE 7 & 8 — TWO-WAY LOOP, UI/UX OVERHAUL, EVALUATION & STANDALONE DEPLOYMENT

### 6.1 Bidirectional Conversational Loop
```
┌──────────────────────────────────────────────────────────────────────────────────────────────────┐
│ FIGURE 7: Two-Way Deaf <-> Hearing Conversational Loop                                           │
│ Forward Loop: Signer -> Dwell Capture -> AI Polish -> Piper / Regional Neural Voice              │
│ Reverse Loop: Hearing Partner -> Microphone -> Whisper STT (<200ms) -> Subtitles & ISL Badges    │
└──────────────────────────────────────────────────────────────────────────────────────────────────┘
```

### 6.2 Camera-First UI/UX Hierarchy
```
┌──────────────────────────────────────────────────────────────────────────────────────────────────┐
│ FIGURE 11: Camera-First Assistive Workspace Wireframe                                            │
│ +---------------------------------------+------------------------------------------------------+ │
│ |                                       | DETECTED SIGN: [ T ] (96%) | Hold 0.8s [======  ]    | │
│ |      60% HERO CAMERA VIEWPORT         | CURRENT WORD: WATER_ | [ 1: WATER ] [ 2: WATCH ]     | │
│ | (30 FPS DirectShow + MediaPipe Mesh)  | SENTENCE: I NEED WATER | [ Translate ] [ Polish ]    | │
│ |                                       | [ 🔊 Speak in Hindi  Enter ]                         | │
│ +---------------------------------------+------------------------------------------------------+ │
│ | ▼ Collapsible Drawer: Two-Way Dialogue Timeline | Live Subtitles | Visual ISL Sign Badges    | │
│ +----------------------------------------------------------------------------------------------+ │
└──────────────────────────────────────────────────────────────────────────────────────────────────┘
```

### 6.3 Quantitative Evaluation and Latency Benchmarks
```
┌──────────────────────────────────────────────────────────────────────────────────────────────────┐
│ FIGURE 6: End-to-End Latency Budget Breakdown (Total Lag: 23.4 ms)                               │
│ 1. Video Frame Acquisition: 8.2 ms (35.0%)                                                       │
│ 2. MediaPipe Landmark Extraction: 4.1 ms (17.5%)                                                 │
│ 3. Coordinate Normalization: 0.1 ms (0.4%)                                                       │
│ 4. ONNX Residual MLP Forward Pass: 1.4 ms (6.0%)                                                 │
│ 5. One-Euro Adaptive Filter: 0.1 ms (0.4%)                                                       │
│ 6. PyQt6 UI Render & Dwell: 1.2 ms (5.1%)                                                        │
│ 7. Piper Neural Voice Synthesis: 8.3 ms (35.5%)                                                  │
└──────────────────────────────────────────────────────────────────────────────────────────────────┘
```

| Dimension | Low-Fidelity Baseline (Phase 1) | High-Fidelity System (v3.1) |
| :--- | :--- | :--- |
| **Model Accuracy** | 42.8% (Dialect clashes) | **99.84% – 99.96% Accuracy** |
| **Inference Latency** | >500 ms (30-frame buffer) | **<1.8 ms per frame** |
| **End-to-End Delay** | >1,000 ms (Lag & stutter) | **23.4 ms total latency** |
| **Vocabulary Reach** | Closed Dictionary (364 words) | **Unlimited Fingerspelling** |
| **Model Footprint** | >150 MB (Heavy video weights) | **556 KB** (`isl_letter_classifier.onnx`) |
| **Ergonomics** | Manual spacebar tapping | **0.8s Steady-Hold Dwell Stabilizer** |
| **Grammar Bridge** | Raw telegraphic glosses | **1-Click AI Grammar Polish (`Ctrl+P`)** |
| **Multilingual Voice** | English only | **8 Indian Regional Languages** |
| **Conversational Flow** | 1-Way (Sign to Speech only) | **Two-Way Conversational Loop** |
| **Cloud Cost** | Monthly API fees | **$0.00 / month (100% Local)** |

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
