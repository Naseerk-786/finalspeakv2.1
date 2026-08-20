# 📐 SignSpeak Universal — Methodologies Followed Document
**Definitive Engineering, Machine Learning & Software Design Methodologies**

---

## 📌 Executive Summary

The development of **SignSpeak Universal** followed six rigorous computer science, machine learning, and software engineering methodologies. Every architectural choice was guided by empirical benchmarking, hardware awareness, data-centric design, and real-time execution constraints. This document details each methodology, its mathematical formulation, and its software implementation across the project lifecycle.

---

## 🔬 1. Data-Centric AI & Multi-Source Harvesting Methodology

### 1.1 Multi-Source Dataset Integration
Rather than relying on a single lab-recorded dataset, we adopted a **Data-Centric AI methodology** that aggregates diverse real-world hand image samples across multiple public repositories (Kaggle ISL, Kaggle ASL, and GitHub ISL).

### 1.2 Automated Multi-Core Quality Control Filtering
* **Problem:** Raw public datasets contain corrupted files, motion-blurred images, and photos where no hands are visible.
* **Methodology:** Implemented an automated multi-core filter using **12 parallel CPU workers** executing MediaPipe Hands. Any frame where MediaPipe fails to detect clean hand keypoints with $\ge 0.40$ confidence is automatically discarded.
* **Empirical Result:** Processed 129,773 raw images, discarding 22,256 unusable frames and retaining **107,517 clean 3D landmark vectors**.

---

## 📐 2. Invariant Feature Normalization Methodology

### 2.1 Spatial Normalization Math
Raw pixel coordinates $(x,y,z)$ vary based on camera resolution, user distance, and position in the frame. We applied a two-stage spatial normalization transform:

1. **Wrist-Centered Origin Translation:**
   Let $\mathbf{P}_0$ be the wrist keypoint. All 21 3D hand keypoints $\mathbf{P}_i$ are translated relative to the wrist origin:
   $$\mathbf{P}'_i = \mathbf{P}_i - \mathbf{P}_0 \quad \forall i \in [0, 20]$$

2. **Hand Span Scale Normalization:**
   Let $\mathbf{P}_9$ be the middle finger MCP joint. The scale factor $S$ is calculated as the Euclidean distance between the wrist and MCP joint:
   $$S = \|\mathbf{P}_9 - \mathbf{P}_0\|_2 + \epsilon$$
   $$\mathbf{P}_{\text{norm}, i} = \frac{\mathbf{P}'_i}{S} \quad \forall i \in [0, 20]$$

### 2.2 Left/Right Active Hand Mirroring
* **Methodology:** When only one hand is visible (either left or right), the active hand's 63-dimensional normalized feature vector is duplicated across both left and right feature slots ($\mathbf{X}_{\text{frame}} = [\mathbf{X}_{\text{active}} \,\|\, \mathbf{X}_{\text{active}}]$).
* **Impact:** Makes the classifier completely invariant to whether the user signs with their left or right hand.

---

## 🧠 3. Neural Network Architecture & Regularization Methodology

### 3.1 Deep Residual MLP Topology (`ISLLetterClassifier`)
* **Methodology:** Implemented a deep Multi-Layer Perceptron equipped with **Residual Skip Connections** ($h_2 = \text{Block}_2(h_1) + h_1$).
* **Why Skip Connections:** Prevents gradient vanishing across deep layers, allowing smooth backpropagation and enabling rapid convergence during 200-epoch GPU training.

### 3.2 Advanced Regularization Techniques
* **Batch Normalization (`BatchNorm1d`):** Stabilizes hidden layer activation distributions and accelerates training.
* **SiLU Activations (Sigmoid Linear Unit):** Replaced standard ReLU with SiLU ($f(x) = x \cdot \sigma(x)$) to ensure smooth gradient flow for negative inputs.
* **Dropout ($0.20$):** Randomly zeroes out 20% of neuron activations during training to prevent co-adaptation and overfitting.
* **Label Smoothing ($0.05$):** Softens target cross-entropy labels ($0.95$ for correct class, $0.05/N$ for remaining classes) to prevent overconfident boundary predictions.

---

## 🏋️ 4. Model Training & Hyperparameter Optimization Methodology

### 4.1 Stratified Train / Validation / Test Splitting
* **Methodology:** Split the 107,517-sample dataset into **80% Training (86,013 samples)**, **10% Validation (10,752 samples)**, and **10% Held-Out Testing (10,752 samples)** using stratified splitting to preserve exact class distribution ratios across all 35 classes.

### 4.2 Learning Rate Scheduling & Optimization
* **Optimizer:** AdamW with weight decay ($\text{lr} = 2\times 10^{-3}$, $\text{weight\_decay} = 1\times 10^{-4}$).
* **Scheduler:** Cosine Annealing Learning Rate Scheduler over **200 CUDA Epochs**, smoothly annealing learning rate down to $1\times 10^{-5}$.
* **Data Augmentation:** Applied random Gaussian landmark jitter ($\sigma = 0.01, p = 0.30$) during training to simulate subtle hand tremors and camera noise.

---

## ⚡ 5. Asynchronous Multi-Threaded Software Engine Methodology

### 5.1 Decoupled 4-Thread Architecture
To maintain a butter-smooth 30 FPS video feed without GUI frame dropping, processing is decoupled across 4 isolated threads:

```
┌─────────────────────────────────────────────────────────────┐
│ Thread 1: Camera Capture & MediaPipe Skeleton Extraction    │
└──────────────────────────────┬──────────────────────────────┘
                               │ (Queue / Signal)
                               ▼
┌─────────────────────────────────────────────────────────────┐
│ Thread 2: ONNX Letter Inference Engine (<1.8ms per frame)   │
└──────────────────────────────┬──────────────────────────────┘
                               │ (Confirmed letter)
                               ▼
┌─────────────────────────────────────────────────────────────┐
│ Thread 3: PyQt6 Main UI Thread (Word Builder & Dashboard)   │
└──────────────────────────────┬──────────────────────────────┘
                               │ (Committed word text)
                               ▼
┌─────────────────────────────────────────────────────────────┐
│ Thread 4: Offline Piper Neural TTS & Audio Playback         │
└─────────────────────────────────────────────────────────────┘
```

---

## 🛡️ 6. Temporal Noise Filtering & Stabilization Methodology

### 6.1 Consecutive Frame Matching Filter
* **Problem:** Single-frame predictions can flicker during rapid hand transitions between letters.
* **Methodology:** Implemented a temporal confirmation buffer. A candidate letter must be predicted with $\ge 0.50$ confidence for **4 consecutive inference frames** ($\sim 150\text{ ms}$) before it is confirmed and appended to the active word builder.

### 6.2 Auto-Commit Idle Timeout
* **Methodology:** Tracks hand presence timestamps ($t_{\text{last\_hand}}$). When no hand is detected for $\ge 1.5$ seconds ($\text{NO\_HAND\_COMMIT\_TIMEOUT} = 1.5\text{s}$), the current word builder automatically commits the formed word to the sentence buffer and sends it to the Piper TTS engine for speech synthesis.
