# SignSpeak Universal — 12-Slide PPT Master Plan

## Purpose of this file

This document is the **source of truth for the presentation**. Every future slide should be created from this plan so that the PPT stays faithful to the actual SignSpeak Universal project and does not introduce unsupported features, technologies, numbers, or claims.

**Presentation structure:** 12 slides total  
- Slide 1 — Introduction / Title
- Slides 2–11 — 10 Main Working Slides
- Slide 12 — Conclusion / Outro

**Primary presentation goal:** maximum understandability and graspability for teachers, students, evaluators, and prospective users.

**Core storytelling rule:** Explain the project as a journey:
**Problem → Why the first approach failed → Key architectural decision → How the system was engineered → How the user interacts with it → What the final system achieved → How it is delivered.**

---

# SLIDE 1 — INTRODUCTION

## Title
**SIGN SPEAK UNIVERSAL**

### Subtitle
**A Real-Time Assistive Communication Cockpit for Indian Sign Language**

### What this slide must communicate
SignSpeak Universal is a camera-first desktop application designed to enable practical, two-way communication using Indian Sign Language fingerspelling.

### Essential information
- Students:
  - Khaja Naseeruddin M — 24MT7021
  - Khan Amaan — 24MT7022
- Guide:
  - Dr. Pallavi Vasant Sapkale
- Department:
  - Computer Science and Engineering
- Institution:
  - Ramrao Adik Institute of Technology
  - Under D. Y. Patil Deemed to be University

### Visual direction
Use the actual application/camera interface as the hero visual if available. Keep institutional details secondary.

### Speaker takeaway
> “This project is not just a sign classifier. We built a complete local communication system around real-time fingerspelling.”

---

# SLIDE 2 — THE PROBLEM WE ARE SOLVING

## Title
**The Communication Gap**

### Main idea
The real problem is not simply recognizing a hand sign. The problem is enabling a Deaf user and a hearing person to communicate naturally in everyday situations.

### Explain in three simple points
1. **Interpreter dependency**
   - Human interpreters are scarce, expensive, and concentrated in major urban areas.
2. **Existing AI limitations**
   - Many academic systems recognize only a fixed vocabulary.
   - This makes names, medicines, addresses, and technical terms difficult or impossible to express.
3. **One-way communication**
   - Sign → Speech alone is incomplete.
   - The Deaf user also needs to understand the hearing person's reply.

### Project goal
**A lightweight, local, two-way communication cockpit using an ordinary laptop camera.**

### Visual
Simple communication loop:
**Deaf User → SignSpeak → Hearing User → SignSpeak → Deaf User**

### Avoid
Do not overload this slide with statistics or technical details. The audience should understand the human problem first.

---

# SLIDE 3 — LOW-FIDELITY PROTOTYPE → HIGH-FIDELITY PROTOTYPE

## Title
**From Low-Fidelity Prototype to High-Fidelity Prototype**

### Main story
The first prototype was **not a failure**. It was the experimental foundation that revealed what a practical system needed to improve.

We began with a **low-fidelity prototype** focused on exploring dynamic word-level recognition. That prototype gave us measurable evidence about latency, vocabulary limits, dataset compatibility, and UI responsiveness.

Those findings directly informed the **high-fidelity prototype**.

### Low-fidelity baseline
**ISL + ASL datasets → 364 word classes → MediaPipe Holistic → 856-D features → 30-frame sequence → ST-GCN**

### What the low-fidelity prototype taught us
#### 1. Dataset compatibility matters
Combining ISL and ASL introduced dialect conflicts.

#### 2. Real-time interaction needs low latency
A 30-frame temporal buffer created a substantial delay.

#### 3. A fixed dictionary limits communication
364 predefined words could not naturally cover names, medicines, addresses, or technical terms.

#### 4. Architecture affects the user experience
A single blocking processing loop reduced the live UI to **5–12 FPS**.

### The upgrade
These observations led to a deliberate architectural redesign:

**Low-Fidelity Exploration**
→ **Evidence from Real-Time Testing**
→ **Design Decisions**
→ **High-Fidelity Prototype**

### Main lesson
> “The low-fidelity prototype was our learning stage. The high-fidelity prototype is the engineered upgrade built from those lessons.”

### Visual
Use a clean **Evolution → Upgrade** graphic rather than a “failure” graphic:
**Low-Fidelity Prototype** → **Learnings** → **High-Fidelity Prototype**

---

# SLIDE 4 — THE HIGH-FIDELITY ARCHITECTURAL BREAKTHROUGH

## Title
**We Changed the Problem Instead of Fighting the Problem**

### Core upgrade
The high-fidelity prototype deliberately moved from:

**Dynamic word recognition**

to:

**Single-frame fingerspelling recognition**

### New recognition space
**35 classes**
- 26 letters: A–Z
- 9 digits: 1–9

### Why this works
#### Faster
One frame removes the 30-frame temporal buffer.

#### Open vocabulary
Letters can be assembled into arbitrary words.

#### More practical
Users can spell:
- Names
- Medicines
- Addresses
- Brand names
- Technical terms

### Key concept
**35 classes → character sequence → unlimited word construction**

### Important wording
Do not claim that the model directly understands every spoken word. It recognizes fingerspelling characters and the application builds language on top of them.

### Visual
Before/after:
**364 fixed words → 35 characters → unlimited combinations**

---

# SLIDE 5 — FROM CAMERA IMAGE TO ROBUST FEATURES

## Title
**The Camera Does Not Feed Raw Images to the Model**

### Main idea
Instead of asking the neural network to learn camera position, hand size, and location, we convert the hand into normalized 3D geometry.

### Pipeline
**Camera Frame**
↓
**MediaPipe Hands**
↓
**21 Hand Landmarks × 3D**
↓
**Wrist-Centered Translation**
↓
**Scale Normalization**
↓
**Left/Right Hand Handling**
↓
**126-Dimensional Feature Vector**

### Explain the mathematics simply

#### Step 1 — Translation invariance
Move the wrist to the origin.

**P′ᵢ = Pᵢ − P₀**

This means the hand can appear anywhere in the camera view.

#### Step 2 — Scale invariance
Normalize using the wrist-to-middle-finger-joint distance.

**S = ||P₉ − P₀||₂**

This reduces the effect of hand size and camera distance.

#### Step 3 — 126-D representation
Two 63-D hand slots are maintained for left/right hand handling.

### Audience takeaway
> “We teach the model the shape of the hand, not where the hand happens to be on the screen.”

### Visual
Use a simple hand-landmark diagram plus a 4-step transformation graphic. Avoid putting full equations on screen unless needed for evaluation.

---

# SLIDE 6 — DATA + MODEL: BUILDING THE RECOGNIZER

## Title
**From 129,773 Images to a Lightweight Neural Model**

### Dataset engineering
- **129,773** raw sign images collected.
- 12-core parallel processing used for landmark extraction.
- Images with detection confidence below **0.40** were rejected.
- **22,256** defective/low-quality images were removed.
- **107,517** validated vectors remained.

### Model
Custom **Deep Residual MLP (`ISLLetterClassifier`)**

### Simplified architecture
**126-D Input**
→ **256**
→ **256 + Residual Skip**
→ **128**
→ **35-Class Output**

### Training result
- Training accuracy: **99.89%**
- Validation accuracy: **99.67%**
- Held-out test accuracy: **99.70%**
- Final reported live/held-out benchmark elsewhere in the compendium: **99.96%**

### Runtime model
- ONNX format
- **556 KB**
- **<1.8 ms CPU execution time**

### Important consistency rule
The compendium reports both a **99.70% held-out test accuracy** in the model-training section and a **99.96% top-1 benchmark** in the final evaluation section. Future slides must label these metrics exactly rather than presenting them as if they were the same experiment.

### Visual
Use a simple funnel:
**Raw Images → QC → 107,517 vectors → Residual MLP → ONNX**

---

# SLIDE 7 — MAKING IT WORK FOR REAL USERS

## Title
**Accuracy Alone Was Not Enough**

### Problem discovered
Different users have different:
- Finger proportions
- Joint flexibility
- Hand sizes
- Camera angles

Complex signs such as **R, U, and V** showed live performance drops.

### Our solution
#### Sign Recorder Studio
- 3-second preparation countdown
- 3-second active recording
- 90 vectors captured at 30 FPS
- User-specific recordings

#### 3D augmentation
We simulate realistic variation using:
- Rotation
- Scale jitter
- Joint noise

### Co-training
- 20% user-recorded augmented data
- 80% global baseline data
- Fine-tuning on RTX 4050 in approximately **8 seconds**

### Result
Live validation accuracy increased to approximately **99.84%–99.96%** without catastrophic forgetting of the global alphabet.

### Key takeaway
> “The system can adapt to the person using it instead of assuming every hand looks identical.”

### Visual
**Global Model → User Recording → 3D Augmentation → Fine-Tuning → Personalized Model**

---

# SLIDE 8 — REAL-TIME SYSTEM ARCHITECTURE

## Title
**How We Keep the Camera Smooth at 30 FPS**

### Main problem
Speech recognition, text-to-speech, and AI processing should never freeze the camera.

### Five-thread architecture
1. **CaptureThread**
   - Camera + MediaPipe landmarks
2. **InferenceThread**
   - ONNX inference + confirmation filter
3. **Main UI**
   - Dwell bar + word builder + sentence cockpit
4. **TTSThread**
   - Regional neural speech
5. **SpeechToTextThread**
   - Microphone + Whisper AI

### Simple system flow
**Camera**
→ **Landmarks**
→ **126-D Features**
→ **ONNX**
→ **Stable Letter**
→ **Word/Sentence**

At the same time:

**Hearing Partner Speech**
→ **Whisper**
→ **Subtitle**
→ **Visual ISL Badges**

### Performance
- UI target: **30 FPS flat**
- Total measured system latency: **23.4 ms**
- ONNX forward pass: **1.4 ms**

### Key takeaway
> “Parallel processing separates the camera experience from slower background tasks.”

### Visual
Use five clean lanes/threads instead of a dense software-engineering diagram.

---

# SLIDE 9 — FROM LETTERS TO NATURAL COMMUNICATION

## Title
**Recognition Becomes Usable Through the Interaction Layer**

### 1. Dwell Stabilizer
A sign must remain steady for **0.80 seconds** at sufficient confidence before the letter is committed.

### 2. Anti-duplication lock
Prevents one held gesture from being entered repeatedly.

### 3. One-Euro filtering
Reduces webcam landmark jitter while preserving fast hand movement.

### 4. Predictive autocomplete
Three context-aware word suggestions help users complete words faster.

### 5. AI Grammar Polish
`Ctrl+P` converts raw sign glosses such as:

**ME WATER DRINK WANT**

into:

**I want to drink water.**

### 6. Multilingual speech
The system supports:
- English
- Hindi
- Telugu
- Tamil
- Marathi
- Kannada
- Bengali
- Gujarati

### Key takeaway
> “The AI is not only recognizing signs; it is reducing the effort required to communicate.”

### Visual
Show:
**Sign → Letter → Word → Sentence → Natural Speech**

---

# SLIDE 10 — THE TWO-WAY CONVERSATION

## Title
**Breaking the One-Way Communication Barrier**

### Main idea
SignSpeak is designed as a complete conversation loop.

### Outgoing path
**Deaf User**
→ Fingerspelling
→ SignSpeak recognition
→ Word/Sentence
→ Grammar polish
→ Neural speech
→ Hearing partner

### Incoming path
**Hearing Partner**
→ Microphone
→ Whisper AI
→ Subtitle
→ Visual ISL fingerspelling badges
→ Deaf user reads the reply

### Supporting features
- Live subtitle card
- ISL visual badges
- Background speech recognition
- Dialogue timeline
- Timestamped transcript export

### Example
Hearing partner says:
**“Do you need help?”**

The application can show the spoken response as visual letter badges such as:

**[ D ] [ O ] [ Y ] [ O ] [ U ] ...**

### Key takeaway
> “Communication is complete only when both people can send and receive information.”

### Visual
Use a circular two-way arrow between:
**Deaf User ↔ SignSpeak ↔ Hearing User**

---

# SLIDE 11 — HIGH-FIDELITY PROTOTYPE: PERFORMANCE + DELIVERY

## Title
**What We Finally Built**

### Final measurable results
| Metric | SignSpeak Universal |
|---|---:|
| Top-1 Accuracy | **99.96%** |
| Macro Precision | **99.88%** |
| Macro Recall | **99.85%** |
| Macro F1 | **99.86%** |
| End-to-End Latency | **23.4 ms** |
| ONNX Model Size | **556 KB** |
| UI Performance | **30 FPS** |
| Regional Languages | **8** |
| Cloud Cost | **$0/month** |

### Evaluation evidence
Use a small secondary strip rather than another dense table:
**10 modular sub-systems tested** → **100% automated integration pass rate** → **measured latency budget** → **standalone EXE delivery**

This slide therefore demonstrates both **performance** and **engineering verification**, directly supporting the coding/testing and data-analysis criteria.

### Productization
The project is packaged as a standalone Windows application using:
- PyInstaller
- ONNX Runtime
- PyQt6
- MediaPipe
- OpenCV
- Piper
- Whisper

### User experience
The final UI is **camera-first**:
- 60% hero camera viewport
- 44–52 px accessible interaction targets
- Detected-sign feedback
- Dwell progress
- Word suggestions
- Sentence cockpit
- Collapsible two-way dialogue drawer

### Final product message
**A local Windows application that turns a standard laptop into a two-way assistive communication cockpit.**

### Visual
Use the strongest authentic screenshot of the final application, with 4–5 small performance callouts.

---

# SLIDE 12 — CONCLUSION / OUTRO

## Title
**From a Failed Prototype to a Practical Communication System**

### Three-part conclusion

#### 1. We learned
Real-world AI is not only about model accuracy.

#### 2. We engineered
We combined:
- Geometric normalization
- Lightweight residual learning
- Personalization
- Parallel processing
- Ergonomic interaction
- Speech and language layers

#### 3. We delivered
A camera-first, local, two-way assistive communication application.

### Final statement
> **SignSpeak Universal shows that practical assistive AI can be fast, lightweight, private, and usable without expensive cloud infrastructure or specialized sensors.**

### Future direction
- WebAssembly browser version
- Android/iOS companion app
- Bluetooth Low Energy hearing-aid streaming

### Closing line
**“The goal was never just to recognize a hand. The goal was to make communication easier.”**

### Visual
Minimal final slide. Use the SignSpeak logo/application image and one strong closing sentence.

---

# HIGH-FIDELITY PROTOTYPE — FEATURE COVERAGE CHECKLIST

The 10 working slides must collectively show that the high-fidelity prototype is a **complete upgrade of the earlier low-fidelity exploration**, not merely a new classifier.

The presentation must preserve and, where appropriate, visually demonstrate these documented features:

## Recognition & AI
- 35-class fingerspelling recognition: A–Z + 1–9
- MediaPipe 3D hand landmarks
- 126-dimensional normalized representation
- Wrist-centered translation
- Scale normalization
- Left/right hand handling
- Deep Residual MLP
- ONNX runtime inference
- Label smoothing during training

## Data & Personalization
- 129,773 raw images
- Automated quality filtering
- 107,517 validated vectors
- Sign Recorder Studio
- 3-second preparation countdown
- 3-second active recording
- 90 vectors at 30 FPS
- User-specific recordings
- 3D rotation augmentation
- Scale jitter
- Gaussian joint jitter
- 20/80 user/global co-training
- ~8-second GPU fine-tuning

## Real-Time Interaction
- Five-thread asynchronous architecture
- 30 FPS camera pipeline
- 4-frame confirmation filter
- 0.8-second dwell stabilizer
- Visual hold/progress bar
- Audio confirmation tick
- Anti-duplication hysteresis lock
- One-Euro adaptive filtering
- Keyboard ergonomics
- Spacebar word commit
- Smart Backspace
- Escape clear

## Language & Communication
- Word builder
- Predictive autocomplete with 3 suggestions
- One-touch suggestion hotkeys
- AI Grammar Polish (`Ctrl+P`)
- Revert with `Ctrl+Z`
- 8 Indian regional languages
- One-click translation (`Ctrl+T`)
- Regional neural speech
- Dynamic Speak control

## Two-Way Conversation
- Microphone input
- Whisper AI speech-to-text
- Incoming subtitle card
- Visual ISL sign/fingerspelling badges
- Two-way conversational loop
- Dialogue timeline
- Timestamped transcript export

## High-Fidelity UI/UX
The PPT should explicitly identify the final interface as the **High-Fidelity Prototype / Camera-First Workspace** and preserve its documented interaction model:
- 60% hero camera viewport
- Camera-centered workspace
- Live hand detection overlay
- Detected sign + confidence
- Hold progress indicator
- Current word area
- Autocomplete strip
- Sentence cockpit
- Language selector
- Translate action
- Polish action
- Regional Speak action
- Collapsible bottom drawer
- Two-way dialogue timeline
- Live subtitles
- Visual ISL badges
- Accessible 44–52 px interaction targets
- Progressive disclosure

## Delivery
- Standalone Windows application
- PyInstaller packaging
- Inno Setup installer
- Local execution architecture
- 556 KB ONNX model binary
- Documented repository structure

### Presentation rule
Do not try to place every feature on one slide. Distribute them across Slides 4–11 so the audience sees the **evolution from low-fidelity prototype to high-fidelity prototype and the complete feature set** without creating an unreadable slide deck.

# PRESENTATION-WIDE DESIGN RULES

## 1. Explain before optimizing
Every technical concept should first be explained in plain language, then supported by the technical detail.

## 2. One idea per slide
Do not turn a slide into a report page.

## 3. Use a consistent visual hierarchy
Each slide should have:
- One strong title
- One central message
- 3–6 supporting elements
- One primary visual

## 4. Prefer diagrams over paragraphs
Convert pipelines, architectures, and processes into visual flows.

## 5. Numbers must have context
Never show a metric without saying what it measures.

## 6. Preserve project terminology
Use the actual project terms from the compendium:
- SignSpeak Universal
- 35-class fingerspelling
- 126-dimensional representation
- Deep Residual MLP
- Sign Recorder Studio
- 5-thread architecture
- Dwell Stabilizer
- One-Euro Filter
- AI Grammar Polish
- Whisper AI
- Camera-first UI/UX

## 7. Do not invent claims
Future slide generation must remain within the evidence in `MASTER_PROJECT_COMPENDIUM.md` unless the user explicitly provides new evidence.

## 8. Do not overload with mathematics
Equations are useful for evaluation, but the main presentation should explain what the mathematics achieves.

## 9. Keep the journey visible
The audience should be able to reconstruct the project:

**Problem**
→ **Low-Fidelity Prototype**
→ **Evidence & Learnings**
→ **High-Fidelity Architectural Upgrade**
→ **Robust Features**
→ **Model**
→ **Personalization**
→ **Real-Time Architecture**
→ **Human Interaction**
→ **Two-Way Communication**
→ **Final Product**

## 10. Protect the actual application
UI redesign and presentation visuals must not imply that existing application capabilities were removed. The presentation should describe the final feature set documented in the compendium.

---

# FACT-CHECK NOTES FOR FUTURE SLIDE GENERATION

### Project naming
Use **SignSpeak Universal**, not “Science Speak.”

### Accuracy values
The source contains multiple evaluation contexts:
- Training: 99.89%
- Validation: 99.67%
- Held-out test: 99.70%
- Final evaluation table: 99.96% Top-1

Always label the context.

### Latency values
- ONNX execution: <1.8 ms in the model/runtime description
- ONNX forward-pass entry in latency budget: 1.4 ms
- Total end-to-end system latency: 23.4 ms

Do not collapse these into one number.

### Vocabulary
The system's recognition layer uses **35 classes (A–Z and 1–9)**. “Infinite vocabulary” refers to the ability to compose arbitrary words from recognized characters, not to an infinite number of directly classified model classes.

### Privacy / local operation
The compendium describes the system as local and $0/month cloud cost. Do not make stronger privacy claims than the source supports.

### EXE
The packaging blueprint uses PyInstaller and Inno Setup for a standalone Windows distribution. Do not claim a particular final installer size unless separately documented.

---

# FINAL SLIDE ORDER AT A GLANCE

1. **SIGN SPEAK UNIVERSAL** — What is it?
2. **The Communication Gap** — Why does it matter?
3. **Our First Model Failed** — What went wrong?
4. **The Architectural Breakthrough** — What changed?
5. **From Camera to Geometry** — How does recognition work?
6. **Data + Model** — How did we train it?
7. **Making It Work for Real Users** — How did we improve robustness?
8. **Real-Time System Architecture** — How does the app stay responsive?
9. **Recognition → Communication** — How does the user actually use it?
10. **The Two-Way Conversation** — How do both people communicate?
11. **Final System: Performance + Delivery** — What did we achieve?
12. **From Prototype to Practical System** — What is the final takeaway?

---


# EVALUATION-RUBRIC ALIGNMENT

The updated compendium adds the official **Mini Project Mock 2 Evaluation Rubric**, so the PPT must be designed not only for visual clarity but also to make the marks-bearing evidence easy for an evaluator to identify.

The rubric has **30 marks** across five areas:
- **Coding and testing (sub-modules): 10 marks**
- **Depth and accuracy of data analysis and interpretation: 5 marks**
- **Clarity and effectiveness of project presentation, written and oral: 5 marks**
- **Compliance with project guidelines, structure and formatting: 5 marks**
- **Potential impact and practical relevance: 5 marks**

The updated compendium explicitly maps these criteria to evidence in the project. fileciteturn1file0L524-L540

## How the 12-slide PPT should demonstrate the rubric

### Coding & Testing — 10 marks
The presentation must visibly establish that this is a working, modular high-fidelity system rather than only a trained model.

Evidence to distribute across Slides 6–11:
- Video Capture + 3D Landmarks
- 126-D coordinate normalization
- Neural inference + ONNX
- Dwell stabilizer + hysteresis
- One-Euro filtering
- AI autocomplete + hotkeys
- AI Grammar Polish + revert
- Multilingual TTS
- Whisper STT + ISL badges
- Sign Studio + GPU co-training

The updated compendium states that these 10 sub-modules were tested and that the automated integration harness achieved a 100% pass rate. fileciteturn1file0L546-L588

**PPT rule:** Do not put “100% verified” everywhere as a decorative claim. Instead, show the actual modules, their purpose, and selected verification evidence.

### Data Analysis & Interpretation — 5 marks
The PPT should show that the project decisions were based on measurable evidence:
- 129,773 raw samples
- 22,256 samples removed through QC
- 107,517 validated vectors
- 246,104 augmented samples
- 35-class distribution
- Training/validation/held-out results
- Confusion between difficult classes such as M/N/S and R/U/V
- 23.4 ms measured latency budget

These are explicitly documented in the updated compendium's evaluation section. fileciteturn1file0L592-L605

### Presentation Clarity — 5 marks
The rubric itself rewards clarity, so the PPT should demonstrate clarity rather than merely claim it.

The updated compendium documents:
- 43-page formal report
- 12 embedded figures and authentic lab photographs
- Structured technical documentation
- A 2-minute elevator pitch
- Prepared viva defense answers

fileciteturn1file0L609-L624

**PPT rule:** Every technical slide must answer one simple question and use diagrams, screenshots, or metrics before dense text.

### Compliance & Formatting — 5 marks
The PPT should visibly follow a consistent academic structure:
- Title / institution / student / guide
- Logical 10-working-slide sequence
- Consistent typography and spacing
- Clear section hierarchy
- Proper figure labels where appropriate
- No unsupported claims
- Consistent terminology
- Final conclusion and future scope

The updated compendium documents institutional template adherence, mandatory academic preliminaries, IEEE citation standards, and structured chapter organization. fileciteturn1file0L628-L640

### Impact & Practical Relevance — 5 marks
The presentation must connect the engineering to actual use:
- Healthcare and hospitals
- Banking and government services
- Public transit / railway counters
- Classrooms and educational institutions
- Local execution on consumer laptops
- Two-way communication
- Standalone Windows deployment

The updated compendium explicitly identifies these application settings and practical-impact claims. fileciteturn1file0L642-L656

**PPT rule:** The conclusion should return to the human problem, not end on model architecture alone.

---

# UPDATED PRESENTATION QUALITY CHECK

Before any slide is finalized, verify:

- [ ] Does this slide have one clear idea?
- [ ] Can a teacher understand the point without reading a paragraph?
- [ ] Does the slide show evidence from the actual project?
- [ ] Is the feature part of the documented high-fidelity prototype?
- [ ] Does the slide contribute to at least one evaluation criterion?
- [ ] Are technical numbers labeled with their correct evaluation context?
- [ ] Are the low-fidelity and high-fidelity stages presented as an engineering evolution?
- [ ] Is there enough visual evidence of the actual application?
- [ ] Does the complete deck demonstrate coding, testing, analysis, presentation quality, compliance, and impact?

# SOURCE OF TRUTH

Primary source used for this presentation plan:

**MASTER_PROJECT_COMPENDIUM.md — SignSpeak Universal: The Master Project Compendium**

This plan intentionally follows the documented project journey and does not introduce external technical claims.

## Required framing for the final PPT

The earlier work must be presented as a **low-fidelity prototype / exploration stage** whose purpose was to test the initial concept and expose real-world requirements.

The current application must be presented as the **high-fidelity prototype**, representing the engineering upgrade that incorporated those learnings into the recognition pipeline, interaction design, real-time architecture, language layer, two-way communication loop, camera-first UI/UX, and Windows application delivery.

**Do not use “failed prototype” as the slide narrative. Use “low-fidelity prototype → learnings → high-fidelity prototype upgrade.”**

