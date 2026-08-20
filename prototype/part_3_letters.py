# SignSpeak Universal Prototype — Real-Time ISL Letter & Fingerspelling Application
# Version: 2.2 (0.8s Steady-Hold Letter Capture & Spacebar Sentence Builder)
# Architecture: Real-Time Hand Landmark Classifier -> 0.8s Dwell Stabilizer -> Spacebar Word Builder -> Piper TTS
# UI Theme: Warm Soft 2D Plushy Design (0.8s Hold-to-Capture, Spacebar Word Commit, Enter Speech)

import os
import sys
import json
import time
import wave
import tempfile
import queue
import threading
import numpy as np
from pathlib import Path
from collections import deque

# Ensure prototype folder is in python path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import cv2
import mediapipe as mp
import onnxruntime as ort
from piper import PiperVoice

from one_euro_filter import LandmarkStreamSmoother

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QTextEdit, QFrame, QGroupBox, QProgressBar
)
from PyQt6.QtCore import Qt, QTimer, pyqtSignal, QThread
from PyQt6.QtGui import QImage, QPixmap, QFont, QColor, QPalette

# ═══════════════════════════════════════════════════════════════
# Constants & Paths
# ═══════════════════════════════════════════════════════════════
BASE_DIR = Path(r"d:\finalspeak")
MODELS_DIR = BASE_DIR / "models"

ONNX_MODEL_PATH = MODELS_DIR / "isl_letter_classifier.onnx"
CLASS_META_PATH = MODELS_DIR / "isl_letter_meta.json"
PIPER_MODEL_PATH = MODELS_DIR / "en_US-lessac-medium.onnx"

CONFIDENCE_THRESHOLD = 0.50   # Min confidence to start dwell hold
DWELL_HOLD_SECONDS = 0.80     # Hold sign steady for 0.80s to capture letter

CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480


# ═══════════════════════════════════════════════════════════════
# Audio Feedback Tone (Non-Blocking)
# ═══════════════════════════════════════════════════════════════
def play_feedback_tone(freq=1200, duration_ms=30):
    """Plays a soft, non-blocking confirmation audio tick on letter capture."""
    def _beep():
        try:
            import winsound
            winsound.Beep(freq, duration_ms)
        except Exception:
            pass
    threading.Thread(target=_beep, daemon=True).start()


# ═══════════════════════════════════════════════════════════════
# MediaPipe Hand Feature Extractor (126-dim)
# ═══════════════════════════════════════════════════════════════
class SingleFrameHandExtractor:
    """Extracts 126-dim normalized hand landmarks from a single video frame."""

    def __init__(self):
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_styles = mp.solutions.drawing_styles
        self.hands = None

    def initialize(self):
        """Initialize MediaPipe inside thread context."""
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

    def extract(self, frame_bgr):
        if self.hands is None:
            return None, None

        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        results = self.hands.process(frame_rgb)

        if not results.multi_hand_landmarks:
            return None, results

        lh_feats = np.zeros((21, 3), dtype=np.float32)
        rh_feats = np.zeros((21, 3), dtype=np.float32)
        has_lh, has_rh = False, False

        for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
            label = handedness.classification[0].label
            coords = np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark], dtype=np.float32)

            wrist = coords[0]
            hand_span = np.linalg.norm(coords[9] - wrist) + 1e-6
            norm_coords = (coords - wrist) / hand_span

            if label == "Left":
                lh_feats = norm_coords
                has_lh = True
            else:
                rh_feats = norm_coords
                has_rh = True

        if has_lh and not has_rh:
            rh_feats = lh_feats.copy()
        elif has_rh and not has_lh:
            lh_feats = rh_feats.copy()

        feature_vector = np.concatenate([lh_feats.flatten(), rh_feats.flatten()])
        return feature_vector, results

    def draw_landmarks(self, frame, results):
        if results and results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                self.mp_drawing.draw_landmarks(
                    frame,
                    hand_landmarks,
                    self.mp_hands.HAND_CONNECTIONS,
                    self.mp_styles.get_default_hand_landmarks_style(),
                    self.mp_styles.get_default_hand_connections_style()
                )
        return frame

    def close(self):
        if self.hands:
            self.hands.close()


# ═══════════════════════════════════════════════════════════════
# Thread 1: Camera Capture & MediaPipe Processing
# ═══════════════════════════════════════════════════════════════
class CaptureThread(QThread):
    frame_ready = pyqtSignal(np.ndarray)       # Rendered frame
    feature_ready = pyqtSignal(object)         # 126-dim vector or None
    status_update = pyqtSignal(str)            # Status log

    def __init__(self):
        super().__init__()
        self.running = False
        self.extractor = None

    def run(self):
        self.running = True
        self.status_update.emit("Initializing MediaPipe Hands Extractor...")
        self.extractor = SingleFrameHandExtractor()
        self.extractor.initialize()

        # Try multiple camera indices and backends with DirectShow first
        cap = None
        for cam_idx in [0, 1, 2]:
            for backend in [cv2.CAP_DSHOW, cv2.CAP_MSMF, cv2.CAP_ANY]:
                test_cap = cv2.VideoCapture(cam_idx, backend)
                if test_cap.isOpened():
                    ret, _ = test_cap.read()
                    if ret:
                        cap = test_cap
                        self.status_update.emit(f"Camera opened: index={cam_idx}, backend={backend}")
                        break
                    else:
                        test_cap.release()
            if cap is not None:
                break

        if cap is None or not cap.isOpened():
            self.status_update.emit("ERROR: Unable to open camera. Please check camera connection.")
            err_frame = np.zeros((CAMERA_HEIGHT, CAMERA_WIDTH, 3), dtype=np.uint8)
            err_frame[:] = (245, 235, 226)
            cv2.putText(err_frame, "CAMERA NOT AVAILABLE", (80, 220),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (60, 50, 45), 2, cv2.LINE_AA)
            cv2.putText(err_frame, "Close other camera applications and restart", (80, 270),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (120, 100, 90), 1, cv2.LINE_AA)
            self.frame_ready.emit(err_frame)
            return

        cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
        self.status_update.emit("Webcam Active at 30 FPS")

        while self.running:
            ret, frame = cap.read()
            if not ret or frame is None:
                time.sleep(0.01)
                continue

            frame = cv2.flip(frame, 1)  # Mirror frame
            feat_vec, results = self.extractor.extract(frame)
            self.feature_ready.emit(feat_vec)

            display_frame = self.extractor.draw_landmarks(frame.copy(), results)
            
            # Draw badge indicator with warm soft colors
            cv2.circle(display_frame, (30, 30), 8, (120, 190, 130) if feat_vec is not None else (100, 100, 210), -1)
            status_text = "HAND DETECTED" if feat_vec is not None else "NO HAND DETECTED"
            color = (80, 170, 90) if feat_vec is not None else (80, 80, 200)
            cv2.putText(display_frame, status_text, (48, 35),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)

            self.frame_ready.emit(display_frame)
            time.sleep(1 / 30)

        cap.release()
        if self.extractor:
            self.extractor.close()

    def stop(self):
        self.running = False


# ═══════════════════════════════════════════════════════════════
# Thread 2: ONNX Letter Inference Engine
# ═══════════════════════════════════════════════════════════════
class InferenceThread(QThread):
    prediction_ready = pyqtSignal(str, float)  # (letter, confidence)
    no_hand_signal = pyqtSignal()
    status_update = pyqtSignal(str)

    def __init__(self, onnx_path, class_meta_path):
        super().__init__()
        self.onnx_path = str(onnx_path)
        with open(class_meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
        self.idx2class = {int(k): v for k, v in meta["idx2class"].items()}
        self.feat_queue = queue.Queue(maxsize=30)
        self.smoother = LandmarkStreamSmoother(dim=126, min_cutoff=0.8, beta=0.01)
        self.running = False

    def enqueue_feature(self, feat_vec):
        if self.feat_queue.full():
            try:
                self.feat_queue.get_nowait()
            except queue.Empty:
                pass
        self.feat_queue.put(feat_vec)

    def run(self):
        self.running = True
        if not os.path.exists(self.onnx_path):
            self.status_update.emit("ERROR: ONNX model binary not found!")
            return

        session = ort.InferenceSession(self.onnx_path)
        input_name = session.get_inputs()[0].name
        self.status_update.emit("ONNX Letter Inference Engine Active (One-Euro Smoothed)")

        while self.running:
            try:
                feat = self.feat_queue.get(timeout=0.1)
            except queue.Empty:
                continue

            if feat is None:
                self.smoother.reset()
                self.no_hand_signal.emit()
                continue

            # Apply adaptive One-Euro jitter filter
            smoothed_feat = self.smoother.smooth(feat)
            input_tensor = smoothed_feat.reshape(1, -1).astype(np.float32)
            outputs = session.run(None, {input_name: input_tensor})
            logits = outputs[0][0]

            exp_logits = np.exp(logits - np.max(logits))
            probs = exp_logits / exp_logits.sum()

            top_idx = int(np.argmax(probs))
            top_conf = float(probs[top_idx])
            letter = self.idx2class.get(top_idx, "?")

            self.prediction_ready.emit(letter, top_conf)

    def stop(self):
        self.running = False


# ═══════════════════════════════════════════════════════════════
# Thread 3: Piper Offline TTS Engine
# ═══════════════════════════════════════════════════════════════
class TTSThread(QThread):
    speech_done = pyqtSignal(str)

    def __init__(self, piper_model_path):
        super().__init__()
        self.piper_model_path = str(piper_model_path)
        self.text_queue = queue.Queue()
        self.running = False

    def enqueue_text(self, text):
        if not self.text_queue.full():
            self.text_queue.put(text)

    def run(self):
        self.running = True
        voice = None
        if os.path.exists(self.piper_model_path):
            try:
                voice = PiperVoice.load(self.piper_model_path)
            except Exception as e:
                print(f"Piper Load Error: {e}")

        while self.running:
            try:
                text = self.text_queue.get(timeout=0.5)
            except queue.Empty:
                continue

            text = text.strip()
            if not text:
                continue

            try:
                if voice:
                    wav_path = os.path.join(tempfile.gettempdir(), "signspeak_letter_tts.wav")
                    with wave.open(wav_path, "w") as wf:
                        voice.synthesize_wav(text, wf)
                    import winsound
                    winsound.PlaySound(wav_path, winsound.SND_FILENAME)
                self.speech_done.emit(text)
            except Exception as e:
                print(f"TTS Synthesis Error: {e}")

    def stop(self):
        self.running = False


# ═══════════════════════════════════════════════════════════════
# Main PyQt6 Desktop Application Window (0.8s Dwell + Spacebar Word Commit)
# ═══════════════════════════════════════════════════════════════
class SignSpeakApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("SignSpeak — Indian Sign Language Studio")
        self.setMinimumSize(1150, 750)
        self.setStyleSheet(self._get_stylesheet())

        # State
        self.is_running = False
        self.current_word_letters = []
        self.sentence_words = []

        self.live_letter = None
        self.live_confidence = 0.0

        # 0.8s Steady-Hold Dwell Tracker
        self.held_candidate = None
        self.dwell_start_time = None
        self.dwell_progress_pct = 0
        self.locked_letter = None         # Prevents repeating same letter while holding same pose

        self._build_ui()
        self._init_threads()

        # Dwell progress animation timer (30 FPS)
        self.dwell_timer = QTimer()
        self.dwell_timer.timeout.connect(self._on_dwell_tick)
        self.dwell_timer.start(30)

        # Enable window key capturing
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)

        # Auto-start pipeline
        QTimer.singleShot(500, self.start_pipeline)

    def _get_stylesheet(self):
        """Warm soft 2D plushy design system (no glowing neon gradients)."""
        return """
            QMainWindow {
                background-color: #F8F3EE;
            }
            QLabel {
                color: #3D3530;
                font-family: 'Segoe UI', sans-serif;
            }
            QGroupBox {
                background-color: #FFFDFB;
                border: 2px solid #EAE0D5;
                border-radius: 14px;
                margin-top: 10px;
                padding: 14px;
                font-family: 'Segoe UI', sans-serif;
                color: #5C4D44;
                font-weight: bold;
                font-size: 13px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 16px;
                padding: 0 8px;
                background-color: #FFFDFB;
                color: #8C6D58;
            }
            QPushButton {
                background-color: #D96B43;
                color: #FFFFFF;
                border: none;
                padding: 11px 20px;
                font-size: 13px;
                font-weight: bold;
                border-radius: 10px;
                font-family: 'Segoe UI', sans-serif;
            }
            QPushButton:hover {
                background-color: #C55A32;
            }
            QPushButton:pressed {
                background-color: #B04A25;
            }
            QPushButton#recordBtn {
                background-color: #D96B43;
                color: #FFFFFF;
                font-size: 14px;
                padding: 12px 22px;
            }
            QPushButton#recordBtn:hover {
                background-color: #C55A32;
            }
            QPushButton#secondaryBtn {
                background-color: #E2B091;
                color: #3D3530;
            }
            QPushButton#secondaryBtn:hover {
                background-color: #D49D7B;
            }
            QPushButton#stopBtn {
                background-color: #C85A5A;
                color: #FFFFFF;
            }
            QPushButton#stopBtn:hover {
                background-color: #B34848;
            }
            QPushButton#commitBtn {
                background-color: #5A8F76;
                color: #FFFFFF;
            }
            QPushButton#commitBtn:hover {
                background-color: #487A62;
            }
            QProgressBar {
                background-color: #EFE7DE;
                border-radius: 8px;
                text-align: center;
                color: #4A3E37;
                font-weight: bold;
                height: 22px;
            }
            QProgressBar::chunk {
                background-color: #D96B43;
                border-radius: 8px;
            }
            QTextEdit {
                background-color: #FAF6F0;
                color: #4A3E37;
                border: 2px solid #E5D7C9;
                border-radius: 10px;
                font-family: 'Consolas', monospace;
                font-size: 12px;
                padding: 8px;
            }
        """

    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QHBoxLayout(central)
        main_layout.setSpacing(18)
        main_layout.setContentsMargins(18, 18, 18, 18)

        # === LEFT COLUMN: Video Stream & Pipeline Controls ===
        left_col = QVBoxLayout()

        header_title = QLabel("SignSpeak ISL Studio")
        header_title.setFont(QFont("Segoe UI", 19, QFont.Weight.Bold))
        header_title.setStyleSheet("color: #7A533D; padding: 2px;")
        header_title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        left_col.addWidget(header_title)

        video_group = QGroupBox("Camera Stream & Hand Landmarks")
        video_layout = QVBoxLayout(video_group)
        self.video_label = QLabel("Starting Video Stream...")
        self.video_label.setFont(QFont("Segoe UI", 13))
        self.video_label.setFixedSize(CAMERA_WIDTH, CAMERA_HEIGHT)
        self.video_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.video_label.setStyleSheet("background-color: #EFE7DE; color: #7A6F68; border-radius: 10px;")
        video_layout.addWidget(self.video_label)
        left_col.addWidget(video_group)

        controls_layout = QHBoxLayout()
        self.start_btn = QPushButton("Start Camera")
        self.start_btn.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.start_btn.clicked.connect(self.start_pipeline)
        controls_layout.addWidget(self.start_btn)

        self.stop_btn = QPushButton("Stop Camera")
        self.stop_btn.setObjectName("stopBtn")
        self.stop_btn.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.stop_btn.clicked.connect(self.stop_pipeline)
        self.stop_btn.setEnabled(False)
        controls_layout.addWidget(self.stop_btn)
        left_col.addLayout(controls_layout)

        main_layout.addLayout(left_col, stretch=3)

        # === RIGHT COLUMN: Sign Recognition & Word Builder ===
        right_col = QVBoxLayout()

        # Current Detected Letter Display + 0.8s Steady-Hold Progress
        letter_group = QGroupBox("Live Detected Sign & 0.8s Hold Capture")
        letter_layout = QVBoxLayout(letter_group)
        
        self.letter_label = QLabel("-")
        self.letter_label.setFont(QFont("Segoe UI", 48, QFont.Weight.Bold))
        self.letter_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.letter_label.setStyleSheet("color: #D96B43; padding: 2px;")
        letter_layout.addWidget(self.letter_label)

        # Confidence Bar
        self.confidence_bar = QProgressBar()
        self.confidence_bar.setMaximum(100)
        self.confidence_bar.setValue(0)
        self.confidence_bar.setTextVisible(True)
        self.confidence_bar.setFormat("Confidence: %p%")
        self.confidence_bar.setStyleSheet("""
            QProgressBar {
                background-color: #EFE7DE;
                border-radius: 8px;
                text-align: center;
                color: #4A3E37;
                font-weight: bold;
                height: 20px;
            }
            QProgressBar::chunk {
                background-color: #E29578;
                border-radius: 8px;
            }
        """)
        letter_layout.addWidget(self.confidence_bar)

        # 0.8s Steady-Hold Progress Bar
        self.dwell_progress_bar = QProgressBar()
        self.dwell_progress_bar.setMaximum(100)
        self.dwell_progress_bar.setValue(0)
        self.dwell_progress_bar.setTextVisible(True)
        self.dwell_progress_bar.setFormat("Hold Steady for 0.8s to Capture")
        self.dwell_progress_bar.setStyleSheet("""
            QProgressBar {
                background-color: #EFE7DE;
                border-radius: 8px;
                text-align: center;
                color: #4A3E37;
                font-weight: bold;
                height: 24px;
            }
            QProgressBar::chunk {
                background-color: #5A8F76;
                border-radius: 8px;
            }
        """)
        letter_layout.addWidget(self.dwell_progress_bar)

        right_col.addWidget(letter_group)

        # Active Word Builder Box (Spacebar = Commit Word)
        word_group = QGroupBox("Active Word Builder (Press Spacebar to Commit)")
        word_layout = QVBoxLayout(word_group)
        
        self.word_label = QLabel("")
        self.word_label.setFont(QFont("Segoe UI", 26, QFont.Weight.Bold))
        self.word_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.word_label.setStyleSheet(
            "color: #4A3E37; background-color: #F4EAE1; border: 2px solid #E5D7C9; "
            "border-radius: 10px; padding: 10px; min-height: 45px;"
        )
        word_layout.addWidget(self.word_label)

        word_btn_layout = QHBoxLayout()

        self.commit_btn = QPushButton("Commit Word (Spacebar)")
        self.commit_btn.setObjectName("commitBtn")
        self.commit_btn.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.commit_btn.clicked.connect(self.commit_word)
        word_btn_layout.addWidget(self.commit_btn)

        del_btn = QPushButton("Delete Letter (Backspace)")
        del_btn.setObjectName("secondaryBtn")
        del_btn.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        del_btn.clicked.connect(self.delete_last_letter)
        word_btn_layout.addWidget(del_btn)

        word_layout.addLayout(word_btn_layout)
        right_col.addWidget(word_group)

        # Sentence Builder & Spoken Output Box
        sentence_group = QGroupBox("Full Sentence Line (Press Enter to Speak)")
        sentence_layout = QVBoxLayout(sentence_group)
        self.sentence_label = QLabel("")
        self.sentence_label.setFont(QFont("Segoe UI", 16))
        self.sentence_label.setWordWrap(True)
        self.sentence_label.setStyleSheet(
            "color: #3D3530; background-color: #FAF6F0; border: 1px solid #EAE0D5; "
            "border-radius: 8px; padding: 10px; min-height: 50px;"
        )
        sentence_layout.addWidget(self.sentence_label)

        sent_btn_layout = QHBoxLayout()
        speak_btn = QPushButton("Speak Full Sentence (Enter)")
        speak_btn.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        speak_btn.clicked.connect(self.speak_full_sentence)
        sent_btn_layout.addWidget(speak_btn)

        clear_btn = QPushButton("Clear All (Esc)")
        clear_btn.setObjectName("secondaryBtn")
        clear_btn.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        clear_btn.clicked.connect(self.clear_all)
        sent_btn_layout.addWidget(clear_btn)

        sentence_layout.addLayout(sent_btn_layout)
        right_col.addWidget(sentence_group)

        # System Log Box
        log_group = QGroupBox("Activity Log")
        log_layout = QVBoxLayout(log_group)
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setMaximumHeight(120)
        log_layout.addWidget(self.log_text)
        right_col.addWidget(log_group)

        self.status_label = QLabel("Status: Ready — Hold sign steady for 0.8s to capture.")
        self.status_label.setStyleSheet("color: #8C7C72; font-size: 12px; padding: 2px;")
        right_col.addWidget(self.status_label)

        main_layout.addLayout(right_col, stretch=2)

    def _init_threads(self):
        self.capture_thread = CaptureThread()
        self.capture_thread.frame_ready.connect(self.update_video)
        self.capture_thread.feature_ready.connect(self.on_feature_ready)
        self.capture_thread.status_update.connect(self.log)

        self.inference_thread = InferenceThread(ONNX_MODEL_PATH, CLASS_META_PATH)
        self.inference_thread.prediction_ready.connect(self.on_prediction)
        self.inference_thread.no_hand_signal.connect(self.on_no_hand)
        self.inference_thread.status_update.connect(self.log)

        self.tts_thread = TTSThread(PIPER_MODEL_PATH)
        self.tts_thread.speech_done.connect(self.on_speech_done)

    def start_pipeline(self):
        if self.is_running:
            return
        self.is_running = True
        self.start_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)

        self.log("Starting ISL Camera Pipeline...")
        self.capture_thread.start()
        self.inference_thread.start()
        self.tts_thread.start()

    def stop_pipeline(self):
        self.is_running = False
        self.capture_thread.stop()
        self.inference_thread.stop()
        self.tts_thread.stop()

        self.capture_thread.wait(3000)
        self.inference_thread.wait(3000)
        self.tts_thread.wait(3000)

        self._init_threads()
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.status_label.setText("Status: Stopped")
        self.log("Pipeline stopped.")

    def update_video(self, frame):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        bytes_per_line = ch * w
        qt_img = QImage(rgb.data, w, h, bytes_per_line, QImage.Format.Format_RGB888).copy()
        pixmap = QPixmap.fromImage(qt_img).scaled(
            self.video_label.size(), Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation
        )
        self.video_label.setPixmap(pixmap)

    def on_feature_ready(self, feat_vec):
        self.inference_thread.enqueue_feature(feat_vec)

    # ═══════════════════════════════════════════════════════════
    # 0.8s Steady-Hold Dwell Capture Engine
    # ═══════════════════════════════════════════════════════════
    def on_prediction(self, letter, confidence):
        now = time.time()
        self.live_letter = letter
        self.live_confidence = confidence

        self.letter_label.setText(letter)
        self.confidence_bar.setValue(min(100, int(confidence * 100)))

        # Dwell logic: Track continuous steady posture
        if confidence >= CONFIDENCE_THRESHOLD and letter not in ("-", "?", "NO_SIGN"):
            if letter == self.held_candidate:
                # Same sign being maintained
                if self.dwell_start_time is None:
                    self.dwell_start_time = now
            else:
                # Switched to a new letter: reset dwell timer and unlock
                self.held_candidate = letter
                self.dwell_start_time = now
                self.dwell_progress_pct = 0
                self.locked_letter = None
        else:
            # Low confidence or transitioning
            self.held_candidate = None
            self.dwell_start_time = None
            self.dwell_progress_pct = 0
            self.locked_letter = None

        self.status_label.setText(f"Live — Sign: {letter} ({confidence:.1%}) | Hold for 0.8s to record | Spacebar = Commit Word")

    def on_no_hand(self):
        self.letter_label.setText("-")
        self.confidence_bar.setValue(0)
        self.live_letter = None
        self.live_confidence = 0.0
        self.held_candidate = None
        self.dwell_start_time = None
        self.dwell_progress_pct = 0
        self.locked_letter = None
        self.dwell_progress_bar.setValue(0)
        self.dwell_progress_bar.setFormat("Hold Steady for 0.8s to Capture")
        self.status_label.setText("Live — Waiting for Hand Gesture...")

    def _on_dwell_tick(self):
        """Timer callback updating the 0.8s progress bar and capturing upon completion."""
        if not self.is_running:
            return

        now = time.time()
        if self.held_candidate and self.dwell_start_time is not None:
            if self.locked_letter != self.held_candidate:
                elapsed = now - self.dwell_start_time
                self.dwell_progress_pct = min(100, int((elapsed / DWELL_HOLD_SECONDS) * 100))

                if elapsed >= DWELL_HOLD_SECONDS:
                    self._capture_letter(self.held_candidate, self.live_confidence)
                    self.locked_letter = self.held_candidate
                    self.dwell_progress_pct = 100
            else:
                self.dwell_progress_pct = 100
        else:
            self.dwell_progress_pct = 0

        self.dwell_progress_bar.setValue(self.dwell_progress_pct)
        if self.dwell_progress_pct >= 100 and self.locked_letter:
            self.dwell_progress_bar.setFormat(f"Captured '{self.locked_letter}'")
        elif self.dwell_progress_pct > 0 and self.held_candidate:
            self.dwell_progress_bar.setFormat(f"Holding '{self.held_candidate}' ({self.dwell_progress_pct}%)")
        else:
            self.dwell_progress_bar.setFormat("Hold Steady for 0.8s to Capture")

    def _capture_letter(self, letter, confidence):
        """Adds confirmed letter to the active word with audio feedback."""
        self.current_word_letters.append(letter)
        word_str = "".join(self.current_word_letters)
        self.word_label.setText(word_str)
        self.log(f"Captured Letter: '{letter}' ({confidence:.1%}) ──► Word: \"{word_str}\"")
        play_feedback_tone(freq=1250, duration_ms=35)

    # ═══════════════════════════════════════════════════════════
    # Word & Sentence Construction
    # ═══════════════════════════════════════════════════════════
    def commit_word(self):
        """Commits the active word to the sentence line (Spacebar)."""
        if self.current_word_letters:
            word = "".join(self.current_word_letters)
            self.sentence_words.append(word)
            self.sentence_label.setText(" ".join(self.sentence_words))
            self.log(f"Committed Word: \"{word}\" (Spacebar)")
            self.current_word_letters.clear()
            self.word_label.setText("")
            play_feedback_tone(freq=900, duration_ms=25)

    def delete_last_letter(self):
        """Deletes last letter, or pulls back last committed word if active word is empty."""
        if self.current_word_letters:
            popped = self.current_word_letters.pop()
            self.word_label.setText("".join(self.current_word_letters))
            self.log(f"Deleted Letter: '{popped}'")
        elif self.sentence_words:
            last_word = self.sentence_words.pop()
            self.sentence_label.setText(" ".join(self.sentence_words))
            self.current_word_letters = list(last_word)
            self.word_label.setText(last_word)
            self.log(f"Restored Word for Editing: \"{last_word}\"")

    def speak_full_sentence(self):
        """Commits any pending word and synthesizes speech for the complete sentence."""
        if self.current_word_letters:
            self.commit_word()

        full_text = self.sentence_label.text().strip()
        if full_text:
            self.tts_thread.enqueue_text(full_text)
            self.log(f"Speaking Full Sentence: \"{full_text}\"")

    def clear_all(self):
        """Clears word and sentence buffers."""
        self.current_word_letters.clear()
        self.sentence_words.clear()
        self.word_label.setText("")
        self.sentence_label.setText("")
        self.held_candidate = None
        self.dwell_start_time = None
        self.dwell_progress_pct = 0
        self.locked_letter = None
        self.dwell_progress_bar.setValue(0)
        self.dwell_progress_bar.setFormat("Hold Steady for 0.8s to Capture")
        self.log("Cleared word and sentence buffers.")

    def keyPressEvent(self, event):
        """Handle global keyboard shortcuts cleanly."""
        if event.key() == Qt.Key.Key_Space:
            self.commit_word()
        elif event.key() == Qt.Key.Key_Backspace:
            self.delete_last_letter()
        elif event.key() in (Qt.Key.Key_Return, Qt.Key.Key_Enter):
            self.speak_full_sentence()
        elif event.key() == Qt.Key.Key_Escape:
            self.clear_all()
        else:
            super().keyPressEvent(event)

    def on_speech_done(self, text):
        self.log(f"Piper TTS Spoke: \"{text}\"")

    def log(self, msg):
        ts = time.strftime("%H:%M:%S")
        self.log_text.append(f"[{ts}] {msg}")

    def closeEvent(self, event):
        self.stop_pipeline()
        event.accept()


# ═══════════════════════════════════════════════════════════════
# Main Entry Point
# ═══════════════════════════════════════════════════════════════
def main():
    app = QApplication(sys.argv)
    app.setStyle("Fusion")

    # Warm Soft 2D Plushy Palette
    palette = QPalette()
    palette.setColor(QPalette.ColorRole.Window, QColor(248, 243, 238))
    palette.setColor(QPalette.ColorRole.WindowText, QColor(61, 53, 48))
    palette.setColor(QPalette.ColorRole.Base, QColor(250, 246, 240))
    palette.setColor(QPalette.ColorRole.AlternateBase, QColor(244, 234, 225))
    palette.setColor(QPalette.ColorRole.Text, QColor(61, 53, 48))
    palette.setColor(QPalette.ColorRole.Button, QColor(217, 107, 67))
    palette.setColor(QPalette.ColorRole.ButtonText, QColor(255, 255, 255))
    palette.setColor(QPalette.ColorRole.Highlight, QColor(226, 149, 120))
    palette.setColor(QPalette.ColorRole.HighlightedText, QColor(255, 255, 255))
    app.setPalette(palette)

    window = SignSpeakApp()
    window.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
