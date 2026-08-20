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
# Thread 4: Non-Blocking AI Autocomplete Worker (Groq Cloud / Offline Fallback)
# ═══════════════════════════════════════════════════════════════
class AIPredictionThread(QThread):
    suggestions_ready = pyqtSignal(list)
    status_update = pyqtSignal(str)

    def __init__(self, config_path=None):
        super().__init__()
        self.config_path = config_path or (BASE_DIR / "ai_config.json")
        self.request_queue = queue.Queue(maxsize=4)
        self.running = True
        self.api_key = None
        self._load_config()

        # Offline common frequency dictionary for zero-delay instant fallback
        self.offline_dict = [
            "ABOUT", "AFTER", "AGAIN", "ALL", "ALWAYS", "AND", "ANY", "ASK", "BAD", "BEAUTIFUL",
            "BECAUSE", "BEFORE", "BEST", "BETTER", "BIG", "BOOK", "BOY", "CALL", "CAN", "CAR",
            "CHANGE", "CHILD", "COME", "COOK", "DAY", "DOCTOR", "DO", "DOG", "DRINK", "DRIVE",
            "EAT", "FAMILY", "FATHER", "FEEL", "FIND", "FOOD", "FOR", "FRIEND", "GIVE", "GO",
            "GOOD", "HAPPY", "HAVE", "HEAR", "HELLO", "HELP", "HERE", "HOME", "HOSPITAL", "HOUSE",
            "HOW", "HURT", "IMPORTANT", "IS", "KNOW", "LANGUAGE", "LEARN", "LIKE", "LISTEN", "LIVE",
            "LOOK", "LOVE", "MAKE", "MAN", "ME", "MEET", "MONEY", "MORE", "MORNING", "MOTHER",
            "NAME", "NEED", "NEW", "NIGHT", "NO", "NOW", "OF", "OFFICE", "OLD", "OPEN",
            "PEOPLE", "PLEASE", "READ", "RIGHT", "ROOM", "SAD", "SAY", "SCHOOL", "SEE", "SIGN",
            "SIT", "SLEEP", "SLOW", "SMALL", "SORRY", "SPEAK", "STAND", "STOP", "STUDY", "TALK",
            "TEACHER", "TELL", "THANK", "THAT", "THE", "THEIR", "THEM", "THINK", "TIME", "TODAY",
            "TOMORROW", "UNDERSTAND", "USE", "WAIT", "WALK", "WANT", "WARM", "WATCH", "WATER",
            "WAY", "WE", "WELCOME", "WHAT", "WHEN", "WHERE", "WHICH", "WHO", "WHY", "WILL",
            "WITH", "WOMAN", "WORD", "WORK", "WORLD", "WRITE", "WRONG", "YES", "YOU", "YOUR"
        ]

    def _load_config(self):
        if os.path.exists(self.config_path):
            try:
                with open(self.config_path, "r", encoding="utf-8") as f:
                    cfg = json.load(f)
                    self.api_key = cfg.get("api_key", "").strip()
            except Exception as e:
                print(f"AI Config Load Notice: {e}")

    def enqueue_prediction(self, prefix, sentence_context=""):
        while not self.request_queue.empty():
            try:
                self.request_queue.get_nowait()
            except queue.Empty:
                break
        self.request_queue.put((prefix.upper().strip(), sentence_context.strip()))

    def run(self):
        self.running = True
        while self.running:
            try:
                item = self.request_queue.get(timeout=0.5)
            except queue.Empty:
                continue

            prefix, context = item
            if not prefix:
                self.suggestions_ready.emit([])
                continue

            # 1. Query Groq Cloud API
            suggestions = self._query_groq(prefix, context)

            # 2. Fallback to offline dictionary if offline or timed out
            if not suggestions:
                suggestions = [w for w in self.offline_dict if w.startswith(prefix) and w != prefix][:3]

            self.suggestions_ready.emit(suggestions)

    def _query_groq(self, prefix, context):
        if not self.api_key:
            return None

        url = "https://api.groq.com/openai/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"
        }
        prompt = f"Suggest up to 3 uppercase English words starting with '{prefix}'."
        if context:
            prompt += f" Previous sentence context: '{context}'."
        prompt += " Return ONLY a JSON list of uppercase strings, e.g. [\"WATER\", \"WATCH\", \"WAIT\"]."

        payload = {
            "model": "groq/compound-mini",
            "messages": [
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.1,
            "max_tokens": 35
        }

        try:
            import urllib.request
            import re
            req = urllib.request.Request(url, data=json.dumps(payload).encode("utf-8"), headers=headers)
            with urllib.request.urlopen(req, timeout=1.8) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                content = data["choices"][0]["message"]["content"]
                match = re.search(r"\[.*?\]", content, re.DOTALL)
                if match:
                    words = json.loads(match.group(0))
                    valid = []
                    for w in words:
                        clean = re.sub(r"[^A-Z]", "", str(w).upper())
                        if clean and clean.startswith(prefix) and clean not in valid:
                            valid.append(clean)
                    return valid[:3]
        except Exception:
            pass
        return None

    def stop(self):
        self.running = False


# ═══════════════════════════════════════════════════════════════
# Main PyQt6 Desktop Application Window (UI/UX Redesigned + AI Autocomplete)
# ═══════════════════════════════════════════════════════════════
class SignSpeakApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("SignSpeak Studio — Indian Sign Language Fingerspelling & Sentence Builder")
        self.setMinimumSize(1220, 800)
        self.setStyleSheet(self._get_stylesheet())

        # State Variables (100% Preserved)
        self.is_running = False
        self.current_word_letters = []
        self.sentence_words = []
        self.current_suggestions = []

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
        """Warm Soft 2D Plushy Design System — Modern, Minimal, High-Contrast & Accessible."""
        return """
            QMainWindow {
                background-color: #F7F4EF;
            }
            QLabel {
                color: #2D2521;
                font-family: 'Segoe UI', -apple-system, BlinkMacSystemFont, 'Helvetica Neue', sans-serif;
            }
            QGroupBox {
                background-color: #FFFFFF;
                border: 1.5px solid #E8DFD5;
                border-radius: 14px;
                margin-top: 12px;
                padding: 14px;
                font-family: 'Segoe UI', -apple-system, BlinkMacSystemFont, sans-serif;
                color: #5C4D44;
                font-weight: 700;
                font-size: 12px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 14px;
                padding: 0 8px;
                background-color: #FFFFFF;
                color: #75655B;
                border-radius: 4px;
            }
            QPushButton {
                background-color: #D96B43;
                color: #FFFFFF;
                border: none;
                padding: 10px 18px;
                font-size: 13px;
                font-weight: 700;
                border-radius: 10px;
                font-family: 'Segoe UI', -apple-system, BlinkMacSystemFont, sans-serif;
            }
            QPushButton:hover {
                background-color: #C55A32;
            }
            QPushButton:pressed {
                background-color: #A84521;
            }
            QPushButton:disabled {
                background-color: #D8CCC0;
                color: #8C7F75;
            }
            QPushButton#secondaryBtn {
                background-color: #EFE7DE;
                color: #4A3E37;
                border: 1.5px solid #D8C9B8;
            }
            QPushButton#secondaryBtn:hover {
                background-color: #E5DACF;
                border-color: #C9B8A5;
            }
            QPushButton#secondaryBtn:pressed {
                background-color: #D8CCC0;
            }
            QPushButton#stopBtn {
                background-color: #C45353;
                color: #FFFFFF;
            }
            QPushButton#stopBtn:hover {
                background-color: #AF4242;
            }
            QPushButton#stopBtn:pressed {
                background-color: #933333;
            }
            QPushButton#commitBtn {
                background-color: #3E8867;
                color: #FFFFFF;
            }
            QPushButton#commitBtn:hover {
                background-color: #337356;
            }
            QPushButton#commitBtn:pressed {
                background-color: #275C44;
            }
            QProgressBar {
                background-color: #EFE7DE;
                border: 1px solid #DFD2C4;
                border-radius: 8px;
                text-align: center;
                color: #2D2521;
                font-weight: 700;
                font-size: 11px;
                font-family: 'Segoe UI', sans-serif;
            }
            QProgressBar::chunk {
                border-radius: 7px;
            }
            QTextEdit {
                background-color: #FAF7F2;
                color: #3D3530;
                border: 1.5px solid #EAE0D5;
                border-radius: 10px;
                font-family: 'Consolas', 'Courier New', monospace;
                font-size: 11px;
                padding: 8px;
                line-height: 1.4;
            }
            QScrollBar:vertical {
                background: #FAF7F2;
                width: 8px;
                border-radius: 4px;
            }
            QScrollBar::handle:vertical {
                background: #D8CCC0;
                border-radius: 4px;
                min-height: 20px;
            }
            QScrollBar::handle:vertical:hover {
                background: #BFAF9E;
            }
        """

    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        root_vbox = QVBoxLayout(central)
        root_vbox.setSpacing(14)
        root_vbox.setContentsMargins(20, 16, 20, 20)

        # ═══════════════════════════════════════════════════════════
        # TOP HEADER APP BAR
        # ═══════════════════════════════════════════════════════════
        header_widget = QWidget()
        header_layout = QHBoxLayout(header_widget)
        header_layout.setContentsMargins(4, 2, 4, 2)

        header_title_layout = QVBoxLayout()
        header_title_layout.setSpacing(2)

        app_title = QLabel("SignSpeak Studio")
        app_title.setFont(QFont("Segoe UI", 20, QFont.Weight.Bold))
        app_title.setStyleSheet("color: #2D2521; font-weight: 800;")
        header_title_layout.addWidget(app_title)

        app_subtitle = QLabel("Indian Sign Language Fingerspelling & Real-Time Sentence Builder")
        app_subtitle.setFont(QFont("Segoe UI", 11))
        app_subtitle.setStyleSheet("color: #75655B;")
        header_title_layout.addWidget(app_subtitle)

        header_layout.addLayout(header_title_layout)
        header_layout.addStretch()

        # Top Status Badges
        badges_layout = QHBoxLayout()
        badges_layout.setSpacing(10)

        ai_badge = QLabel("⚡ AI Autocomplete")
        ai_badge.setFont(QFont("Segoe UI", 11, QFont.Weight.Bold))
        ai_badge.setStyleSheet(
            "background-color: #EFE7DE; color: #5C4D44; border: 1px solid #D8C9B8; "
            "border-radius: 8px; padding: 6px 12px;"
        )
        badges_layout.addWidget(ai_badge)

        offline_badge = QLabel("🔊 Neural Piper Voice")
        offline_badge.setFont(QFont("Segoe UI", 11, QFont.Weight.Bold))
        offline_badge.setStyleSheet(
            "background-color: #EFE7DE; color: #5C4D44; border: 1px solid #D8C9B8; "
            "border-radius: 8px; padding: 6px 12px;"
        )
        badges_layout.addWidget(offline_badge)

        self.engine_status_badge = QLabel("● Engine Active")
        self.engine_status_badge.setFont(QFont("Segoe UI", 11, QFont.Weight.Bold))
        self.engine_status_badge.setStyleSheet(
            "background-color: #E3F1E9; color: #2D704F; border: 1px solid #B8DCBE; "
            "border-radius: 8px; padding: 6px 12px;"
        )
        badges_layout.addWidget(self.engine_status_badge)

        header_layout.addLayout(badges_layout)
        root_vbox.addWidget(header_widget)

        # ═══════════════════════════════════════════════════════════
        # MAIN DUAL-COLUMN BODY
        # ═══════════════════════════════════════════════════════════
        body_layout = QHBoxLayout()
        body_layout.setSpacing(18)

        # ───────────────────────────────────────────────────────────
        # LEFT COLUMN: Camera-First Viewport & Controls
        # ───────────────────────────────────────────────────────────
        left_col = QVBoxLayout()
        left_col.setSpacing(12)

        video_group = QGroupBox("Camera Viewport & Skeleton Tracking")
        video_layout = QVBoxLayout(video_group)
        video_layout.setContentsMargins(10, 14, 10, 10)

        self.video_label = QLabel("Initializing Video Stream...")
        self.video_label.setFont(QFont("Segoe UI", 12))
        self.video_label.setFixedSize(CAMERA_WIDTH, CAMERA_HEIGHT)
        self.video_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.video_label.setStyleSheet(
            "background-color: #EBE3D8; color: #75655B; border-radius: 12px; "
            "border: 1.5px solid #E2D7CB;"
        )
        video_layout.addWidget(self.video_label)
        left_col.addWidget(video_group)

        # Camera Controls Row
        controls_layout = QHBoxLayout()
        controls_layout.setSpacing(12)

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

        # Keyboard Cheat Sheet Reference Card
        shortcut_box = QGroupBox("Keyboard Navigation Guide")
        shortcut_layout = QVBoxLayout(shortcut_box)
        shortcut_layout.setContentsMargins(12, 12, 12, 12)
        shortcut_layout.setSpacing(6)

        shortcuts = [
            ("Hold Sign (0.8s)", "Captures letter into active word"),
            ("Keys [ 1 / 2 / 3 ]", "Accepts AI Autocomplete suggestion"),
            ("Spacebar", "Commits active word to sentence line"),
            ("Backspace", "Deletes last letter (or restores last word)"),
            ("Enter", "Synthesizes speech for full sentence"),
            ("Escape", "Clears both word and sentence buffers")
        ]

        for key_text, desc_text in shortcuts:
            row = QHBoxLayout()
            row.setSpacing(8)
            k_lbl = QLabel(key_text)
            k_lbl.setFont(QFont("Segoe UI", 10, QFont.Weight.Bold))
            k_lbl.setStyleSheet(
                "background-color: #EFE7DE; color: #4A3E37; border: 1px solid #D8C9B8; "
                "border-radius: 5px; padding: 2px 8px; min-width: 120px;"
            )
            d_lbl = QLabel(desc_text)
            d_lbl.setFont(QFont("Segoe UI", 10))
            d_lbl.setStyleSheet("color: #75655B;")
            row.addWidget(k_lbl)
            row.addWidget(d_lbl, stretch=1)
            shortcut_layout.addLayout(row)

        left_col.addWidget(shortcut_box)
        body_layout.addLayout(left_col, stretch=3)

        # ───────────────────────────────────────────────────────────
        # RIGHT COLUMN: Recognition, Word & Sentence Cockpit
        # ───────────────────────────────────────────────────────────
        right_col = QVBoxLayout()
        right_col.setSpacing(12)

        # ── Card 1: Live Detected Sign & 0.8s Steady Hold ──
        letter_group = QGroupBox("Live Detected Sign & 0.8s Hold Capture")
        letter_layout = QHBoxLayout(letter_group)
        letter_layout.setContentsMargins(14, 14, 14, 14)
        letter_layout.setSpacing(16)

        # Big Sign Letter Tile
        letter_card = QFrame()
        letter_card.setFixedSize(96, 96)
        letter_card.setStyleSheet(
            "background-color: #FDF9F5; border: 2px solid #E8DACB; border-radius: 14px;"
        )
        letter_card_layout = QVBoxLayout(letter_card)
        letter_card_layout.setContentsMargins(0, 0, 0, 0)
        self.letter_label = QLabel("-")
        self.letter_label.setFont(QFont("Segoe UI", 52, QFont.Weight.Bold))
        self.letter_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.letter_label.setStyleSheet("color: #D96B43; border: none; background: transparent;")
        letter_card_layout.addWidget(self.letter_label)
        letter_layout.addWidget(letter_card)

        # Meters Stack
        meters_vbox = QVBoxLayout()
        meters_vbox.setSpacing(8)

        # Model Confidence Progress Bar
        self.confidence_bar = QProgressBar()
        self.confidence_bar.setMaximum(100)
        self.confidence_bar.setValue(0)
        self.confidence_bar.setTextVisible(True)
        self.confidence_bar.setFormat("Model Confidence: %p%")
        self.confidence_bar.setFixedHeight(20)
        self.confidence_bar.setStyleSheet("""
            QProgressBar {
                background-color: #EFE7DE;
                border: 1px solid #DFD2C4;
                border-radius: 6px;
                text-align: center;
                color: #2D2521;
                font-weight: bold;
                font-size: 10px;
            }
            QProgressBar::chunk {
                background-color: #E29578;
                border-radius: 5px;
            }
        """)
        meters_vbox.addWidget(self.confidence_bar)

        # 0.8s Steady-Hold Dwell Bar
        self.dwell_progress_bar = QProgressBar()
        self.dwell_progress_bar.setMaximum(100)
        self.dwell_progress_bar.setValue(0)
        self.dwell_progress_bar.setTextVisible(True)
        self.dwell_progress_bar.setFormat("Hold Steady for 0.8s to Capture")
        self.dwell_progress_bar.setFixedHeight(24)
        self.dwell_progress_bar.setStyleSheet("""
            QProgressBar {
                background-color: #EFE7DE;
                border: 1px solid #DFD2C4;
                border-radius: 6px;
                text-align: center;
                color: #2D2521;
                font-weight: bold;
                font-size: 11px;
            }
            QProgressBar::chunk {
                background-color: #3E8867;
                border-radius: 5px;
            }
        """)
        meters_vbox.addWidget(self.dwell_progress_bar)

        letter_layout.addLayout(meters_vbox, stretch=1)
        right_col.addWidget(letter_group)

        # ── Card 2: Active Word Builder + AI Autocomplete Strip ──
        word_group = QGroupBox("Active Word Builder")
        word_layout = QVBoxLayout(word_group)
        word_layout.setContentsMargins(14, 14, 14, 14)
        word_layout.setSpacing(8)

        # 💡 Gboard-Style 3 Suggestion Pills Strip
        self.suggestions_container = QWidget()
        self.suggestions_layout = QHBoxLayout(self.suggestions_container)
        self.suggestions_layout.setContentsMargins(0, 0, 0, 2)
        self.suggestions_layout.setSpacing(8)

        self.sug_title_lbl = QLabel("💡 AI Suggestions:")
        self.sug_title_lbl.setFont(QFont("Segoe UI", 10, QFont.Weight.Bold))
        self.sug_title_lbl.setStyleSheet("color: #8C6D58;")
        self.suggestions_layout.addWidget(self.sug_title_lbl)

        self.pill_buttons = []
        for i in range(3):
            btn = QPushButton(f"[{i+1}] -")
            btn.setFocusPolicy(Qt.FocusPolicy.NoFocus)
            btn.setStyleSheet("""
                QPushButton {
                    background-color: #EFE7DE;
                    color: #4A3E37;
                    border: 1.5px solid #D8C9B8;
                    border-radius: 8px;
                    padding: 5px 12px;
                    font-size: 11px;
                    font-weight: bold;
                    font-family: 'Segoe UI', sans-serif;
                }
                QPushButton:hover {
                    background-color: #D96B43;
                    color: #FFFFFF;
                    border-color: #C55A32;
                }
                QPushButton:pressed {
                    background-color: #B04A25;
                }
            """)
            btn.clicked.connect(lambda checked, idx=i: self._on_suggestion_clicked(idx))
            btn.setVisible(False)
            self.suggestions_layout.addWidget(btn)
            self.pill_buttons.append(btn)

        self.suggestions_layout.addStretch()
        self.suggestions_container.setVisible(False)
        word_layout.addWidget(self.suggestions_container)

        self.word_label = QLabel("")
        self.word_label.setFont(QFont("Segoe UI", 26, QFont.Weight.Bold))
        self.word_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.word_label.setFixedHeight(50)
        self.word_label.setStyleSheet(
            "color: #2D2521; background-color: #FAF4EE; border: 1.5px solid #E2D7CB; "
            "border-radius: 12px; padding: 4px 12px; letter-spacing: 2px;"
        )
        word_layout.addWidget(self.word_label)

        word_btn_layout = QHBoxLayout()
        word_btn_layout.setSpacing(10)

        self.commit_btn = QPushButton("Commit Word [ Space ]")
        self.commit_btn.setObjectName("commitBtn")
        self.commit_btn.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.commit_btn.clicked.connect(self.commit_word)
        word_btn_layout.addWidget(self.commit_btn, stretch=3)

        del_btn = QPushButton("Delete Letter [ ⌫ ]")
        del_btn.setObjectName("secondaryBtn")
        del_btn.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        del_btn.clicked.connect(self.delete_last_letter)
        word_btn_layout.addWidget(del_btn, stretch=2)

        word_layout.addLayout(word_btn_layout)
        right_col.addWidget(word_group)

        # ── Card 3: Full Sentence Builder ──
        sentence_group = QGroupBox("Full Spoken Sentence Line")
        sentence_layout = QVBoxLayout(sentence_group)
        sentence_layout.setContentsMargins(14, 14, 14, 14)
        sentence_layout.setSpacing(10)

        self.sentence_label = QLabel("")
        self.sentence_label.setFont(QFont("Segoe UI", 16))
        self.sentence_label.setWordWrap(True)
        self.sentence_label.setMinimumHeight(52)
        self.sentence_label.setStyleSheet(
            "color: #2D2521; background-color: #FFFFFF; border: 1.5px solid #E2D7CB; "
            "border-radius: 12px; padding: 10px 14px;"
        )
        sentence_layout.addWidget(self.sentence_label)

        sent_btn_layout = QHBoxLayout()
        sent_btn_layout.setSpacing(10)

        speak_btn = QPushButton("Speak Full Sentence [ Enter ]")
        speak_btn.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        speak_btn.clicked.connect(self.speak_full_sentence)
        sent_btn_layout.addWidget(speak_btn, stretch=3)

        clear_btn = QPushButton("Clear All [ Esc ]")
        clear_btn.setObjectName("secondaryBtn")
        clear_btn.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        clear_btn.clicked.connect(self.clear_all)
        sent_btn_layout.addWidget(clear_btn, stretch=2)

        sentence_layout.addLayout(sent_btn_layout)
        right_col.addWidget(sentence_group)

        # ── Card 4: Activity Stream & Status ──
        log_group = QGroupBox("Activity Stream")
        log_layout = QVBoxLayout(log_group)
        log_layout.setContentsMargins(12, 12, 12, 12)
        log_layout.setSpacing(8)

        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setFixedHeight(105)
        log_layout.addWidget(self.log_text)

        self.status_label = QLabel("Status: Ready — Hold sign steady for 0.8s to capture | Keys [1/2/3] for AI Autocomplete.")
        self.status_label.setStyleSheet("color: #75655B; font-size: 11px; padding: 0 2px;")
        log_layout.addWidget(self.status_label)

        right_col.addWidget(log_group)
        body_layout.addLayout(right_col, stretch=2)

        root_vbox.addLayout(body_layout)

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

        self.ai_thread = AIPredictionThread(BASE_DIR / "ai_config.json")
        self.ai_thread.suggestions_ready.connect(self.on_suggestions_ready)

    def start_pipeline(self):
        if self.is_running:
            return
        self.is_running = True
        self.start_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.engine_status_badge.setText("● Camera Active (30 FPS)")
        self.engine_status_badge.setStyleSheet(
            "background-color: #E3F1E9; color: #2D704F; border: 1px solid #B8DCBE; "
            "border-radius: 8px; padding: 6px 12px;"
        )

        self.log("Starting ISL Camera Pipeline...")
        self.capture_thread.start()
        self.inference_thread.start()
        self.tts_thread.start()
        self.ai_thread.start()

    def stop_pipeline(self):
        self.is_running = False
        self.capture_thread.stop()
        self.inference_thread.stop()
        self.tts_thread.stop()
        self.ai_thread.stop()

        self.capture_thread.wait(3000)
        self.inference_thread.wait(3000)
        self.tts_thread.wait(3000)
        self.ai_thread.wait(3000)

        self._init_threads()
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.engine_status_badge.setText("○ Camera Stopped")
        self.engine_status_badge.setStyleSheet(
            "background-color: #F8E8E8; color: #8A3333; border: 1px solid #E0BDB8; "
            "border-radius: 8px; padding: 6px 12px;"
        )
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
    # 0.8s Steady-Hold Dwell Capture Engine (100% Preserved)
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
        self._trigger_ai_prediction()

    # ═══════════════════════════════════════════════════════════
    # AI Autocomplete & Gboard Suggestion Handlers
    # ═══════════════════════════════════════════════════════════
    def _trigger_ai_prediction(self):
        """Dispatches non-blocking suggestion lookup for active prefix."""
        prefix = "".join(self.current_word_letters).strip()
        context = " ".join(self.sentence_words).strip()
        if prefix:
            self.ai_thread.enqueue_prediction(prefix, context)
        else:
            self.on_suggestions_ready([])

    def on_suggestions_ready(self, suggestions):
        """Updates the 3 Gboard suggestion pills smoothly."""
        self.current_suggestions = suggestions
        prefix = "".join(self.current_word_letters).strip()
        if suggestions and prefix:
            self.suggestions_container.setVisible(True)
            for i in range(3):
                if i < len(suggestions):
                    self.pill_buttons[i].setText(f"[{i+1}] {suggestions[i]}")
                    self.pill_buttons[i].setVisible(True)
                else:
                    self.pill_buttons[i].setVisible(False)
        else:
            for btn in self.pill_buttons:
                btn.setVisible(False)
            self.suggestions_container.setVisible(False)

    def _on_suggestion_clicked(self, idx):
        """Triggered when clicking a suggestion pill."""
        if self.current_suggestions and idx < len(self.current_suggestions):
            self._accept_autocomplete(self.current_suggestions[idx])

    def _accept_autocomplete(self, word):
        """Commits chosen autocomplete suggestion directly into the sentence line."""
        self.sentence_words.append(word)
        self.sentence_label.setText(" ".join(self.sentence_words))
        self.current_word_letters.clear()
        self.word_label.setText("")
        self.on_suggestions_ready([])
        self.log(f"AI Autocomplete: \"{word}\" committed to sentence.")
        play_feedback_tone(freq=1100, duration_ms=40)

    # ═══════════════════════════════════════════════════════════
    # Word & Sentence Construction (100% Preserved)
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
            self.on_suggestions_ready([])
            play_feedback_tone(freq=900, duration_ms=25)

    def delete_last_letter(self):
        """Deletes last letter, or pulls back last committed word if active word is empty."""
        if self.current_word_letters:
            popped = self.current_word_letters.pop()
            self.word_label.setText("".join(self.current_word_letters))
            self.log(f"Deleted Letter: '{popped}'")
            self._trigger_ai_prediction()
        elif self.sentence_words:
            last_word = self.sentence_words.pop()
            self.sentence_label.setText(" ".join(self.sentence_words))
            self.current_word_letters = list(last_word)
            self.word_label.setText(last_word)
            self.log(f"Restored Word for Editing: \"{last_word}\"")
            self._trigger_ai_prediction()

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
        self.on_suggestions_ready([])
        self.log("Cleared word and sentence buffers.")

    def keyPressEvent(self, event):
        """Handle global keyboard shortcuts cleanly (100% Preserved + Keys 1/2/3)."""
        if event.key() == Qt.Key.Key_1 and self.current_suggestions and len(self.current_suggestions) >= 1:
            self._accept_autocomplete(self.current_suggestions[0])
        elif event.key() == Qt.Key.Key_2 and self.current_suggestions and len(self.current_suggestions) >= 2:
            self._accept_autocomplete(self.current_suggestions[1])
        elif event.key() == Qt.Key.Key_3 and self.current_suggestions and len(self.current_suggestions) >= 3:
            self._accept_autocomplete(self.current_suggestions[2])
        elif event.key() == Qt.Key.Key_Space:
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


# ═══════════════════════════════════════════════════════════
# Main Entry Point
# ═══════════════════════════════════════════════════════════
def main():
    app = QApplication(sys.argv)
    app.setStyle("Fusion")

    # Warm Soft 2D Plushy Palette (Global)
    palette = QPalette()
    palette.setColor(QPalette.ColorRole.Window, QColor(247, 244, 239))
    palette.setColor(QPalette.ColorRole.WindowText, QColor(45, 37, 33))
    palette.setColor(QPalette.ColorRole.Base, QColor(255, 255, 255))
    palette.setColor(QPalette.ColorRole.AlternateBase, QColor(245, 240, 234))
    palette.setColor(QPalette.ColorRole.Text, QColor(45, 37, 33))
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

