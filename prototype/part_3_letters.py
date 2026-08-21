# SignSpeak Universal Prototype — Real-Time ISL Letter & Fingerspelling Application
# Version: 2.6 (Two-Way Deaf <-> Hearing Communication Loop & Multilingual Voice Engine)
# Architecture: Real-Time Hand Landmark Classifier -> 0.8s Dwell Stabilizer -> Spacebar Word Builder -> Multilingual Piper/Regional Voice -> Live Whisper STT -> ISL Sign Display
# UI Theme: Warm Soft 2D Plushy Design (Two-Way Dialogue Timeline & Live Sign Visualizer)

import os
import sys
import io
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
    QLabel, QPushButton, QTextEdit, QFrame, QGroupBox, QProgressBar,
    QDialog, QCheckBox, QComboBox, QTabWidget, QScrollArea
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

        for lms, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
            label = handedness.classification[0].label
            coords = np.array([[lm.x, lm.y, lm.z] for lm in lms.landmark], dtype=np.float32)

            wrist = coords[0:1, :]
            coords_centered = coords - wrist
            span = np.linalg.norm(coords[9, :2] - coords[0, :2]) + 1e-6
            coords_norm = coords_centered / span

            if label == "Left":
                lh_feats = coords_norm
                has_lh = True
            elif label == "Right":
                rh_feats = coords_norm
                has_rh = True

        if has_lh and not has_rh:
            rh_feats = lh_feats.copy()
        elif has_rh and not has_lh:
            lh_feats = rh_feats.copy()

        feat_126 = np.concatenate([lh_feats.flatten(), rh_feats.flatten()], axis=0).astype(np.float32)
        return feat_126, results

    def close(self):
        if self.hands:
            self.hands.close()
            self.hands = None


# ═══════════════════════════════════════════════════════════════
# Thread 1: Video Capture + Landmark Processing
# ═══════════════════════════════════════════════════════════════
class CaptureThread(QThread):
    frame_ready = pyqtSignal(np.ndarray)
    feature_ready = pyqtSignal(np.ndarray)
    status_update = pyqtSignal(str)

    def __init__(self, camera_index=0):
        super().__init__()
        self.camera_index = camera_index
        self.running = False
        self.extractor = SingleFrameHandExtractor()
        self.smoother = LandmarkStreamSmoother(dim=126, min_cutoff=1.0, beta=0.007)

    def run(self):
        self.running = True
        cap = cv2.VideoCapture(self.camera_index, cv2.CAP_DSHOW)
        if not cap.isOpened():
            cap = cv2.VideoCapture(self.camera_index)

        if not cap.isOpened():
            self.status_update.emit(f"Error: Unable to open Camera {self.camera_index}.")
            return

        cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
        cap.set(cv2.CAP_PROP_FPS, 30)

        self.extractor.initialize()
        self.status_update.emit("Camera feed initialized at 30 FPS.")

        last_time = time.time()
        fps = 0.0

        while self.running:
            ret, frame = cap.read()
            if not ret:
                time.sleep(0.01)
                continue

            frame = cv2.flip(frame, 1)
            now = time.time()
            dt = now - last_time
            last_time = now
            if dt > 0:
                fps = 0.9 * fps + 0.1 * (1.0 / dt)

            feat_126, results = self.extractor.extract(frame)

            annotated = frame.copy()
            if results and results.multi_hand_landmarks:
                for hand_lms in results.multi_hand_landmarks:
                    self.extractor.mp_drawing.draw_landmarks(
                        annotated,
                        hand_lms,
                        self.extractor.mp_hands.HAND_CONNECTIONS,
                        self.extractor.mp_styles.get_default_hand_landmarks_style(),
                        self.extractor.mp_styles.get_default_hand_connections_style()
                    )

            cv2.putText(
                annotated, f"FPS: {fps:.1f}", (15, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (45, 112, 79), 2
            )

            if feat_126 is not None:
                smooth_feat = self.smoother.smooth(feat_126, now)
                self.feature_ready.emit(smooth_feat)
            else:
                self.smoother.reset()

            self.frame_ready.emit(annotated)
            time.sleep(0.005)

        cap.release()
        self.extractor.close()
        self.status_update.emit("Camera feed stopped.")

    def stop(self):
        self.running = False


# ═══════════════════════════════════════════════════════════════
# Thread 2: Sub-2ms ONNX Inference Engine
# ═══════════════════════════════════════════════════════════════
class InferenceThread(QThread):
    prediction_ready = pyqtSignal(str, float)
    no_hand_signal = pyqtSignal()
    status_update = pyqtSignal(str)

    def __init__(self, model_path, meta_path):
        super().__init__()
        self.model_path = str(model_path)
        self.meta_path = str(meta_path)
        self.feature_queue = queue.Queue(maxsize=2)
        self.running = False

        self.idx2class = {}
        if os.path.exists(self.meta_path):
            with open(self.meta_path, "r", encoding="utf-8") as f:
                meta = json.load(f)
                self.idx2class = {int(k): v for k, v in meta["idx2class"].items()}

        self.temporal_history = deque(maxlen=4)

    def enqueue_feature(self, feat_126):
        if not self.feature_queue.full():
            self.feature_queue.put(feat_126)

    def run(self):
        self.running = True

        if not os.path.exists(self.model_path):
            self.status_update.emit(f"Error: ONNX model not found at {self.model_path}")
            return

        opts = ort.SessionOptions()
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        opts.intra_op_num_threads = 2
        session = ort.InferenceSession(self.model_path, opts, providers=["CPUExecutionProvider"])
        input_name = session.get_inputs()[0].name
        output_name = session.get_outputs()[0].name

        self.status_update.emit("ONNX Inference Engine ready (Latency: <1.8ms).")

        while self.running:
            try:
                feat = self.feature_queue.get(timeout=0.1)
            except queue.Empty:
                self.temporal_history.clear()
                self.no_hand_signal.emit()
                continue

            x = feat.reshape(1, 126).astype(np.float32)
            logits = session.run([output_name], {input_name: x})[0]

            exp_logits = np.exp(logits - np.max(logits, axis=1, keepdims=True))
            probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)

            pred_idx = int(np.argmax(probs[0]))
            confidence = float(probs[0, pred_idx])
            pred_letter = self.idx2class.get(pred_idx, "?")

            self.temporal_history.append((pred_letter, confidence))

            if len(self.temporal_history) == self.temporal_history.maxlen:
                counts = {}
                conf_acc = {}
                for ltr, conf in self.temporal_history:
                    counts[ltr] = counts.get(ltr, 0) + 1
                    conf_acc[ltr] = conf_acc.get(ltr, 0.0) + conf

                best_ltr = max(counts, key=counts.get)
                if counts[best_ltr] >= 3:
                    avg_conf = conf_acc[best_ltr] / counts[best_ltr]
                    self.prediction_ready.emit(best_ltr, avg_conf)
                else:
                    self.prediction_ready.emit(pred_letter, confidence)
            else:
                self.prediction_ready.emit(pred_letter, confidence)

    def stop(self):
        self.running = False


# ═══════════════════════════════════════════════════════════════
# Thread 3: Piper & Multilingual Neural Voice TTS Engine
# ═══════════════════════════════════════════════════════════════
class TTSThread(QThread):
    speech_done = pyqtSignal(str, str)  # (text, lang_code)

    def __init__(self, piper_model_path):
        super().__init__()
        self.piper_model_path = str(piper_model_path)
        self.text_queue = queue.Queue()
        self.running = False

    def enqueue_text(self, text, lang_code="en"):
        if not self.text_queue.full():
            self.text_queue.put((text, lang_code))

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
                item = self.text_queue.get(timeout=0.5)
            except queue.Empty:
                continue

            text, lang_code = item
            text = text.strip()
            if not text:
                continue

            try:
                if lang_code == "en":
                    # English: Use Local Offline Piper Neural Voice
                    if voice:
                        wav_path = os.path.join(tempfile.gettempdir(), "signspeak_letter_tts.wav")
                        with wave.open(wav_path, "w") as wf:
                            voice.synthesize_wav(text, wf)
                        import winsound
                        winsound.PlaySound(wav_path, winsound.SND_FILENAME)
                    else:
                        import pyttsx3
                        eng = pyttsx3.init()
                        eng.say(text)
                        eng.runAndWait()
                else:
                    # Regional Indian Language Speech (Hindi, Telugu, Tamil, Marathi, etc.)
                    self._synthesize_regional(text, lang_code)

                self.speech_done.emit(text, lang_code)
            except Exception as e:
                print(f"TTS Synthesis Error: {e}")

    def _synthesize_regional(self, text, lang_code):
        """Synthesizes and plays regional Indian neural audio cleanly."""
        try:
            import urllib.request
            import urllib.parse
            import ctypes
            encoded = urllib.parse.quote(text)
            url = f"https://translate.google.com/translate_tts?ie=UTF-8&tl={lang_code}&client=tw-ob&q={encoded}"
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"})
            with urllib.request.urlopen(req, timeout=3.5) as resp:
                audio_bytes = resp.read()

            mp3_path = os.path.join(tempfile.gettempdir(), f"signspeak_voice_{lang_code}.mp3")
            with open(mp3_path, "wb") as f:
                f.write(audio_bytes)

            winmm = ctypes.windll.winmm
            alias = f"sign_audio_{int(time.time() * 1000)}"
            winmm.mciSendStringW(f'open "{mp3_path}" type mpegvideo alias {alias}', None, 0, None)
            winmm.mciSendStringW(f'play {alias} wait', None, 0, None)
            winmm.mciSendStringW(f'close {alias}', None, 0, None)
        except Exception as e:
            print(f"Regional Voice Fallback Error: {e}")
            try:
                import pyttsx3
                eng = pyttsx3.init()
                eng.say(text)
                eng.runAndWait()
            except Exception:
                pass

    def stop(self):
        self.running = False


# ═══════════════════════════════════════════════════════════════
# Thread 4: Non-Blocking AI Autocomplete & Grammar Polish Worker (Groq Cloud / Offline Fallback)
# ═══════════════════════════════════════════════════════════════
class AIPredictionThread(QThread):
    suggestions_ready = pyqtSignal(list)
    polish_ready = pyqtSignal(str)
    translation_ready = pyqtSignal(str, str, str)  # (translated_text, target_lang_code, original_text)
    status_update = pyqtSignal(str)

    def __init__(self, config_path=None):
        super().__init__()
        self.config_path = config_path or (BASE_DIR / "ai_config.json")
        self.request_queue = queue.Queue(maxsize=12)
        self.running = True
        self.api_key = None
        self._load_config()

        # Offline common frequency dictionary for instant <0.1ms fallback
        self.offline_dict = [
            "ABOUT", "ABOVE", "ACTION", "AFTER", "AGAIN", "ALL", "ALWAYS", "AND", "ANIMAL", "ANSWER",
            "ANY", "APPLE", "APPOINTMENT", "ASK", "BABY", "BACK", "BAD", "BAG", "BALL", "BALLOON",
            "BANANA", "BEAUTIFUL", "BECAUSE", "BED", "BEFORE", "BEGIN", "BEHIND", "BELIEVE", "BEST", "BETTER",
            "BICYCLE", "BIG", "BIRD", "BIRTHDAY", "BLACK", "BLANKET", "BLUE", "BOARD", "BOOK", "BOTTLE",
            "BOX", "BOY", "BREAD", "BREAK", "BRIDGE", "BRIGHT", "BROTHER", "BROWN", "BUILD", "BUS",
            "CALL", "CAN", "CAR", "CAT", "CHAIR", "CHANGE", "CHILD", "CHILDREN", "CITY", "CLEAN",
            "CLOSE", "CLOTHES", "COLD", "COLOR", "COME", "COMPUTER", "COOK", "COUNTRY", "CRY", "CUT",
            "DANCE", "DARK", "DAUGHTER", "DAY", "DEAD", "DEAR", "DEAF", "DECIDE", "DELICIOUS", "DOCTOR",
            "DO", "DOG", "DOOR", "DRAW", "DREAM", "DRESS", "DRINK", "DRIVE", "EAR", "EARLY",
            "EAT", "EDUCATION", "EGG", "EIGHT", "ELEPHANT", "EMERGENCY", "EMPTY", "END", "ENGINE", "ENJOY",
            "ENOUGH", "ENTER", "EVENING", "EVERY", "EVERYONE", "EVERYTHING", "EYE", "FACE", "FALL", "FAMILY",
            "FAR", "FAST", "FATHER", "FEEL", "FEW", "FIND", "FINE", "FINGER", "FINISH", "FIRE",
            "FIRST", "FISH", "FIVE", "FLOWER", "FLY", "FOOD", "FOOT", "FOR", "FOREST", "FORGET",
            "FOUR", "FREE", "FRESH", "FRIEND", "FROM", "FRONT", "FRUIT", "FULL", "FUNNY", "GAME",
            "GARDEN", "GIRL", "GIVE", "GIVEN", "GIVING", "GLASS", "GO", "GOOD", "GOODBYE", "GRASS",
            "GREAT", "GREEN", "GROUND", "GROUP", "GROW", "HAND", "HAPPY", "HARD", "HAT", "HAVE",
            "HE", "HEAD", "HEAR", "HEART", "HEAVY", "HELLO", "HELP", "HERE", "HIGH", "HILL",
            "HISTORY", "HOLD", "HOME", "HOPE", "HORSE", "HOSPITAL", "HOT", "HOTEL", "HOUR", "HOUSE",
            "HOW", "HUNDRED", "HUNGRY", "HURT", "ICE", "IDEA", "IMPORTANT", "IN", "INSIDE", "INTERNET",
            "IS", "ISLAND", "JOB", "JOIN", "JUMP", "JUST", "KEEP", "KEY", "KICK", "KID",
            "KIND", "KING", "KITCHEN", "KNOW", "LAND", "LANGUAGE", "LARGE", "LAST", "LATE", "LAUGH",
            "LEARN", "LEAVE", "LEFT", "LEG", "LESSON", "LETTER", "LIGHT", "LIKE", "LINE", "LIP",
            "LISTEN", "LITTLE", "LIVE", "LONG", "LOOK", "LOVE", "LUNCH", "MAKE", "MAN", "MANY",
            "MARKET", "MATCH", "ME", "MEAL", "MEDICINE", "MEET", "MEMBER", "MESSAGE", "MIDDLE", "MILK",
            "MIND", "MINUTE", "MISS", "MOMENT", "MONEY", "MONTH", "MOON", "MORE", "MORNING", "MOTHER",
            "MOUTH", "MOVE", "MOVIE", "MUSIC", "MUST", "NAME", "NEAR", "NEED", "NEIGHBOR", "NEVER",
            "NEW", "NEWS", "NEXT", "NICE", "NIGHT", "NINE", "NO", "NOISE", "NOON", "NORTH",
            "NOSE", "NOT", "NOTE", "NOW", "NUMBER", "NURSE", "OFFICE", "OFTEN", "OLD", "ONCE",
            "ONE", "ONLY", "OPEN", "ORANGE", "ORDER", "OTHER", "OUR", "OUT", "OUTSIDE", "PAGE",
            "PAIN", "PAINT", "PAPER", "PARENT", "PARK", "PART", "PASS", "PAST", "PATH", "PAY",
            "PEACE", "PEN", "PEOPLE", "PERSON", "PHONE", "PHOTO", "PICTURE", "PIECE", "PLACE", "PLAN",
            "PLANE", "PLANT", "PLAY", "PLEASE", "POINT", "POLICE", "POOR", "POSSIBLE", "POWER", "PRACTICE",
            "PRESENT", "PRETTY", "PRICE", "PROBLEM", "PROMISE", "PUPIL", "PUSH", "PUT", "QUEEN", "QUESTION",
            "QUICK", "QUIET", "RADIO", "RAIN", "READ", "READY", "REAL", "REASON", "RED", "REMEMBER",
            "REST", "RICE", "RICH", "RIDE", "RIGHT", "RING", "RIVER", "ROAD", "ROOM", "ROUND",
            "RUN", "SAD", "SAFE", "SAME", "SAVE", "SAY", "SCHOOL", "SCIENCE", "SEA", "SEASON",
            "SEAT", "SECOND", "SECRET", "SEE", "SEED", "SEEM", "SEND", "SENTENCE", "SEVEN", "SEVERAL",
            "SHARE", "SHOE", "SHOP", "SHORT", "SHOULD", "SHOW", "SHUT", "SICK", "SIDE", "SIGN",
            "SILENT", "SIMPLE", "SING", "SISTER", "SIT", "SIX", "SKIN", "SKY", "SLEEP", "SLOW",
            "SMALL", "SMILE", "SMOKE", "SNOW", "SO", "SOFT", "SOME", "SOMEONE", "SOMETHING", "SOMETIMES",
            "SON", "SONG", "SOON", "SORRY", "SOUND", "SOUTH", "SPACE", "SPEAK", "SPECIAL", "SPEED",
            "SPELL", "SPEND", "SPORT", "SPRING", "STAND", "STAR", "START", "STATION", "STAY", "STEP",
            "STICK", "STILL", "STOP", "STORE", "STORY", "STREET", "STRONG", "STUDENT", "STUDY", "SUCH",
            "SUDDEN", "SUGAR", "SUMMER", "SUN", "SUNDAY", "SURE", "SWEET", "SWIM", "TABLE", "TAKE",
            "TALK", "TALL", "TASTE", "TEA", "TEACH", "TEACHER", "TEAM", "TELL", "TEN", "TEST",
            "THANK", "THANKS", "THAT", "THE", "THEIR", "THEM", "THEN", "THERE", "THESE", "THEY",
            "THICK", "THIN", "THING", "THINK", "THIRD", "THIS", "THOSE", "THOUGH", "THOUGHT", "THREE",
            "THROUGH", "THROW", "TIME", "TIRED", "TO", "TODAY", "TOGETHER", "TOMORROW", "TONIGHT", "TOO",
            "TOOTH", "TOP", "TOUCH", "TOWN", "TRAIN", "TRAVEL", "TREE", "TRIP", "TRUE", "TRUST",
            "TRY", "TURN", "TWO", "UNCLE", "UNDER", "UNDERSTAND", "UNTIL", "UP", "UPON", "US",
            "USE", "USEFUL", "VACATION", "VERY", "VILLAGE", "VISIT", "VOICE", "WAIT", "WAKE", "WALK",
            "WALL", "WANT", "WAR", "WARM", "WASH", "WATCH", "WATER", "WAVE", "WAY", "WE",
            "WEAK", "WEAR", "WEATHER", "WEEK", "WELCOME", "WELL", "WEST", "WET", "WHAT", "WHEAT",
            "WHEEL", "WHEN", "WHERE", "WHICH", "WHILE", "WHITE", "WHO", "WHOLE", "WHOM", "WHOSE",
            "WHY", "WIDE", "WIFE", "WILD", "WILL", "WIN", "WIND", "WINDOW", "WINTER", "WISH",
            "WITH", "WITHOUT", "WOMAN", "WOMEN", "WONDER", "WOOD", "WORD", "WORK", "WORKER", "WORLD",
            "WORRY", "WRITE", "WRONG", "YARD", "YEAR", "YELLOW", "YES", "YESTERDAY", "YET", "YOU",
            "YOUNG", "YOUR", "ZERO", "ZOO"
        ]

        # Context bigram transition suggestions for next words
        self.next_word_map = {
            "BALL": ["GIVE", "PLAY", "THROW", "CATCH"],
            "GIVE": ["ME", "TO", "PLEASE", "HIM", "HER"],
            "WANT": ["WATER", "FOOD", "HELP", "TO GO", "TO EAT"],
            "I": ["WANT", "NEED", "AM", "LIKE", "FEEL"],
            "ME": ["PLEASE", "HELP", "WATER", "NOW"],
            "YOU": ["ARE", "WANT", "NEED", "CAN", "HELP"],
            "PLEASE": ["HELP", "GIVE", "COME", "WAIT", "SIT"],
            "THANK": ["YOU", "VERY MUCH", "AGAIN", "ALL"],
            "HELLO": ["HOW ARE YOU", "MY FRIEND", "GOOD MORNING", "EVERYONE"],
            "GOOD": ["MORNING", "EVENING", "NIGHT", "JOB", "DAY"],
            "HELP": ["ME", "PLEASE", "NOW", "HIM"],
            "WHERE": ["IS", "ARE", "HOSPITAL", "DOCTOR", "BATHROOM"],
            "HOW": ["ARE YOU", "MUCH", "MANY", "IS IT"],
            "DOCTOR": ["APPOINTMENT", "HELP", "HOSPITAL", "MEDICINE"],
            "WATER": ["DRINK", "PLEASE", "BOTTLE", "COLD"],
            "FOOD": ["EAT", "DELICIOUS", "PLEASE", "HUNGRY"]
        }

        # Offline dictionary fallback for Indian languages
        self.offline_translations = {
            "hi": {
                "HELLO": "नमस्ते", "THANK YOU": "धन्यवाद", "PLEASE": "कृपया", "HELP": "मदद करें",
                "WATER": "पानी", "FOOD": "खाना", "DOCTOR": "डॉक्टर", "YES": "हाँ", "NO": "नहीं",
                "GOOD": "अच्छा", "I WANT WATER": "मुझे पानी चाहिए", "PLEASE HELP ME": "कृपया मेरी मदद करें"
            },
            "te": {
                "HELLO": "నమస్కారం", "THANK YOU": "ధన్యవాదాలు", "PLEASE": "దయచేసి", "HELP": "సహాయం చేయండి",
                "WATER": "నీరు", "FOOD": "ఆహారం", "DOCTOR": "వైద్యుడు", "YES": "అవును", "NO": "కాదు",
                "GOOD": "మంచిది", "I WANT WATER": "నాకు నీరు కావాలి", "PLEASE HELP ME": "దయచేసి నాకు సహాయం చేయండి"
            },
            "ta": {
                "HELLO": "வணக்கம்", "THANK YOU": "நன்றி", "PLEASE": "தயவுசெய்து", "HELP": "உதவி செய்யுங்கள்",
                "WATER": "தண்ணீர்", "FOOD": "உணவு", "DOCTOR": "மருத்துவர்", "YES": "ஆம்", "NO": "இல்லை",
                "GOOD": "நல்லது", "I WANT WATER": "எனக்கு தண்ணீர் வேண்டும்", "PLEASE HELP ME": "தயவுசெய்து எனக்கு உதவுங்கள்"
            },
            "mr": {
                "HELLO": "नमस्कार", "THANK YOU": "धन्यवाद", "PLEASE": "कृपया", "HELP": "मदत करा",
                "WATER": "पाणी", "FOOD": "अन्न", "DOCTOR": "डॉक्टर", "YES": "होय", "NO": "नाही",
                "GOOD": "छान", "I WANT WATER": "मला पाणी हवे आहे", "PLEASE HELP ME": "कृपया मला मदत करा"
            },
            "kn": {
                "HELLO": "ನಮಸ್ಕಾರ", "THANK YOU": "ಧನ್ಯವಾದಗಳು", "PLEASE": "ದಯವಿಟ್ಟು", "HELP": "ಸಹಾಯ ಮಾಡಿ",
                "WATER": "ನೀರು", "FOOD": "ಆಹಾರ", "DOCTOR": "ವೈದ್ಯರು", "YES": "ಹೌದು", "NO": "ಇಲ್ಲ",
                "GOOD": "ಉತ್ತಮ", "I WANT WATER": "ನನಗೆ ನೀರು ಬೇಕು", "PLEASE HELP ME": "ದಯವಿಟ್ಟು ನನಗೆ ಸಹಾಯ ಮಾಡಿ"
            },
            "bn": {
                "HELLO": "নমস্কার", "THANK YOU": "ধন্যবাদ", "PLEASE": "দয়া করে", "HELP": "সাহায্য করুন",
                "WATER": "জল", "FOOD": "খাবার", "DOCTOR": "ডাক্তার", "YES": "হ্যাঁ", "NO": "না",
                "GOOD": "ভালো", "I WANT WATER": "আমার জল চাই", "PLEASE HELP ME": "দয়া করে আমাকে সাহায্য করুন"
            },
            "gu": {
                "HELLO": "નમસ્તે", "THANK YOU": "આભાર", "PLEASE": "કૃપા કરીને", "HELP": "મદદ કરો",
                "WATER": "પાણી", "FOOD": "ખોરાક", "DOCTOR": "ડૉક્ટર", "YES": "હા", "NO": "ના",
                "GOOD": "સારું", "I WANT WATER": "મને પાણી જોઈએ છે", "PLEASE HELP ME": "કૃપા કરીને મને મદદ કરો"
            }
        }

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
                item = self.request_queue.get_nowait()
                if item[0] in ("polish", "translate"):
                    self.request_queue.put(item)
                    break
            except queue.Empty:
                break
        self.request_queue.put(("autocomplete", prefix.upper().strip(), sentence_context.strip()))

    def enqueue_polish(self, raw_sentence):
        self.request_queue.put(("polish", raw_sentence.strip(), ""))

    def enqueue_translate(self, raw_sentence, target_lang_code, target_lang_name):
        self.request_queue.put(("translate", raw_sentence.strip(), (target_lang_code, target_lang_name)))

    def run(self):
        self.running = True
        while self.running:
            try:
                task_type, arg1, arg2 = self.request_queue.get(timeout=0.5)
            except queue.Empty:
                continue

            if task_type == "autocomplete":
                prefix, context = arg1, arg2
                suggestions = self._query_groq(prefix, context)

                if not suggestions:
                    if prefix:
                        suggestions = [w for w in self.offline_dict if w.startswith(prefix) and w != prefix][:3]
                    elif context:
                        last_word = context.split()[-1].upper() if context.split() else ""
                        suggestions = self.next_word_map.get(last_word, ["PLEASE", "THANK YOU", "HELP"])[:3]

                if suggestions:
                    self.suggestions_ready.emit(suggestions)

            elif task_type == "polish":
                raw_text = arg1
                polished = self._polish_groq(raw_text)
                if not polished:
                    polished = raw_text.strip().capitalize()
                    if not polished.endswith((".", "!", "?")):
                        polished += "."
                self.polish_ready.emit(polished)

            elif task_type == "translate":
                raw_text = arg1
                lang_code, lang_name = arg2
                translated = self._translate_groq(raw_text, lang_name)
                if not translated:
                    # Offline fallback dictionary lookup
                    clean_key = raw_text.upper().strip().rstrip(".!?")
                    lang_dict = self.offline_translations.get(lang_code, {})
                    translated = lang_dict.get(clean_key, raw_text)
                self.translation_ready.emit(translated, lang_code, raw_text)

    def _query_groq(self, prefix, context):
        if not self.api_key:
            return None

        url = "https://api.groq.com/openai/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"
        }

        if prefix:
            prompt = f"Suggest up to 3 uppercase English words starting with '{prefix}'."
            if context:
                prompt += f" Previous sentence words: '{context}'."
        else:
            prompt = f"Given sign sentence '{context}', suggest the 3 most likely NEXT words to follow."

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
                        clean = re.sub(r"[^A-Z ]", "", str(w).upper()).strip()
                        if clean and clean not in valid:
                            if prefix:
                                if clean.startswith(prefix):
                                    valid.append(clean)
                            else:
                                valid.append(clean)
                    if valid:
                        return valid[:3]
        except Exception:
            pass
        return None

    def _polish_groq(self, raw_text):
        if not self.api_key or not raw_text:
            return None

        url = "https://api.groq.com/openai/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"
        }
        prompt = (
            f"You are an expert Indian Sign Language translator. "
            f"Translate this sign gloss: '{raw_text}' into a single natural, fluent English sentence. "
            f"Output ONLY the sentence. No explanations, no quotes, no markdown."
        )

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
            with urllib.request.urlopen(req, timeout=2.5) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                content = data["choices"][0]["message"]["content"].strip()
                content = re.sub(r'^["\']|["\']$', '', content).strip()
                if content:
                    return content
        except Exception:
            pass
        return None

    def _translate_groq(self, raw_text, target_lang_name):
        """Translates English sentence to regional Indian language with high fluency."""
        if not self.api_key or not raw_text:
            return None

        url = "https://api.groq.com/openai/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"
        }
        prompt = (
            f"Translate this sentence to natural, everyday {target_lang_name}: '{raw_text}'. "
            f"Output ONLY the translated sentence in {target_lang_name} native script. "
            f"No transliteration, no English translation, no explanations, no quotes."
        )

        payload = {
            "model": "groq/compound-mini",
            "messages": [
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.1,
            "max_tokens": 60
        }

        try:
            import urllib.request
            import re
            req = urllib.request.Request(url, data=json.dumps(payload).encode("utf-8"), headers=headers)
            with urllib.request.urlopen(req, timeout=2.5) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                content = data["choices"][0]["message"]["content"].strip()
                content = re.sub(r'^["\']|["\']$', '', content).strip()
                if content:
                    return content
        except Exception:
            pass
        return None

    def stop(self):
        self.running = False


# ═══════════════════════════════════════════════════════════════
# Thread 5: Live Audio Listener & Speech-to-Text Worker (Groq Whisper / sounddevice)
# ═══════════════════════════════════════════════════════════════
class SpeechToTextThread(QThread):
    transcript_ready = pyqtSignal(str)
    audio_level = pyqtSignal(int)          # 0-100 for mic volume bar
    status_changed = pyqtSignal(str)       # "Listening...", "Transcribing...", "Idle"

    def __init__(self, config_path=None):
        super().__init__()
        self.config_path = config_path or (BASE_DIR / "ai_config.json")
        self.running = False
        self.is_listening = False
        self.api_key = None
        self._load_config()
        self.audio_frames = []
        self._lock = threading.Lock()
        self.sample_rate = 16000
        self.block_size = 1024

    def _load_config(self):
        if os.path.exists(self.config_path):
            try:
                with open(self.config_path, "r", encoding="utf-8") as f:
                    cfg = json.load(f)
                    self.api_key = cfg.get("api_key", "").strip()
            except Exception as e:
                print(f"STT AI Config Load Notice: {e}")

    def toggle_listening(self):
        """Toggle recording state on/off."""
        if self.is_listening:
            self.stop_listening()
        else:
            self.start_listening()

    def start_listening(self):
        with self._lock:
            self.audio_frames = []
            self.is_listening = True
        self.status_changed.emit("Listening to hearing partner voice...")

    def stop_listening(self):
        with self._lock:
            if not self.is_listening:
                return
            self.is_listening = False
            frames_to_process = list(self.audio_frames)
            self.audio_frames = []

        self.audio_level.emit(0)
        if len(frames_to_process) >= 5:  # At least ~0.3s of audio
            self.status_changed.emit("Transcribing speech via Whisper AI...")
            threading.Thread(target=self._transcribe_audio, args=(frames_to_process,), daemon=True).start()
        else:
            self.status_changed.emit("Listening stopped (too short).")

    def run(self):
        self.running = True
        import sounddevice as sd

        def audio_callback(indata, frame_count, time_info, status):
            if not self.running:
                return
            with self._lock:
                if self.is_listening:
                    self.audio_frames.append(indata.copy())
            # Calculate RMS energy for audio visualizer
            try:
                rms = np.sqrt(np.mean(indata.astype(np.float32)**2))
                level = int(min(100, max(0, rms / 150.0)))
                self.audio_level.emit(level)
            except Exception:
                pass

        try:
            with sd.InputStream(samplerate=self.sample_rate, channels=1, dtype='int16',
                                blocksize=self.block_size, callback=audio_callback):
                while self.running:
                    time.sleep(0.05)
        except Exception as e:
            print(f"SoundDevice Audio InputStream Notice: {e}")

    def _transcribe_audio(self, frames):
        """Encodes audio into in-memory WAV and queries Groq Whisper endpoint."""
        try:
            audio_data = np.concatenate(frames, axis=0)
            wav_io = io.BytesIO()
            with wave.open(wav_io, 'wb') as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(self.sample_rate)
                wf.writeframes(audio_data.tobytes())

            audio_bytes = wav_io.getvalue()
            if not audio_bytes or len(audio_bytes) < 4000:
                self.status_changed.emit("No clear speech detected.")
                return

            if self.api_key:
                boundary = "----WebKitFormBoundarySignSpeakAudioSTT"
                body = bytearray()
                body.extend(f"--{boundary}\r\n".encode("utf-8"))
                body.extend(b'Content-Disposition: form-data; name="file"; filename="audio.wav"\r\n')
                body.extend(b'Content-Type: audio/wav\r\n\r\n')
                body.extend(audio_bytes)
                body.extend(b"\r\n")

                body.extend(f"--{boundary}\r\n".encode("utf-8"))
                body.extend(b'Content-Disposition: form-data; name="model"\r\n\r\n')
                body.extend(b"whisper-large-v3-turbo\r\n")

                body.extend(f"--{boundary}\r\n".encode("utf-8"))
                body.extend(b'Content-Disposition: form-data; name="response_format"\r\n\r\n')
                body.extend(b"json\r\n")

                body.extend(f"--{boundary}--\r\n".encode("utf-8"))

                req = urllib.request.Request(
                    "https://api.groq.com/openai/v1/audio/transcriptions",
                    data=bytes(body),
                    headers={
                        "Authorization": f"Bearer {self.api_key}",
                        "Content-Type": f"multipart/form-data; boundary={boundary}",
                        "User-Agent": "SignSpeak-Universal/2.6"
                    }
                )
                with urllib.request.urlopen(req, timeout=8.0) as resp:
                    res = json.loads(resp.read().decode("utf-8"))
                    raw_text = res.get("text", "").strip()
                    if raw_text:
                        self.transcript_ready.emit(raw_text)
                        self.status_changed.emit(f"Transcribed: \"{raw_text}\"")
                        return

            self.status_changed.emit("Transcription complete (empty text).")
        except Exception as e:
            print(f"STT Transcription Error: {e}")
            self.status_changed.emit("Speech recognition error.")

    def stop(self):
        self.running = False
        self.is_listening = False


# ═══════════════════════════════════════════════════════════════
# Shortcuts & Controls Quick Help Popup Dialog
# ═══════════════════════════════════════════════════════════════
class ShortcutsHelpDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("SignSpeak Studio — Controls & Shortcuts Reference")
        self.setFixedSize(600, 560)
        self.setStyleSheet("""
            QDialog {
                background-color: #F7F4EF;
            }
            QLabel {
                color: #2D2521;
                font-family: 'Segoe UI', -apple-system, sans-serif;
            }
            QPushButton {
                background-color: #D96B43;
                color: #FFFFFF;
                border: none;
                padding: 9px 24px;
                font-size: 13px;
                font-weight: 700;
                border-radius: 9px;
                font-family: 'Segoe UI', sans-serif;
            }
            QPushButton:hover {
                background-color: #C55A32;
            }
        """)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(24, 20, 24, 20)
        layout.setSpacing(12)

        title = QLabel("Keyboard Shortcuts & System Controls")
        title.setFont(QFont("Segoe UI", 16, QFont.Weight.Bold))
        title.setStyleSheet("color: #2D2521; font-weight: 800;")
        layout.addWidget(title)

        subtitle = QLabel("Complete reference of gestures, key bindings, and two-way interaction controls.")
        subtitle.setFont(QFont("Segoe UI", 10))
        subtitle.setStyleSheet("color: #75655B;")
        layout.addWidget(subtitle)

        # Card container with rows
        card = QFrame()
        card.setStyleSheet("background-color: #FFFFFF; border: 1.5px solid #E8DFD5; border-radius: 12px; padding: 12px;")
        card_layout = QVBoxLayout(card)
        card_layout.setSpacing(7)

        shortcuts_list = [
            ("Hold Sign (0.8s)", "Auto-captures recognized letter into active word builder"),
            ("Keys [ 1 / 2 / 3 ]", "Selects autocomplete suggestion or next-word prediction"),
            ("Spacebar", "Commits active word into full spoken sentence line"),
            ("Backspace", "Deletes last letter (or restores previous word for editing)"),
            ("Ctrl + P", "Applies AI sentence grammar polish (raw gloss to fluent English)"),
            ("Ctrl + Z", "Reverts grammar polish to original signed words"),
            ("Voice Dropdown", "Selects Indian regional speech language (Hindi, Telugu, Tamil, etc.)"),
            ("Enter", "Vocalizes full sentence in selected regional voice"),
            ("Ctrl + M / F2", "Toggles microphone listening to hearing partner's spoken voice"),
            ("Export Transcript", "Saves full two-way conversation session to text file"),
            ("Escape", "Clears both word and sentence buffers instantly"),
            ("F1", "Opens this shortcuts and controls guide")
        ]

        for key_text, desc_text in shortcuts_list:
            row = QHBoxLayout()
            row.setSpacing(10)
            k_lbl = QLabel(key_text)
            k_lbl.setFont(QFont("Segoe UI", 10, QFont.Weight.Bold))
            k_lbl.setStyleSheet(
                "background-color: #EFE7DE; color: #4A3E37; border: 1px solid #D8C9B8; "
                "border-radius: 6px; padding: 3px 8px; min-width: 130px;"
            )
            d_lbl = QLabel(desc_text)
            d_lbl.setFont(QFont("Segoe UI", 10))
            d_lbl.setStyleSheet("color: #5C4D44;")
            d_lbl.setWordWrap(True)
            row.addWidget(k_lbl)
            row.addWidget(d_lbl, stretch=1)
            card_layout.addLayout(row)

        layout.addWidget(card)

        btn_box = QHBoxLayout()
        btn_box.addStretch()
        close_btn = QPushButton("Close")
        close_btn.clicked.connect(self.accept)
        btn_box.addWidget(close_btn)
        layout.addLayout(btn_box)


# ═══════════════════════════════════════════════════════════════
# Main PyQt6 Desktop Application Window (Two-Way Communication Engine)
# ═══════════════════════════════════════════════════════════════
class SignSpeakApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("SignSpeak Studio — Indian Sign Language Two-Way Communication Cockpit")
        self.setMinimumSize(1260, 840)
        self.setStyleSheet(self._get_stylesheet())

        # State Variables (100% Preserved + Two-Way Dialogue History)
        self.is_running = False
        self.current_word_letters = []
        self.sentence_words = []
        self.raw_sentence_words = []
        self.is_polished = False
        self._speak_after_polish = False
        self.current_suggestions = ["HELLO", "PLEASE", "THANK YOU"]

        self.live_letter = None
        self.live_confidence = 0.0

        # Two-Way Conversation History Log
        self.conversation_history = []
        self.is_listening_mic = False

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
            QPushButton#micBtn {
                background-color: #2A6F97;
                color: #FFFFFF;
            }
            QPushButton#micBtn:hover {
                background-color: #014F86;
            }
            QPushButton#micBtn:pressed {
                background-color: #013A63;
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
                font-family: 'Segoe UI', -apple-system, sans-serif;
                font-size: 12px;
                padding: 8px;
                line-height: 1.4;
            }
            QTabWidget::pane {
                border: 1.5px solid #E8DFD5;
                border-radius: 12px;
                background-color: #FFFFFF;
                top: -1px;
                padding: 4px;
            }
            QTabBar::tab {
                background-color: #EFE7DE;
                color: #5C4D44;
                font-weight: 700;
                font-size: 11px;
                font-family: 'Segoe UI', sans-serif;
                border: 1.5px solid #D8C9B8;
                border-bottom: none;
                border-top-left-radius: 8px;
                border-top-right-radius: 8px;
                padding: 6px 14px;
                margin-right: 4px;
            }
            QTabBar::tab:selected {
                background-color: #FFFFFF;
                color: #D96B43;
                border-color: #E8DFD5;
            }
            QTabBar::tab:hover:!selected {
                background-color: #E5DACF;
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

        app_subtitle = QLabel("Indian Sign Language Two-Way Communication Cockpit")
        app_subtitle.setFont(QFont("Segoe UI", 11))
        app_subtitle.setStyleSheet("color: #75655B;")
        header_title_layout.addWidget(app_subtitle)

        header_layout.addLayout(header_title_layout)
        header_layout.addStretch()

        # Top Status Badges & Quick Help Button
        badges_layout = QHBoxLayout()
        badges_layout.setSpacing(10)

        self.help_btn = QPushButton("Keyboard Shortcuts (F1)")
        self.help_btn.setObjectName("secondaryBtn")
        self.help_btn.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.help_btn.clicked.connect(self.show_shortcuts_guide)
        badges_layout.addWidget(self.help_btn)

        self.two_way_badge = QLabel("Two-Way Loop: Active")
        self.two_way_badge.setFont(QFont("Segoe UI", 11, QFont.Weight.Bold))
        self.two_way_badge.setStyleSheet(
            "background-color: #E2EBF0; color: #1E4D6B; border: 1px solid #C4D7E2; "
            "border-radius: 8px; padding: 6px 12px;"
        )
        badges_layout.addWidget(self.two_way_badge)

        ai_badge = QLabel("AI Autocomplete: Active")
        ai_badge.setFont(QFont("Segoe UI", 11, QFont.Weight.Bold))
        ai_badge.setStyleSheet(
            "background-color: #EFE7DE; color: #5C4D44; border: 1px solid #D8C9B8; "
            "border-radius: 8px; padding: 6px 12px;"
        )
        badges_layout.addWidget(ai_badge)

        self.engine_status_badge = QLabel("Camera: 30 FPS Active")
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
        # LEFT COLUMN: Camera-First Viewport & Hearing Partner Loop
        # ───────────────────────────────────────────────────────────
        left_col = QVBoxLayout()
        left_col.setSpacing(12)

        video_group = QGroupBox("Camera Feed & Skeleton Tracking")
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

        # ── Hearing Partner Voice & ISL Visual Translation Card ──
        partner_group = QGroupBox("Hearing Partner Voice (Incoming Speech-to-Sign Loop)")
        partner_layout = QVBoxLayout(partner_group)
        partner_layout.setContentsMargins(12, 12, 12, 12)
        partner_layout.setSpacing(8)

        # Mic Control & Volume Meter Row
        mic_row = QHBoxLayout()
        mic_row.setSpacing(10)

        self.listen_btn = QPushButton("Listen Hearing Voice [ Ctrl+M ]")
        self.listen_btn.setObjectName("micBtn")
        self.listen_btn.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.listen_btn.clicked.connect(self.toggle_mic_listening)
        mic_row.addWidget(self.listen_btn, stretch=3)

        self.mic_level_bar = QProgressBar()
        self.mic_level_bar.setMaximum(100)
        self.mic_level_bar.setValue(0)
        self.mic_level_bar.setTextVisible(False)
        self.mic_level_bar.setFixedHeight(18)
        self.mic_level_bar.setStyleSheet("""
            QProgressBar {
                background-color: #EFE7DE;
                border: 1px solid #DFD2C4;
                border-radius: 5px;
            }
            QProgressBar::chunk {
                background-color: #2A6F97;
                border-radius: 4px;
            }
        """)
        mic_row.addWidget(self.mic_level_bar, stretch=2)

        partner_layout.addLayout(mic_row)

        self.mic_status_lbl = QLabel("Microphone: Idle (Press Ctrl+M or click to listen)")
        self.mic_status_lbl.setFont(QFont("Segoe UI", 10))
        self.mic_status_lbl.setStyleSheet("color: #75655B; font-weight: 600;")
        partner_layout.addWidget(self.mic_status_lbl)

        # Subtitles Box
        self.incoming_speech_label = QLabel("Waiting for hearing partner to speak...")
        self.incoming_speech_label.setFont(QFont("Segoe UI", 12, QFont.Weight.Bold))
        self.incoming_speech_label.setWordWrap(True)
        self.incoming_speech_label.setMinimumHeight(44)
        self.incoming_speech_label.setStyleSheet(
            "background-color: #EEF4F8; color: #1E3D59; border: 1.5px solid #D0E1ED; "
            "border-radius: 10px; padding: 8px 12px;"
        )
        partner_layout.addWidget(self.incoming_speech_label)

        # Visual ISL Sign Translation Strip
        sign_label_row = QHBoxLayout()
        sign_label_row.setSpacing(6)
        sign_title = QLabel("ISL Fingerspelling Visuals:")
        sign_title.setFont(QFont("Segoe UI", 10, QFont.Weight.Bold))
        sign_title.setStyleSheet("color: #75655B;")
        sign_label_row.addWidget(sign_title)
        sign_label_row.addStretch()
        partner_layout.addLayout(sign_label_row)

        self.incoming_signs_widget = QWidget()
        self.incoming_signs_layout = QHBoxLayout(self.incoming_signs_widget)
        self.incoming_signs_layout.setContentsMargins(0, 2, 0, 2)
        self.incoming_signs_layout.setSpacing(4)
        self.incoming_signs_layout.addStretch()
        partner_layout.addWidget(self.incoming_signs_widget)

        left_col.addWidget(partner_group)

        # Keyboard Cheat Sheet Reference Card
        shortcut_box = QGroupBox("Keyboard Navigation Guide")
        shortcut_layout = QVBoxLayout(shortcut_box)
        shortcut_layout.setContentsMargins(12, 10, 12, 10)
        shortcut_layout.setSpacing(5)

        shortcuts = [
            ("Hold Sign (0.8s)", "Captures letter into active word"),
            ("Keys [ 1 / 2 / 3 ]", "Accepts autocomplete suggestion"),
            ("Spacebar", "Commits active word to sentence line"),
            ("Backspace", "Deletes last letter (or restores last word)"),
            ("Ctrl + P / Ctrl + Z", "Grammar polish / Revert to raw"),
            ("Ctrl + M / F2", "Toggles mic listening for hearing voice"),
            ("Enter / Escape", "Synthesize speech / Clear buffers")
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

        # ── Card 1: Live Detected Gesture & 0.8s Steady Hold ──
        letter_group = QGroupBox("Real-Time Gesture Recognition & Hold Capture")
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
        self.confidence_bar.setFormat("Confidence: %p%")
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
        word_group = QGroupBox("Active Word Construction")
        word_layout = QVBoxLayout(word_group)
        word_layout.setContentsMargins(14, 14, 14, 14)
        word_layout.setSpacing(8)

        # Gboard-Style 3 Suggestion Pills Strip (Permanently Visible)
        self.suggestions_container = QWidget()
        self.suggestions_layout = QHBoxLayout(self.suggestions_container)
        self.suggestions_layout.setContentsMargins(0, 0, 0, 2)
        self.suggestions_layout.setSpacing(8)

        self.sug_title_lbl = QLabel("Suggestions:")
        self.sug_title_lbl.setFont(QFont("Segoe UI", 10, QFont.Weight.Bold))
        self.sug_title_lbl.setStyleSheet("color: #8C6D58;")
        self.suggestions_layout.addWidget(self.sug_title_lbl)

        self.pill_buttons = []
        initial_pills = ["HELLO", "PLEASE", "THANK YOU"]
        for i in range(3):
            btn = QPushButton(f"[{i+1}] {initial_pills[i]}")
            btn.setFocusPolicy(Qt.FocusPolicy.NoFocus)
            btn.setStyleSheet("""
                QPushButton {
                    background-color: #EFE7DE;
                    color: #3D3530;
                    border: 1.5px solid #D8C9B8;
                    border-radius: 8px;
                    padding: 5px 12px;
                    font-size: 11px;
                    font-weight: 700;
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
            self.suggestions_layout.addWidget(btn)
            self.pill_buttons.append(btn)

        self.suggestions_layout.addStretch()
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

        del_btn = QPushButton("Delete Letter [ Backspace ]")
        del_btn.setObjectName("secondaryBtn")
        del_btn.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        del_btn.clicked.connect(self.delete_last_letter)
        word_btn_layout.addWidget(del_btn, stretch=2)

        word_layout.addLayout(word_btn_layout)
        right_col.addWidget(word_group)

        # ── Card 3: Full Sentence Builder + AI Sign Grammar Polish ──
        sentence_group = QGroupBox("Spoken Sentence Builder")
        sentence_layout = QVBoxLayout(sentence_group)
        sentence_layout.setContentsMargins(14, 14, 14, 14)
        sentence_layout.setSpacing(10)

        self.sentence_label = QLabel("")
        self.sentence_label.setFont(QFont("Segoe UI", 16))
        self.sentence_label.setWordWrap(True)
        self.sentence_label.setMinimumHeight(48)
        self.sentence_label.setStyleSheet(
            "color: #2D2521; background-color: #FFFFFF; border: 1.5px solid #E2D7CB; "
            "border-radius: 12px; padding: 8px 12px;"
        )
        sentence_layout.addWidget(self.sentence_label)

        # AI Polish Controls Row
        polish_row = QHBoxLayout()
        polish_row.setSpacing(10)

        self.polish_btn = QPushButton("Grammar Polish [ Ctrl+P ]")
        self.polish_btn.setObjectName("secondaryBtn")
        self.polish_btn.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.polish_btn.clicked.connect(self.toggle_ai_polish)
        polish_row.addWidget(self.polish_btn, stretch=3)

        self.auto_polish_checkbox = QCheckBox("Auto-polish before speaking")
        self.auto_polish_checkbox.setChecked(True)
        self.auto_polish_checkbox.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.auto_polish_checkbox.setStyleSheet("color: #5C4D44; font-weight: 700; font-size: 11px;")
        polish_row.addWidget(self.auto_polish_checkbox, stretch=2)

        sentence_layout.addLayout(polish_row)

        # Voice & Regional Indian Language Selection Row
        voice_row = QHBoxLayout()
        voice_row.setSpacing(8)

        lang_icon_lbl = QLabel("Voice Language:")
        lang_icon_lbl.setFont(QFont("Segoe UI", 10, QFont.Weight.Bold))
        lang_icon_lbl.setStyleSheet("color: #75655B;")
        voice_row.addWidget(lang_icon_lbl)

        self.language_combo = QComboBox()
        self.language_combo.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.language_combo.setStyleSheet("""
            QComboBox {
                background-color: #FAF4EE;
                color: #2D2521;
                border: 1.5px solid #D8C9B8;
                border-radius: 8px;
                padding: 4px 10px;
                font-family: 'Segoe UI', sans-serif;
                font-size: 11px;
                font-weight: 700;
            }
            QComboBox:hover {
                border-color: #D96B43;
            }
            QComboBox QAbstractItemView {
                background-color: #FFFFFF;
                color: #2D2521;
                selection-background-color: #E29578;
                selection-color: #FFFFFF;
                border: 1px solid #D8C9B8;
                border-radius: 6px;
                padding: 4px;
            }
        """)

        # Supported Regional Languages
        self.languages = [
            ("en", "English (Piper Neural Voice)"),
            ("hi", "Hindi (हिन्दी)"),
            ("te", "Telugu (తెలుగు)"),
            ("ta", "Tamil (தமிழ்)"),
            ("mr", "Marathi (मराठी)"),
            ("kn", "Kannada (ಕನ್ನಡ)"),
            ("bn", "Bengali (বাংলা)"),
            ("gu", "Gujarati (ગુજરાતી)")
        ]
        for code, label in self.languages:
            self.language_combo.addItem(label, code)

        voice_row.addWidget(self.language_combo, stretch=1)
        sentence_layout.addLayout(voice_row)

        sent_btn_layout = QHBoxLayout()
        sent_btn_layout.setSpacing(10)

        speak_btn = QPushButton("Synthesize Speech [ Enter ]")
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

        # ── Card 4: Tabbed Cockpit (Two-Way Conversation & Activity Stream) ──
        self.tab_widget = QTabWidget()
        self.tab_widget.setFocusPolicy(Qt.FocusPolicy.NoFocus)

        # Tab 1: Two-Way Dialogue Timeline
        dialogue_tab = QWidget()
        dialogue_tab_layout = QVBoxLayout(dialogue_tab)
        dialogue_tab_layout.setContentsMargins(8, 8, 8, 8)
        dialogue_tab_layout.setSpacing(6)

        self.conversation_view = QTextEdit()
        self.conversation_view.setReadOnly(True)
        self.conversation_view.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.conversation_view.setFixedHeight(120)
        dialogue_tab_layout.addWidget(self.conversation_view)

        dialogue_btn_row = QHBoxLayout()
        dialogue_btn_row.setSpacing(8)

        self.export_chat_btn = QPushButton("Export Transcript (.txt)")
        self.export_chat_btn.setObjectName("secondaryBtn")
        self.export_chat_btn.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.export_chat_btn.clicked.connect(self.export_conversation_transcript)
        dialogue_btn_row.addWidget(self.export_chat_btn)

        self.copy_chat_btn = QPushButton("Copy Dialogue")
        self.copy_chat_btn.setObjectName("secondaryBtn")
        self.copy_chat_btn.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.copy_chat_btn.clicked.connect(self.copy_conversation_transcript)
        dialogue_btn_row.addWidget(self.copy_chat_btn)

        self.clear_chat_btn = QPushButton("Clear Dialogue")
        self.clear_chat_btn.setObjectName("secondaryBtn")
        self.clear_chat_btn.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.clear_chat_btn.clicked.connect(self.clear_conversation_transcript)
        dialogue_btn_row.addWidget(self.clear_chat_btn)

        dialogue_tab_layout.addLayout(dialogue_btn_row)
        self.tab_widget.addTab(dialogue_tab, "Two-Way Dialogue Timeline")

        # Tab 2: Activity Stream Log
        log_tab = QWidget()
        log_tab_layout = QVBoxLayout(log_tab)
        log_tab_layout.setContentsMargins(8, 8, 8, 8)
        log_tab_layout.setSpacing(6)

        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.log_text.setFixedHeight(120)
        log_tab_layout.addWidget(self.log_text)

        self.tab_widget.addTab(log_tab, "Activity Stream")
        right_col.addWidget(self.tab_widget)

        self.status_label = QLabel("System Status: Ready — Hold sign steady for 0.8s to capture | Keys [1/2/3] for Autocomplete | Ctrl+M for Mic Listener.")
        self.status_label.setStyleSheet("color: #75655B; font-size: 11px; padding: 0 2px;")
        right_col.addWidget(self.status_label)

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
        self.ai_thread.polish_ready.connect(self.on_sentence_polished)
        self.ai_thread.translation_ready.connect(self.on_sentence_translated)

        # Thread 5: Speech-to-Text Listener
        self.stt_thread = SpeechToTextThread(BASE_DIR / "ai_config.json")
        self.stt_thread.transcript_ready.connect(self.on_incoming_transcript)
        self.stt_thread.audio_level.connect(self.on_mic_level)
        self.stt_thread.status_changed.connect(self.on_stt_status)

    def show_shortcuts_guide(self):
        """Displays the interactive Keyboard & Controls guide modal."""
        dialog = ShortcutsHelpDialog(self)
        dialog.exec()

    def start_pipeline(self):
        if self.is_running:
            return
        self.is_running = True
        self.start_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.engine_status_badge.setText("Camera: 30 FPS Active")
        self.engine_status_badge.setStyleSheet(
            "background-color: #E3F1E9; color: #2D704F; border: 1px solid #B8DCBE; "
            "border-radius: 8px; padding: 6px 12px;"
        )

        self.log("Starting ISL Camera Pipeline & Two-Way Loop...")
        self.capture_thread.start()
        self.inference_thread.start()
        self.tts_thread.start()
        self.ai_thread.start()
        self.stt_thread.start()

    def stop_pipeline(self):
        self.is_running = False
        self.capture_thread.stop()
        self.inference_thread.stop()
        self.tts_thread.stop()
        self.ai_thread.stop()
        self.stt_thread.stop()

        self.capture_thread.wait(3000)
        self.inference_thread.wait(3000)
        self.tts_thread.wait(3000)
        self.ai_thread.wait(3000)
        self.stt_thread.wait(3000)

        self._init_threads()
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.engine_status_badge.setText("Camera: Disconnected")
        self.engine_status_badge.setStyleSheet(
            "background-color: #F8E8E8; color: #8A3333; border: 1px solid #E0BDB8; "
            "border-radius: 8px; padding: 6px 12px;"
        )
        self.status_label.setText("System Status: Stopped")
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
        self._update_suggestions_instant()
        self._trigger_ai_prediction()

    # ═══════════════════════════════════════════════════════════
    # AI Autocomplete & Suggestion Handlers
    # ═══════════════════════════════════════════════════════════
    def _update_suggestions_instant(self):
        """Instantly updates suggestion pills (<0.1ms) using local lexicon."""
        prefix = "".join(self.current_word_letters).strip()
        context = " ".join(self.sentence_words).strip()

        if prefix:
            self.sug_title_lbl.setText("Suggestions:")
            matches = [w for w in self.ai_thread.offline_dict if w.startswith(prefix) and w != prefix][:3]
            if not matches:
                matches = [prefix]
            self.on_suggestions_ready(matches)
        elif context:
            self.sug_title_lbl.setText("Next Word:")
            last_word = self.sentence_words[-1].upper() if self.sentence_words else ""
            matches = self.ai_thread.next_word_map.get(last_word, ["PLEASE", "THANK YOU", "HELP"])[:3]
            self.on_suggestions_ready(matches)
        else:
            self.sug_title_lbl.setText("Quick Starters:")
            self.on_suggestions_ready(["HELLO", "PLEASE", "THANK YOU"])

    def _trigger_ai_prediction(self):
        """Dispatches non-blocking suggestion lookup to Groq Cloud in background."""
        prefix = "".join(self.current_word_letters).strip()
        context = " ".join(self.sentence_words).strip()
        self.ai_thread.enqueue_prediction(prefix, context)

    def on_suggestions_ready(self, suggestions):
        """Updates the 3 suggestion pills smoothly."""
        if not suggestions:
            return
        self.current_suggestions = suggestions
        for i in range(3):
            if i < len(suggestions):
                self.pill_buttons[i].setText(f"[{i+1}] {suggestions[i]}")
                self.pill_buttons[i].setEnabled(True)
            else:
                self.pill_buttons[i].setText(f"[{i+1}] -")
                self.pill_buttons[i].setEnabled(False)

    def _on_suggestion_clicked(self, idx):
        """Triggered when clicking a suggestion pill."""
        if self.current_suggestions and idx < len(self.current_suggestions):
            self._accept_autocomplete(self.current_suggestions[idx])

    def _accept_autocomplete(self, word):
        """Commits chosen autocomplete suggestion directly into the sentence line."""
        if not word or word == "-":
            return
        self.sentence_words.append(word)
        self.sentence_label.setText(" ".join(self.sentence_words))
        self.current_word_letters.clear()
        self.word_label.setText("")
        self.is_polished = False
        self.polish_btn.setText("Grammar Polish [ Ctrl+P ]")
        self.log(f"AI Autocomplete: \"{word}\" committed to sentence.")
        play_feedback_tone(freq=1100, duration_ms=40)
        self._update_suggestions_instant()
        self._trigger_ai_prediction()

    # ═══════════════════════════════════════════════════════════
    # 1-Click AI Sign Grammar Polish Engine
    # ═══════════════════════════════════════════════════════════
    def toggle_ai_polish(self):
        """Toggles AI grammar polish between polished natural sentence and raw sign gloss."""
        if self.is_polished and self.raw_sentence_words:
            # Revert to raw sign gloss
            self.sentence_words = list(self.raw_sentence_words)
            self.sentence_label.setText(" ".join(self.sentence_words))
            self.is_polished = False
            self.polish_btn.setText("Grammar Polish [ Ctrl+P ]")
            self.log(f"Restored Raw Sign Sequence: \"{' '.join(self.sentence_words)}\"")
            play_feedback_tone(freq=950, duration_ms=30)
        else:
            # Trigger AI Polish
            full_text = self.sentence_label.text().strip()
            if not full_text:
                return
            self.raw_sentence_words = list(self.sentence_words)
            self.polish_btn.setEnabled(False)
            self.polish_btn.setText("Polishing Grammar...")
            self.log(f"Applying AI Grammar Polish: \"{full_text}\"...")
            self.ai_thread.enqueue_polish(full_text)

    def on_sentence_polished(self, polished_text):
        """Callback when AI Grammar Polish completes."""
        self.polish_btn.setEnabled(True)
        if polished_text:
            self.sentence_label.setText(polished_text)
            self.is_polished = True
            self.polish_btn.setText("Revert to Raw [ Ctrl+Z ]")
            self.log(f"AI Polished Sentence: \"{polished_text}\"")
            play_feedback_tone(freq=1350, duration_ms=45)

            # If user triggered speak with auto-polish on:
            if self._speak_after_polish:
                self._speak_after_polish = False
                self.tts_thread.enqueue_text(polished_text)
                self.log(f"Speaking Polished Sentence: \"{polished_text}\"")
                self._append_to_conversation("You (Signer)", polished_text, is_signer=True)

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
            self.is_polished = False
            self.polish_btn.setText("Grammar Polish [ Ctrl+P ]")
            play_feedback_tone(freq=900, duration_ms=25)
            self._update_suggestions_instant()
            self._trigger_ai_prediction()

    def delete_last_letter(self):
        """Deletes last letter, or pulls back last committed word if active word is empty."""
        if self.current_word_letters:
            popped = self.current_word_letters.pop()
            self.word_label.setText("".join(self.current_word_letters))
            self.log(f"Deleted Letter: '{popped}'")
            self._update_suggestions_instant()
            self._trigger_ai_prediction()
        elif self.sentence_words:
            last_word = self.sentence_words.pop()
            self.sentence_label.setText(" ".join(self.sentence_words))
            self.current_word_letters = list(last_word)
            self.word_label.setText(last_word)
            self.is_polished = False
            self.polish_btn.setText("Grammar Polish [ Ctrl+P ]")
            self.log(f"Restored Word for Editing: \"{last_word}\"")
            self._update_suggestions_instant()
            self._trigger_ai_prediction()

    def speak_full_sentence(self):
        """Commits any pending word and synthesizes speech for the complete sentence in the selected language."""
        if self.current_word_letters:
            self.commit_word()

        full_text = self.sentence_label.text().strip()
        if not full_text:
            return

        lang_code = self.language_combo.currentData() or "en"
        lang_name = self.language_combo.currentText()

        if lang_code == "en":
            # English Speech Flow
            if self.auto_polish_checkbox.isChecked() and not self.is_polished:
                self.raw_sentence_words = list(self.sentence_words)
                self._speak_after_polish = True
                self.polish_btn.setEnabled(False)
                self.polish_btn.setText("Polishing Grammar...")
                self.log(f"Auto-Polishing before Speech: \"{full_text}\"...")
                self.ai_thread.enqueue_polish(full_text)
            else:
                self.tts_thread.enqueue_text(full_text, lang_code="en")
                self.log(f"Speaking Full Sentence (English): \"{full_text}\"")
                self._append_to_conversation("You (Signer)", full_text, is_signer=True)
        else:
            # Regional Indian Language Speech Flow (Hindi, Telugu, Tamil, Marathi, etc.)
            self.polish_btn.setEnabled(False)
            self.polish_btn.setText("Translating...")
            self.log(f"Translating to {lang_name}: \"{full_text}\"...")
            self.ai_thread.enqueue_translate(full_text, lang_code, lang_name)

    def on_sentence_translated(self, translated_text, lang_code, original_text):
        """Callback when regional language translation completes."""
        self.polish_btn.setEnabled(True)
        self.polish_btn.setText("Grammar Polish [ Ctrl+P ]")
        if translated_text:
            self.sentence_label.setText(translated_text)
            self.log(f"Translated to [{lang_code.upper()}]: \"{translated_text}\" (Original: \"{original_text}\")")
            self.tts_thread.enqueue_text(translated_text, lang_code=lang_code)
            self._append_to_conversation(f"You ({lang_code.upper()})", translated_text, is_signer=True)

    def clear_all(self):
        """Clears word and sentence buffers."""
        self.current_word_letters.clear()
        self.sentence_words.clear()
        self.raw_sentence_words.clear()
        self.is_polished = False
        self._speak_after_polish = False
        self.polish_btn.setText("Grammar Polish [ Ctrl+P ]")
        self.word_label.setText("")
        self.sentence_label.setText("")
        self.held_candidate = None
        self.dwell_start_time = None
        self.dwell_progress_pct = 0
        self.locked_letter = None
        self.dwell_progress_bar.setValue(0)
        self.dwell_progress_bar.setFormat("Hold Steady for 0.8s to Capture")
        self._update_suggestions_instant()
        self.log("Cleared word and sentence buffers.")

    # ═══════════════════════════════════════════════════════════
    # Two-Way Microphone Listener & Hearing Speech Handlers
    # ═══════════════════════════════════════════════════════════
    def toggle_mic_listening(self):
        """Toggles microphone recording state for hearing person voice input."""
        if not self.is_running:
            self.start_pipeline()

        self.stt_thread.toggle_listening()
        if self.stt_thread.is_listening:
            self.listen_btn.setText("Stop Listening [ Ctrl+M ]")
            self.listen_btn.setStyleSheet("""
                QPushButton {
                    background-color: #C45353;
                    color: #FFFFFF;
                    font-weight: bold;
                    border-radius: 10px;
                    padding: 8px 14px;
                }
                QPushButton:hover { background-color: #AF4242; }
            """)
            self.mic_status_lbl.setText("Microphone: Recording hearing voice...")
            self.two_way_badge.setText("Two-Way Loop: Listening Mic...")
            self.two_way_badge.setStyleSheet(
                "background-color: #F8E8E8; color: #8A3333; border: 1px solid #E0BDB8; "
                "border-radius: 8px; padding: 6px 12px;"
            )
            self.log("Microphone Listener: Started (Listening to hearing partner)...")
            play_feedback_tone(freq=1450, duration_ms=40)
        else:
            self.listen_btn.setText("Listen Hearing Voice [ Ctrl+M ]")
            self.listen_btn.setStyleSheet("""
                QPushButton {
                    background-color: #2A6F97;
                    color: #FFFFFF;
                    font-weight: bold;
                    border-radius: 10px;
                    padding: 8px 14px;
                }
                QPushButton:hover { background-color: #014F86; }
            """)
            self.mic_status_lbl.setText("Microphone: Processing audio...")
            self.two_way_badge.setText("Two-Way Loop: Active")
            self.two_way_badge.setStyleSheet(
                "background-color: #E2EBF0; color: #1E4D6B; border: 1px solid #C4D7E2; "
                "border-radius: 8px; padding: 6px 12px;"
            )
            self.log("Microphone Listener: Stopped (Transcribing audio via Whisper AI)...")
            play_feedback_tone(freq=1050, duration_ms=40)

    def on_mic_level(self, level):
        """Updates mic volume meter in real time."""
        if hasattr(self, "mic_level_bar"):
            self.mic_level_bar.setValue(level)

    def on_stt_status(self, msg):
        """Callback when STT thread updates status."""
        if hasattr(self, "mic_status_lbl"):
            self.mic_status_lbl.setText(f"Microphone: {msg}")
        self.log(f"STT Engine: {msg}")

    def on_incoming_transcript(self, transcript_text):
        """Callback when hearing partner's spoken speech is transcribed by Whisper."""
        clean_text = transcript_text.strip()
        if not clean_text:
            return

        self.incoming_speech_label.setText(f"\"{clean_text}\"")
        self._render_incoming_isl_signs(clean_text)
        self._append_to_conversation("Hearing Partner", clean_text, is_signer=False)
        self.log(f"Hearing Partner Spoke: \"{clean_text}\"")
        play_feedback_tone(freq=1350, duration_ms=50)

    def _render_incoming_isl_signs(self, text):
        """Renders visual ISL fingerspelling letter chips for incoming words."""
        while self.incoming_signs_layout.count():
            item = self.incoming_signs_layout.takeAt(0)
            widget = item.widget()
            if widget:
                widget.deleteLater()

        clean_chars = [c.upper() for c in text if c.isalnum() or c.isspace()][:24]
        for char in clean_chars:
            if char.isspace():
                space_lbl = QLabel(" ")
                space_lbl.setFixedWidth(8)
                self.incoming_signs_layout.addWidget(space_lbl)
            else:
                chip = QLabel(char)
                chip.setFixedSize(26, 26)
                chip.setAlignment(Qt.AlignmentFlag.AlignCenter)
                chip.setFont(QFont("Segoe UI", 11, QFont.Weight.Bold))
                chip.setStyleSheet("""
                    background-color: #E29578;
                    color: #FFFFFF;
                    border-radius: 6px;
                    font-weight: bold;
                """)
                self.incoming_signs_layout.addWidget(chip)

        self.incoming_signs_layout.addStretch()

    def _clear_incoming_isl_signs(self):
        """Clears the visual sign strip."""
        while self.incoming_signs_layout.count():
            item = self.incoming_signs_layout.takeAt(0)
            widget = item.widget()
            if widget:
                widget.deleteLater()
        self.incoming_signs_layout.addStretch()

    # ═══════════════════════════════════════════════════════════
    # Two-Way Dialogue History & Export System
    # ═══════════════════════════════════════════════════════════
    def _append_to_conversation(self, sender, text, is_signer=True):
        """Appends dialogue turn to conversation timeline."""
        ts = time.strftime("%H:%M:%S")
        self.conversation_history.append({
            "sender": sender,
            "text": text,
            "timestamp": ts,
            "is_signer": is_signer
        })
        self._refresh_conversation_view()

    def _refresh_conversation_view(self):
        """Renders formatted HTML dialogue bubbles in the conversation view."""
        html_parts = []
        for item in self.conversation_history[-30:]:
            if item["is_signer"]:
                # Green/Teal bubble for signer
                html_parts.append(
                    f"<div style='margin-bottom: 6px;'>"
                    f"<span style='color: #2D704F; font-weight: bold; font-size: 11px;'>🟢 {item['sender']} [{item['timestamp']}]:</span><br>"
                    f"<div style='background-color: #EBF5EE; color: #1E4631; padding: 6px 10px; border-radius: 8px; border: 1px solid #D2E7DA; margin-top: 2px; font-weight: 600; font-size: 12px;'>"
                    f"{item['text']}</div></div>"
                )
            else:
                # Blue bubble for hearing partner
                html_parts.append(
                    f"<div style='margin-bottom: 6px;'>"
                    f"<span style='color: #1E4D6B; font-weight: bold; font-size: 11px;'>🔵 {item['sender']} [{item['timestamp']}]:</span><br>"
                    f"<div style='background-color: #EEF4F8; color: #153A52; padding: 6px 10px; border-radius: 8px; border: 1px solid #D0E1ED; margin-top: 2px; font-weight: 600; font-size: 12px;'>"
                    f"{item['text']}</div></div>"
                )

        self.conversation_view.setHtml("".join(html_parts))
        scrollbar = self.conversation_view.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())

    def export_conversation_transcript(self):
        """Exports the complete dialogue session to a clean timestamped text file."""
        if not self.conversation_history:
            self.log("Notice: No conversation dialogue to export.")
            return

        transcripts_dir = BASE_DIR / "transcripts"
        transcripts_dir.mkdir(parents=True, exist_ok=True)
        file_name = f"dialogue_{time.strftime('%Y%m%d_%H%M%S')}.txt"
        out_path = transcripts_dir / file_name

        lines = [
            "===========================================================",
            " SignSpeak Universal - Two-Way Conversation Transcript",
            f" Session Date: {time.strftime('%Y-%m-%d %H:%M:%S')}",
            "===========================================================\n"
        ]
        for item in self.conversation_history:
            role = "SIGNER (ISL)" if item["is_signer"] else "HEARING PARTNER (VOICE)"
            lines.append(f"[{item['timestamp']}] {role}:\n  \"{item['text']}\"\n")
        lines.append("===========================================================\n")

        try:
            with open(out_path, "w", encoding="utf-8") as f:
                f.write("\n".join(lines))
            self.log(f"Exported Transcript to: transcripts/{file_name}")
            play_feedback_tone(freq=1500, duration_ms=45)
        except Exception as e:
            self.log(f"Export Transcript Error: {e}")

    def copy_conversation_transcript(self):
        """Copies formatted conversation transcript to the system clipboard."""
        if not self.conversation_history:
            self.log("Notice: No dialogue to copy.")
            return

        text_content = "\n".join([
            f"[{item['timestamp']}] {'You (Signer)' if item['is_signer'] else 'Hearing Partner'}: \"{item['text']}\""
            for item in self.conversation_history
        ])
        clipboard = QApplication.clipboard()
        if clipboard:
            clipboard.setText(text_content)
        self.log("Copied full dialogue transcript to clipboard.")
        play_feedback_tone(freq=1200, duration_ms=25)

    def clear_conversation_transcript(self):
        """Clears the dialogue history."""
        self.conversation_history.clear()
        self.conversation_view.clear()
        self.incoming_speech_label.setText("Waiting for hearing partner to speak...")
        self._clear_incoming_isl_signs()
        self.log("Cleared two-way dialogue history.")
        play_feedback_tone(freq=900, duration_ms=25)

    def keyPressEvent(self, event):
        """Handle global keyboard shortcuts cleanly (100% Preserved + F1/Ctrl+P/Ctrl+Z/Ctrl+M/Keys 1,2,3 & Numpad)."""
        key = event.key()
        if key == Qt.Key.Key_F1:
            self.show_shortcuts_guide()
        elif (event.modifiers() & Qt.KeyboardModifier.ControlModifier and key == Qt.Key.Key_M) or key == Qt.Key.Key_F2:
            self.toggle_mic_listening()
        elif event.modifiers() & Qt.KeyboardModifier.ControlModifier and key == Qt.Key.Key_P:
            self.toggle_ai_polish()
        elif event.modifiers() & Qt.KeyboardModifier.ControlModifier and key == Qt.Key.Key_Z:
            if self.is_polished:
                self.toggle_ai_polish()
        elif key in (Qt.Key.Key_1, Qt.Key.Key_Numpad1) and self.current_suggestions and len(self.current_suggestions) >= 1:
            self._accept_autocomplete(self.current_suggestions[0])
        elif key in (Qt.Key.Key_2, Qt.Key.Key_Numpad2) and self.current_suggestions and len(self.current_suggestions) >= 2:
            self._accept_autocomplete(self.current_suggestions[1])
        elif key in (Qt.Key.Key_3, Qt.Key.Key_Numpad3) and self.current_suggestions and len(self.current_suggestions) >= 3:
            self._accept_autocomplete(self.current_suggestions[2])
        elif key == Qt.Key.Key_Space:
            self.commit_word()
        elif key == Qt.Key.Key_Backspace:
            self.delete_last_letter()
        elif key in (Qt.Key.Key_Return, Qt.Key.Key_Enter):
            self.speak_full_sentence()
        elif key == Qt.Key.Key_Escape:
            self.clear_all()
        else:
            super().keyPressEvent(event)

    def on_speech_done(self, text, lang_code="en"):
        lang_tag = lang_code.upper() if lang_code else "EN"
        self.log(f"Speech Vocalized [{lang_tag}]: \"{text}\"")

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
