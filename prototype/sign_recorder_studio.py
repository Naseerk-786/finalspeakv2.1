# SignSpeak Studio — Interactive ISL Sign Recorder & Personalized Trainer
# Features: 3s Preparation Countdown (Red) -> 3s Auto-Record (Green) -> Co-Training Fast GPU Fine-Tuner -> Live Test Mode

import os
import sys
import json
import time
import queue
import numpy as np
from pathlib import Path

import cv2
import mediapipe as mp
import onnxruntime as ort

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QComboBox, QTableWidget, QTableWidgetItem,
    QHeaderView, QProgressBar, QFrame, QGroupBox, QTextEdit, QSplitter,
    QMessageBox, QRadioButton, QButtonGroup, QLineEdit
)
from PyQt6.QtCore import Qt, QTimer, pyqtSignal, QThread
from PyQt6.QtGui import QImage, QPixmap, QFont, QColor, QPalette, QBrush

import fine_tune_engine

BASE_DIR = Path(r"d:\finalspeak")
DATA_DIR = BASE_DIR / "data"
USER_RECORDED_DIR = DATA_DIR / "user_recorded"
MODELS_DIR = BASE_DIR / "models"
ONNX_MODEL_PATH = MODELS_DIR / "isl_letter_classifier.onnx"
CLASS_META_PATH = MODELS_DIR / "isl_letter_meta.json"

USER_RECORDED_DIR.mkdir(parents=True, exist_ok=True)

# Standard Classes: A-Z, 1-9, and NO_SIGN
BASE_CLASSES = [chr(c) for c in range(ord('A'), ord('Z')+1)] + [str(i) for i in range(1, 10)] + ["NO_SIGN"]


# ═══════════════════════════════════════════════════════════════
# MediaPipe Hand Feature Extractor
# ═══════════════════════════════════════════════════════════════
class HandLandmarkProcessor:
    def __init__(self):
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_styles = mp.solutions.drawing_styles
        self.hands = None

    def initialize(self):
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

    def process(self, frame_bgr):
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

    def draw_landmarks(self, frame, results, is_recording=False):
        if results and results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Custom high-visibility connections
                landmark_color = (46, 204, 113) if is_recording else (255, 170, 0)
                self.mp_drawing.draw_landmarks(
                    frame,
                    hand_landmarks,
                    self.mp_hands.HAND_CONNECTIONS,
                    self.mp_drawing.DrawingSpec(color=landmark_color, thickness=2, circle_radius=3),
                    self.mp_drawing.DrawingSpec(color=(255, 255, 255), thickness=2)
                )
        return frame

    def close(self):
        if self.hands:
            self.hands.close()


# ═══════════════════════════════════════════════════════════════
# Asynchronous Camera Capture Thread
# ═══════════════════════════════════════════════════════════════
class CameraThread(QThread):
    frame_ready = pyqtSignal(np.ndarray, object, object)  # (raw_frame, feat_vec, results)
    error_signal = pyqtSignal(str)

    def __init__(self):
        super().__init__()
        self.running = False
        self.processor = HandLandmarkProcessor()

    def run(self):
        self.running = True
        self.processor.initialize()

        cap = None
        for cam_idx in [0, 1, 2]:
            for backend in [cv2.CAP_DSHOW, cv2.CAP_MSMF, cv2.CAP_ANY]:
                test_cap = cv2.VideoCapture(cam_idx, backend)
                if test_cap.isOpened():
                    ret, _ = test_cap.read()
                    if ret:
                        cap = test_cap
                        break
                    test_cap.release()
            if cap is not None:
                break

        if cap is None:
            self.error_signal.emit("Could not open camera. Check connections.")
            return

        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        while self.running:
            ret, frame = cap.read()
            if not ret or frame is None:
                time.sleep(0.01)
                continue

            frame = cv2.flip(frame, 1)
            feat_vec, results = self.processor.process(frame)
            self.frame_ready.emit(frame, feat_vec, results)
            time.sleep(1 / 30)

        cap.release()
        self.processor.close()

    def stop(self):
        self.running = False


# ═══════════════════════════════════════════════════════════════
# Asynchronous Fine-Tuning Background Worker
# ═══════════════════════════════════════════════════════════════
class FineTuneWorker(QThread):
    epoch_progress = pyqtSignal(int, int, str, float, float) # (epoch, total_epochs, log_msg, train_acc, val_acc)
    finished_signal = pyqtSignal(float, dict) # (best_val_acc, meta_info)
    error_signal = pyqtSignal(str)

    def __init__(self, epochs=35):
        super().__init__()
        self.epochs = epochs

    def run(self):
        try:
            def callback(epoch, total_epochs, msg, train_acc, val_acc):
                self.epoch_progress.emit(epoch, total_epochs, msg, train_acc, val_acc)

            best_val_acc, meta_info = fine_tune_engine.run_fine_tuning(
                epochs=self.epochs,
                progress_callback=callback
            )
            self.finished_signal.emit(best_val_acc, meta_info)
        except Exception as e:
            self.error_signal.emit(str(e))


# ═══════════════════════════════════════════════════════════════
# SignSpeak Studio Main Application Window
# ═══════════════════════════════════════════════════════════════
class SignSpeakStudio(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("SignSpeak Studio — Interactive ISL Sign Recorder & Personalized Trainer")
        self.setMinimumSize(1280, 820)
        self.setStyleSheet(self._get_stylesheet())

        # State Variables
        self.current_mode = "RECORD"  # "RECORD" or "TEST"
        self.record_state = "IDLE"    # "IDLE", "COUNTDOWN", "RECORDING"

        self.selected_class = "A"
        self.countdown_remaining = 3.0
        self.recording_remaining = 3.0
        self.recorded_features_buffer = []

        self.last_timer_tick = time.time()
        self.onnx_session = None
        self.idx2class = {}
        self.class2idx = {}

        self._load_onnx_model()
        self._build_ui()
        self._init_camera()
        self._refresh_dataset_table()

        # UI Update Timer (30 FPS ticker for countdown & record state machine)
        self.state_timer = QTimer()
        self.state_timer.timeout.connect(self._on_state_tick)
        self.state_timer.start(50)  # 20 ticks per second

    def _get_stylesheet(self):
        return """
            QMainWindow {
                background-color: #121417;
            }
            QLabel {
                color: #ECEFF1;
                font-family: 'Segoe UI', sans-serif;
            }
            QGroupBox {
                background-color: #1A1D24;
                border: 1px solid #2B303C;
                border-radius: 12px;
                margin-top: 14px;
                padding: 14px;
                font-family: 'Segoe UI', sans-serif;
                color: #64B5F6;
                font-weight: bold;
                font-size: 13px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 14px;
                padding: 0 8px;
                background-color: #1A1D24;
                color: #90CAF9;
            }
            QPushButton {
                background-color: #1976D2;
                color: #FFFFFF;
                border: none;
                padding: 10px 18px;
                font-size: 13px;
                font-weight: bold;
                border-radius: 8px;
                font-family: 'Segoe UI', sans-serif;
            }
            QPushButton:hover {
                background-color: #2196F3;
            }
            QPushButton:pressed {
                background-color: #1565C0;
            }
            QPushButton#recordBtn {
                background-color: #E53935;
                color: #FFFFFF;
                font-size: 15px;
                font-weight: bold;
                padding: 14px 24px;
                border-radius: 10px;
            }
            QPushButton#recordBtn:hover {
                background-color: #F44336;
            }
            QPushButton#trainBtn {
                background-color: #2E7D32;
                color: #FFFFFF;
                font-size: 14px;
                font-weight: bold;
                padding: 12px 20px;
                border-radius: 8px;
            }
            QPushButton#trainBtn:hover {
                background-color: #388E3C;
            }
            QPushButton#deleteBtn {
                background-color: #374151;
                color: #EF5350;
            }
            QPushButton#deleteBtn:hover {
                background-color: #4B5563;
            }
            QTableWidget {
                background-color: #15181E;
                color: #E0E0E0;
                gridline-color: #262C38;
                border: 1px solid #2B303C;
                border-radius: 8px;
                font-family: 'Segoe UI', sans-serif;
                font-size: 12px;
            }
            QHeaderView::section {
                background-color: #1E232D;
                color: #90CAF9;
                padding: 6px;
                font-weight: bold;
                border: 1px solid #2B303C;
            }
            QProgressBar {
                background-color: #212631;
                border-radius: 6px;
                text-align: center;
                color: #FFFFFF;
                font-weight: bold;
                height: 20px;
            }
            QProgressBar::chunk {
                background-color: #2ECC71;
                border-radius: 6px;
            }
            QComboBox {
                background-color: #212631;
                color: #FFFFFF;
                border: 1px solid #374151;
                border-radius: 6px;
                padding: 6px 12px;
                font-size: 13px;
                font-weight: bold;
            }
            QComboBox QAbstractItemView {
                background-color: #1E232D;
                color: #FFFFFF;
                selection-background-color: #1976D2;
            }
            QTextEdit {
                background-color: #0E1013;
                color: #81C784;
                border: 1px solid #262C38;
                border-radius: 8px;
                font-family: 'Consolas', monospace;
                font-size: 11px;
            }
            QLineEdit {
                background-color: #212631;
                color: #FFFFFF;
                border: 1px solid #374151;
                border-radius: 6px;
                padding: 6px 10px;
                font-size: 13px;
            }
        """

    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QHBoxLayout(central)
        main_layout.setContentsMargins(16, 16, 16, 16)
        main_layout.setSpacing(16)

        # ═══════════════════════════════════════════════════════════
        # LEFT COLUMN: Live Camera Feed & Interactive HUD
        # ═══════════════════════════════════════════════════════════
        left_col = QVBoxLayout()

        header_layout = QHBoxLayout()
        title_label = QLabel("SignSpeak Studio — Personal ISL Trainer")
        title_label.setFont(QFont("Segoe UI", 16, QFont.Weight.Bold))
        header_layout.addWidget(title_label)

        # Mode Selector Buttons
        self.mode_record_btn = QPushButton("Recording Studio")
        self.mode_record_btn.setCheckable(True)
        self.mode_record_btn.setChecked(True)
        self.mode_record_btn.clicked.connect(lambda: self._set_mode("RECORD"))
        header_layout.addWidget(self.mode_record_btn)

        self.mode_test_btn = QPushButton("Live Test Mode")
        self.mode_test_btn.setCheckable(True)
        self.mode_test_btn.clicked.connect(lambda: self._set_mode("TEST"))
        header_layout.addWidget(self.mode_test_btn)

        left_col.addLayout(header_layout)

        # Video Preview Display
        video_group = QGroupBox("Camera Stream & Hand Landmark HUD")
        video_layout = QVBoxLayout(video_group)
        self.video_label = QLabel("Initializing Camera...")
        self.video_label.setFixedSize(640, 480)
        self.video_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.video_label.setStyleSheet("background-color: #0E1013; border-radius: 10px; border: 2px solid #2B303C;")
        video_layout.addWidget(self.video_label)

        # State Banner Indicator
        self.status_banner = QLabel("READY — Select a sign and click 'Start 3s Recording'")
        self.status_banner.setFont(QFont("Segoe UI", 12, QFont.Weight.Bold))
        self.status_banner.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.status_banner.setStyleSheet(
            "background-color: #212631; color: #90CAF9; padding: 10px; border-radius: 8px;"
        )
        video_layout.addWidget(self.status_banner)

        left_col.addWidget(video_group)

        # Recording Control Panel
        self.control_group = QGroupBox("Sign Recording Controls (3s Preparation -> 3s Auto-Capture)")
        ctrl_layout = QVBoxLayout(self.control_group)

        sign_select_layout = QHBoxLayout()
        sign_select_layout.addWidget(QLabel("Target Sign Class:"))

        self.class_dropdown = QComboBox()
        for cls_name in BASE_CLASSES:
            self.class_dropdown.addItem(cls_name)
        self.class_dropdown.currentTextChanged.connect(self._on_class_selected)
        sign_select_layout.addWidget(self.class_dropdown)

        sign_select_layout.addWidget(QLabel("or Custom Sign:"))
        self.custom_sign_input = QLineEdit()
        self.custom_sign_input.setPlaceholderText("e.g., HELLO, WATER")
        sign_select_layout.addWidget(self.custom_sign_input)

        self.add_custom_btn = QPushButton("Add")
        self.add_custom_btn.clicked.connect(self._add_custom_sign)
        sign_select_layout.addWidget(self.add_custom_btn)

        ctrl_layout.addLayout(sign_select_layout)

        # Action Buttons
        action_layout = QHBoxLayout()
        self.record_trigger_btn = QPushButton("START 3s RECORDING (Spacebar)")
        self.record_trigger_btn.setObjectName("recordBtn")
        self.record_trigger_btn.clicked.connect(self.start_countdown)
        action_layout.addWidget(self.record_trigger_btn)

        self.cancel_btn = QPushButton("Cancel")
        self.cancel_btn.setObjectName("deleteBtn")
        self.cancel_btn.clicked.connect(self.cancel_recording)
        self.cancel_btn.setEnabled(False)
        action_layout.addWidget(self.cancel_btn)

        ctrl_layout.addLayout(action_layout)

        # Visual Countdowns Progress Bar
        self.action_progress_bar = QProgressBar()
        self.action_progress_bar.setMaximum(100)
        self.action_progress_bar.setValue(0)
        self.action_progress_bar.setTextVisible(True)
        self.action_progress_bar.setFormat("Ready")
        ctrl_layout.addWidget(self.action_progress_bar)

        left_col.addWidget(self.control_group)

        # Live Test Panel (Hidden in Record Mode)
        self.test_group = QGroupBox("Live Prediction HUD")
        test_layout = QVBoxLayout(self.test_group)
        self.test_pred_label = QLabel("-")
        self.test_pred_label.setFont(QFont("Segoe UI", 48, QFont.Weight.Bold))
        self.test_pred_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.test_pred_label.setStyleSheet("color: #4CAF50;")
        test_layout.addWidget(self.test_pred_label)

        self.test_conf_bar = QProgressBar()
        self.test_conf_bar.setMaximum(100)
        self.test_conf_bar.setValue(0)
        self.test_conf_bar.setFormat("Confidence: %p%")
        test_layout.addWidget(self.test_conf_bar)

        self.test_group.setVisible(False)
        left_col.addWidget(self.test_group)

        main_layout.addLayout(left_col, stretch=3)

        # ═══════════════════════════════════════════════════════════
        # RIGHT COLUMN: Dataset Manager & GPU Auto-Trainer
        # ═══════════════════════════════════════════════════════════
        right_col = QVBoxLayout()

        # Dataset Inventory Table
        table_group = QGroupBox("Dataset Inventory & Recorded Classes")
        table_layout = QVBoxLayout(table_group)

        self.dataset_table = QTableWidget()
        self.dataset_table.setColumnCount(4)
        self.dataset_table.setHorizontalHeaderLabels(["Class", "Base Samples", "Your Samples", "Status"])
        self.dataset_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.dataset_table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.dataset_table.itemClicked.connect(self._on_table_row_clicked)
        table_layout.addWidget(self.dataset_table)

        table_action_layout = QHBoxLayout()
        self.delete_samples_btn = QPushButton("Clear Selected Class Data")
        self.delete_samples_btn.setObjectName("deleteBtn")
        self.delete_samples_btn.clicked.connect(self._delete_selected_samples)
        table_action_layout.addWidget(self.delete_samples_btn)

        self.refresh_table_btn = QPushButton("Refresh Table")
        self.refresh_table_btn.clicked.connect(self._refresh_dataset_table)
        table_action_layout.addWidget(self.refresh_table_btn)
        table_layout.addLayout(table_action_layout)

        right_col.addWidget(table_group, stretch=3)

        # Fine-Tuning Control Panel
        train_group = QGroupBox("Personalized GPU Fine-Tuner (Zero Forgetting)")
        train_layout = QVBoxLayout(train_group)

        train_desc = QLabel("Trains on RTX 4050 GPU in ~8s by blending base 107k samples with your recorded camera signs.")
        train_desc.setFont(QFont("Segoe UI", 11))
        train_desc.setStyleSheet("color: #B0BEC5;")
        train_desc.setWordWrap(True)
        train_layout.addWidget(train_desc)

        self.train_btn = QPushButton("FINE-TUNE MODEL (CUDA GPU)")
        self.train_btn.setObjectName("trainBtn")
        self.train_btn.clicked.connect(self.start_fine_tuning)
        train_layout.addWidget(self.train_btn)

        self.train_progress_bar = QProgressBar()
        self.train_progress_bar.setMaximum(100)
        self.train_progress_bar.setValue(0)
        self.train_progress_bar.setFormat("Trainer Idle")
        train_layout.addWidget(self.train_progress_bar)

        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setMaximumHeight(130)
        train_layout.addWidget(self.log_text)

        right_col.addWidget(train_group, stretch=2)

        main_layout.addLayout(right_col, stretch=2)

    def _init_camera(self):
        self.camera_thread = CameraThread()
        self.camera_thread.frame_ready.connect(self._on_frame_ready)
        self.camera_thread.error_signal.connect(self.log)
        self.camera_thread.start()

    def _load_onnx_model(self):
        if ONNX_MODEL_PATH.exists() and CLASS_META_PATH.exists():
            try:
                self.onnx_session = ort.InferenceSession(str(ONNX_MODEL_PATH))
                with open(CLASS_META_PATH, "r", encoding="utf-8") as f:
                    meta = json.load(f)
                self.idx2class = {int(k): v for k, v in meta["idx2class"].items()}
                self.class2idx = {v: int(k) for k, v in self.idx2class.items()}
                print(f"[OK] Loaded ONNX session with {len(self.idx2class)} classes.")
            except Exception as e:
                print(f"[!] Warning loading ONNX: {e}")

    def _set_mode(self, mode):
        self.current_mode = mode
        if mode == "RECORD":
            self.mode_record_btn.setChecked(True)
            self.mode_test_btn.setChecked(False)
            self.control_group.setVisible(True)
            self.test_group.setVisible(False)
            self.status_banner.setText("RECORD MODE — Select a sign and click 'Start 3s Recording'")
            self.status_banner.setStyleSheet("background-color: #212631; color: #90CAF9; padding: 10px; border-radius: 8px;")
        else:
            self.mode_record_btn.setChecked(False)
            self.mode_test_btn.setChecked(True)
            self.control_group.setVisible(False)
            self.test_group.setVisible(True)
            self.status_banner.setText("LIVE TEST MODE — Show signs to your webcam to test model accuracy")
            self.status_banner.setStyleSheet("background-color: #1B5E20; color: #A5D6A7; padding: 10px; border-radius: 8px;")

    def _on_class_selected(self, cls_name):
        self.selected_class = cls_name
        self.record_trigger_btn.setText(f"START 3s RECORDING FOR '{cls_name}' (Spacebar)")

    def _add_custom_sign(self):
        custom_name = self.custom_sign_input.text().strip().upper()
        if custom_name:
            if self.class_dropdown.findText(custom_name) == -1:
                self.class_dropdown.addItem(custom_name)
            self.class_dropdown.setCurrentText(custom_name)
            self.custom_sign_input.clear()
            self._refresh_dataset_table()
            self.log(f"Added custom sign class: '{custom_name}'")

    def _on_table_row_clicked(self, item):
        row = item.row()
        cls_name = self.dataset_table.item(row, 0).text()
        self.class_dropdown.setCurrentText(cls_name)

    def _refresh_dataset_table(self):
        # Load meta if available
        base_samples_map = {}
        if CLASS_META_PATH.exists():
            try:
                with open(CLASS_META_PATH, "r", encoding="utf-8") as f:
                    meta = json.load(f)
                avg_base = meta.get("base_samples", 107517) // max(1, meta.get("num_classes", 35))
                for cls_name in BASE_CLASSES:
                    base_samples_map[cls_name] = f"~{avg_base:,}"
            except Exception:
                pass

        # Scan user recorded directory
        user_counts = {}
        for f in USER_RECORDED_DIR.glob("*.npz"):
            cls_name = f.stem
            try:
                data = np.load(str(f))
                user_counts[cls_name] = len(data["features"])
            except Exception:
                user_counts[cls_name] = 0

        # Combine all known classes
        all_classes = list(BASE_CLASSES)
        for cls_name in user_counts.keys():
            if cls_name not in all_classes:
                all_classes.append(cls_name)

        self.dataset_table.setRowCount(len(all_classes))
        for row, cls_name in enumerate(all_classes):
            base_count = base_samples_map.get(cls_name, "0" if cls_name == "NO_SIGN" else "~3,000")
            user_count = user_counts.get(cls_name, 0)
            status = f"✓ Ready ({user_count})" if user_count > 0 else "Pending"

            item_cls = QTableWidgetItem(cls_name)
            item_base = QTableWidgetItem(base_count)
            item_user = QTableWidgetItem(str(user_count))
            item_status = QTableWidgetItem(status)

            if user_count > 0:
                item_status.setForeground(QBrush(QColor("#4CAF50")))
            else:
                item_status.setForeground(QBrush(QColor("#9E9E9E")))

            self.dataset_table.setItem(row, 0, item_cls)
            self.dataset_table.setItem(row, 1, item_base)
            self.dataset_table.setItem(row, 2, item_user)
            self.dataset_table.setItem(row, 3, item_status)

    def _delete_selected_samples(self):
        selected_rows = self.dataset_table.selectionModel().selectedRows()
        if not selected_rows:
            QMessageBox.information(self, "Select Row", "Please select a class row to delete its recorded samples.")
            return

        cls_name = self.dataset_table.item(selected_rows[0].row(), 0).text()
        npz_file = USER_RECORDED_DIR / f"{cls_name}.npz"
        if npz_file.exists():
            npz_file.unlink()
            self.log(f"Deleted user samples for class: '{cls_name}'")
            self._refresh_dataset_table()
        else:
            QMessageBox.information(self, "No Data", f"No custom recorded samples found for '{cls_name}'.")

    # ═══════════════════════════════════════════════════════════
    # 3-Second Preparation -> 3-Second Recording State Machine
    # ═══════════════════════════════════════════════════════════
    def start_countdown(self):
        if self.record_state != "IDLE":
            return

        self.record_state = "COUNTDOWN"
        self.countdown_remaining = 3.0
        self.recording_remaining = 3.0
        self.recorded_features_buffer.clear()
        self.last_timer_tick = time.time()

        self.record_trigger_btn.setEnabled(False)
        self.cancel_btn.setEnabled(True)
        self.log(f"Starting 3s countdown for sign '{self.selected_class}'... Get ready!")

    def cancel_recording(self):
        self.record_state = "IDLE"
        self.recorded_features_buffer.clear()
        self.record_trigger_btn.setEnabled(True)
        self.cancel_btn.setEnabled(False)
        self.action_progress_bar.setValue(0)
        self.action_progress_bar.setFormat("Cancelled")
        self.status_banner.setText("RECORD MODE — Select a sign and click 'Start 3s Recording'")
        self.status_banner.setStyleSheet("background-color: #212631; color: #90CAF9; padding: 10px; border-radius: 8px;")
        self.log("Recording cancelled.")

    def _on_state_tick(self):
        now = time.time()
        dt = now - self.last_timer_tick
        self.last_timer_tick = now

        if self.record_state == "COUNTDOWN":
            self.countdown_remaining -= dt
            pct = max(0, int(((3.0 - self.countdown_remaining) / 3.0) * 100))
            secs = max(0.1, self.countdown_remaining)

            self.action_progress_bar.setValue(pct)
            self.action_progress_bar.setFormat(f"PREPARING SIGN: {secs:.1f}s")
            self.action_progress_bar.setStyleSheet("QProgressBar::chunk { background-color: #E53935; }")

            self.status_banner.setText(f"GET READY FOR '{self.selected_class}' IN {int(np.ceil(secs))} SECONDS...")
            self.status_banner.setStyleSheet("background-color: #B71C1C; color: #FFCDD2; padding: 10px; border-radius: 8px;")

            if self.countdown_remaining <= 0.0:
                # Transition to RECORDING state!
                self.record_state = "RECORDING"
                self.recording_remaining = 3.0
                self.recorded_features_buffer.clear()
                self.log(f"RECORDING NOW: Hold sign '{self.selected_class}' and move hand slightly for angle diversity!")

        elif self.record_state == "RECORDING":
            self.recording_remaining -= dt
            pct = max(0, int(((3.0 - self.recording_remaining) / 3.0) * 100))
            secs = max(0.0, self.recording_remaining)

            self.action_progress_bar.setValue(pct)
            self.action_progress_bar.setFormat(f"RECORDING: {secs:.1f}s ({len(self.recorded_features_buffer)} frames)")
            self.action_progress_bar.setStyleSheet("QProgressBar::chunk { background-color: #2ECC71; }")

            self.status_banner.setText(f"RECORDING '{self.selected_class}' — HOLD STEADY ({secs:.1f}s left)")
            self.status_banner.setStyleSheet("background-color: #1B5E20; color: #A5D6A7; padding: 10px; border-radius: 8px;")

            if self.recording_remaining <= 0.0:
                self._save_recorded_sign()

    def _save_recorded_sign(self):
        self.record_state = "IDLE"
        self.record_trigger_btn.setEnabled(True)
        self.cancel_btn.setEnabled(False)

        count = len(self.recorded_features_buffer)
        if count < 10:
            QMessageBox.warning(self, "Low Sample Count", f"Only {count} valid hand frames detected. Please keep your hand clearly in front of the camera and re-record.")
            self.action_progress_bar.setValue(0)
            self.action_progress_bar.setFormat("Failed: No Hand Detected")
            return

        # Save or Append to user_recorded/<class_name>.npz
        new_features = np.array(self.recorded_features_buffer, dtype=np.float32)
        save_path = USER_RECORDED_DIR / f"{self.selected_class}.npz"
        if save_path.exists():
            try:
                old_data = np.load(str(save_path))
                old_features = old_data["features"] if "features" in old_data else old_data["X"]
                features_array = np.concatenate([old_features, new_features], axis=0)
            except Exception:
                features_array = new_features
        else:
            features_array = new_features

        np.savez_compressed(str(save_path), features=features_array)
        total_samples_for_class = len(features_array)

        self.action_progress_bar.setValue(100)
        self.action_progress_bar.setFormat(f"Saved {count} new frames (Total: {total_samples_for_class}) for '{self.selected_class}'!")
        self.status_banner.setText(f"SUCCESS — '{self.selected_class}' now has {total_samples_for_class} total samples!")
        self.status_banner.setStyleSheet("background-color: #2E7D32; color: #C8E6C9; padding: 10px; border-radius: 8px;")

        self.log(f"SUCCESS: Added {count} frames to '{self.selected_class}' (Accumulated Total: {total_samples_for_class} vectors)")
        self._refresh_dataset_table()

    # ═══════════════════════════════════════════════════════════
    # Live Camera Rendering & HUD Overlays
    # ═══════════════════════════════════════════════════════════
    def _on_frame_ready(self, raw_frame, feat_vec, results):
        is_rec = (self.record_state == "RECORDING")
        display_frame = self.camera_thread.processor.draw_landmarks(raw_frame.copy(), results, is_recording=is_rec)
        h, w, _ = display_frame.shape

        # Buffer feature if recording
        if is_rec and feat_vec is not None:
            self.recorded_features_buffer.append(feat_vec)

        # ── Overlay HUD based on State ──
        if self.record_state == "COUNTDOWN":
            # Red/Amber Preparation Banner & Countdown
            secs_num = int(np.ceil(max(0.1, self.countdown_remaining)))
            cv2.rectangle(display_frame, (0, 0), (w, h), (0, 0, 220), 8) # Red Border

            # Dark semi-transparent box in center
            overlay = display_frame.copy()
            cv2.rectangle(overlay, (w//2 - 160, h//2 - 100), (w//2 + 160, h//2 + 100), (20, 20, 30), -1)
            cv2.addWeighted(overlay, 0.7, display_frame, 0.3, 0, display_frame)

            cv2.putText(display_frame, f"GET READY FOR '{self.selected_class}'", (w//2 - 140, h//2 - 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 220, 255), 2, cv2.LINE_AA)
            cv2.putText(display_frame, str(secs_num), (w//2 - 25, h//2 + 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 2.5, (50, 50, 255), 6, cv2.LINE_AA)

        elif self.record_state == "RECORDING":
            # Bright Emerald Green Recording Frame
            cv2.rectangle(display_frame, (0, 0), (w, h), (46, 204, 113), 10) # Green Border

            # Pulsing Green REC Dot
            cv2.circle(display_frame, (35, 35), 10, (46, 204, 113), -1)
            cv2.putText(display_frame, "REC", (52, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (46, 204, 113), 2, cv2.LINE_AA)

            # Timer & Frame Counter
            timer_text = f"Sign: '{self.selected_class}' | Time: {self.recording_remaining:.1f}s | Frames: {len(self.recorded_features_buffer)}"
            cv2.putText(display_frame, timer_text, (110, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)

        elif self.current_mode == "TEST":
            # Real-Time Inference Mode
            if feat_vec is not None and self.onnx_session:
                input_name = self.onnx_session.get_inputs()[0].name
                inp = feat_vec.reshape(1, -1).astype(np.float32)
                logits = self.onnx_session.run(None, {input_name: inp})[0][0]
                exp_l = np.exp(logits - np.max(logits))
                probs = exp_l / exp_l.sum()

                top_idx = int(np.argmax(probs))
                top_conf = float(probs[top_idx])
                top_cls = self.idx2class.get(top_idx, "?")

                self.test_pred_label.setText(top_cls)
                self.test_conf_bar.setValue(int(top_conf * 100))

                # HUD Overlay on video
                color = (46, 204, 113) if top_conf >= 0.6 else (0, 165, 255)
                cv2.putText(display_frame, f"Sign: {top_cls} ({top_conf:.1%})", (20, 45),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2, cv2.LINE_AA)
            else:
                self.test_pred_label.setText("-")
                self.test_conf_bar.setValue(0)
                cv2.putText(display_frame, "Waiting for hand sign...", (20, 45),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (150, 150, 150), 2, cv2.LINE_AA)

        else:
            # Idle Mode: Hand Presence Indicator
            dot_color = (46, 204, 113) if feat_vec is not None else (0, 0, 220)
            status_txt = "HAND DETECTED — READY" if feat_vec is not None else "NO HAND DETECTED"
            cv2.circle(display_frame, (30, 30), 8, dot_color, -1)
            cv2.putText(display_frame, status_txt, (48, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (220, 220, 220), 1, cv2.LINE_AA)

        # Convert to QPixmap
        rgb = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
        qt_img = QImage(rgb.data, w, h, w * 3, QImage.Format.Format_RGB888).copy()
        pixmap = QPixmap.fromImage(qt_img).scaled(
            self.video_label.size(), Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation
        )
        self.video_label.setPixmap(pixmap)

    # ═══════════════════════════════════════════════════════════
    # Fine-Tuning Execution
    # ═══════════════════════════════════════════════════════════
    def start_fine_tuning(self):
        self.train_btn.setEnabled(False)
        self.train_progress_bar.setValue(0)
        self.log("Initializing CUDA GPU Fine-Tuning Worker...")

        self.worker = FineTuneWorker(epochs=35)
        self.worker.epoch_progress.connect(self._on_epoch_progress)
        self.worker.finished_signal.connect(self._on_fine_tune_finished)
        self.worker.error_signal.connect(self._on_fine_tune_error)
        self.worker.start()

    def _on_epoch_progress(self, epoch, total_epochs, msg, train_acc, val_acc):
        pct = int((epoch / total_epochs) * 100)
        self.train_progress_bar.setValue(pct)
        self.train_progress_bar.setFormat(f"Epoch {epoch}/{total_epochs} | Val Acc: {val_acc:.2%}")
        self.log(msg)

    def _on_fine_tune_finished(self, best_val_acc, meta_info):
        self.train_btn.setEnabled(True)
        self.train_progress_bar.setValue(100)
        self.train_progress_bar.setFormat(f"Trained! Best Val Accuracy: {best_val_acc:.2%}")

        # Reload ONNX session
        self._load_onnx_model()
        self._refresh_dataset_table()

        msg = f"Model fine-tuned successfully on RTX 4050 GPU!\nValidation Accuracy: {best_val_acc:.2%}\nTotal Samples: {meta_info['total_samples']:,}"
        QMessageBox.information(self, "Fine-Tuning Complete", msg)
        self.log(f"SUCCESS: {msg}")

    def _on_fine_tune_error(self, err_msg):
        self.train_btn.setEnabled(True)
        self.train_progress_bar.setValue(0)
        self.train_progress_bar.setFormat("Error")
        QMessageBox.critical(self, "Fine-Tuning Error", f"Training failed: {err_msg}")
        self.log(f"ERROR: {err_msg}")

    def keyPressEvent(self, event):
        if event.key() == Qt.Key.Key_Space:
            if self.record_state == "IDLE" and self.current_mode == "RECORD":
                self.start_countdown()
        elif event.key() == Qt.Key.Key_Escape:
            if self.record_state != "IDLE":
                self.cancel_recording()
        else:
            super().keyPressEvent(event)

    def log(self, text):
        ts = time.strftime("%H:%M:%S")
        self.log_text.append(f"[{ts}] {text}")

    def closeEvent(self, event):
        if self.camera_thread:
            self.camera_thread.stop()
            self.camera_thread.wait(2000)
        event.accept()


# ═══════════════════════════════════════════════════════════════
# Application Entry Point
# ═══════════════════════════════════════════════════════════
def main():
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    window = SignSpeakStudio()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
