import cv2
import time
import math
import numpy as np
import tkinter as tk
import threading
import os
import pickle
import tensorflow as tf
import mediapipe as mp
import ctypes

# Prevent pyautogui from messing with Windows DPI scaling so Tkinter looks big like test_tracking.py
if os.name == 'nt':
    try:
        ctypes.windll.user32.SetProcessDPIAware = lambda: None
    except AttributeError:
        pass

import pyautogui
pyautogui.FAILSAFE = False
pyautogui.PAUSE = 0

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BEST_SKLEARN_PATH = os.path.join(BASE_DIR, "models", "best_sklearn_model.pkl")

# Calibration settings
CALIBRATION_LAYOUT = "16"      # Options: "9", "12", or "16"
CALIBRATION_SAMPLES = 90       # Frames to collect per point
CALIBRATION_SETTLE_FRAMES = 35 # Wait frames before collecting (let the eyes settle)
CALIBRATION_MIN_VALID_SAMPLES = 40
CALIBRATION_MAX_ERROR_PX = 140  # Acceptable average calibration error before fallback
EAR_BLINK_THRESHOLD = 0.18     # Below → blink → skip frame
MIRROR_X = False               # Keep this off because the webcam frame is mirrored before inference to match training data
MIRROR_Y = False               # No vertical mirroring needed

# ==============================================================================
# Model & Feature Extraction
# ==============================================================================
def load_tflite_model(model_path=os.path.join(BASE_DIR, "models", "eye_model.tflite")):
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    return interpreter, interpreter.get_input_details(), interpreter.get_output_details()

def load_best_model():
    with open(os.path.join(BASE_DIR, "models", "feature_scaler.pkl"), "rb") as f:
        scaler = pickle.load(f)

    metadata_path = os.path.join(BASE_DIR, "models", "training_meta.pkl")
    best_model_name = "unknown"
    if os.path.exists(metadata_path):
        with open(metadata_path, "rb") as f:
            metadata = pickle.load(f)
        best_model_name = metadata.get("best_model", "unknown")

    if os.path.exists(BEST_SKLEARN_PATH):
        with open(BEST_SKLEARN_PATH, "rb") as f:
            model = pickle.load(f)
        print(f"[OK] Loaded best sklearn model: {BEST_SKLEARN_PATH} ({best_model_name})")
        return {"type": "sklearn", "model": model, "scaler": scaler}

    interpreter, input_details, output_details = load_tflite_model()
    print(f"[OK] Loaded TFLite model: {os.path.join(BASE_DIR, 'models', 'eye_model.tflite')}")
    return {
        "type": "tflite",
        "model": interpreter,
        "input_details": input_details,
        "output_details": output_details,
        "scaler": scaler,
    }


def predict_model(model_backend, X):
    if model_backend["type"] == "sklearn":
        pred = model_backend["model"].predict(X)
        return pred[0]

    interpreter = model_backend["model"]
    input_details = model_backend["input_details"]
    output_details = model_backend["output_details"]
    interpreter.set_tensor(input_details[0]['index'], X.astype(np.float32))
    interpreter.invoke()
    return interpreter.get_tensor(output_details[0]['index'])[0]

import pickle

# MediaPipe Initialization
options = mp.tasks.vision.FaceLandmarkerOptions(
    base_options=mp.tasks.BaseOptions(model_asset_path=os.path.join(BASE_DIR, "models", "face_landmarker.task")),
    running_mode=mp.tasks.vision.RunningMode.VIDEO,
    output_facial_transformation_matrixes=True
)
landmarker = mp.tasks.vision.FaceLandmarker.create_from_options(options)

# Landmark Indices
LEFT_IRIS, RIGHT_IRIS = [474, 475, 476, 477], [469, 470, 471, 472]
LEFT_EAR_IDX, RIGHT_EAR_IDX = (263, 387, 385, 362, 380, 373), (33, 160, 158, 133, 153, 144)
LEFT_EYE_CONTOUR = [263, 466, 388, 387, 386, 385, 384, 398, 362, 382, 381, 380, 374, 373, 390, 249]
RIGHT_EYE_CONTOUR = [33, 246, 161, 160, 159, 158, 157, 173, 133, 155, 154, 153, 145, 144, 163, 7]
FACE_OVAL = [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136, 172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109]

def get_normalized_center(landmarks, indices):
    return (sum(landmarks[i].x for i in indices)/len(indices), sum(landmarks[i].y for i in indices)/len(indices), sum(landmarks[i].z for i in indices)/len(indices))

def compute_gaze_ratio(landmarks, iris_idx, eye_contour_idx):
    ix, iy, _ = get_normalized_center(landmarks, iris_idx)
    xs, ys = [landmarks[i].x for i in eye_contour_idx], [landmarks[i].y for i in eye_contour_idx]
    hx, hy = (ix - min(xs)) / max(max(xs) - min(xs), 1e-7), (iy - min(ys)) / max(max(ys) - min(ys), 1e-7)
    return float(np.clip(hx, 0, 1)), float(np.clip(hy, 0, 1))

def compute_ear(landmarks, indices):
    pts = [np.array([landmarks[i].x, landmarks[i].y]) for i in indices]
    return float((np.linalg.norm(pts[1] - pts[5]) + np.linalg.norm(pts[2] - pts[4])) / (2.0 * np.linalg.norm(pts[0] - pts[3]) + 1e-7))

def compute_face_area(landmarks):
    pts = np.array([(landmarks[i].x, landmarks[i].y) for i in FACE_OVAL])
    x, y = pts[:, 0], pts[:, 1]
    return float(0.5 * abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1))))

# ==============================================================================
# One-Euro Filter
# ==============================================================================
class OneEuroFilter:
    def __init__(self, min_cutoff=0.7, beta=0.7, d_cutoff=1.0):
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.d_cutoff = d_cutoff
        self.x_prev = None
        self.dx_prev = None
        self.t_prev = None

    def _alpha(self, cutoff, dt):
        if cutoff <= 0.0:
            return 1.0
        tau = 1.0 / cutoff
        return 1.0 / (1.0 + tau / max(dt, 1e-6))

    def update(self, x, t=None):
        if t is None:
            t = time.perf_counter()

        if self.t_prev is None:
            self.t_prev = t
            self.x_prev = x
            self.dx_prev = 0.0
            return x

        dt = max(t - self.t_prev, 1e-6)
        self.t_prev = t

        dx = (x - self.x_prev) / dt if self.x_prev is not None else 0.0
        alpha_dx = self._alpha(self.d_cutoff, dt)
        dx_hat = alpha_dx * dx + (1.0 - alpha_dx) * self.dx_prev

        cutoff = self.min_cutoff + self.beta * abs(dx_hat)
        alpha_x = self._alpha(cutoff, dt)
        x_hat = alpha_x * x + (1.0 - alpha_x) * self.x_prev

        self.x_prev = x_hat
        self.dx_prev = dx_hat
        return x_hat

# ==============================================================================
# Gaze Mouse Controller — identical to GazeVisualizer but moves mouse cursor
# ==============================================================================
class GazeVisualizer:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Gaze AI - Mouse Controller")
        self.root.attributes("-fullscreen", True)
        self.root.configure(background='#0B0F19') # Deep Navy background
        
        self.screen_w = self.root.winfo_screenwidth()
        self.screen_h = self.root.winfo_screenheight()
        
        self.canvas = tk.Canvas(self.root, width=self.screen_w, height=self.screen_h, bg='#0B0F19', highlightthickness=0)
        self.canvas.pack()
        
        # 1. Background Grid
        for i in range(0, self.screen_w, 60):
            self.canvas.create_line(i, 0, i, self.screen_h, fill="#1E293B", dash=(4, 4))
        for i in range(0, self.screen_h, 60):
            self.canvas.create_line(0, i, self.screen_w, i, fill="#1E293B", dash=(4, 4))
            
        # 3. HUD Info Panel
        self.hud_bg = self.canvas.create_rectangle(20, 20, 320, 220, outline="#38BDF8", width=2, fill="#0F172A", stipple="gray50")
        self.hud_title = self.canvas.create_text(170, 45, text="GAZE AI ONLINE", fill="#38BDF8", font=("Consolas", 18, "bold"))
        self.hud_line = self.canvas.create_line(40, 65, 300, 65, fill="#38BDF8")
        
        self.txt_status = self.canvas.create_text(40, 90, anchor="w", text="Status: SEARCHING...", fill="#94A3B8", font=("Consolas", 14))
        self.txt_coords = self.canvas.create_text(40, 120, anchor="w", text="Target: X: 0 | Y: 0", fill="#F8FAFC", font=("Consolas", 14))
        self.txt_pose = self.canvas.create_text(40, 150, anchor="w", text="Pose  : P: 0° | Y: 0°", fill="#F8FAFC", font=("Consolas", 14))
        self.txt_ear = self.canvas.create_text(40, 180, anchor="w", text="EAR   : 0.00", fill="#F8FAFC", font=("Consolas", 14))
        
        self.txt_exit = self.canvas.create_text(self.screen_w/2, self.screen_h - 40, text="PRESS [ ESC ] TO INITIATE SHUTDOWN", fill="#64748B", font=("Consolas", 14, "bold"))
        
        # 4. Crosshair Elements (kept for consistency but won't be visible after overlay hides)
        self.trail_positions = []
        self.trail_dots = []
        for _ in range(8):
            self.trail_dots.append(self.canvas.create_oval(-10,-10,-10,-10, fill="#0284C7", outline=""))
            
        self.crosshair_outer = self.canvas.create_oval(-10,-10,-10,-10, outline="#38BDF8", width=2)
        self.crosshair_inner = self.canvas.create_oval(-10,-10,-10,-10, fill="#38BDF8", outline="")
        self.crosshair_h = self.canvas.create_line(-10,-10,-10,-10, fill="#38BDF8", width=2)
        self.crosshair_v = self.canvas.create_line(-10,-10,-10,-10, fill="#38BDF8", width=2)
        
        self.root.bind("<Escape>", self.quit)
        
        self.running = True
        self.target_x, self.target_y = self.screen_w / 2, self.screen_h / 2
        self.current_x, self.current_y = self.screen_w / 2, self.screen_h / 2
        self.smoothed_target_x = self.screen_w / 2
        self.smoothed_target_y = self.screen_h / 2
        
        # One-Euro Filter with heavier smoothing (min_cutoff=0.02) and less motion responsiveness (beta=0.1) for smoother tracking
        self.one_euro_filter_x = OneEuroFilter(min_cutoff=0.02, beta=0.1)
        self.one_euro_filter_y = OneEuroFilter(min_cutoff=0.02, beta=0.1)
        
        # Fixation Deadzone Lock state (prevents jitter while staring)
        self.fixation_anchor_x = self.screen_w / 2
        self.fixation_anchor_y = self.screen_h / 2
        
        # Real-time state
        self.face_detected = False
        self.pitch, self.yaw, self.ear = 0, 0, 0
        self.raw_pred_x, self.raw_pred_y = 0.5, 0.5
        self.calibrated = False
        self.calibration_matrix = np.eye(3)
        self.axis_map = None
        self.polynomial_map = None
        self.calibration_layout = CALIBRATION_LAYOUT
        self.calibration_points = self._build_calibration_points()
        self.calibration_range_min = np.array([0.0, 0.0], dtype=np.float32)
        self.calibration_range_scale = np.array([1.0, 1.0], dtype=np.float32)
        self.calibration_target = self.canvas.create_oval(-20, -20, -20, -20, outline="#F59E0B", width=4)
        self.calibration_label = self.canvas.create_text(self.screen_w/2, 60, text="", fill="#FBBF24", font=("Consolas", 18, "bold"))
        self.calibration_countdown = self.canvas.create_text(self.screen_w/2, 100, text="", fill="#A7F3D0", font=("Consolas", 16, "bold"))
        
        # Dwell to click state
        self.dwell_start_time = time.time()
        self.dwell_x = self.screen_w / 2
        self.dwell_y = self.screen_h / 2

        print("Loading Model...")
        self.model_backend = load_best_model()
        self.scaler = self.model_backend["scaler"]
        
        # Start Threads
        self.cam_thread = threading.Thread(target=self.camera_loop, daemon=True)
        self.cam_thread.start()
        
        self.calibrate()

        # ---- MOUSE MODE: hide the overlay after calibration ----
        self.root.withdraw()
        print("\n" + "=" * 50)
        print("  GAZE -> MOUSE  *  TRACKING ACTIVE")
        print("  Press ESC at any time to quit")
        print("=" * 50 + "\n")

        self.update_ui()
        self.root.mainloop()

    def update_ui(self):
        # ---- MOUSE MODE: check for ESC globally via Win32 API ----
        # VK_ESCAPE = 0x1B.  GetAsyncKeyState returns high bit set if key is down.
        if ctypes.windll.user32.GetAsyncKeyState(0x1B) & 0x8000:
            self.quit()
            return

        # Smooth lerp (0.10) to glide cleanly to fixation targets (lowered for more smoothing)
        smoothing = 0.10
        self.current_x += (self.target_x - self.current_x) * smoothing
        self.current_y += (self.target_y - self.current_y) * smoothing
        
        self.trail_positions.append((self.current_x, self.current_y))
        if len(self.trail_positions) > 8:
            self.trail_positions.pop(0)
            
        for i, (tx, ty) in enumerate(self.trail_positions):
            r = (i + 1) * 1.2
            self.canvas.coords(self.trail_dots[i], tx - r, ty - r, tx + r, ty + r)
            
        cx, cy = self.current_x, self.current_y
        r_out, r_in = 25, 4
        self.canvas.coords(self.crosshair_outer, cx - r_out, cy - r_out, cx + r_out, cy + r_out)
        self.canvas.coords(self.crosshair_inner, cx - r_in, cy - r_in, cx + r_in, cy + r_in)
        self.canvas.coords(self.crosshair_h, cx - 35, cy, cx + 35, cy)
        self.canvas.coords(self.crosshair_v, cx, cy - 35, cx, cy + 35)

        # ---- MOUSE MODE: move the mouse cursor instead of just the crosshair ----
        mx = max(0, min(int(cx), self.screen_w - 1))
        my = max(0, min(int(cy), self.screen_h - 1))
        pyautogui.moveTo(mx, my, _pause=False)
        
        # ---- MOUSE MODE: check for DWELL to click ----
        if self.calibrated and self.face_detected:
            dist = math.hypot(mx - self.dwell_x, my - self.dwell_y)
            DWELL_RADIUS = 75  # Stare within this pixel radius (approx size of a large icon)
            DWELL_TIME = 3.0   # 3 seconds to trigger click

            if dist > DWELL_RADIUS:
                # Gaze broke out of the dwell radius, reset anchor and timer
                self.dwell_x = mx
                self.dwell_y = my
                self.dwell_start_time = time.time()
            else:
                # We are dwelling...
                duration = time.time() - self.dwell_start_time
                if duration >= DWELL_TIME:
                    pyautogui.click(mx, my)
                    print(f"[*] Dwell click triggered at {mx}, {my} (stared for {DWELL_TIME}s)")
                    # Reset timer so it doesn't double-click immediately
                    self.dwell_start_time = time.time()
        
        if self.face_detected:
            self.canvas.itemconfig(self.txt_status, text="Status: TRACKING ACTIVE", fill="#34D399")
            self.canvas.itemconfig(self.txt_coords, text=f"Mouse: X: {mx:04d} | Y: {my:04d}")
            self.canvas.itemconfig(self.txt_pose, text=f"Pose  : P: {self.pitch:02.0f}° | Y: {self.yaw:02.0f}°")
            self.canvas.itemconfig(self.txt_ear, text=f"EAR   : {self.ear:.2f}")
        else:
            self.canvas.itemconfig(self.txt_status, text="Status: FACE NOT FOUND", fill="#F87171")
            
        if self.running:
            self.root.after(16, self.update_ui)

    def _build_calibration_points(self):
        if self.calibration_layout == "16":
            cols, rows = 4, 4
        elif self.calibration_layout == "12":
            cols, rows = 4, 3
        else:
            cols, rows = 3, 3

        padding = 0.06
        xs = np.linspace(padding, 1.0 - padding, cols)
        ys = np.linspace(padding, 1.0 - padding, rows)
        points = [(float(x), float(y)) for y in ys for x in xs]

        # Ensure the four corners are explicitly included in the 16-point grid.
        if self.calibration_layout == "16":
            corners = [(padding, padding), (padding, 1.0 - padding), (1.0 - padding, padding), (1.0 - padding, 1.0 - padding)]
            points = [p for p in points if p not in corners] + corners

        return points

    def calibrate(self):
        screen_points = []
        raw_points = []

        self.canvas.itemconfig(self.txt_status, text="Status: CALIBRATING...", fill="#FBBF24")
        self.canvas.itemconfig(self.txt_coords, text="Look at the moving dot and hold still")
        self.root.update()

        for index, (x_norm, y_norm) in enumerate(self.calibration_points, start=1):
            target_x = x_norm * self.screen_w
            target_y = y_norm * self.screen_h
            self._show_calibration_target(target_x, target_y, index)
            self._wait_frames(CALIBRATION_SETTLE_FRAMES)

            samples = []
            while len(samples) < CALIBRATION_SAMPLES and self.running:
                remaining = CALIBRATION_SAMPLES - len(samples)
                self.canvas.itemconfig(self.calibration_countdown, text=f"Sampling: {remaining} frames")
                self.root.update()
                self._update_calibration_target_color(x_norm, y_norm)
                if self.face_detected and self.ear > EAR_BLINK_THRESHOLD:
                    samples.append((self.raw_pred_x, self.raw_pred_y))
                time.sleep(0.02)
            self.canvas.itemconfig(self.calibration_countdown, text="")

            if len(samples) >= CALIBRATION_MIN_VALID_SAMPLES:
                sample_array = np.array(samples, dtype=np.float32)
                sample_array = sample_array[np.isfinite(sample_array).all(axis=1)]
                if len(sample_array) >= CALIBRATION_MIN_VALID_SAMPLES:
                    raw_points.append(np.median(sample_array, axis=0))
                    screen_points.append((target_x, target_y))
                else:
                    print(f"[WARN] Calibration point skipped at {x_norm:.2f},{y_norm:.2f} due to poor tracking")
            else:
                print(f"[WARN] Calibration point skipped at {x_norm:.2f},{y_norm:.2f} due to poor tracking")

            self._hide_calibration_target()
            time.sleep(0.1)

        if len(raw_points) >= 3:
            raw_points = np.array(raw_points, dtype=np.float32)
            screen_points = np.array(screen_points, dtype=np.float32)
            affine_matrix = self._fit_affine_mapping(raw_points, screen_points)
            affine_error = self._compute_avg_error(raw_points, screen_points, affine_matrix)
            axis_map = self._fit_axis_mapping(raw_points, screen_points)
            axis_error = self._compute_axis_error(raw_points, screen_points, axis_map)
            polynomial_map = self._fit_polynomial_mapping(raw_points, screen_points)
            polynomial_error = self._compute_polynomial_error(raw_points, screen_points, polynomial_map)

            if polynomial_error + 1e-6 < min(axis_error, affine_error) and polynomial_error < CALIBRATION_MAX_ERROR_PX:
                self.polynomial_map = polynomial_map
                self.axis_map = None
                self.calibration_matrix = np.eye(3)
                self.calibrated = True
                self._set_calibration_range(raw_points)
                print(f"[OK] Calibration complete using quadratic mapping (error={polynomial_error:.1f}px)")
            elif axis_error + 1e-6 < affine_error or affine_error > CALIBRATION_MAX_ERROR_PX:
                self.axis_map = axis_map
                self.polynomial_map = None
                self.calibration_matrix = np.eye(3)
                self.calibrated = True
                self._set_calibration_range(raw_points)
                print(f"[OK] Calibration complete using axis-aligned mapping (error={axis_error:.1f}px)")
            else:
                self.calibration_matrix = affine_matrix
                self.axis_map = None
                self.polynomial_map = None
                self.calibrated = True
                self._set_calibration_range(raw_points)
                print(f"[OK] Calibration complete using affine mapping (error={affine_error:.1f}px)")
        else:
            print("[WARN] Not enough calibration samples collected; using default mapping")

        self.canvas.itemconfig(self.txt_status, text="Status: TRACKING ACTIVE", fill="#34D399")
        self.canvas.itemconfig(self.txt_coords, text="Calibration complete")
        self._hide_calibration_target()
        self.root.update()

    def _show_calibration_target(self, x, y, index):
        size = 32
        self.canvas.coords(self.calibration_target, x - size, y - size, x + size, y + size)
        self.canvas.itemconfig(self.calibration_label, text=f"Calibration {index}/{len(self.calibration_points)}")
        self.root.update()

    def _hide_calibration_target(self):
        self.canvas.coords(self.calibration_target, -20, -20, -20, -20)
        self.canvas.itemconfig(self.calibration_label, text="")
        self.canvas.itemconfig(self.calibration_countdown, text="")
        self.root.update()

    def _update_calibration_target_color(self, target_norm_x, target_norm_y):
        color = "#F59E0B"
        if self.face_detected:
            distance = np.hypot(self.raw_pred_x - target_norm_x, self.raw_pred_y - target_norm_y)
            if distance < 0.08:
                color = "#22C55E"
        self.canvas.itemconfig(self.calibration_target, outline=color)

    def _wait_frames(self, count):
        for _ in range(count):
            self.root.update()
            time.sleep(0.02)

    def _fit_affine_mapping(self, raw_pts, screen_pts):
        ones = np.ones((raw_pts.shape[0], 1), dtype=np.float32)
        raw_aug = np.hstack([raw_pts, ones])
        solution, *_ = np.linalg.lstsq(raw_aug, screen_pts, rcond=None)
        matrix = np.vstack([solution.T, [0.0, 0.0, 1.0]])
        return matrix

    def _apply_calibration(self, raw_x, raw_y):
        # The calibration fit is already expressed in screen-pixel space.
        # Apply it directly to the normalized predictions and clip to the screen.
        if self.polynomial_map is not None:
            mapped = self._apply_polynomial_mapping(raw_x, raw_y)
        elif self.axis_map is not None:
            mapped = self._apply_axis_mapping(raw_x, raw_y)
        else:
            mapped = self.calibration_matrix.dot(np.array([raw_x, raw_y, 1.0], dtype=np.float32))[:2]

        mapped = np.clip(mapped, 0.0, np.array([self.screen_w, self.screen_h], dtype=np.float32))
        return float(mapped[0]), float(mapped[1])

    def _set_calibration_range(self, raw_pts):
        mapped = np.array([self._raw_calibration_map(x, y) for x, y in raw_pts], dtype=np.float32)
        min_mapped = np.min(mapped, axis=0)
        max_mapped = np.max(mapped, axis=0)
        range_vals = max_mapped - min_mapped
        range_vals = np.where(range_vals < 1e-3, 1.0, range_vals)
        self.calibration_range_min = min_mapped
        self.calibration_range_scale = np.array([1.0, 1.0], dtype=np.float32)
        print(f"[DEBUG] Calibration mapped range: x={min_mapped[0]:.1f}..{max_mapped[0]:.1f}, y={min_mapped[1]:.1f}..{max_mapped[1]:.1f}")

    def _raw_calibration_map(self, raw_x, raw_y):
        if self.polynomial_map is not None:
            return np.array(self._apply_polynomial_mapping(raw_x, raw_y), dtype=np.float32)
        if self.axis_map is not None:
            return np.array(self._apply_axis_mapping(raw_x, raw_y), dtype=np.float32)
        mapped = self.calibration_matrix.dot(np.array([raw_x, raw_y, 1.0], dtype=np.float32))
        return mapped[:2]

    def _fit_axis_mapping(self, raw_pts, screen_pts):
        x_coeffs = np.polyfit(raw_pts[:, 0], screen_pts[:, 0], deg=1)
        y_coeffs = np.polyfit(raw_pts[:, 1], screen_pts[:, 1], deg=1)
        return np.stack([x_coeffs, y_coeffs])

    def _apply_axis_mapping(self, raw_x, raw_y):
        x_scale, x_bias = self.axis_map[0]
        y_scale, y_bias = self.axis_map[1]
        return float(raw_x * x_scale + x_bias), float(raw_y * y_scale + y_bias)

    def _fit_polynomial_mapping(self, raw_pts, screen_pts):
        def solve_axis(raw_vals, target_vals):
            design = np.column_stack([
                np.ones(len(raw_vals)),
                raw_vals[:, 0],
                raw_vals[:, 1],
                raw_vals[:, 0] * raw_vals[:, 1],
                raw_vals[:, 0] ** 2,
                raw_vals[:, 1] ** 2,
            ])
            coeffs, *_ = np.linalg.lstsq(design, target_vals, rcond=None)
            return coeffs

        x_coeffs = solve_axis(raw_pts, screen_pts[:, 0])
        y_coeffs = solve_axis(raw_pts, screen_pts[:, 1])
        return np.stack([x_coeffs, y_coeffs])

    def _apply_polynomial_mapping(self, raw_x, raw_y):
        coeffs = self.polynomial_map
        if coeffs is None:
            return float(raw_x), float(raw_y)

        x_coeffs = coeffs[0]
        y_coeffs = coeffs[1]
        x_val = (
            x_coeffs[0]
            + x_coeffs[1] * raw_x
            + x_coeffs[2] * raw_y
            + x_coeffs[3] * raw_x * raw_y
            + x_coeffs[4] * raw_x ** 2
            + x_coeffs[5] * raw_y ** 2
        )
        y_val = (
            y_coeffs[0]
            + y_coeffs[1] * raw_x
            + y_coeffs[2] * raw_y
            + y_coeffs[3] * raw_x * raw_y
            + y_coeffs[4] * raw_x ** 2
            + y_coeffs[5] * raw_y ** 2
        )
        return float(x_val), float(y_val)

    def _compute_polynomial_error(self, raw_pts, screen_pts, polynomial_map):
        mapped = np.array([self._apply_polynomial_mapping(x, y) for x, y in raw_pts], dtype=np.float32)
        return float(np.mean(np.linalg.norm(mapped - screen_pts, axis=1)))

    def _compute_avg_error(self, raw_pts, screen_pts, matrix):
        mapped = np.dot(np.hstack([raw_pts, np.ones((raw_pts.shape[0], 1), dtype=np.float32)]), matrix.T)
        mapped_xy = mapped[:, :2]
        return float(np.mean(np.linalg.norm(mapped_xy - screen_pts, axis=1)))

    def _compute_axis_error(self, raw_pts, screen_pts, axis_map):
        x_scale, x_bias = axis_map[0]
        y_scale, y_bias = axis_map[1]
        mapped_x = raw_pts[:, 0] * x_scale + x_bias
        mapped_y = raw_pts[:, 1] * y_scale + y_bias
        return float(np.mean(np.linalg.norm(np.vstack([mapped_x, mapped_y]).T - screen_pts, axis=1)))

    def camera_loop(self):
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        while self.running:
            success, frame = cap.read()
            if not success: continue
            
            # Match the training collector's preprocessing: mirror the webcam frame
            # before feature extraction so the model sees the same geometry it learned from.
            frame = cv2.flip(frame, 1)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
            res = landmarker.detect_for_video(mp_image, int(time.time() * 1000))
            
            if not res.face_landmarks:
                self.face_detected = False
                continue
            
            self.face_detected = True
            lm, mat = res.face_landmarks[0], res.facial_transformation_matrixes[0]
            
            pitch, yaw, roll = 0, 0, 0
            if mat is not None:
                pose_mat = mat[:3, :3]
                sy = np.sqrt(pose_mat[0,0]**2 + pose_mat[1,0]**2)
                if sy > 1e-6:
                    x, y, z = np.arctan2(pose_mat[2,1], pose_mat[2,2]), np.arctan2(-pose_mat[2,0], sy), np.arctan2(pose_mat[1,0], pose_mat[0,0])
                else:
                    x, y, z = np.arctan2(-pose_mat[1,2], pose_mat[1,1]), np.arctan2(-pose_mat[2,0], sy), 0
                pitch, yaw, roll = np.degrees(x), np.degrees(y), np.degrees(z)

            lx, ly, lz = get_normalized_center(lm, LEFT_IRIS)
            rx, ry, rz = get_normalized_center(lm, RIGHT_IRIS)
            iod = np.sqrt((lx - rx)**2 + (ly - ry)**2 + (lz - rz)**2)
            lgh, lgv = compute_gaze_ratio(lm, LEFT_IRIS, LEFT_EYE_CONTOUR)
            rgh, rgv = compute_gaze_ratio(lm, RIGHT_IRIS, RIGHT_EYE_CONTOUR)
            l_ear, r_ear = compute_ear(lm, LEFT_EAR_IDX), compute_ear(lm, RIGHT_EAR_IDX)
            face_area = compute_face_area(lm)

            self.pitch, self.yaw, self.ear = pitch, yaw, (l_ear + r_ear)/2

            avg_ix, avg_iy, avg_iz = (lx+rx)/2, (ly+ry)/2, (lz+rz)/2
            avg_gh, avg_gv = (lgh+rgh)/2, (lgv+rgv)/2
            
            features = [
                pitch, yaw, roll, lx, ly, lz, rx, ry, rz, iod, lgh, lgv, rgh, rgv,
                l_ear, r_ear, face_area, avg_ix, avg_iy, avg_iz, lx-rx, ly-ry,
                avg_gh, avg_gv, lgh-rgh, lgv-rgv, (l_ear+r_ear)/2, l_ear-r_ear,
                yaw*avg_ix, pitch*avg_iy, yaw*avg_gh, pitch*avg_gv,
                avg_ix/max(iod, 1e-7), avg_iy/max(iod, 1e-7),
                avg_ix**2, avg_iy**2, yaw**2, pitch**2
            ]

            X_scaled = np.nan_to_num(self.scaler.transform(np.array(features).reshape(1, -1)), nan=0.0, posinf=0.0, neginf=0.0)
            pred = predict_model(self.model_backend, X_scaled)
            
            # ---------------------------------------------------------
            # Use the model output directly and let calibration map it.
            # The calibration fit should be the only coordinate transform.
            # ---------------------------------------------------------
            raw_x, raw_y = pred[0], pred[1]
            
            # The frame is already mirrored before inference to match training data.
            # Keep the model output in the same coordinate convention.
            if MIRROR_X:
                raw_x = 1.0 - raw_x
            if MIRROR_Y:
                raw_y = 1.0 - raw_y

            norm_x = np.clip(raw_x, 0.0, 1.0)
            norm_y = np.clip(raw_y, 0.0, 1.0)

            self.raw_pred_x, self.raw_pred_y = norm_x, norm_y
            predicted_x, predicted_y = self._apply_calibration(norm_x, norm_y) if self.calibrated else (norm_x * self.screen_w, norm_y * self.screen_h)

            # Apply One-Euro filter for smooth and responsive tracking
            t_now = time.perf_counter()
            euro_x = self.one_euro_filter_x.update(predicted_x, t_now)
            euro_y = self.one_euro_filter_y.update(predicted_y, t_now)
            
            # ---------------------------------------------------------
            # Fixation Deadzone Lock: Freeze position when staring at a target
            # ---------------------------------------------------------
            dist_to_anchor = math.hypot(euro_x - self.fixation_anchor_x, euro_y - self.fixation_anchor_y)
            DEADZONE_RADIUS = 35.0  # Pixels: micro-jitter within this circle is completely locked

            if dist_to_anchor > DEADZONE_RADIUS:
                # Gaze moved to a new focus area - update fixation anchor
                self.fixation_anchor_x = euro_x
                self.fixation_anchor_y = euro_y

            # Target position is set to the stable fixation anchor
            self.target_x = self.fixation_anchor_x
            self.target_y = self.fixation_anchor_y
            
            # Constraint
            self.target_x = max(0, min(self.screen_w, self.target_x))
            self.target_y = max(0, min(self.screen_h, self.target_y))
            
        cap.release()

    def quit(self, event=None):
        self.running = False
        self.root.destroy()
        os._exit(0)

if __name__ == "__main__":
    app = GazeVisualizer()
