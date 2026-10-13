"""
Real-Time Gaze Mouse Controller (v2 — with Calibration)
=========================================================
Uses the trained gaze estimation model + per-session affine calibration
to move the mouse cursor based on where you're looking.

Flow:
  1. Load model + camera
  2. Calibration phase: user looks at 9 on-screen dots
  3. Compute affine correction matrix from calibration data
  4. Live tracking with corrected predictions

Usage:
    python src/gaze_mouse.py                 # Preview mode (default)
    python src/gaze_mouse.py --move-mouse    # Actually control cursor
    python src/gaze_mouse.py --use-keras     # Use Keras instead of TFLite
"""

import argparse
import math
import os
import pickle
import sys
import time
from collections import deque

import cv2
import mediapipe as mp_lib
import numpy as np
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(BASE_DIR, "models")
LANDMARKER_PATH = os.path.join(MODEL_DIR, "face_landmarker.task")
TFLITE_PATH = os.path.join(MODEL_DIR, "eye_model.tflite")
KERAS_PATH = os.path.join(MODEL_DIR, "gaze_model.keras")
BEST_SKLEARN_PATH = os.path.join(MODEL_DIR, "best_sklearn_model.pkl")
SCALER_PATH = os.path.join(MODEL_DIR, "feature_scaler.pkl")

# ---------------------------------------------------------------------------
# MediaPipe landmark indices (must match web_collector.py exactly)
# ---------------------------------------------------------------------------
LEFT_IRIS = [474, 475, 476, 477]
RIGHT_IRIS = [469, 470, 471, 472]

LEFT_EAR_IDX = (263, 387, 385, 362, 380, 373)
RIGHT_EAR_IDX = (33, 160, 158, 133, 153, 144)

# Full eye contour (for robust bounding box — matches web_collector.py)
LEFT_EYE_CONTOUR = [263, 466, 388, 387, 386, 385, 384, 398, 362, 382, 381, 380, 374, 373, 390, 249]
RIGHT_EYE_CONTOUR = [33, 246, 161, 160, 159, 158, 157, 173, 133, 155, 154, 153, 145, 144, 163, 7]

FACE_OVAL = [
    10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288,
    397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136,
    172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109,
]

# ---------------------------------------------------------------------------
# Tuning constants
# ---------------------------------------------------------------------------
SMOOTHING_ALPHA = 0.35         # EMA weight (higher = more responsive, jittery)
SMOOTHING_WINDOW = 6           # Rolling avg window
EAR_BLINK_THRESHOLD = 0.18    # Below → blink → freeze cursor

# Calibration settings
CALIBRATION_POINTS = 9         # 3x3 grid
CALIBRATION_SAMPLES = 40       # Frames to collect per point
CALIBRATION_SETTLE_FRAMES = 20 # Wait frames before collecting (let eyes settle)


# ===================================================================
# Feature extraction (mirrors web_collector.py + train_gaze_model.py)
# ===================================================================
def extract_head_pose(transformation_matrix):
    r_mat = transformation_matrix[:3, :3]
    angles = cv2.RQDecomp3x3(r_mat)[0]
    return float(angles[0]), float(angles[1]), float(angles[2])


def get_normalized_center(landmarks, indices):
    n = len(indices)
    x = sum(landmarks[i].x for i in indices) / n
    y = sum(landmarks[i].y for i in indices) / n
    z = sum(landmarks[i].z for i in indices) / n
    return x, y, z


def compute_ear(landmarks, idx):
    pts = [np.array([landmarks[i].x, landmarks[i].y]) for i in idx]
    horiz = np.linalg.norm(pts[0] - pts[3])
    if horiz < 1e-7:
        return 0.0
    v1 = np.linalg.norm(pts[1] - pts[5])
    v2 = np.linalg.norm(pts[2] - pts[4])
    return float((v1 + v2) / (2.0 * horiz))


def compute_gaze_ratio(landmarks, iris_idx, eye_contour_idx):
    """Normalised iris position within the eye contour bounding box."""
    ix = sum(landmarks[i].x for i in iris_idx) / len(iris_idx)
    iy = sum(landmarks[i].y for i in iris_idx) / len(iris_idx)
    xs = [landmarks[i].x for i in eye_contour_idx]
    ys = [landmarks[i].y for i in eye_contour_idx]
    x_lo, x_hi = min(xs), max(xs)
    y_lo, y_hi = min(ys), max(ys)
    h = (ix - x_lo) / max(x_hi - x_lo, 1e-7)
    v = (iy - y_lo) / max(y_hi - y_lo, 1e-7)
    return float(np.clip(h, 0, 1)), float(np.clip(v, 0, 1))


def compute_face_area(landmarks):
    pts = np.array([(landmarks[i].x, landmarks[i].y) for i in FACE_OVAL])
    x, y = pts[:, 0], pts[:, 1]
    return float(0.5 * abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1))))


def extract_features(landmarks, transformation_matrix):
    """Extract the full 38-element feature vector (matching training pipeline)."""
    pitch, yaw, roll = extract_head_pose(transformation_matrix)
    lx, ly, lz = get_normalized_center(landmarks, LEFT_IRIS)
    rx, ry, rz = get_normalized_center(landmarks, RIGHT_IRIS)
    iod = math.sqrt((lx - rx)**2 + (ly - ry)**2 + (lz - rz)**2)

    l_gaze_h, l_gaze_v = compute_gaze_ratio(landmarks, LEFT_IRIS, LEFT_EYE_CONTOUR)
    r_gaze_h, r_gaze_v = compute_gaze_ratio(landmarks, RIGHT_IRIS, RIGHT_EYE_CONTOUR)

    l_ear = compute_ear(landmarks, LEFT_EAR_IDX)
    r_ear = compute_ear(landmarks, RIGHT_EAR_IDX)
    face_area = compute_face_area(landmarks)

    # 17 raw features
    raw = [
        pitch, yaw, roll,
        lx, ly, lz,
        rx, ry, rz,
        iod,
        l_gaze_h, l_gaze_v,
        r_gaze_h, r_gaze_v,
        l_ear, r_ear,
        face_area,
    ]

    # 21 engineered features (must match train_gaze_model.py exactly)
    avg_x = (lx + rx) / 2
    avg_y = (ly + ry) / 2
    avg_z = (lz + rz) / 2
    diff_x = lx - rx
    diff_y = ly - ry
    avg_gh = (l_gaze_h + r_gaze_h) / 2
    avg_gv = (l_gaze_v + r_gaze_v) / 2
    diff_gh = l_gaze_h - r_gaze_h
    diff_gv = l_gaze_v - r_gaze_v
    avg_ear = (l_ear + r_ear) / 2
    ear_diff = l_ear - r_ear
    iod_safe = iod if iod > 1e-6 else 1e-6

    engineered = [
        avg_x, avg_y, avg_z,
        diff_x, diff_y,
        avg_gh, avg_gv, diff_gh, diff_gv,
        avg_ear, ear_diff,
        yaw * avg_x,
        pitch * avg_y,
        yaw * avg_gh,
        pitch * avg_gv,
        avg_x / iod_safe,
        avg_y / iod_safe,
        avg_x ** 2,
        avg_y ** 2,
        yaw ** 2,
        pitch ** 2,
    ]

    return np.array(raw + engineered, dtype=np.float32), l_ear, r_ear


# ===================================================================
# Model loading & inference
# ===================================================================
def load_model(use_keras=False):
    import tensorflow as tf

    with open(SCALER_PATH, "rb") as f:
        scaler = pickle.load(f)

    if use_keras and os.path.exists(KERAS_PATH):
        model = tf.keras.models.load_model(KERAS_PATH, compile=False)
        print(f"[OK] Loaded Keras model: {KERAS_PATH}")
        return {"type": "keras", "model": model, "scaler": scaler}

    if os.path.exists(BEST_SKLEARN_PATH):
        with open(BEST_SKLEARN_PATH, "rb") as f:
            model = pickle.load(f)
        print(f"[OK] Loaded best sklearn model: {BEST_SKLEARN_PATH}")
        return {"type": "sklearn", "model": model, "scaler": scaler}

    if os.path.exists(TFLITE_PATH):
        interpreter = tf.lite.Interpreter(model_path=TFLITE_PATH)
        interpreter.allocate_tensors()
        input_details = interpreter.get_input_details()[0]
        output_details = interpreter.get_output_details()[0]
        print(f"[OK] Loaded TFLite model: {TFLITE_PATH}")
        return {
            "type": "tflite",
            "model": interpreter,
            "input_details": input_details,
            "output_details": output_details,
            "scaler": scaler,
        }

    print("[ERROR] No model found!")
    sys.exit(1)


def predict_gaze(model_backend, features):
    scaler = model_backend["scaler"]
    X = scaler.transform(features.reshape(1, -1))
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

    if model_backend["type"] == "keras":
        pred = model_backend["model"].predict(X, verbose=0)
        return float(np.clip(pred[0][0], 0, 1)), float(np.clip(pred[0][1], 0, 1))

    if model_backend["type"] == "sklearn":
        pred = model_backend["model"].predict(X)
        return float(np.clip(pred[0][0], 0, 1)), float(np.clip(pred[0][1], 0, 1))

    interpreter = model_backend["model"]
    input_details = model_backend["input_details"]
    output_details = model_backend["output_details"]
    interpreter.set_tensor(input_details["index"], X)
    interpreter.invoke()
    pred = interpreter.get_tensor(output_details["index"])
    return float(np.clip(pred[0][0], 0, 1)), float(np.clip(pred[0][1], 0, 1))


# ===================================================================
# Affine calibration
# ===================================================================
def compute_affine_correction(raw_predictions, target_positions):
    """Compute a 2D affine transform that maps raw model predictions to
    actual screen positions.  Solves:  target = A @ [pred_x, pred_y, 1]^T

    This corrects for: offset, scaling, rotation, mirroring, and skew.
    """
    n = len(raw_predictions)
    assert n == len(target_positions) and n >= 3

    # Build matrices  [pred_x, pred_y, 1] -> target_x/y
    A_mat = np.zeros((2 * n, 6))
    b_vec = np.zeros(2 * n)
    for i, (pred, tgt) in enumerate(zip(raw_predictions, target_positions)):
        px, py = pred
        tx, ty = tgt
        A_mat[2 * i]     = [px, py, 1, 0, 0, 0]
        A_mat[2 * i + 1] = [0, 0, 0, px, py, 1]
        b_vec[2 * i]     = tx
        b_vec[2 * i + 1] = ty

    # Least-squares solve
    result, _, _, _ = np.linalg.lstsq(A_mat, b_vec, rcond=None)
    # result = [a, b, c, d, e, f]  where:
    #   corrected_x = a*px + b*py + c
    #   corrected_y = d*px + e*py + f
    transform = result.reshape(2, 3)
    return transform


def apply_affine(transform, pred_x, pred_y):
    """Apply the affine correction."""
    vec = np.array([pred_x, pred_y, 1.0])
    corrected = transform @ vec
    return float(corrected[0]), float(corrected[1])


# ===================================================================
# Cursor smoothing
# ===================================================================
class GazeSmoother:
    def __init__(self, window=SMOOTHING_WINDOW, alpha=SMOOTHING_ALPHA):
        self.history_x = deque(maxlen=window)
        self.history_y = deque(maxlen=window)
        self.alpha = alpha
        self.prev_x = 0.5
        self.prev_y = 0.5

    def update(self, raw_x, raw_y):
        sx = self.alpha * raw_x + (1 - self.alpha) * self.prev_x
        sy = self.alpha * raw_y + (1 - self.alpha) * self.prev_y
        self.history_x.append(sx)
        self.history_y.append(sy)
        avg_x = float(np.mean(self.history_x))
        avg_y = float(np.mean(self.history_y))
        self.prev_x = avg_x
        self.prev_y = avg_y
        return avg_x, avg_y

    def reset(self):
        self.history_x.clear()
        self.history_y.clear()
        self.prev_x = 0.5
        self.prev_y = 0.5


# ===================================================================
# Calibration phase (fullscreen)
# ===================================================================
def run_calibration(model_backend, landmarker, cap, screen_w, screen_h, start_time=None):
    """Show calibration dots and collect raw predictions for each."""
    # Generate 3x3 grid of calibration points
    padding = 80
    cols, rows = 3, 3
    cal_targets = []
    for r in range(rows):
        for c in range(cols):
            x = padding + (screen_w - 2 * padding) * c / (cols - 1)
            y = padding + (screen_h - 2 * padding) * r / (rows - 1)
            cal_targets.append((x, y))

    window_name = "Gaze Calibration"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    if start_time is None:
        start_time = time.perf_counter()
    raw_predictions = []   # Per-point average raw model prediction
    target_positions = []  # Normalised target positions

    for pt_idx, (tx, ty) in enumerate(cal_targets):
        # Normalised target
        norm_tx = tx / screen_w
        norm_ty = ty / screen_h

        frame_count = 0
        settle_count = 0
        point_preds_x = []
        point_preds_y = []

        while True:
            ret, frame = cap.read()
            if not ret:
                continue

            # NOTE: Do NOT flip frame. Training data was collected with unmirrored
            # browser canvas, so inference must also use unmirrored frames.
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp_lib.Image(image_format=mp_lib.ImageFormat.SRGB, data=rgb)
            ts_ms = int((time.perf_counter() - start_time) * 1000)
            result = landmarker.detect_for_video(mp_image, ts_ms)

            # Draw calibration screen
            display = np.zeros((screen_h, screen_w, 3), dtype=np.uint8)

            # Draw all points dimly
            for i, (px, py) in enumerate(cal_targets):
                if i < pt_idx:
                    cv2.circle(display, (int(px), int(py)), 8, (0, 80, 0), -1)  # done
                elif i == pt_idx:
                    pass  # drawn below
                else:
                    cv2.circle(display, (int(px), int(py)), 6, (40, 40, 40), -1)  # upcoming

            # Draw active calibration target with pulsing animation
            pulse = int(15 + 8 * math.sin(time.perf_counter() * 5))
            if settle_count < CALIBRATION_SETTLE_FRAMES:
                color = (0, 200, 255)  # orange = settling
            else:
                color = (0, 255, 0)    # green = collecting
            cv2.circle(display, (int(tx), int(ty)), pulse, color, -1)
            cv2.circle(display, (int(tx), int(ty)), 3, (255, 255, 255), -1)

            # Status text
            status = f"Point {pt_idx + 1}/{len(cal_targets)}"
            if settle_count < CALIBRATION_SETTLE_FRAMES:
                status += " - Look at the dot..."
            else:
                progress = len(point_preds_x)
                status += f" - Collecting ({progress}/{CALIBRATION_SAMPLES})"

            cv2.putText(display, status, (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
            cv2.putText(display, "Look at the GREEN dot. Keep your head still.",
                        (20, screen_h - 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (180, 180, 180), 1)

            cv2.imshow(window_name, display)
            key = cv2.waitKey(1) & 0xFF
            if key == 27:
                cv2.destroyWindow(window_name)
                return None  # User cancelled

            # Process face
            if result.face_landmarks and result.facial_transformation_matrixes:
                landmarks = result.face_landmarks[0]
                matrix = result.facial_transformation_matrixes[0]
                features, l_ear, r_ear = extract_features(landmarks, matrix)

                # Skip blinks
                if (l_ear + r_ear) / 2 < EAR_BLINK_THRESHOLD:
                    continue

                settle_count += 1
                if settle_count <= CALIBRATION_SETTLE_FRAMES:
                    continue  # Let eyes settle on the dot

                # Collect prediction
                pred_x, pred_y = predict_gaze(model_backend, features)
                point_preds_x.append(pred_x)
                point_preds_y.append(pred_y)

                if len(point_preds_x) >= CALIBRATION_SAMPLES:
                    break

        # Average prediction for this point
        avg_pred_x = float(np.median(point_preds_x))
        avg_pred_y = float(np.median(point_preds_y))
        raw_predictions.append((avg_pred_x, avg_pred_y))
        target_positions.append((norm_tx, norm_ty))

        print(f"  Cal point {pt_idx + 1}: target=({norm_tx:.2f}, {norm_ty:.2f})  "
              f"raw_pred=({avg_pred_x:.3f}, {avg_pred_y:.3f})")

    cv2.destroyWindow(window_name)

    # Compute affine correction
    transform = compute_affine_correction(raw_predictions, target_positions)
    print(f"\n[OK] Calibration complete! Affine transform:\n{transform}")

    return transform


# ===================================================================
# Main tracking loop
# ===================================================================
def run_tracking(model_backend, landmarker, cap, transform, screen_w, screen_h,
                 move_mouse=False, start_time=None):
    """Live gaze tracking with affine-corrected predictions."""

    if move_mouse:
        import pyautogui
        pyautogui.FAILSAFE = True
        pyautogui.PAUSE = 0

    smoother = GazeSmoother()
    if start_time is None:
        start_time = time.perf_counter()
    frame_count = 0
    fps = 0
    fps_timer = time.perf_counter()

    window_name = "Gaze Tracker"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 640, 480)
    cv2.moveWindow(window_name, 50, 50)

    print("\n" + "=" * 50)
    print("  LIVE GAZE TRACKING - ACTIVE")
    print("=" * 50)
    print("  ESC/Q  - Quit")
    print("  R      - Recalibrate")
    print("=" * 50 + "\n")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # NOTE: Do NOT flip frame. Must match training data (unmirrored).
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp_lib.Image(image_format=mp_lib.ImageFormat.SRGB, data=rgb)
        ts_ms = int((time.perf_counter() - start_time) * 1000)
        result = landmarker.detect_for_video(mp_image, ts_ms)

        gaze_x, gaze_y = 0.5, 0.5
        is_blinking = False
        face_found = False

        if result.face_landmarks and result.facial_transformation_matrixes:
            face_found = True
            landmarks = result.face_landmarks[0]
            matrix = result.facial_transformation_matrixes[0]
            features, l_ear, r_ear = extract_features(landmarks, matrix)

            avg_ear = (l_ear + r_ear) / 2
            is_blinking = avg_ear < EAR_BLINK_THRESHOLD

            if not is_blinking:
                raw_x, raw_y = predict_gaze(model_backend, features)

                # Apply affine correction from calibration
                corr_x, corr_y = apply_affine(transform, raw_x, raw_y)
                corr_x = float(np.clip(corr_x, 0, 1))
                corr_y = float(np.clip(corr_y, 0, 1))

                gaze_x, gaze_y = smoother.update(corr_x, corr_y)

                if move_mouse:
                    mx = int(gaze_x * screen_w)
                    my = int(gaze_y * screen_h)
                    mx = max(0, min(mx, screen_w - 1))
                    my = max(0, min(my, screen_h - 1))
                    try:
                        pyautogui.moveTo(mx, my, _pause=False)
                    except Exception:
                        pass

        # --- Visualization ---
        display = frame.copy()
        cam_h, cam_w = display.shape[:2]

        cx = int(gaze_x * cam_w)
        cy = int(gaze_y * cam_h)
        color = (0, 0, 255) if is_blinking else (0, 255, 0)
        cv2.drawMarker(display, (cx, cy), color, cv2.MARKER_CROSS, 30, 2)
        cv2.circle(display, (cx, cy), 8, color, 2)

        status = "BLINK" if is_blinking else ("TRACKING" if face_found else "NO FACE")
        status_color = (0, 0, 255) if is_blinking or not face_found else (0, 255, 0)
        cv2.putText(display, status, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, status_color, 2)
        cv2.putText(display, f"Gaze: ({gaze_x:.3f}, {gaze_y:.3f})", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        cv2.putText(display, f"FPS: {fps}", (10, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

        if move_mouse and face_found and not is_blinking:
            cv2.putText(display, f"Cursor: ({mx}, {my})", (10, 120),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 1)

        cv2.imshow(window_name, display)

        frame_count += 1
        if time.perf_counter() - fps_timer >= 1.0:
            fps = frame_count
            frame_count = 0
            fps_timer = time.perf_counter()

        key = cv2.waitKey(1) & 0xFF
        if key in (27, ord('q'), ord('Q')):
            break
        elif key in (ord('r'), ord('R')):
            return "recalibrate"

    return "quit"


# ===================================================================
# Main
# ===================================================================
def main():
    parser = argparse.ArgumentParser(description="Gaze-controlled mouse cursor")
    parser.add_argument("--use-keras", action="store_true",
                        help="Use Keras model instead of TFLite")
    parser.add_argument("--move-mouse", action="store_true",
                        help="Actually move the mouse cursor")
    parser.add_argument("--camera", type=int, default=0,
                        help="Camera device index")
    args = parser.parse_args()

    # Load model
    print("Loading gaze model...")
    model_backend = load_model(use_keras=args.use_keras)

    # Setup MediaPipe
    print("Loading MediaPipe Face Landmarker...")
    base_options = python.BaseOptions(model_asset_path=LANDMARKER_PATH)
    options = vision.FaceLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.VIDEO,
        output_face_blendshapes=False,
        output_facial_transformation_matrixes=True,
        num_faces=1,
    )
    landmarker = vision.FaceLandmarker.create_from_options(options)

    # Open camera
    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        print(f"[ERROR] Cannot open camera {args.camera}")
        return

    cam_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    cam_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"[OK] Camera: {cam_w}x{cam_h}")

    # Detect screen resolution via a temporary fullscreen window
    temp_win = "__detect_screen__"
    cv2.namedWindow(temp_win, cv2.WINDOW_NORMAL)
    cv2.setWindowProperty(temp_win, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    cv2.imshow(temp_win, np.zeros((100, 100, 3), dtype=np.uint8))
    cv2.waitKey(100)
    _, _, screen_w, screen_h = cv2.getWindowImageRect(temp_win)
    cv2.destroyWindow(temp_win)
    cv2.waitKey(1)
    print(f"[OK] Screen: {screen_w}x{screen_h}")

    session_start = time.perf_counter()

    try:
        while True:
            # --- Calibration ---
            print("\n--- Starting Calibration ---")
            print("Look at each GREEN dot as it appears. Keep head still.")
            transform = run_calibration(model_backend, landmarker, cap,
                                        screen_w, screen_h,
                                        start_time=session_start)
            if transform is None:
                print("Calibration cancelled.")
                break

            # --- Live tracking ---
            result = run_tracking(model_backend, landmarker, cap, transform,
                                  screen_w, screen_h,
                                  move_mouse=args.move_mouse,
                                  start_time=session_start)

            if result == "recalibrate":
                print("\n--- Recalibrating... ---")
                continue
            else:
                break
    finally:
        cap.release()
        landmarker.close()
        cv2.destroyAllWindows()
        print("\nGaze mouse controller stopped.")


if __name__ == "__main__":
    main()
