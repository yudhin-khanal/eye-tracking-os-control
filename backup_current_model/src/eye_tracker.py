import argparse
import os
import time
from collections import deque

import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(BASE_DIR, "models")
TFLITE_MODEL_PATH = os.path.join(MODEL_DIR, "eye_model.tflite")
KERAS_MODEL_PATH = os.path.join(MODEL_DIR, "best_eye_model.keras")
LANDMARKER_PATH = os.path.join(MODEL_DIR, "face_landmarker.task")

# Prediction tuning
SMOOTHING_WINDOW = 5
OPEN_THRESHOLD = 0.55
CLOSE_THRESHOLD = 0.45

# Geometry fusion tuning
EAR_CLOSED_REF = 0.16
EAR_OPEN_REF = 0.30
EAR_MIN_RANGE = 0.04
MODEL_FUSION_WEIGHT = 0.75
UNCERTAIN_MARGIN = 0.12

# Runtime behavior
DEFAULT_CALIBRATION_SECONDS = 2.5
MIN_CALIBRATION_SAMPLES = 25
CALIBRATION_OPEN_MARGIN = 0.20
CALIBRATION_HYSTERESIS_GAP = 0.10
AUTO_RESTART_DELAY_SEC = 1.0

# Eye landmark sets (MediaPipe Face Mesh indices)
LEFT_EYE_CONTOUR = [33, 246, 161, 160, 159, 158, 157, 173, 133, 155, 154, 153, 145, 144, 163, 7]
RIGHT_EYE_CONTOUR = [263, 466, 388, 387, 386, 385, 384, 398, 362, 382, 381, 380, 374, 373, 390, 249]
LEFT_EYE_CORNERS = (33, 133)
RIGHT_EYE_CORNERS = (263, 362)
LEFT_EAR_POINTS = (33, 160, 158, 133, 153, 144)
RIGHT_EAR_POINTS = (263, 387, 385, 362, 380, 373)

def setup_landmarker():
    base_options = python.BaseOptions(model_asset_path=LANDMARKER_PATH)
    options = vision.FaceLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.VIDEO,
        output_face_blendshapes=False,
        output_facial_transformation_matrixes=False,
        num_faces=1)
    detector = vision.FaceLandmarker.create_from_options(options)
    return detector

def resolve_model_backend(prefer_keras=True):
    """Load the best available model backend.

    Keras is preferred when present because full-precision inference is often
    more accurate than post-training quantized TFLite models.
    """
    if prefer_keras and os.path.exists(KERAS_MODEL_PATH):
        keras_model = tf.keras.models.load_model(KERAS_MODEL_PATH, compile=False)
        input_shape = keras_model.input_shape
        if isinstance(input_shape, list):
            input_shape = input_shape[0]
        return {
            "type": "keras",
            "model": keras_model,
            "input_shape": tuple(input_shape),
            "input_dtype": np.float32,
            "output_quant": (0.0, 0),
        }

    interpreter = tf.lite.Interpreter(model_path=TFLITE_MODEL_PATH)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()[0]
    output_details = interpreter.get_output_details()[0]

    return {
        "type": "tflite",
        "model": interpreter,
        "input_details": input_details,
        "output_details": output_details,
        "input_shape": tuple(input_details["shape"]),
        "input_dtype": input_details["dtype"],
        "input_quant": input_details.get("quantization", (0.0, 0)),
        "output_quant": output_details.get("quantization", (0.0, 0)),
    }


def _safe_hwc(input_shape):
    # Supports shapes like (1, H, W, C), (H, W, C), or dynamic values.
    shape = []
    for dim in input_shape:
        try:
            value = int(dim)
        except (TypeError, ValueError):
            value = 1
        shape.append(value if value > 0 else 1)
    if len(shape) >= 4:
        return int(shape[-3]), int(shape[-2]), int(shape[-1])
    if len(shape) == 3:
        return int(shape[0]), int(shape[1]), int(shape[2])
    raise ValueError(f"Unsupported input shape: {input_shape}")


def get_eye_region(image, landmarks, eye_indices, corner_indices, padding_scale=1.8):
    """Extract a square eye crop centered on the eye with contour masking."""
    h, w = image.shape[:2]
    eye_pts = np.array([(landmarks[i].x * w, landmarks[i].y * h) for i in eye_indices], dtype=np.float32)

    c0 = np.array([landmarks[corner_indices[0]].x * w, landmarks[corner_indices[0]].y * h], dtype=np.float32)
    c1 = np.array([landmarks[corner_indices[1]].x * w, landmarks[corner_indices[1]].y * h], dtype=np.float32)

    center = eye_pts.mean(axis=0)
    eye_width = np.linalg.norm(c1 - c0)
    crop_size = max(12, int(eye_width * padding_scale))

    x1 = max(0, int(center[0] - crop_size / 2))
    y1 = max(0, int(center[1] - crop_size / 2))
    x2 = min(w, x1 + crop_size)
    y2 = min(h, y1 + crop_size)

    if x2 <= x1 or y2 <= y1:
        return np.array([], dtype=np.uint8), (x1, y1, x2, y2)

    eye_crop = image[y1:y2, x1:x2].copy()

    # Keep only the polygon area to reduce eyebrow/skin noise.
    contour = eye_pts.copy()
    contour[:, 0] -= x1
    contour[:, 1] -= y1
    mask = np.zeros((y2 - y1, x2 - x1), dtype=np.uint8)
    cv2.fillConvexPoly(mask, contour.astype(np.int32), 255)
    eye_crop = cv2.bitwise_and(eye_crop, eye_crop, mask=mask)

    return eye_crop, (x1, y1, x2, y2)


def _pt_xy(landmarks, idx, width, height):
    return np.array([landmarks[idx].x * width, landmarks[idx].y * height], dtype=np.float32)


def compute_ear(landmarks, ear_indices, width, height):
    """Compute eye aspect ratio from 6 eye landmarks."""
    p1 = _pt_xy(landmarks, ear_indices[0], width, height)
    p2 = _pt_xy(landmarks, ear_indices[1], width, height)
    p3 = _pt_xy(landmarks, ear_indices[2], width, height)
    p4 = _pt_xy(landmarks, ear_indices[3], width, height)
    p5 = _pt_xy(landmarks, ear_indices[4], width, height)
    p6 = _pt_xy(landmarks, ear_indices[5], width, height)

    horizontal = np.linalg.norm(p1 - p4)
    if horizontal < 1e-6:
        return None

    vertical_1 = np.linalg.norm(p2 - p6)
    vertical_2 = np.linalg.norm(p3 - p5)
    return float((vertical_1 + vertical_2) / (2.0 * horizontal))


def ear_to_open_score(ear, history):
    if ear is None:
        return None

    history.append(ear)
    low_ref = EAR_CLOSED_REF
    high_ref = EAR_OPEN_REF

    if len(history) >= 20:
        values = np.asarray(history, dtype=np.float32)
        low = float(np.percentile(values, 15))
        high = float(np.percentile(values, 85))
        if (high - low) >= EAR_MIN_RANGE:
            low_ref = low
            high_ref = high

    return float(np.clip((ear - low_ref) / max(1e-6, high_ref - low_ref), 0.0, 1.0))


def fuse_open_scores(model_score, geometry_score):
    if model_score is None:
        return geometry_score
    if geometry_score is None:
        return model_score

    # Trust geometry slightly more when the model sits near the decision boundary.
    if abs(model_score - 0.5) < UNCERTAIN_MARGIN:
        weight = 0.55
    else:
        weight = MODEL_FUSION_WEIGHT

    return float(weight * model_score + (1.0 - weight) * geometry_score)


def _letterbox_resize(image, target_w, target_h):
    """Resize while preserving aspect ratio and pad to exact model size."""
    h, w = image.shape[:2]
    if h == 0 or w == 0:
        return np.zeros((target_h, target_w, image.shape[2] if image.ndim == 3 else 1), dtype=np.uint8)

    scale = min(target_w / w, target_h / h)
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))

    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
    if resized.ndim == 2:
        canvas = np.zeros((target_h, target_w), dtype=np.uint8)
    else:
        canvas = np.zeros((target_h, target_w, resized.shape[2]), dtype=np.uint8)

    x_off = (target_w - new_w) // 2
    y_off = (target_h - new_h) // 2
    canvas[y_off:y_off + new_h, x_off:x_off + new_w] = resized
    return canvas


def preprocess_eye(eye_img, input_shape, input_dtype, input_quant):
    in_h, in_w, in_c = _safe_hwc(input_shape)

    if eye_img.size == 0:
        return None

    # Normalize local contrast to improve robustness in changing light.
    eye = _letterbox_resize(eye_img, in_w, in_h)
    eye = cv2.GaussianBlur(eye, (3, 3), 0)

    if eye.ndim == 3:
        lch = cv2.cvtColor(eye, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lch)
        l = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4)).apply(l)
        eye = cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2RGB)

    if in_c == 1 and eye.ndim == 3:
        eye = cv2.cvtColor(eye, cv2.COLOR_RGB2GRAY)
        eye = np.expand_dims(eye, axis=-1)
    elif in_c == 3 and eye.ndim == 2:
        eye = np.repeat(eye[..., np.newaxis], 3, axis=-1)

    if np.issubdtype(input_dtype, np.floating):
        tensor = eye.astype(np.float32) / 255.0
        tensor = np.expand_dims(tensor, axis=0).astype(input_dtype)
        return tensor

    scale, zero_point = input_quant
    if scale and scale > 0:
        # Use [0, 1] as the real-value domain and quantize to model dtype.
        real_tensor = eye.astype(np.float32) / 255.0
        quantized = np.round(real_tensor / scale + zero_point)
    else:
        quantized = eye.astype(np.float32)

    if input_dtype == np.uint8:
        quantized = np.clip(quantized, 0, 255).astype(np.uint8)
    elif input_dtype == np.int8:
        quantized = np.clip(quantized, -128, 127).astype(np.int8)
    else:
        quantized = quantized.astype(input_dtype)

    return np.expand_dims(quantized, axis=0)


def parse_open_score(raw_output):
    values = np.asarray(raw_output).astype(np.float32).reshape(-1)
    if values.size == 0:
        return 0.0
    if values.size == 1:
        return float(np.clip(values[0], 0.0, 1.0))

    # For 2-class outputs, use the second logit/probability as "open".
    if values.size == 2:
        probs_sum = float(values[0] + values[1])
        if 0.8 <= probs_sum <= 1.2 and np.all(values >= 0.0):
            return float(np.clip(values[1], 0.0, 1.0))

        # If logits are returned, convert to probability by logistic margin.
        margin = float(values[1] - values[0])
        return float(1.0 / (1.0 + np.exp(-margin)))

    return float(np.clip(np.mean(values), 0.0, 1.0))


def infer_eye_open_score(model_backend, eye_img):
    input_tensor = preprocess_eye(
        eye_img,
        input_shape=model_backend["input_shape"],
        input_dtype=model_backend["input_dtype"],
        input_quant=model_backend.get("input_quant", (0.0, 0)),
    )
    if input_tensor is None:
        return None

    if model_backend["type"] == "keras":
        pred = model_backend["model"].predict(input_tensor, verbose=0)
        return parse_open_score(pred)

    interpreter = model_backend["model"]
    input_details = model_backend["input_details"]
    output_details = model_backend["output_details"]

    interpreter.set_tensor(input_details["index"], input_tensor)
    interpreter.invoke()
    output = interpreter.get_tensor(output_details["index"])

    out_scale, out_zero = model_backend.get("output_quant", (0.0, 0))
    if np.issubdtype(output.dtype, np.integer) and out_scale and out_scale > 0:
        output = (output.astype(np.float32) - out_zero) * out_scale

    return parse_open_score(output)


def smooth_score(queue, score):
    queue.append(score)
    return float(np.mean(queue))


def classify_with_hysteresis(score, prev_state, open_threshold, close_threshold):
    if prev_state == "Open":
        return "Closed" if score < close_threshold else "Open"
    return "Open" if score > open_threshold else "Closed"


def calibrate_thresholds(open_samples):
    if len(open_samples) < MIN_CALIBRATION_SAMPLES:
        return OPEN_THRESHOLD, CLOSE_THRESHOLD

    values = np.asarray(open_samples, dtype=np.float32)
    baseline_open = float(np.percentile(values, 35))
    close_threshold = float(np.clip(baseline_open - CALIBRATION_OPEN_MARGIN, 0.20, 0.70))
    open_threshold = float(np.clip(close_threshold + CALIBRATION_HYSTERESIS_GAP, close_threshold + 0.05, 0.95))
    return open_threshold, close_threshold


def run_tracking_session(model_backend, max_frames=0, calibration_seconds=DEFAULT_CALIBRATION_SECONDS):
    window_name = "Eye Tracking"
    landmarker = setup_landmarker()
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open camera 0.")
        landmarker.close()
        return "camera-open-failed"

    left_scores = deque(maxlen=SMOOTHING_WINDOW)
    right_scores = deque(maxlen=SMOOTHING_WINDOW)
    left_ear_history = deque(maxlen=120)
    right_ear_history = deque(maxlen=120)
    left_state = "Open"
    right_state = "Open"

    open_threshold = OPEN_THRESHOLD
    close_threshold = CLOSE_THRESHOLD
    calibration_samples = []
    calibrating = calibration_seconds > 0
    calibration_start = time.perf_counter()

    start_t = time.perf_counter()
    frame_count = 0
    exit_reason = "unknown"

    try:
        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                exit_reason = "camera-read-failed"
                break

            # Mirror view gives more natural interaction.
            frame = cv2.flip(frame, 1)

            # Convert to RGB for MediaPipe
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

            # Detect landmarks
            timestamp_ms = int((time.perf_counter() - start_t) * 1000)
            detection_result = landmarker.detect_for_video(mp_image, timestamp_ms)

            if detection_result.face_landmarks:
                landmarks = detection_result.face_landmarks[0]
                frame_h, frame_w = rgb_frame.shape[:2]

                left_eye, l_box = get_eye_region(
                    rgb_frame,
                    landmarks,
                    LEFT_EYE_CONTOUR,
                    LEFT_EYE_CORNERS,
                )
                right_eye, r_box = get_eye_region(
                    rgb_frame,
                    landmarks,
                    RIGHT_EYE_CONTOUR,
                    RIGHT_EYE_CORNERS,
                )

                score_l = infer_eye_open_score(model_backend, left_eye)
                score_r = infer_eye_open_score(model_backend, right_eye)
                ear_l = compute_ear(landmarks, LEFT_EAR_POINTS, frame_w, frame_h)
                ear_r = compute_ear(landmarks, RIGHT_EAR_POINTS, frame_w, frame_h)
                geom_l = ear_to_open_score(ear_l, left_ear_history)
                geom_r = ear_to_open_score(ear_r, right_ear_history)
                score_l = fuse_open_scores(score_l, geom_l)
                score_r = fuse_open_scores(score_r, geom_r)

                # Visualization
                cv2.rectangle(frame, (l_box[0], l_box[1]), (l_box[2], l_box[3]), (0, 255, 0), 1)
                cv2.rectangle(frame, (r_box[0], r_box[1]), (r_box[2], r_box[3]), (0, 255, 0), 1)

                if score_l is not None and score_r is not None:
                    smooth_l = smooth_score(left_scores, score_l)
                    smooth_r = smooth_score(right_scores, score_r)
                    avg_score = (smooth_l + smooth_r) / 2.0

                    if calibrating:
                        calibration_samples.append(avg_score)
                        elapsed = time.perf_counter() - calibration_start
                        progress = float(np.clip(elapsed / max(1e-6, calibration_seconds), 0.0, 1.0))
                        bar_width = int(260 * progress)

                        cv2.putText(
                            frame,
                            "Calibrating: keep eyes naturally open",
                            (20, 30),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.6,
                            (0, 220, 255),
                            2,
                        )
                        cv2.rectangle(frame, (20, 42), (280, 58), (80, 80, 80), 1)
                        cv2.rectangle(frame, (20, 42), (20 + bar_width, 58), (0, 220, 255), -1)

                        enough_samples = len(calibration_samples) >= MIN_CALIBRATION_SAMPLES
                        reached_time = elapsed >= calibration_seconds
                        timeout = elapsed >= calibration_seconds * 2.0

                        if (reached_time and enough_samples) or timeout:
                            open_threshold, close_threshold = calibrate_thresholds(calibration_samples)
                            calibrating = False
                            left_scores.clear()
                            right_scores.clear()
                            left_state = "Open"
                            right_state = "Open"
                            print(
                                f"Calibration complete: open>{open_threshold:.2f}, close<{close_threshold:.2f} "
                                f"from {len(calibration_samples)} samples"
                            )
                    else:
                        left_state = classify_with_hysteresis(
                            smooth_l,
                            left_state,
                            open_threshold=open_threshold,
                            close_threshold=close_threshold,
                        )
                        right_state = classify_with_hysteresis(
                            smooth_r,
                            right_state,
                            open_threshold=open_threshold,
                            close_threshold=close_threshold,
                        )

                        color_l = (0, 255, 0) if left_state == "Open" else (0, 0, 255)
                        color_r = (0, 255, 0) if right_state == "Open" else (0, 0, 255)

                        cv2.putText(
                            frame,
                            f"L: {left_state} ({smooth_l:.2f})",
                            (l_box[0], max(20, l_box[1] - 10)),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.5,
                            color_l,
                            1,
                        )
                        cv2.putText(
                            frame,
                            f"R: {right_state} ({smooth_r:.2f})",
                            (r_box[0], max(20, r_box[1] - 10)),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.5,
                            color_r,
                            1,
                        )

                        cv2.putText(
                            frame,
                            f"Avg Score: {avg_score:.2f}",
                            (20, 45),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.9,
                            (255, 180, 0),
                            2,
                        )

                        cv2.putText(
                            frame,
                            f"Thr O:{open_threshold:.2f} C:{close_threshold:.2f}",
                            (20, 75),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.55,
                            (240, 240, 240),
                            1,
                        )

            frame_count += 1
            if max_frames > 0 and frame_count >= max_frames:
                exit_reason = "max-frames"
                break

            cv2.imshow(window_name, frame)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                exit_reason = "user-exit"
                break

            try:
                if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
                    exit_reason = "window-closed"
                    break
            except cv2.error:
                exit_reason = "window-closed"
                break

        if exit_reason == "unknown":
            exit_reason = "session-ended"
        return exit_reason
    finally:
        cap.release()
        landmarker.close()
        cv2.destroyAllWindows()


def main(
    prefer_keras=True,
    max_frames=0,
    auto_restart=True,
    restart_delay=AUTO_RESTART_DELAY_SEC,
    calibration_seconds=DEFAULT_CALIBRATION_SECONDS,
):
    if not os.path.exists(LANDMARKER_PATH):
        print(f"Error: {LANDMARKER_PATH} not found.")
        return

    if not os.path.exists(TFLITE_MODEL_PATH) and not os.path.exists(KERAS_MODEL_PATH):
        print(f"Error: Neither {TFLITE_MODEL_PATH} nor {KERAS_MODEL_PATH} found.")
        return

    print("Initializing models...")
    model_backend = resolve_model_backend(prefer_keras=prefer_keras)

    in_h, in_w, in_c = _safe_hwc(model_backend["input_shape"])
    print(f"Model backend: {model_backend['type']}")
    print(f"Model input shape: (H={in_h}, W={in_w}, C={in_c})")
    print("Controls: press Q or ESC to quit. Closing the window will reopen automatically.")

    while True:
        reason = run_tracking_session(
            model_backend=model_backend,
            max_frames=max_frames,
            calibration_seconds=calibration_seconds,
        )

        if reason in ("user-exit", "max-frames"):
            break

        if not auto_restart:
            print(f"Session ended ({reason}). Auto-restart disabled.")
            break

        if reason not in ("window-closed", "camera-read-failed"):
            print(f"Session ended ({reason}). Not restarting automatically.")
            break

        print(f"Session ended ({reason}). Restarting in {restart_delay:.1f}s...")
        time.sleep(max(0.0, restart_delay))


def parse_args():
    parser = argparse.ArgumentParser(description="Realtime eye open/closed tracker")
    parser.add_argument(
        "--prefer-tflite",
        action="store_true",
        help="Use TFLite model even when best_eye_model.keras is available.",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=0,
        help="Stop automatically after N frames (0 means run until user quits).",
    )
    parser.add_argument(
        "--no-auto-restart",
        action="store_true",
        help="Disable auto-restart after closing the display window.",
    )
    parser.add_argument(
        "--restart-delay",
        type=float,
        default=AUTO_RESTART_DELAY_SEC,
        help="Delay in seconds before auto-restarting after window close.",
    )
    parser.add_argument(
        "--calibration-seconds",
        type=float,
        default=DEFAULT_CALIBRATION_SECONDS,
        help="Startup calibration duration in seconds (0 disables calibration).",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(
        prefer_keras=not args.prefer_tflite,
        max_frames=args.max_frames,
        auto_restart=not args.no_auto_restart,
        restart_delay=args.restart_delay,
        calibration_seconds=max(0.0, args.calibration_seconds),
    )
