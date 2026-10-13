"""
Enhanced Gaze Data Collection Server
=====================================
Flask + SocketIO backend for browser-based gaze data collection.

Key improvements over the original:
- Server-side target generation (grid / random / mixed / edges modes)
- Enhanced feature extraction: gaze ratios, EAR, face area, brightness
- Automatic quality filtering: rejects blinks, extreme poses, bad frames
- Dataset statistics API endpoint
- Richer CSV output (28 columns)
"""

import base64
import csv
import glob
import math
import os
import random
import sys
import time

import cv2
import mediapipe as mp
import numpy as np
from flask import Flask, render_template, jsonify
from flask_socketio import SocketIO, emit
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------
app = Flask(__name__)
app.config["SECRET_KEY"] = os.urandom(24).hex()
socketio = SocketIO(
    app,
    cors_allowed_origins="*",
    max_http_buffer_size=10 * 1024 * 1024,  # 10 MB per message
)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LANDMARKER_PATH = os.path.join(BASE_DIR, "models", "face_landmarker.task")
DATA_DIR = os.path.join(BASE_DIR, "dataset")

# ---------------------------------------------------------------------------
# MediaPipe landmark indices
# ---------------------------------------------------------------------------
LEFT_IRIS = [474, 475, 476, 477]
RIGHT_IRIS = [469, 470, 471, 472]

# EAR: (outer, upper1, upper2, inner, lower1, lower2)
LEFT_EAR_IDX = (263, 387, 385, 362, 380, 373)
RIGHT_EAR_IDX = (33, 160, 158, 133, 153, 144)

# Full eye contour (for robust bounding box)
LEFT_EYE_CONTOUR = [263, 466, 388, 387, 386, 385, 384, 398, 362, 382, 381, 380, 374, 373, 390, 249]
RIGHT_EYE_CONTOUR = [33, 246, 161, 160, 159, 158, 157, 173, 133, 155, 154, 153, 145, 144, 163, 7]

# Face-oval landmarks (for approximate face area)
FACE_OVAL = [
    10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288,
    397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136,
    172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109,
]

# ---------------------------------------------------------------------------
# CSV header for saved data
# ---------------------------------------------------------------------------
CSV_HEADER = [
    "timestamp", "user", "session_type",
    "screen_w", "screen_h", "cam_w", "cam_h",
    "target_x", "target_y", "target_type",
    "head_pitch", "head_yaw", "head_roll",
    "l_iris_x", "l_iris_y", "l_iris_z",
    "r_iris_x", "r_iris_y", "r_iris_z",
    "inter_ocular_dist",
    "l_gaze_ratio_x", "l_gaze_ratio_y",
    "r_gaze_ratio_x", "r_gaze_ratio_y",
    "l_ear", "r_ear",
    "face_area", "frame_brightness",
]

# ---------------------------------------------------------------------------
# Module-level state
# ---------------------------------------------------------------------------
landmarker = None

_default_session = dict(
    data=[],
    user="",
    session_type="",
    session_name="",
    collection_mode="mixed",
    targets=[],
    frames_per_target=15,
    is_recording=False,
    quality_stats=dict(good=0, poor=0, rejected=0),
    start_time=None,
)
session_state: dict = dict(_default_session)


# ===================================================================
# Helper: MediaPipe setup
# ===================================================================
def setup_landmarker():
    base_options = python.BaseOptions(model_asset_path=LANDMARKER_PATH)
    options = vision.FaceLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.IMAGE,
        output_face_blendshapes=False,
        output_facial_transformation_matrixes=True,
        num_faces=1,
    )
    return vision.FaceLandmarker.create_from_options(options)


# ===================================================================
# Feature extraction helpers
# ===================================================================
def extract_head_pose(transformation_matrix):
    """Pitch, yaw, roll from the 4×4 transformation matrix."""
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
    """Eye Aspect Ratio from 6 landmarks (normalized coords)."""
    pts = [np.array([landmarks[i].x, landmarks[i].y]) for i in idx]
    horiz = np.linalg.norm(pts[0] - pts[3])
    if horiz < 1e-7:
        return 0.0
    v1 = np.linalg.norm(pts[1] - pts[5])
    v2 = np.linalg.norm(pts[2] - pts[4])
    return float((v1 + v2) / (2.0 * horiz))


def compute_gaze_ratio(landmarks, iris_idx, eye_contour_idx):
    """Normalised iris position within the eye contour bounding box.

    Uses the FULL eye contour (16 landmarks) to compute the bounding box,
    ensuring the iris always falls within it.

    Returns (h_ratio, v_ratio) each in [0, 1].
    h_ratio ~ 0 = looking left in image, h_ratio ~ 1 = looking right in image.
    v_ratio ~ 0 = looking up, v_ratio ~ 1 = looking down.
    """
    ix = sum(landmarks[i].x for i in iris_idx) / len(iris_idx)
    iy = sum(landmarks[i].y for i in iris_idx) / len(iris_idx)

    xs = [landmarks[i].x for i in eye_contour_idx]
    ys = [landmarks[i].y for i in eye_contour_idx]
    x_lo, x_hi = min(xs), max(xs)
    y_lo, y_hi = min(ys), max(ys)

    h = (ix - x_lo) / max(x_hi - x_lo, 1e-7)
    v = (iy - y_lo) / max(y_hi - y_lo, 1e-7)
    return float(np.clip(h, 0, 1)), float(np.clip(v, 0, 1))


def compute_frame_brightness(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
    return float(np.mean(gray))


def compute_face_area(landmarks):
    """Approximate area of the face oval (normalised coordinates)."""
    pts = np.array([(landmarks[i].x, landmarks[i].y) for i in FACE_OVAL])
    x, y = pts[:, 0], pts[:, 1]
    return float(0.5 * abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1))))


# ===================================================================
# Target generation
# ===================================================================
def _grid_points(count, w, h, pad):
    """Even grid covering the screen."""
    aspect = w / h
    cols = max(2, int(math.sqrt(count * aspect)))
    rows = max(2, round(count / cols))
    pts = []
    for r in range(rows):
        for c in range(cols):
            x = pad + (w - 2 * pad) * c / max(1, cols - 1)
            y = pad + (h - 2 * pad) * r / max(1, rows - 1)
            pts.append(dict(x=round(x, 1), y=round(y, 1), type="grid"))
    return pts


def generate_targets(mode, count, screen_w, screen_h, padding=80):
    """Build the target sequence for a session."""
    targets = []

    if mode == "grid":
        targets = _grid_points(count, screen_w, screen_h, padding)
        random.shuffle(targets)

    elif mode == "random":
        for _ in range(count):
            targets.append(dict(
                x=random.randint(padding, screen_w - padding),
                y=random.randint(padding, screen_h - padding),
                type="random",
            ))

    elif mode == "mixed":
        # 60 % grid  +  40 % random
        grid_n = max(4, int(count * 0.6))
        rand_n = count - grid_n
        targets = _grid_points(grid_n, screen_w, screen_h, padding)
        for _ in range(rand_n):
            targets.append(dict(
                x=random.randint(padding, screen_w - padding),
                y=random.randint(padding, screen_h - padding),
                type="random",
            ))
        random.shuffle(targets)

    elif mode == "edges":
        # Corners + edge midpoints + centre, then random fill
        corners = [
            (padding, padding), (screen_w - padding, padding),
            (padding, screen_h - padding), (screen_w - padding, screen_h - padding),
        ]
        midpoints = [
            (screen_w // 2, padding), (screen_w // 2, screen_h - padding),
            (padding, screen_h // 2), (screen_w - padding, screen_h // 2),
            (screen_w // 2, screen_h // 2),
        ]
        for x, y in corners:
            targets.append(dict(x=x, y=y, type="corner"))
        for x, y in midpoints:
            targets.append(dict(x=x, y=y, type="edge"))

        while len(targets) < count:
            if random.random() < 0.5:
                side = random.choice(["t", "b", "l", "r"])
                if side == "t":
                    x, y = random.randint(padding, screen_w - padding), padding + random.randint(0, 60)
                elif side == "b":
                    x, y = random.randint(padding, screen_w - padding), screen_h - padding - random.randint(0, 60)
                elif side == "l":
                    x, y = padding + random.randint(0, 60), random.randint(padding, screen_h - padding)
                else:
                    x, y = screen_w - padding - random.randint(0, 60), random.randint(padding, screen_h - padding)
                targets.append(dict(x=x, y=y, type="edge"))
            else:
                targets.append(dict(
                    x=random.randint(padding, screen_w - padding),
                    y=random.randint(padding, screen_h - padding),
                    type="random",
                ))
        random.shuffle(targets)

    return targets[:count]


# ===================================================================
# Flask routes
# ===================================================================
@app.route("/")
def index():
    return render_template("index.html")

@app.after_request
def add_header(response):
    response.headers['Cache-Control'] = 'no-store, no-cache, must-revalidate, post-check=0, pre-check=0, max-age=0'
    response.headers['Pragma'] = 'no-cache'
    response.headers['Expires'] = '-1'
    return response


@app.route("/api/dataset_stats")
def dataset_stats():
    """Return aggregate statistics about all collected data files."""
    os.makedirs(DATA_DIR, exist_ok=True)
    csv_files = glob.glob(os.path.join(DATA_DIR, "*.csv"))

    stats = dict(total_sessions=len(csv_files), total_frames=0,
                 users={}, session_types={}, files=[])

    for path in csv_files:
        try:
            with open(path, "r", encoding="utf-8-sig") as fh:
                reader = csv.DictReader(fh)
                rows = list(reader)
            n = len(rows)
            stats["total_frames"] += n

            basename = os.path.basename(path)
            stats["files"].append(dict(
                name=basename, frames=n,
                size_kb=round(os.path.getsize(path) / 1024, 1),
            ))

            if rows:
                user = rows[0].get("user", "unknown")
                stats["users"][user] = stats["users"].get(user, 0) + n

                stype = rows[0].get("session_type") or rows[0].get("lighting", "unknown")
                stats["session_types"][stype] = stats["session_types"].get(stype, 0) + n
        except Exception:
            continue

    # ------------------------------------------------------------------
    # Collection plan: define target frame counts per condition
    # ------------------------------------------------------------------
    CONDITION_TARGETS = {
        "standard":        5000,
    }
    OVERALL_TARGET = 5000
    MIN_USERS = 1

    plan = dict(
        overall_target=OVERALL_TARGET,
        overall_collected=stats["total_frames"],
        overall_remaining=max(0, OVERALL_TARGET - stats["total_frames"]),
        overall_pct=min(100, round(stats["total_frames"] / OVERALL_TARGET * 100, 1)) if OVERALL_TARGET else 0,
        min_users=MIN_USERS,
        current_users=len(stats["users"]),
        conditions=[],
    )

    for cond, target in CONDITION_TARGETS.items():
        collected = stats["session_types"].get(cond, 0)
        plan["conditions"].append(dict(
            name=cond,
            label=cond.replace("_", " ").title(),
            target=target,
            collected=collected,
            remaining=max(0, target - collected),
            pct=min(100, round(collected / target * 100, 1)) if target else 0,
            done=collected >= target,
        ))

    stats["plan"] = plan
    return jsonify(stats)


# ===================================================================
# SocketIO events
# ===================================================================
@socketio.on("start_session")
def handle_start_session(data):
    global session_state

    screen_w = data.get("screen_w", 1920)
    screen_h = data.get("screen_h", 1080)
    mode = data.get("collection_mode", "mixed")
    target_count = data.get("target_count", 80)
    frames_per_target = data.get("frames_per_target", 15)

    targets = generate_targets(mode, target_count, screen_w, screen_h)

    session_state = dict(
        data=[],
        user=data.get("user", "unknown"),
        session_type=data.get("session_type", "standard"),
        session_name=data.get("session_name", "session"),
        collection_mode=mode,
        targets=targets,
        frames_per_target=frames_per_target,
        is_recording=True,
        quality_stats=dict(good=0, poor=0, rejected=0),
        start_time=time.time(),
    )

    print(f"[SESSION] {session_state['session_type']} ({mode}) by {session_state['user']}")
    print(f"          {len(targets)} targets × {frames_per_target} frames")

    emit("session_started", dict(
        status="success",
        targets=targets,
        frames_per_target=frames_per_target,
    ))


@socketio.on("discard_session")
def handle_discard_session():
    global session_state
    session_state["data"] = []
    session_state["is_recording"] = False
    session_state["quality_stats"] = dict(good=0, poor=0, rejected=0)
    print("[SESSION] Discarded")
    emit("session_discarded", dict(status="success"))


@socketio.on("save_session")
def handle_save_session():
    global session_state
    session_state["is_recording"] = False

    if not session_state["data"]:
        emit("session_saved", dict(status="empty"))
        return

    os.makedirs(DATA_DIR, exist_ok=True)
    user = session_state["user"].replace(" ", "_")
    stype = session_state["session_type"].replace(" ", "_")
    sname = session_state["session_name"].replace(" ", "_")
    filename = f"gaze_data_{user}_{stype}_{sname}_{int(time.time())}.csv"
    filepath = os.path.join(DATA_DIR, filename)

    with open(filepath, "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(CSV_HEADER)
        writer.writerows(session_state["data"])

    duration = time.time() - (session_state["start_time"] or time.time())
    qs = session_state["quality_stats"]
    total = qs["good"] + qs["poor"] + qs["rejected"]
    quality_pct = round(qs["good"] / max(1, total) * 100, 1)

    result = dict(
        status="success",
        file=filename,
        frames=len(session_state["data"]),
        duration=round(duration, 1),
        quality_percent=quality_pct,
        quality_stats=qs,
    )
    session_state["data"] = []
    print(f"[SAVED] {filename} — {result['frames']} frames, {quality_pct}% good")
    emit("session_saved", result)


@socketio.on("process_frame")
def handle_process_frame(data):
    """Decode one webcam frame, extract all features, store in buffer."""
    global landmarker, session_state

    if not session_state["is_recording"]:
        return

    # --- decode image -------------------------------------------------------
    try:
        _, encoded = data["image"].split(",", 1)
        img_bytes = base64.b64decode(encoded)
        frame = cv2.imdecode(np.frombuffer(img_bytes, np.uint8), cv2.IMREAD_COLOR)
        if frame is None:
            raise ValueError("decode returned None")
    except Exception:
        session_state["quality_stats"]["rejected"] += 1
        emit("frame_result", dict(success=False, reason="decode_failed"))
        return

    brightness = compute_frame_brightness(frame)

    # --- MediaPipe detection ------------------------------------------------
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    result = landmarker.detect(mp_img)

    if not result.face_landmarks or not result.facial_transformation_matrixes:
        session_state["quality_stats"]["rejected"] += 1
        emit("frame_result", dict(success=False, reason="no_face"))
        return

    lm = result.face_landmarks[0]
    mat = result.facial_transformation_matrixes[0]

    # --- head pose ----------------------------------------------------------
    pitch, yaw, roll = extract_head_pose(mat)

    # Reject extreme head poses
    if abs(yaw) > 35 or abs(pitch) > 30:
        session_state["quality_stats"]["rejected"] += 1
        emit("frame_result", dict(success=False, reason="extreme_pose"))
        return

    # --- iris & inter-ocular ------------------------------------------------
    lx, ly, lz = get_normalized_center(lm, LEFT_IRIS)
    rx, ry, rz = get_normalized_center(lm, RIGHT_IRIS)
    iod = math.sqrt((lx - rx) ** 2 + (ly - ry) ** 2 + (lz - rz) ** 2)

    # --- gaze ratios --------------------------------------------------------
    lg_h, lg_v = compute_gaze_ratio(lm, LEFT_IRIS, LEFT_EYE_CONTOUR)
    rg_h, rg_v = compute_gaze_ratio(lm, RIGHT_IRIS, RIGHT_EYE_CONTOUR)

    # DEBUG: print actual values on first frame of session
    if session_state.get("frames_collected", 0) == 0 and not session_state.get("_debug_printed"):
        session_state["_debug_printed"] = True
        l_ix = sum(lm[i].x for i in LEFT_IRIS) / len(LEFT_IRIS)
        l_xs = [lm[i].x for i in LEFT_EYE_CONTOUR]
        r_ix = sum(lm[i].x for i in RIGHT_IRIS) / len(RIGHT_IRIS)
        r_xs = [lm[i].x for i in RIGHT_EYE_CONTOUR]
        print(f"[DEBUG] LEFT: iris_x={l_ix:.6f}, contour_x=[{min(l_xs):.6f}, {max(l_xs):.6f}], h={lg_h:.6f}")
        print(f"[DEBUG] RIGHT: iris_x={r_ix:.6f}, contour_x=[{min(r_xs):.6f}, {max(r_xs):.6f}], h={rg_h:.6f}")
        print(f"[DEBUG] LEFT contour x values: {[f'{x:.5f}' for x in l_xs]}")
        print(f"[DEBUG] RIGHT contour x values: {[f'{x:.5f}' for x in r_xs]}")

    # --- EAR ----------------------------------------------------------------
    l_ear = compute_ear(lm, LEFT_EAR_IDX)
    r_ear = compute_ear(lm, RIGHT_EAR_IDX)

    # Reject blinks
    if l_ear < 0.15 and r_ear < 0.15:
        session_state["quality_stats"]["rejected"] += 1
        emit("frame_result", dict(success=False, reason="blink"))
        return

    # --- face area ----------------------------------------------------------
    face_area = compute_face_area(lm)

    # --- quality label ------------------------------------------------------
    is_good = (
        abs(yaw) < 20 and abs(pitch) < 15
        and l_ear > 0.18 and r_ear > 0.18
        and 20 < brightness < 235
        and face_area > 0.01
    )
    session_state["quality_stats"]["good" if is_good else "poor"] += 1

    # --- store row ----------------------------------------------------------
    target_x = data["target_x"]
    target_y = data["target_y"]

    session_state["data"].append([
        int(time.time() * 1000),
        session_state["user"],
        session_state["session_type"],
        data["screen_w"], data["screen_h"],
        data["cam_w"], data["cam_h"],
        target_x, target_y,
        data.get("target_type", "unknown"),
        round(pitch, 4), round(yaw, 4), round(roll, 4),
        round(lx, 6), round(ly, 6), round(lz, 6),
        round(rx, 6), round(ry, 6), round(rz, 6),
        round(iod, 6),
        round(lg_h, 6), round(lg_v, 6),
        round(rg_h, 6), round(rg_v, 6),
        round(l_ear, 6), round(r_ear, 6),
        round(face_area, 6), round(brightness, 1),
    ])

    emit("frame_result", dict(
        success=True,
        quality="good" if is_good else "poor",
        brightness=round(brightness, 1),
        head_yaw=round(yaw, 1),
        head_pitch=round(pitch, 1),
        l_ear=round(l_ear, 3),
        r_ear=round(r_ear, 3),
    ))


# ===================================================================
# Entrypoint
# ===================================================================
if __name__ == "__main__":
    if not os.path.exists(LANDMARKER_PATH):
        print(f"[ERROR] {LANDMARKER_PATH} not found!")
        print("Download face_landmarker.task and place it in the models/ directory.")
    else:
        landmarker = setup_landmarker()
        host = os.getenv("EYE_TRACK_HOST", "0.0.0.0")
        port = int(os.getenv("EYE_TRACK_PORT", "5000"))
        argv = sys.argv[1:]
        use_https = (
            os.getenv("EYE_TRACK_USE_HTTPS", "0") == "1"
            or "--https" in argv
            or os.getenv("EYE_TRACK_USE_HTTP", "0") != "1"
            and host not in {"127.0.0.1", "localhost"}
        )

        print(f"[OK] MediaPipe loaded from {LANDMARKER_PATH}")
        print(f"[OK] Dataset dir: {DATA_DIR}")
        if use_https:
            print(f"[OK] Server starting -> https://{host}:{port} (adhoc cert)")
            socketio.run(
                app,
                host=host,
                port=port,
                ssl_context="adhoc",
                allow_unsafe_werkzeug=True,
            )
        else:
            print(f"[OK] Server starting -> http://{host}:{port}")
            socketio.run(app, host=host, port=port, allow_unsafe_werkzeug=True)
