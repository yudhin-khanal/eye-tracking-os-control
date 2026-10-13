"""Debug: check actual iris vs eye contour positions from a live webcam frame."""
import cv2, math, numpy as np, mediapipe as mp, os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "models", "face_landmarker.task")

LEFT_IRIS = [474, 475, 476, 477]
RIGHT_IRIS = [469, 470, 471, 472]
LEFT_EYE_CONTOUR = [33, 246, 161, 160, 159, 158, 157, 173, 133, 155, 154, 153, 145, 144, 163, 7]
RIGHT_EYE_CONTOUR = [263, 466, 388, 387, 386, 385, 384, 398, 362, 382, 381, 380, 374, 373, 390, 249]

# Old 2-point method
LEFT_EYE_INNER, LEFT_EYE_OUTER = 133, 33
RIGHT_EYE_INNER, RIGHT_EYE_OUTER = 362, 263

BaseOptions = mp.tasks.BaseOptions
FaceLandmarker = mp.tasks.vision.FaceLandmarker
FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

options = FaceLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=MODEL_PATH),
    running_mode=VisionRunningMode.IMAGE,
    output_face_blendshapes=False,
    output_facial_transformation_matrixes=True,
    num_faces=1,
    min_face_detection_confidence=0.5,
    min_face_presence_confidence=0.5,
)

landmarker = FaceLandmarker.create_from_options(options)

cap = cv2.VideoCapture(0)
ret, frame = cap.read()
cap.release()
if not ret:
    print("ERROR: Could not read from camera")
    exit(1)

rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
result = landmarker.detect(mp_image)

if not result.face_landmarks:
    print("ERROR: No face detected")
    exit(1)

lm = result.face_landmarks[0]

print(f"Frame shape: {frame.shape}")
print(f"\n=== LEFT EYE ===")
iris_x = sum(lm[i].x for i in LEFT_IRIS) / len(LEFT_IRIS)
iris_y = sum(lm[i].y for i in LEFT_IRIS) / len(LEFT_IRIS)
print(f"  Iris center: ({iris_x:.6f}, {iris_y:.6f})")

# Old method
inner_x = lm[LEFT_EYE_INNER].x
outer_x = lm[LEFT_EYE_OUTER].x
print(f"  OLD method: inner(133).x={inner_x:.6f}  outer(33).x={outer_x:.6f}")
print(f"  OLD x_lo={min(inner_x, outer_x):.6f}  x_hi={max(inner_x, outer_x):.6f}")
old_h = (iris_x - min(inner_x, outer_x)) / max(max(inner_x, outer_x) - min(inner_x, outer_x), 1e-7)
print(f"  OLD h_ratio = {old_h:.6f}  clipped = {np.clip(old_h, 0, 1):.6f}")

# New contour method
xs = [lm[i].x for i in LEFT_EYE_CONTOUR]
ys = [lm[i].y for i in LEFT_EYE_CONTOUR]
print(f"\n  CONTOUR method:")
for i, idx in enumerate(LEFT_EYE_CONTOUR):
    print(f"    lm[{idx}] x={lm[idx].x:.6f} y={lm[idx].y:.6f}")
print(f"  contour x_lo={min(xs):.6f}  x_hi={max(xs):.6f}")
print(f"  contour y_lo={min(ys):.6f}  y_hi={max(ys):.6f}")
new_h = (iris_x - min(xs)) / max(max(xs) - min(xs), 1e-7)
new_v = (iris_y - min(ys)) / max(max(ys) - min(ys), 1e-7)
print(f"  NEW h_ratio = {new_h:.6f}  clipped = {np.clip(new_h, 0, 1):.6f}")
print(f"  NEW v_ratio = {new_v:.6f}  clipped = {np.clip(new_v, 0, 1):.6f}")

print(f"\n=== RIGHT EYE ===")
iris_x_r = sum(lm[i].x for i in RIGHT_IRIS) / len(RIGHT_IRIS)
iris_y_r = sum(lm[i].y for i in RIGHT_IRIS) / len(RIGHT_IRIS)
print(f"  Iris center: ({iris_x_r:.6f}, {iris_y_r:.6f})")

inner_x_r = lm[RIGHT_EYE_INNER].x
outer_x_r = lm[RIGHT_EYE_OUTER].x
print(f"  OLD method: inner(362).x={inner_x_r:.6f}  outer(263).x={outer_x_r:.6f}")
old_h_r = (iris_x_r - min(inner_x_r, outer_x_r)) / max(max(inner_x_r, outer_x_r) - min(inner_x_r, outer_x_r), 1e-7)
print(f"  OLD h_ratio = {old_h_r:.6f}  clipped = {np.clip(old_h_r, 0, 1):.6f}")

xs_r = [lm[i].x for i in RIGHT_EYE_CONTOUR]
ys_r = [lm[i].y for i in RIGHT_EYE_CONTOUR]
print(f"\n  CONTOUR method:")
for i, idx in enumerate(RIGHT_EYE_CONTOUR):
    print(f"    lm[{idx}] x={lm[idx].x:.6f} y={lm[idx].y:.6f}")
print(f"  contour x_lo={min(xs_r):.6f}  x_hi={max(xs_r):.6f}")
new_h_r = (iris_x_r - min(xs_r)) / max(max(xs_r) - min(xs_r), 1e-7)
new_v_r = (iris_y_r - min(ys_r)) / max(max(ys_r) - min(ys_r), 1e-7)
print(f"  NEW h_ratio = {new_h_r:.6f}  clipped = {np.clip(new_h_r, 0, 1):.6f}")
print(f"  NEW v_ratio = {new_v_r:.6f}  clipped = {np.clip(new_v_r, 0, 1):.6f}")

print(f"\n=== IRIS LANDMARKS DETAIL ===")
for idx in LEFT_IRIS:
    print(f"  LEFT lm[{idx}]: x={lm[idx].x:.6f}  y={lm[idx].y:.6f}  z={lm[idx].z:.6f}")
for idx in RIGHT_IRIS:
    print(f"  RIGHT lm[{idx}]: x={lm[idx].x:.6f}  y={lm[idx].y:.6f}  z={lm[idx].z:.6f}")
