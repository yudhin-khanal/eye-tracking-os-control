# Model Inference Pipeline

## 1. Input
- A webcam frame is captured.
- Face and eye landmarks are detected with MediaPipe.

## 2. Compute gaze ratios
- Left and right iris positions are normalized inside each eye contour.
- Result:
  - `l_gaze_h`, `l_gaze_v`
  - `r_gaze_h`, `r_gaze_v`
- This gives a simple 0–1 measure of iris position inside the eye.

## 3. Compute eye openness
- Eye Aspect Ratio (EAR) measures whether the eye is open or blinking.
- Used to skip bad frames.

## 4. Build feature vector
- Head pose: `pitch`, `yaw`, `roll`
- Iris centers: `lx`, `ly`, `lz`, `rx`, `ry`, `rz`
- Gaze ratios: left and right horizontal/vertical
- EAR values and face area
- A few combined features such as average gaze, gaze difference, and yaw/pitch interactions

## 5. Scale features
- The feature vector is normalized with a saved scaler before inference.

## 6. Predict raw gaze point
- The model outputs a normalized gaze point:
  - `pred_x`, `pred_y`
- These values are clipped to stay in `[0, 1]`.

## 7. Calibrate to screen
- A per-session affine transform maps raw predictions to real screen coordinates.
- This corrects for offset, scale, rotation, and skew.

## 8. Optional smoothing
- A simple smoothing filter makes the final gaze position steadier.

## Simple flow
1. Capture frame
2. Detect landmarks
3. Calculate gaze ratios and eye openness
4. Create feature vector
5. Scale features
6. Run model prediction
7. Apply calibration transform
8. Smooth final output
