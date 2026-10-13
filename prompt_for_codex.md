# Context Brief & Instructions for Codex: Building a State-of-the-Art Gaze Estimation Model

**Hey Codex!** I am building a highly accurate, webcam-based gaze estimation model. I have a custom dataset and a working pipeline, but I need you to help me train the absolute *perfect* model for my specific data. Please read this entire briefing, analyze our findings, and provide your expert architecture, code, and training strategies.

---

## 1. Project Overview & Goal
- **Input**: 38 features extracted from MediaPipe Face Landmarks (iris center coords, eye contours, head pose, eye aspect ratio, face area, etc.).
- **Output**: Normalized screen coordinates `(x, y)` between `0.0` and `1.0`.
- **Goal**: Achieve maximum possible accuracy (sub-degree error) so the cursor perfectly tracks the user's eyes on a screen.

## 2. Dataset Profile
- **Size**: ~50,000 frames.
- **Users**: 3 unique users (with varied lighting conditions and head movements).
- **Features Include**:
  - `head_pitch`, `head_yaw`, `head_roll` (via RQDecomp3x3 of transformation matrix)
  - `l_iris_x`, `l_iris_y`, `r_iris_x`, `r_iris_y` (normalized bounding box centers)
  - `l_gaze_ratio_x`, `l_gaze_ratio_y`, `r_gaze_ratio_x`, `r_gaze_ratio_y` (iris position relative to eye contour)
  - `l_ear`, `r_ear` (Eye Aspect Ratios)
  - `inter_ocular_dist`, `face_area`
  - Plus 21 engineered polynomial/interaction features (e.g., `yaw * avg_iris_x`).

## 3. Critical Findings from Our Data Analysis
We have already done a deep statistical analysis on the dataset. Here is what you MUST account for:

1. **The Y-Axis Problem**: Horizontal gaze (`target_x`) correlates strongly with `gaze_ratio_x` (r = -0.87). However, vertical gaze (`target_y`) is fundamentally harder. `gaze_ratio_y` has a terrible correlation (r = -0.19). Vertical gaze instead relies heavily on `head_pitch` and `EAR` (squinting). **X and Y axes need different feature representations.**
2. **Resolution Noise**: The dataset contains samples collected across multiple screen resolutions (1536x864, 1920x1080, etc.). A physical angle mapping to a normalized pixel changes depending on screen aspect ratio.
3. **Cross-User vs. Same-User Gap**: A Ridge baseline gets ~239px Euclidean error on unseen users (Leave-One-User-Out), but drops to ~112px on an 80/20 split of the *same* user. **Personalized calibration is the biggest lever for accuracy.**
4. **Edge Bias**: The model predicts screen centers well (150px error), but struggles heavily at the extreme edges (<0.15 or >0.85 screen coordinates), jumping to >216px error.

## 4. State of the Art & Current V6 Architecture
Based on 2024/2025 research for MediaPipe gaze regression, here is what we've found works best:

- **Loss Function**: **Wing Loss** is vastly superior to MSE or Huber for coordinate regression because it amplifies gradients for small errors, preventing the model from getting "lazy" near the target.
- **Dual-Pathway MLP**: A neural network that splits into two separate heads (X-Head and Y-Head) after a shared backbone. This allows the Y-head to allocate more capacity to complex non-linear combinations of head pose and EAR.
- **Gradient Boosting**: **XGBoost (MultiOutputRegressor)** performs exceptionally well on this structured, tabular data, often beating basic MLPs due to its handling of non-linear feature interactions without needing scaling.
- **Calibration Layer**: Post-prediction, a 9-point calibration step using a **Polynomial Regression or Affine Transform** to correct user-specific anatomical offsets (e.g., angle Kappa).
- **Temporal Smoothing**: Applying a **One-Euro Filter** or **Kalman Filter** to the final predictions to eliminate MediaPipe jitter.

---

## 5. Your Task (Instructions for Codex)

Please act as a Senior Machine Learning Engineer and provide the following. **Above all else, I need you to figure out the absolute best possible way to train this model given the dataset profile and findings — think outside the box if necessary to achieve perfection.**

1. **Model Architecture Recommendations**:
   - Provide the PyTorch or TensorFlow/Keras code for a state-of-the-art **Dual-Pathway MLP** that perfectly handles the X/Y asymmetry mentioned in Finding #1.
   - Suggest the exact hyperparameters (depth, width, dropout, batch norm) we should use.

2. **Feature Engineering & Selection**:
   - Based on our features, what specific feature interactions should we create?
   - Should we drop the engineered polynomial features if they are causing overfitting on the cross-user validation?

3. **Loss Function Implementation**:
   - Provide a robust implementation of **Wing Loss** or **Adaptive Wing Loss** customized for `(x, y)` screen coordinates where the Y-axis is given a higher penalty weight.

4. **Training Strategy**:
   - How should we implement Curriculum Learning (e.g., training on center-screen data first, then edge data)?
   - What data augmentation strategies work best for 1D tabular landmark data (e.g., Gaussian noise injection, feature masking)?

5. **Calibration Code**:
   - Provide a Python snippet for a robust 2nd-order polynomial personal calibration step that takes raw model `(x, y)` predictions and 9 user-collected calibration points, mapping them to the true screen coordinates.

*I am counting on your expertise to synthesize this data and give me the code and mathematical strategies needed to achieve flawless gaze tracking!*
