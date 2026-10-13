# Comprehensive Training Overview

## 1. Training Theory

The project is designed to learn a mapping from facial/eye geometry to normalized screen gaze coordinates.
It uses features derived from MediaPipe face landmarks, including head pose, iris center positions, gaze ratios, eye openness, and face size.
Because gaze prediction is inherently noisy, the pipeline emphasizes quality filtering, feature scaling, and robust regression.

## 2. Data Preparation

### 2.1 Data loading
- All CSV files in `dataset/` are combined into a single training set.
- Rows are dropped when any required gaze or target fields are missing.

### 2.2 Cleaning and filtering
- Blink frames are removed using left/right eye aspect ratios (EAR).
- Extremely dark or bright frames are removed using frame brightness.
- Extreme head poses are removed by limiting yaw and pitch.
- Invalid iris coordinates outside the normalized image range are removed.
- Very small face areas are removed to reject poorly framed faces.

### 2.3 Target normalization
- `target_x` and `target_y` are converted into normalized coordinates:
  - `norm_target_x = target_x / screen_w`
  - `norm_target_y = target_y / screen_h`
- Only targets within `[0, 1]` are retained.

### 2.4 Outlier filtering
- Samples whose gaze-derived residuals are extreme compared to the target are removed.
- The pipeline uses an IQR-style filter on the difference between average iris position and normalized target.

## 3. Feature Engineering

### 3.1 Core features
- `head_pitch`, `head_yaw`, `head_roll`
- left/right iris 3D centers: `l_iris_x/y/z`, `r_iris_x/y/z`
- `inter_ocular_dist`

### 3.2 Extended features
- left/right gaze ratios: `l_gaze_ratio_x/y`, `r_gaze_ratio_x/y`
- eye aspect ratios: `l_ear`, `r_ear`
- face size: `face_area`

### 3.3 Derived features
- average iris position and iris difference
- average gaze ratio and gaze ratio asymmetry
- average EAR and EAR difference
- head-pose interaction terms with iris and gaze ratios
- iris position normalized by inter-ocular distance
- squared terms for non-linear modeling

## 4. Scaling and Split

### 4.1 Train/Test split
- A fixed training set size is reserved using `sklearn.model_selection.train_test_split`.
- In the current training code, `45000` samples are used for training and the remainder for testing.
- Shuffling is enabled with a fixed random seed for reproducibility.

### 4.2 Scaling
- `StandardScaler` is fit only on the training features.
- Both training and testing features are transformed by the fitted scaler.
- Any NaN or infinite values after scaling are replaced with finite values.

## 5. Model Configuration

### 5.1 Candidate model families
- Ridge Regression: linear baseline with L2 regularization.
- Random Forest: tree-based ensemble regression.
- Gradient Boosting: boosted tree ensemble with learning rate control.
- MLP Small: shallow dense neural network.
- MLP Large: deeper residual MLP with dropout and batch normalization.

### 5.2 Neural network details
- Both MLP variants output two values in `[0, 1]` using a sigmoid activation.
- Loss function: Huber loss (`delta=0.05`), which is robust to noisy gaze labels.
- Optimizer: Adam with learning rate `1e-3`.
- Training uses a small validation split of `5%` from the training data.
- MLP Small runs for 10 epochs; MLP Large runs for 6 epochs.

## 6. Evaluation Metrics

### 6.1 Normalized coordinate error
- Mean Absolute Error (MAE) is computed separately for x and y:
  - `MAE-X = mean(|y_true_x - y_pred_x|)`
  - `MAE-Y = mean(|y_true_y - y_pred_y|)`
- These values are in normalized screen units.

### 6.2 Pixel error
- Normalized MAE is converted into pixels using the median screen width and height:
  - `px_x = MAE-X * screen_w`
  - `px_y = MAE-Y * screen_h`

### 6.3 Euclidean gaze error
- The 2D Euclidean pixel error is computed per sample:
  - `dx = (true_x - pred_x) * screen_w`
  - `dy = (true_y - pred_y) * screen_h`
  - `euclidean_error = sqrt(dx^2 + dy^2)`
- The pipeline reports:
  - mean Euclidean error
  - median Euclidean error
  - 90th percentile Euclidean error
  - 95th percentile Euclidean error

## 7. Error interpretation

- `MAE-X` and `MAE-Y` measure bias in the predicted normalized gaze coordinates per axis.
- Pixel errors show how far the predicted gaze would be on-screen in real distance.
- Euclidean error summarizes the final gaze miss distance, which is the most direct real-world measure of gaze accuracy.

## 8. Model export and inference

- After training, the selected model and fitted scaler are saved to `models/`.
- At inference time, live gaze features are scaled with the same scaler before prediction.
- The runtime output is then calibrated and optionally smoothed for on-screen control.
