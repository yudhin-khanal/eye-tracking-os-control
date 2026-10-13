# Project Methodology

## 1. Data Collection
- Data is collected through the Flask-based web collector in `src/web_collector.py`.
- The collector captures webcam frames while the user follows on-screen gaze targets.
- MediaPipe FaceLandmarker extracts landmarks for eyes, iris, and face.
- Each recorded frame is saved as a CSV row with session metadata, target coordinates, landmark-derived features, and frame quality metrics.

## 2. Data Cleaning
- All CSV files in the `dataset/` directory are loaded and merged.
- Rows with missing critical values are dropped, especially the main eye/head features and target coordinates.
- User identifiers are normalized by trimming whitespace and converting to lowercase.
- Quality filters are applied to remove bad frames:
  - blink frames are removed using eye aspect ratio (EAR) thresholds,
  - extreme brightness frames are removed,
  - extreme head poses are removed,
  - invalid iris positions outside the normalized [0, 1] range are removed,
  - tiny face areas are removed to reject distant or partially visible faces.
- Target coordinates are normalized by screen width and height, and only valid normalized targets within [0, 1] are kept.
- Outlier removal uses residual-based filtering on the difference between target position and average iris position.

## 3. Feature Engineering
- Core features include head pose, left/right iris 3D centers, and inter-ocular distance.
- Extended features include left/right gaze ratios, eye aspect ratios, and face area.
- Additional derived features are constructed from the base measurements to better capture gaze geometry.

## 4. Scaling
- The feature matrix is converted to numeric arrays.
- A `StandardScaler` is fit on the training split only.
- Both training and test feature sets are transformed with the same scaler.
- Any NaNs or infinities are replaced with finite values after scaling.

## 5. Train/Test Split
- A fixed training split is created using `sklearn.model_selection.train_test_split`.
- The code reserves a specific training set size (45,000 samples) and uses the remainder for testing.
- Shuffling is enabled with a fixed random seed for reproducible splits.

## 6. Model Training and Evaluation
- The cleaned, scaled training data is used to fit regression models.
- The trained model predicts normalized gaze coordinates on the held-out test set.
- Evaluation is performed by comparing predicted normalized gaze positions to actual normalized targets.

## 7. Model Export
- The resulting scaler and trained model are saved to the `models/` directory.
- The saved scaler is reused at inference time so runtime features are normalized consistently.
- The trained model is later loaded for live gaze prediction and calibration.
