# Eye-Tracking Model Explanation and Defense Notes

## 1. Project overview

This project builds a webcam-based gaze tracking system that predicts where a user is looking on the screen.

The pipeline is:

1. Capture webcam frames.
2. Detect the face and eye landmarks using MediaPipe.
3. Convert those landmarks into gaze-related features.
4. Feed those features into a regression model.
5. Convert the raw prediction into screen coordinates using calibration.
6. Smooth the final cursor movement to reduce jitter.

The core idea is simple: the model learns a relationship between facial and eye features and the user’s target location on the screen.

---

## 2. What the model learns

The training pipeline uses features such as:

- head pose: pitch, yaw, roll
- iris positions: left and right eye iris centers
- gaze ratios: how the eye is looking relative to the eyelid contour
- eye openness: EAR values
- face area
- engineered interaction features: combinations of head pose and iris position

These features are used because they contain meaningful cues about where the user is looking.

For example:

- If the iris center moves to the right, the gaze is likely more rightward.
- If the head is tilted, the eye position alone may not be enough, so head pose helps the model compensate.
- Eye openness and gaze ratios add more information about the direction of attention.

The target is normalized screen coordinates:

- target_x / screen_width
- target_y / screen_height

So the model predicts a point in normalized space, not a raw pixel location directly.

---

## 3. How the data is prepared

### 3.1 Data collection

The system collects many samples from webcam sessions. Each sample contains:

- visual features from the face and eyes
- the actual target point the user looked at
- screen dimensions

### 3.2 Cleaning and filtering

Before training, the project removes poor samples.

Examples of removal include:

- frames with low eye openness (possible blink)
- extreme lighting conditions
- extreme head pose values
- impossible iris coordinates
- very small face area

This helps prevent the model from learning bad examples.

### 3.3 Feature scaling

The model is trained on standardized features.

This is done with a scaler saved as:

- [models/feature_scaler.pkl](models/feature_scaler.pkl)

Why scaling matters:

- Some features are naturally larger than others.
- Without scaling, one feature could dominate the model.
- Scaling makes training more stable and improves optimization.

### 3.4 Train/test split

The dataset is split into training and test data.

This is important because it allows us to measure generalization.

The model is trained on the training set, and then evaluated on the held-out test set. That tells us whether the model learned real patterns or just memorized the data.

---

## 4. Models used in the training pipeline

The training script compares several model families.

### 4.1 Ridge Regression

Ridge Regression is the baseline model.

It learns a linear mapping from features to target coordinates.

Why it is useful:

- very fast to train and infer
- simple and stable
- easy to explain
- often works well as a strong baseline

In this project, Ridge Regression is important because:

- it is lightweight for real-time use
- it provides a strong baseline against more complex models
- it is the model currently used by the live tracker through [models/best_sklearn_model.pkl](models/best_sklearn_model.pkl)

Even though it is simple, it can perform surprisingly well for this kind of structured tabular data.

### 4.2 Random Forest

Random Forest builds many decision trees and averages their predictions.

Why it can help:

- it captures nonlinear relationships
- it is robust to noise
- it does not require the same level of feature engineering as linear models

However, it is usually slower than Ridge Regression and may be less smooth in real-time applications.

### 4.3 Gradient Boosting

Gradient Boosting builds trees sequentially, where each next tree corrects the mistakes of earlier ones.

Why it can help:

- often strong performance on tabular data
- can capture nonlinear patterns well
- often better than basic linear models

It can be powerful but is typically heavier than Ridge Regression and may be less convenient for low-latency inference.

### 4.4 MLP Small

The MLP Small model is a small neural network.

It contains a few dense layers and learns a nonlinear mapping from features to gaze coordinates.

Why it is useful:

- it can capture more complex relationships than linear models
- it can be more expressive when the problem is nonlinear

Why it may not always win:

- more sensitive to training setup
- may require more careful tuning
- can be slower at inference than a simple sklearn model

### 4.5 MLP Large

The MLP Large model is a deeper neural network with more layers and residual blocks.

It is designed to learn richer patterns.

Why it is useful:

- stronger nonlinear modeling power
- can fit more complicated mappings
- may outperform simpler models if the data is rich enough

Why it may not always be the best choice:

- larger models are usually slower
- they can overfit if the dataset is limited
- they may need more careful regularization and tuning

---

## 5. Why Ridge Regression still matters here

Ridge Regression is not “just a weak baseline.” In this project, it plays a very practical role.

It is useful because:

- the task is structured and tabular, not image-to-image
- the features are hand-engineered and meaningful
- the dataset is not huge enough to always justify a very deep network
- real-time inference demands a lightweight model

In many eye-tracking systems, the relationship between gaze and engineered features is fairly smooth and linear enough that a regularized linear model performs well.

So Ridge Regression acts as:

- a simple and reliable first model
- a strong benchmark
- a fast runtime model for live tracking

This is why the tracker currently loads [models/best_sklearn_model.pkl](models/best_sklearn_model.pkl) and uses it for live inference.

---

## 6. How the model is inferred at runtime

Inference means taking a new frame and producing a screen coordinate prediction.

### 6.1 Frame-to-feature pipeline

For a new webcam frame:

1. MediaPipe detects the face and landmarks.
2. The eye and iris points are extracted.
3. Features are computed from those landmarks.
4. The feature vector is built in the same order used during training.
5. The feature vector is scaled using the saved scaler.
6. The scaled features are passed into the model.
7. The model outputs normalized coordinates.

### 6.2 Why the feature order matters

The model expects features in a fixed order.

That is why the training metadata is stored in:

- [models/training_meta.pkl](models/training_meta.pkl)

If the order changes, the model may receive the wrong input and give poor predictions.

### 6.3 Why the scaler is needed

The scaler transforms new features into the same scale used during training.

Without the scaler, the model may see values in a very different range and produce unstable predictions.

---

## 7. How smoothing works

The raw output from the model can be noisy. A gaze predictor may jump around a little from frame to frame.

That is why smoothing is added.

### 7.1 Current method: exponential smoothing

The current tracker uses a simple exponential smoothing rule:

- new_value = 0.75 * previous_value + 0.25 * predicted_value

This makes the cursor move more smoothly and reduces jitter.

### 7.2 Why smoothing is useful

Smoothing helps with:

- small prediction noise
- unstable landmark detection
- sudden frame-to-frame fluctuations

However, smoothing does not fix a fundamentally bad model prediction. It only makes the trajectory look more stable.

### 7.3 One-Euro filter

A One-Euro filter is a more advanced smoothing method.

It adapts the smoothing strength based on how quickly the signal changes.

In simple terms:

- if the gaze is moving quickly, it allows faster response
- if the gaze is noisy, it smooths more aggressively

This can make the cursor feel more responsive and less jittery.

In this project, One-Euro smoothing was added for testing and can be enabled as an alternative to the existing smoothing method.

---

## 8. How error is calculated

This is one of the most important parts of the project.

### 8.1 Ground truth and prediction

During training, the model learns from examples where:

- the input features are known
- the true target point is known

The model predicts a normalized point.

Then the prediction is compared against the true normalized point.

### 8.2 Mean Absolute Error (MAE)

MAE measures the average absolute distance between prediction and true value.

It is computed separately for X and Y:

- MAE-X
- MAE-Y

This gives a sense of how much the model is off on each axis.

### 8.3 Pixel error

Because the screen size matters, the normalized error is converted into pixel error.

This is done by multiplying the normalized error by the screen width or height.

So if the model is off by 0.01 in normalized X and the screen width is 1920, the error is about 19.2 pixels along X.

### 8.4 Euclidean error

The most common overall error measure is Euclidean distance.

This combines X and Y error into one number using the Pythagorean formula:

- Euclidean error = sqrt((dx)^2 + (dy)^2)

This is very useful because it tells us the overall distance between the predicted point and the true point.

### 8.5 Why median, p90, and p95 are reported

The mean error can be affected by a few bad samples.

So the training script also reports:

- median error: the middle value of the error distribution
- p90: 90th percentile of error
- p95: 95th percentile of error

These metrics tell us:

- typical performance
- how often the model performs poorly
- how robust it is to outliers

### 8.6 Why the reported error and live error are different

There are two different kinds of error in this project:

1. Offline test error
   - measured on held-out test data during training
   - shows how well the model generalizes to unseen samples

2. Live calibration error
   - measured during calibration in the tracker
   - shows how well the calibration mapping fits the real-world screen mapping

A model can have good offline error but still show larger live error because of:

- lighting changes
- head movement
- camera distortion
- calibration mismatch
- user-specific behavior

So offline error and live tracking accuracy are related, but not identical.

---

## 9. How calibration makes the model better

The trained model predicts normalized gaze coordinates.

But the raw model output is not always directly aligned with the actual screen point the user is looking at.

That is why calibration is essential.

### 9.1 Why calibration is needed

Calibration learns a mapping from the model’s raw output to the actual screen position.

For example:

- the model might predict a point a bit too far left
- the mapping might need a shift, rotation, or scaling

Calibration corrects these differences.

### 9.2 What calibration does

Calibration collects samples at known screen points and learns a transformation.

This may involve:

- translation
- scaling
- rotation
- affine mapping
- more complex nonlinear mapping

In this project, the calibration routine uses a grid of points and learns how to map the raw model output onto the screen.

### 9.3 Why calibration helps

Calibration improves the system because:

- it reduces systematic bias
- it corrects user-specific offset
- it adapts the model output to the individual user
- it makes the cursor more accurate and stable

Without calibration, the model may be directionally correct but still offset from the real target.

### 9.4 Why calibration is not magic

Calibration cannot fully fix a poor model.

If the model is fundamentally inaccurate, calibration may only partially correct it.

So the best system uses both:

- a good predictive model
- a good calibration step

---

## 10. Why the system can still be imperfect

There are several reasons the system may not be perfect:

- camera quality and lighting
- face landmark detection noise
- user movement
- head pose variation
- calibration points not sampled well
- model mismatch to the specific user

That is why the error is often much larger in live tracking than in offline evaluation.

---

## 11. Short summary

The system works like this:

1. Extract features from the face and eyes.
2. Use a trained regression model to estimate where the user is looking.
3. Use calibration to map that estimate onto the screen.
4. Smooth the output so the cursor feels stable.

The model quality matters, but calibration and smoothing are also crucial for making the experience feel accurate and usable.

---

## 12. Defense-style Q&A

### Q1: Why is Ridge Regression used if the problem is nonlinear?

Because it is simple, fast, explainable, and often works well as a baseline. It is also practical for real-time use. The system compares it against more complex models and uses the one that is most suitable for live inference.

### Q2: Why is smoothing necessary?

Because raw model predictions can be noisy from frame to frame. Smoothing makes the cursor more stable and improves the user experience.

### Q3: Does smoothing improve the true model accuracy?

Not directly. Smoothing improves stability and perceived quality, but it does not fix the underlying model error.

### Q4: Why is calibration important?

Because the raw model output is not always directly aligned with the actual screen position. Calibration adapts the mapping to the user and the screen geometry.

### Q5: Why is error measured on a test set?

Because it shows whether the model generalizes to unseen data. A model that looks good on training data but fails on new data is not useful.

### Q6: Why do we report MAE, median, and p95?

Because each metric tells a different story. MAE captures average error, median gives typical error, and p95 shows robustness to larger failures.

### Q7: Why might live tracking error be larger than offline test error?

Because live tracking includes real-world factors such as lighting, head movement, landmark noise, and calibration mismatch.

### Q8: Why use both model prediction and calibration?

Because the model gives a good estimate of gaze direction, while calibration converts that estimate into the correct screen position for each user.

### Q9: Why is the system still not perfect?

Because gaze estimation from a single webcam is a difficult problem and many factors can introduce noise. The system is a practical and real-time approximation rather than a perfect biometric system.

### Q10: Why is this approach still valuable?

Because it shows a working pipeline for real-time gaze estimation, provides a clear training and inference workflow, and can be improved further with better calibration and more robust modeling.

---

## 13. A polished defense statement

This system combines a trained gaze regression model, a calibration mapping, and smoothing to provide a practical real-time eye-tracking experience. The model learns from engineered facial and eye features, Ridge Regression serves as a strong and efficient baseline, and more complex models such as Random Forests, Gradient Boosting, and MLPs are compared to improve performance. The model’s accuracy is evaluated using held-out test data with metrics such as MAE, Euclidean pixel error, median error, and percentile error. Calibration then adapts the raw predictions to the user and screen geometry, while smoothing improves stability and reduces jitter. Together, these components form a usable and explainable gaze-tracking pipeline.
