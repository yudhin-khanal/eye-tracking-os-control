# MLP Architectures

## 1. MLP Small

- Name: `MLP_Small`
- Input: full engineered feature vector
- Layers:
  1. Dense(128, activation="relu")
  2. BatchNormalization
  3. Dropout(0.2)
  4. Dense(64, activation="relu")
  5. BatchNormalization
  6. Dense(2, activation="sigmoid")
- Loss: Huber(delta=0.05)
- Optimizer: Adam(lr=1e-3)
- Metric: MAE

## 2. MLP Large V1

- Name: `MLP_Large_Residual`
- Input: full engineered feature vector
- Layers:
  1. Dense(512, activation="relu")
  2. BatchNormalization
  3. Dropout(0.3)

  4. Residual block 1:
     - Dense(256, activation="relu")
     - BatchNormalization
     - Dropout(0.25)
     - Dense(256, activation="relu")
     - BatchNormalization
     - Add skip connection from the block input
     - Activation("relu")

  5. Residual block 2:
     - Dense(128, activation="relu")
     - BatchNormalization
     - Dropout(0.2)
     - Dense(128, activation="relu")
     - BatchNormalization
     - Add skip connection from the block input
     - Activation("relu")

  6. Dense(64, activation="relu")
  7. BatchNormalization
  8. Dense(32, activation="relu")
  9. Dense(2, activation="sigmoid")
- Loss: Huber(delta=0.05)
- Optimizer: Adam(lr=1e-3)
- Metric: MAE

## 3. MLP Large V2

- Name: `DualPathway_V6`
- Input: full engineered feature vector

### Shared backbone
  1. Dense(512) -> BatchNormalization -> Activation("relu") -> Dropout(0.3)

  2. Residual block 1:
     - Dense(256, activation="relu")
     - BatchNormalization
     - Dropout(0.2)
     - Dense(256, activation="relu")
     - BatchNormalization
     - Add skip connection
     - Activation("relu")

  3. Residual block 2:
     - Dense(128, activation="relu")
     - BatchNormalization
     - Dropout(0.15)
     - Dense(128, activation="relu")
     - BatchNormalization
     - Add skip connection
     - Activation("relu")

### X head (horizontal gaze)
  1. Dense(96, activation="relu")
  2. BatchNormalization
  3. Dropout(0.1)
  4. Dense(48, activation="relu")
  5. BatchNormalization
  6. Dense(24, activation="relu")
  7. Dense(1, activation="sigmoid")

### Y head (vertical gaze)
  1. Dense(128, activation="relu")
  2. BatchNormalization
  3. Dropout(0.1)
  4. Dense(64, activation="relu")
  5. BatchNormalization
  6. Dense(32, activation="relu")
  7. BatchNormalization
  8. Dense(16, activation="relu")
  9. Dense(1, activation="sigmoid")

- Final output: Concatenate X and Y head outputs into 2 values
- This design separates horizontal and vertical predictions for specialized learning.
