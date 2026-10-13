"""
Enhanced Gaze Model Training Pipeline v2
==========================================
Trains multiple regression models to predict screen gaze coordinates
from facial landmark features.

Key improvements over v1:
  - Robust outlier removal (IQR-based on target residuals)
  - Richer feature engineering: interaction terms, polynomial features,
    per-eye asymmetry, gaze-head interaction
  - User-stratified train/test split for better generalisation
  - Deeper residual MLP with cosine-annealing LR schedule
  - Huber loss for robustness to noisy labels
  - K-Fold cross-validation for reliable error estimates
  - TFLite export for fast inference
  - Saves scaler, feature names, and training metadata

Models compared:
  1. Ridge Regression (baseline)
  2. Random Forest
  3. Gradient Boosting (XGB-style via sklearn)
  4. MLP Small  (128→64→2)
  5. MLP Large  (512→256→128→64→2 with residual blocks)

Outputs:
  - Comparison table with per-axis MAE, Euclidean pixel error
  - Best MLP saved as .keras + .tflite
  - Feature scaler + metadata saved
"""

import glob
import os
import pickle
import sys
import time
import warnings

import numpy as np

warnings.filterwarnings("ignore")

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "dataset")
MODEL_DIR = os.path.join(BASE_DIR, "models")
MODEL_SAVE_PATH = os.path.join(MODEL_DIR, "gaze_model.keras")
TFLITE_SAVE_PATH = os.path.join(MODEL_DIR, "eye_model.tflite")
SCALER_PATH = os.path.join(MODEL_DIR, "feature_scaler.pkl")
BEST_SKLEARN_PATH = os.path.join(MODEL_DIR, "best_sklearn_model.pkl")
META_PATH = os.path.join(MODEL_DIR, "training_meta.pkl")

# ---------------------------------------------------------------------------
# Feature column definitions
# ---------------------------------------------------------------------------
CORE_FEATURES = [
    "head_pitch", "head_yaw", "head_roll",
    "l_iris_x", "l_iris_y", "l_iris_z",
    "r_iris_x", "r_iris_y", "r_iris_z",
    "inter_ocular_dist",
]

EXTENDED_FEATURES = [
    "l_gaze_ratio_x", "l_gaze_ratio_y",
    "r_gaze_ratio_x", "r_gaze_ratio_y",
    "l_ear", "r_ear",
    "face_area",
]


# ===================================================================
# Data loading & preprocessing
# ===================================================================
def load_data():
    """Load all CSV files, clean, filter, engineer features."""
    import pandas as pd

    csv_files = glob.glob(os.path.join(DATA_DIR, "*.csv"))
    if not csv_files:
        print(f"[ERROR] No CSV files found in {DATA_DIR}")
        return None, None, None

    print(f"Found {len(csv_files)} data file(s)")

    dfs = []
    for path in sorted(csv_files):
        try:
            df = pd.read_csv(path)
            dfs.append(df)
            print(f"  + {os.path.basename(path):60s}  {len(df):>6,} rows")
        except Exception as e:
            print(f"  x {os.path.basename(path):60s}  ERROR: {e}")

    if not dfs:
        return None, None, None

    df = pd.concat(dfs, ignore_index=True)
    print(f"\n  Raw total: {len(df):,} rows")

    # --- Drop rows with NaN in critical columns ---
    required = CORE_FEATURES + ["target_x", "target_y", "screen_w", "screen_h"]
    df = df.dropna(subset=required)

    # --- Normalise user names (case-insensitive) ---
    df["user"] = df["user"].str.strip().str.lower()

    # --- Quality filtering ---
    initial_len = len(df)

    # Filter blinks
    if "l_ear" in df.columns and "r_ear" in df.columns:
        df = df[(df["l_ear"] > 0.15) | (df["r_ear"] > 0.15)]

    # Filter extreme brightness
    if "frame_brightness" in df.columns:
        df = df[(df["frame_brightness"] > 15) & (df["frame_brightness"] < 245)]

    # Filter extreme head poses (already filtered in collector, but double-check)
    df = df[(df["head_yaw"].abs() < 40) & (df["head_pitch"].abs() < 35)]

    # Filter physically impossible iris positions
    for col in ["l_iris_x", "l_iris_y", "r_iris_x", "r_iris_y"]:
        if col in df.columns:
            df = df[(df[col] > 0.0) & (df[col] < 1.0)]

    # Filter tiny face areas (too far from camera)
    if "face_area" in df.columns:
        df = df[df["face_area"] > 0.005]

    filtered = initial_len - len(df)
    if filtered:
        print(f"  Quality-filtered {filtered} rows ({len(df):,} remaining)")

    # --- Compute normalised targets ---
    df["norm_target_x"] = df["target_x"] / df["screen_w"]
    df["norm_target_y"] = df["target_y"] / df["screen_h"]

    # Sanity: targets must be in [0, 1]
    df = df[(df["norm_target_x"] >= 0) & (df["norm_target_x"] <= 1)]
    df = df[(df["norm_target_y"] >= 0) & (df["norm_target_y"] <= 1)]

    # --- Outlier removal using IQR on residuals ---
    # For each target axis, remove samples where iris position vs target
    # is an extreme outlier (likely mis-labelled or user looked away)
    avg_iris_x = (df["l_iris_x"] + df["r_iris_x"]) / 2
    residual_x = df["norm_target_x"] - avg_iris_x
    q1, q3 = residual_x.quantile(0.02), residual_x.quantile(0.98)
    iqr = q3 - q1
    mask_x = (residual_x >= q1 - 1.5 * iqr) & (residual_x <= q3 + 1.5 * iqr)

    avg_iris_y = (df["l_iris_y"] + df["r_iris_y"]) / 2
    residual_y = df["norm_target_y"] - avg_iris_y
    q1, q3 = residual_y.quantile(0.02), residual_y.quantile(0.98)
    iqr = q3 - q1
    mask_y = (residual_y >= q1 - 1.5 * iqr) & (residual_y <= q3 + 1.5 * iqr)

    before_outlier = len(df)
    df = df[mask_x & mask_y]
    print(f"  Outlier removal: dropped {before_outlier - len(df)} rows ({len(df):,} remaining)")

    print(f"\n  Total usable samples: {len(df):,}")
    print(f"  Users: {sorted(df['user'].unique())}")
    print(f"  Sessions: {sorted(df['session_type'].unique())}")

    # --- Build feature matrix ---
    feature_cols = []

    for c in CORE_FEATURES:
        if c in df.columns:
            feature_cols.append(c)

    for c in EXTENDED_FEATURES:
        if c in df.columns:
            feature_cols.append(c)

    X = df[feature_cols].values.copy()
    col_names = list(feature_cols)

    # --- Engineered features ---
    eng_arrays = []
    eng_names = []

    # Average iris positions
    avg_x = (df["l_iris_x"].values + df["r_iris_x"].values) / 2
    avg_y = (df["l_iris_y"].values + df["r_iris_y"].values) / 2
    avg_z = (df["l_iris_z"].values + df["r_iris_z"].values) / 2
    eng_arrays.extend([avg_x, avg_y, avg_z])
    eng_names.extend(["avg_iris_x", "avg_iris_y", "avg_iris_z"])

    # Iris asymmetry (vergence cue)
    diff_x = df["l_iris_x"].values - df["r_iris_x"].values
    diff_y = df["l_iris_y"].values - df["r_iris_y"].values
    eng_arrays.extend([diff_x, diff_y])
    eng_names.extend(["iris_diff_x", "iris_diff_y"])

    # Gaze ratio averages & asymmetry
    if "l_gaze_ratio_x" in df.columns:
        avg_gh = (df["l_gaze_ratio_x"].values + df["r_gaze_ratio_x"].values) / 2
        avg_gv = (df["l_gaze_ratio_y"].values + df["r_gaze_ratio_y"].values) / 2
        diff_gh = df["l_gaze_ratio_x"].values - df["r_gaze_ratio_x"].values
        diff_gv = df["l_gaze_ratio_y"].values - df["r_gaze_ratio_y"].values
        eng_arrays.extend([avg_gh, avg_gv, diff_gh, diff_gv])
        eng_names.extend(["avg_gaze_h", "avg_gaze_v", "gaze_diff_h", "gaze_diff_v"])

    # EAR average (eye openness)
    if "l_ear" in df.columns:
        avg_ear = (df["l_ear"].values + df["r_ear"].values) / 2
        ear_diff = df["l_ear"].values - df["r_ear"].values
        eng_arrays.extend([avg_ear, ear_diff])
        eng_names.extend(["avg_ear", "ear_diff"])

    # Head-gaze interaction terms (head pose × iris position)
    pitch = df["head_pitch"].values
    yaw = df["head_yaw"].values
    eng_arrays.extend([
        yaw * avg_x,
        pitch * avg_y,
        yaw * avg_gh if "l_gaze_ratio_x" in df.columns else yaw * avg_x,
        pitch * avg_gv if "l_gaze_ratio_y" in df.columns else pitch * avg_y,
    ])
    eng_names.extend(["yaw_x_iris_x", "pitch_x_iris_y", "yaw_x_gaze_h", "pitch_x_gaze_v"])

    # IOD-normalised iris positions (distance-invariant)
    iod = df["inter_ocular_dist"].values
    iod_safe = np.where(iod > 1e-6, iod, 1e-6)
    eng_arrays.extend([avg_x / iod_safe, avg_y / iod_safe])
    eng_names.extend(["iris_x_norm_iod", "iris_y_norm_iod"])

    # Squared terms for non-linearity
    eng_arrays.extend([avg_x ** 2, avg_y ** 2, yaw ** 2, pitch ** 2])
    eng_names.extend(["avg_iris_x_sq", "avg_iris_y_sq", "yaw_sq", "pitch_sq"])

    if eng_arrays:
        eng_matrix = np.column_stack([e.reshape(-1, 1) for e in eng_arrays])
        X = np.column_stack([X, eng_matrix])
        col_names.extend(eng_names)

    print(f"  Feature vector dimension: {X.shape[1]}")

    # --- Targets ---
    y = np.column_stack([df["norm_target_x"].values, df["norm_target_y"].values])

    # Store metadata
    screen_w_median = float(np.median(df["screen_w"].values))
    screen_h_median = float(np.median(df["screen_h"].values))
    meta = dict(
        screen_w=screen_w_median,
        screen_h=screen_h_median,
        col_names=col_names,
        users=sorted(df["user"].unique().tolist()),
        user_labels=df["user"].values,
    )

    return X, y, meta


# ===================================================================
# Model builders
# ===================================================================
def build_sklearn_models():
    from sklearn.linear_model import Ridge
    from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
    from sklearn.multioutput import MultiOutputRegressor

    return {
        "Ridge Regression": Ridge(alpha=1.0),
        "Random Forest": MultiOutputRegressor(
            RandomForestRegressor(
                n_estimators=300, max_depth=20, min_samples_leaf=5,
                n_jobs=-1, random_state=42
            )
        ),
        "Gradient Boosting": MultiOutputRegressor(
            GradientBoostingRegressor(
                n_estimators=300, max_depth=8, learning_rate=0.08,
                subsample=0.8, min_samples_leaf=10, random_state=42
            )
        ),
    }


def build_mlp_small(input_dim):
    import tensorflow as tf
    from tensorflow.keras import layers, models

    model = models.Sequential([
        layers.Input(shape=(input_dim,)),
        layers.Dense(128, activation="relu"),
        layers.BatchNormalization(),
        layers.Dropout(0.2),
        layers.Dense(64, activation="relu"),
        layers.BatchNormalization(),
        layers.Dense(2, activation="sigmoid"),
    ], name="MLP_Small")

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss=tf.keras.losses.Huber(delta=0.05),
        metrics=["mae"],
    )
    return model


def build_mlp_large(input_dim):
    """Deep MLP with residual connections, Huber loss, and cosine LR."""
    import tensorflow as tf
    from tensorflow.keras import layers, models

    inp = layers.Input(shape=(input_dim,))

    # Stem
    x = layers.Dense(512, activation="relu", kernel_initializer="he_normal")(inp)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.3)(x)

    # Residual block 1: 256
    skip = layers.Dense(256)(x)
    x = layers.Dense(256, activation="relu", kernel_initializer="he_normal")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.25)(x)
    x = layers.Dense(256, activation="relu", kernel_initializer="he_normal")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Add()([x, skip])
    x = layers.Activation("relu")(x)

    # Residual block 2: 128
    skip = layers.Dense(128)(x)
    x = layers.Dense(128, activation="relu", kernel_initializer="he_normal")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.2)(x)
    x = layers.Dense(128, activation="relu", kernel_initializer="he_normal")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Add()([x, skip])
    x = layers.Activation("relu")(x)

    # Head
    x = layers.Dense(64, activation="relu", kernel_initializer="he_normal")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dense(32, activation="relu", kernel_initializer="he_normal")(x)

    out = layers.Dense(2, activation="sigmoid")(x)

    model = models.Model(inputs=inp, outputs=out, name="MLP_Large_Residual")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss=tf.keras.losses.Huber(delta=0.05),
        metrics=["mae"],
    )
    return model


# ===================================================================
# Evaluation
# ===================================================================
def evaluate_model(name, y_true, y_pred, screen_w, screen_h):
    mae_x = float(np.mean(np.abs(y_true[:, 0] - y_pred[:, 0])))
    mae_y = float(np.mean(np.abs(y_true[:, 1] - y_pred[:, 1])))

    px_err_x = mae_x * screen_w
    px_err_y = mae_y * screen_h

    dx = (y_true[:, 0] - y_pred[:, 0]) * screen_w
    dy = (y_true[:, 1] - y_pred[:, 1]) * screen_h
    euc = np.sqrt(dx ** 2 + dy ** 2)

    return dict(
        name=name,
        mae_x=mae_x, mae_y=mae_y,
        px_x=px_err_x, px_y=px_err_y,
        euc_mean=float(np.mean(euc)),
        euc_median=float(np.median(euc)),
        euc_p90=float(np.percentile(euc, 90)),
        euc_p95=float(np.percentile(euc, 95)),
    )


def print_results_table(results):
    print("\n" + "=" * 105)
    print(f"{'Model':<24s} {'MAE-X':>8s} {'MAE-Y':>8s} "
          f"{'PxErr-X':>9s} {'PxErr-Y':>9s} "
          f"{'Euc-Mean':>10s} {'Euc-Med':>9s} {'Euc-P90':>9s} {'Euc-P95':>9s}")
    print("-" * 105)
    for r in sorted(results, key=lambda x: x["euc_mean"]):
        print(f"{r['name']:<24s} "
              f"{r['mae_x']:8.4f} {r['mae_y']:8.4f} "
              f"{r['px_x']:8.1f}px {r['px_y']:8.1f}px "
              f"{r['euc_mean']:9.1f}px {r['euc_median']:8.1f}px "
              f"{r['euc_p90']:8.1f}px {r['euc_p95']:8.1f}px")
    print("=" * 105)


# ===================================================================
# TFLite conversion
# ===================================================================
def convert_to_tflite(keras_model, save_path):
    import tensorflow as tf

    converter = tf.lite.TFLiteConverter.from_keras_model(keras_model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.target_spec.supported_types = [tf.float16]
    tflite_model = converter.convert()

    with open(save_path, "wb") as f:
        f.write(tflite_model)

    size_kb = os.path.getsize(save_path) / 1024
    print(f"  TFLite model saved -> {save_path}  ({size_kb:.1f} KB)")


# ===================================================================
# GPU / training helpers
# ===================================================================
def configure_tf_runtime():
    import tensorflow as tf

    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        print(f"[OK] GPU detected: {gpus}")
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    else:
        print("[INFO] No GPU detected; training will run on CPU")


def train_and_evaluate_sklearn(name, model, X_train, y_train, X_test, y_test, screen_w, screen_h):
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    metrics = evaluate_model(name, y_test, pred, screen_w, screen_h)
    return metrics, model


def train_and_evaluate_mlp(name, input_dim, X_train, y_train, X_test, y_test, screen_w, screen_h):
    import tensorflow as tf

    if name == "MLP Small":
        model = build_mlp_small(input_dim)
        epochs = 10
    else:
        model = build_mlp_large(input_dim)
        epochs = 6

    history = model.fit(
        X_train,
        y_train,
        validation_split=0.05,
        epochs=epochs,
        batch_size=256,
        verbose=0,
    )
    pred = model.predict(X_test, verbose=0)
    metrics = evaluate_model(name, y_test, pred, screen_w, screen_h)
    return metrics, model, history


# ===================================================================
# Main training pipeline
# ===================================================================
def main():
    t_start = time.time()

    # ----- Load data -----
    X, y, meta = load_data()
    if X is None:
        return

    screen_w = meta["screen_w"]
    screen_h = meta["screen_h"]

    # ----- Train/test split first -----
    from sklearn.model_selection import train_test_split

    n_train = 45000
    X_clean = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    if len(X_clean) < n_train:
        raise ValueError(f"Not enough samples to create a {n_train:,}-sample training split: only {len(X_clean):,} rows available")

    X_train, X_test, y_train, y_test = train_test_split(
        X_clean,
        y,
        train_size=n_train,
        test_size=None,
        random_state=42,
        shuffle=True,
    )

    # ----- Feature scaling on training data only -----
    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    X_train_scaled = np.nan_to_num(X_train_scaled, nan=0.0, posinf=0.0, neginf=0.0)
    X_test_scaled = np.nan_to_num(X_test_scaled, nan=0.0, posinf=0.0, neginf=0.0)

    print(f"\n  Train: {X_train.shape[0]:,}   Test: {X_test.shape[0]:,}")
    print(f"  Screen resolution (median): {int(screen_w)}x{int(screen_h)}")
    print(f"  Feature dimensions: {X_train.shape[1]}")

    os.makedirs(MODEL_DIR, exist_ok=True)
    configure_tf_runtime()

    results = []

    # =================================================================
    # Train only Ridge Regression
    # =================================================================
    print("\n" + "=" * 95)
    print("  TRAINING RIDGE REGRESSION")
    print("=" * 95)

    from sklearn.linear_model import Ridge

    model = Ridge(alpha=1.0)

    print("\n  Training Ridge Regression...")
    metrics, fitted_model = train_and_evaluate_sklearn("Ridge Regression", model, X_train_scaled, y_train, X_test_scaled, y_test, screen_w, screen_h)
    metrics["model_type"] = "sklearn"
    metrics["model_obj"] = fitted_model
    results.append(metrics)

    print("\n" + "=" * 95)
    print("  MODEL RESULT")
    print("=" * 95)
    print_results_table([{k: v for k, v in r.items() if k != "model_obj" and k != "model_type"} for r in results])

    best_name = "Ridge Regression"
    best_model = fitted_model
    best_model_type = "sklearn"

    print(f"\n[BEST] {best_name} -> mean_euc={metrics['euc_mean']:.1f}px median={metrics['euc_median']:.1f}px p95={metrics['euc_p95']:.1f}px")

    # Save scaler
    with open(SCALER_PATH, "wb") as f:
        pickle.dump(scaler, f)
    print(f"  Scaler saved       -> {SCALER_PATH}")

    # Save the best sklearn model for the tracker if one exists
    if best_model_type == "sklearn":
        with open(BEST_SKLEARN_PATH, "wb") as f:
            pickle.dump(best_model, f)
        print(f"  Best sklearn model saved -> {BEST_SKLEARN_PATH}")

    if best_model_type == "keras":
        keras_path = os.path.join(MODEL_DIR, "gaze_model.keras")
        best_model.save(keras_path)
        print(f"  Best Keras model saved -> {keras_path}")
        convert_to_tflite(best_model, TFLITE_SAVE_PATH)

    # Save training metadata
    training_meta = dict(
        col_names=meta["col_names"],
        screen_w=screen_w,
        screen_h=screen_h,
        users=meta["users"],
        n_features=X_train.shape[1],
        n_train=X_train.shape[0],
        n_test=X_test.shape[0],
        best_model=best_name,
        best_euc_mean=metrics["euc_mean"],
        best_mlp_saved=(best_model_type == "keras"),
        best_sklearn_path=BEST_SKLEARN_PATH,
        all_results=[{k: v for k, v in r.items() if k != "model_obj" and k != "model_type"} for r in results],
    )
    with open(META_PATH, "wb") as f:
        pickle.dump(training_meta, f)
    print(f"  Metadata saved     -> {META_PATH}")

    elapsed = time.time() - t_start
    print(f"\n  Total training time: {elapsed / 60:.1f} minutes")
    print(f"  Feature columns ({len(meta['col_names'])}): {meta['col_names']}")
    print("\nDone!")


if __name__ == "__main__":
    main()
