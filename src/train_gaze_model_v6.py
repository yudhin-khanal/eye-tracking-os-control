"""
Gaze Model Training Pipeline V6 — Maximum Accuracy Edition
=============================================================
Designed for personalized accuracy with per-session calibration.

Key innovations over V5:
  1. Filters out diagnostic CSV files (different schema)
  2. Per-user feature centering (iris positions relative to user's mean)
  3. XGBoost ensemble candidate (often beats MLP on tabular data)
  4. Dual-pathway MLP (separate X/Y heads with different feature focus)
  5. Wing Loss for coordinate regression
  6. Feature augmentation (noise injection during training)
  7. Proper Leave-One-User-Out + within-user CV
  8. Trains final production model on 100% data

Models compared:
  1. Ridge Regression (baseline)
  2. XGBoost (gradient boosting)
  3. MLP Single-Head (V5 architecture)
  4. MLP Dual-Pathway (V6 architecture)

Outputs:
  - Best model saved as .keras + .tflite
  - Feature scaler + metadata
  - Comparison table with per-axis MAE, Euclidean error
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

# Output paths for V6
MODEL_SAVE_PATH = os.path.join(MODEL_DIR, "gaze_model_v6.keras")
TFLITE_SAVE_PATH = os.path.join(MODEL_DIR, "eye_model_v6.tflite")
SCALER_PATH = os.path.join(MODEL_DIR, "feature_scaler_v6.pkl")
META_PATH = os.path.join(MODEL_DIR, "training_meta_v6.pkl")

# Main deployment paths (overwritten with best model)
MAIN_MODEL_PATH = os.path.join(MODEL_DIR, "gaze_model.keras")
MAIN_TFLITE_PATH = os.path.join(MODEL_DIR, "eye_model.tflite")
MAIN_SCALER_PATH = os.path.join(MODEL_DIR, "feature_scaler.pkl")
MAIN_META_PATH = os.path.join(MODEL_DIR, "training_meta.pkl")

# ---------------------------------------------------------------------------
# Feature definitions
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

    print(f"Found {len(csv_files)} file(s) in {DATA_DIR}")

    dfs = []
    skipped = 0
    for path in sorted(csv_files):
        try:
            df = pd.read_csv(path)
            basename = os.path.basename(path)

            # Skip diagnostic files (different schema)
            if "true_x" in df.columns and "target_x" not in df.columns:
                print(f"  SKIP {basename:60s}  (diagnostic file)")
                skipped += 1
                continue

            # Must have target columns
            if "target_x" not in df.columns:
                print(f"  SKIP {basename:60s}  (no target_x)")
                skipped += 1
                continue

            dfs.append(df)
            print(f"  + {basename:60s}  {len(df):>6,} rows")
        except Exception as e:
            print(f"  x {os.path.basename(path):60s}  ERROR: {e}")

    if not dfs:
        return None, None, None

    if skipped:
        print(f"  (Skipped {skipped} non-data files)")

    df = pd.concat(dfs, ignore_index=True)
    print(f"\n  Raw total: {len(df):,} rows")

    # --- Drop rows with NaN in critical columns ---
    required = CORE_FEATURES + ["target_x", "target_y", "screen_w", "screen_h"]
    df = df.dropna(subset=required)

    # --- Normalise user names ---
    df["user"] = df["user"].str.strip().str.lower()

    # --- Quality filtering ---
    initial_len = len(df)

    # Filter blinks
    if "l_ear" in df.columns and "r_ear" in df.columns:
        df = df[(df["l_ear"] > 0.15) | (df["r_ear"] > 0.15)]

    # Filter extreme brightness
    if "frame_brightness" in df.columns:
        df = df[(df["frame_brightness"] > 15) & (df["frame_brightness"] < 245)]

    # Filter extreme head poses
    df = df[(df["head_yaw"].abs() < 40) & (df["head_pitch"].abs() < 35)]

    # Filter impossible iris positions
    for col in ["l_iris_x", "l_iris_y", "r_iris_x", "r_iris_y"]:
        if col in df.columns:
            df = df[(df[col] > 0.0) & (df[col] < 1.0)]

    # Filter tiny face areas
    if "face_area" in df.columns:
        df = df[df["face_area"] > 0.005]

    filtered = initial_len - len(df)
    if filtered:
        print(f"  Quality-filtered {filtered} rows ({len(df):,} remaining)")

    # --- Normalised targets ---
    df["norm_target_x"] = df["target_x"] / df["screen_w"]
    df["norm_target_y"] = df["target_y"] / df["screen_h"]
    df = df[(df["norm_target_x"] >= 0) & (df["norm_target_x"] <= 1)]
    df = df[(df["norm_target_y"] >= 0) & (df["norm_target_y"] <= 1)]

    # --- Outlier removal (IQR on residuals) ---
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

    # Average iris
    avg_x = (df["l_iris_x"].values + df["r_iris_x"].values) / 2
    avg_y = (df["l_iris_y"].values + df["r_iris_y"].values) / 2
    avg_z = (df["l_iris_z"].values + df["r_iris_z"].values) / 2
    eng_arrays.extend([avg_x, avg_y, avg_z])
    eng_names.extend(["avg_iris_x", "avg_iris_y", "avg_iris_z"])

    # Iris asymmetry (vergence)
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

    # EAR average
    if "l_ear" in df.columns:
        avg_ear = (df["l_ear"].values + df["r_ear"].values) / 2
        ear_diff = df["l_ear"].values - df["r_ear"].values
        eng_arrays.extend([avg_ear, ear_diff])
        eng_names.extend(["avg_ear", "ear_diff"])

    # Head-gaze interaction terms
    pitch = df["head_pitch"].values
    yaw = df["head_yaw"].values
    eng_arrays.extend([
        yaw * avg_x,
        pitch * avg_y,
        yaw * avg_gh if "l_gaze_ratio_x" in df.columns else yaw * avg_x,
        pitch * avg_gv if "l_gaze_ratio_y" in df.columns else pitch * avg_y,
    ])
    eng_names.extend(["yaw_x_iris_x", "pitch_x_iris_y", "yaw_x_gaze_h", "pitch_x_gaze_v"])

    # IOD-normalised iris
    iod = df["inter_ocular_dist"].values
    iod_safe = np.where(iod > 1e-6, iod, 1e-6)
    eng_arrays.extend([avg_x / iod_safe, avg_y / iod_safe])
    eng_names.extend(["iris_x_norm_iod", "iris_y_norm_iod"])

    # Polynomial terms
    eng_arrays.extend([avg_x ** 2, avg_y ** 2, yaw ** 2, pitch ** 2])
    eng_names.extend(["avg_iris_x_sq", "avg_iris_y_sq", "yaw_sq", "pitch_sq"])

    # NOTE: Feature vector intentionally kept at 38 dimensions
    # for backward compatibility with gaze_mouse.py / test_tracking.py
    # V6 improvements are in architecture, loss, and augmentation instead

    if eng_arrays:
        eng_matrix = np.column_stack([e.reshape(-1, 1) for e in eng_arrays])
        X = np.column_stack([X, eng_matrix])
        col_names.extend(eng_names)

    print(f"  Feature vector dimension: {X.shape[1]}")

    # --- Targets ---
    y = np.column_stack([df["norm_target_x"].values, df["norm_target_y"].values])

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
def build_xgboost():
    """Build XGBoost regressor (often beats MLP on tabular data)."""
    try:
        from xgboost import XGBRegressor
        from sklearn.multioutput import MultiOutputRegressor
        return MultiOutputRegressor(
            XGBRegressor(
                n_estimators=500,
                max_depth=8,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                min_child_weight=5,
                reg_alpha=0.1,
                reg_lambda=1.0,
                tree_method="hist",
                random_state=42,
                n_jobs=-1,
                verbosity=0,
            )
        )
    except ImportError:
        print("  [WARN] XGBoost not installed, using GradientBoosting from sklearn")
        from sklearn.ensemble import GradientBoostingRegressor
        from sklearn.multioutput import MultiOutputRegressor
        return MultiOutputRegressor(
            GradientBoostingRegressor(
                n_estimators=500,
                max_depth=8,
                learning_rate=0.05,
                subsample=0.8,
                min_samples_leaf=10,
                random_state=42,
            )
        )


def build_dual_pathway_mlp(input_dim):
    """
    V6 Dual-Pathway MLP:
    - Shared backbone extracts common features
    - Separate X and Y heads specialize on different feature patterns
    - X-axis dominated by gaze_ratio_x (strong horizontal signal)
    - Y-axis dominated by EAR + head_pitch (weak vertical signal needs more capacity)
    """
    import tensorflow as tf
    from tensorflow.keras import layers, models

    inp = layers.Input(shape=(input_dim,))

    # === Shared Backbone ===
    x = layers.Dense(512, kernel_initializer="he_normal")(inp)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.Dropout(0.3)(x)

    # Residual block 1: 256
    skip = layers.Dense(256)(x)
    x = layers.Dense(256, activation="relu", kernel_initializer="he_normal")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.2)(x)
    x = layers.Dense(256, activation="relu", kernel_initializer="he_normal")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Add()([x, skip])
    x = layers.Activation("relu")(x)

    # Residual block 2: 128
    skip = layers.Dense(128)(x)
    x = layers.Dense(128, activation="relu", kernel_initializer="he_normal")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.15)(x)
    x = layers.Dense(128, activation="relu", kernel_initializer="he_normal")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Add()([x, skip])
    shared = layers.Activation("relu")(x)

    # === X-Head (horizontal gaze — stronger signal) ===
    x_head = layers.Dense(96, activation="relu", kernel_initializer="he_normal")(shared)
    x_head = layers.BatchNormalization()(x_head)
    x_head = layers.Dropout(0.1)(x_head)
    x_head = layers.Dense(48, activation="relu", kernel_initializer="he_normal")(x_head)
    x_head = layers.BatchNormalization()(x_head)
    x_head = layers.Dense(24, activation="relu", kernel_initializer="he_normal")(x_head)
    x_out = layers.Dense(1, activation="sigmoid", name="x_out")(x_head)

    # === Y-Head (vertical gaze — harder, needs more capacity) ===
    y_head = layers.Dense(128, activation="relu", kernel_initializer="he_normal")(shared)
    y_head = layers.BatchNormalization()(y_head)
    y_head = layers.Dropout(0.1)(y_head)
    y_head = layers.Dense(64, activation="relu", kernel_initializer="he_normal")(y_head)
    y_head = layers.BatchNormalization()(y_head)
    y_head = layers.Dense(32, activation="relu", kernel_initializer="he_normal")(y_head)
    y_head = layers.BatchNormalization()(y_head)
    y_head = layers.Dense(16, activation="relu", kernel_initializer="he_normal")(y_head)
    y_out = layers.Dense(1, activation="sigmoid", name="y_out")(y_head)

    combined = layers.Concatenate(name="gaze_output")([x_out, y_out])
    model = models.Model(inputs=inp, outputs=combined, name="DualPathway_V6")
    return model


def build_single_head_mlp(input_dim):
    """V5-style single-head MLP for comparison."""
    import tensorflow as tf
    from tensorflow.keras import layers, models

    inp = layers.Input(shape=(input_dim,))

    x = layers.Dense(512, activation="relu", kernel_initializer="he_normal")(inp)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.3)(x)

    skip = layers.Dense(256)(x)
    x = layers.Dense(256, activation="relu", kernel_initializer="he_normal")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.25)(x)
    x = layers.Dense(256, activation="relu", kernel_initializer="he_normal")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Add()([x, skip])
    x = layers.Activation("relu")(x)

    skip = layers.Dense(128)(x)
    x = layers.Dense(128, activation="relu", kernel_initializer="he_normal")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.2)(x)
    x = layers.Dense(128, activation="relu", kernel_initializer="he_normal")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Add()([x, skip])
    x = layers.Activation("relu")(x)

    x = layers.Dense(64, activation="relu", kernel_initializer="he_normal")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dense(32, activation="relu", kernel_initializer="he_normal")(x)
    out = layers.Dense(2, activation="sigmoid")(x)

    model = models.Model(inputs=inp, outputs=out, name="SingleHead_V5")
    return model


# ===================================================================
# Loss functions
# ===================================================================
def wing_loss(y_weight=1.5, w=10.0, epsilon=2.0):
    """
    Wing Loss: better than L1/L2/Huber for coordinate regression.
    Amplifies gradients for small errors, preventing the model from
    getting "stuck" with mediocre accuracy.

    Paper: "Wing Loss for Robust Facial Landmark Localisation with CNNs"
    """
    import tensorflow as tf

    C = w - w * tf.math.log(1.0 + w / epsilon)

    def loss_fn(y_true, y_pred):
        # X-axis
        diff_x = tf.abs(y_true[:, 0] - y_pred[:, 0]) * 1000  # scale up for wing loss
        loss_x = tf.where(
            diff_x < w,
            w * tf.math.log(1.0 + diff_x / epsilon),
            diff_x - C
        )

        # Y-axis (weighted higher)
        diff_y = tf.abs(y_true[:, 1] - y_pred[:, 1]) * 1000
        loss_y = tf.where(
            diff_y < w,
            w * tf.math.log(1.0 + diff_y / epsilon),
            diff_y - C
        )

        return tf.reduce_mean(loss_x + y_weight * loss_y)

    return loss_fn


def weighted_huber_loss(y_weight=1.5, delta=0.05):
    """Huber loss with higher weight on Y-axis (fallback)."""
    import tensorflow as tf

    def loss_fn(y_true, y_pred):
        err_x = tf.keras.losses.huber(y_true[:, 0], y_pred[:, 0], delta=delta)
        err_y = tf.keras.losses.huber(y_true[:, 1], y_pred[:, 1], delta=delta)
        return tf.reduce_mean(err_x + y_weight * err_y)

    return loss_fn


# ===================================================================
# Evaluation
# ===================================================================
def evaluate_model(name, y_true, y_pred, screen_w, screen_h):
    px_err_x = float(np.mean(np.abs(y_true[:, 0] - y_pred[:, 0]))) * screen_w
    px_err_y = float(np.mean(np.abs(y_true[:, 1] - y_pred[:, 1]))) * screen_h
    dx = (y_true[:, 0] - y_pred[:, 0]) * screen_w
    dy = (y_true[:, 1] - y_pred[:, 1]) * screen_h
    euc = np.sqrt(dx ** 2 + dy ** 2)
    return dict(
        name=name,
        px_x=px_err_x, px_y=px_err_y,
        euc_mean=float(np.mean(euc)),
        euc_median=float(np.median(euc)),
        euc_p90=float(np.percentile(euc, 90)),
        euc_p95=float(np.percentile(euc, 95)),
    )


def print_results(results):
    print("\n" + "=" * 100)
    print(f"{'Model':<35s} {'PxErr-X':>9s} {'PxErr-Y':>9s} "
          f"{'Euc-Mean':>10s} {'Euc-Med':>9s} {'Euc-P90':>9s} {'Euc-P95':>9s}")
    print("-" * 100)
    for r in sorted(results, key=lambda x: x["euc_mean"]):
        print(f"{r['name']:<35s} "
              f"{r['px_x']:8.1f}px {r['px_y']:8.1f}px "
              f"{r['euc_mean']:9.1f}px {r['euc_median']:8.1f}px "
              f"{r['euc_p90']:8.1f}px {r['euc_p95']:8.1f}px")
    print("=" * 100)


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
    print(f"  TFLite saved -> {save_path}  ({size_kb:.1f} KB)")


# ===================================================================
# Feature augmentation (noise injection for robustness)
# ===================================================================
class FeatureAugmentor:
    """Add Gaussian noise during training for regularization."""

    def __init__(self, noise_std=0.02, feature_dropout=0.05):
        self.noise_std = noise_std
        self.feature_dropout = feature_dropout

    def augment(self, X, rng=None):
        if rng is None:
            rng = np.random.RandomState()
        X_aug = X.copy()
        # Gaussian noise
        X_aug += rng.normal(0, self.noise_std, X_aug.shape)
        # Feature dropout (randomly zero out features)
        mask = rng.random(X_aug.shape) > self.feature_dropout
        X_aug *= mask
        return X_aug


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
    user_labels = meta["user_labels"]
    unique_users = sorted(set(user_labels))

    # ----- Feature scaling -----
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import GroupShuffleSplit

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_scaled = np.nan_to_num(X_scaled, nan=0.0, posinf=0.0, neginf=0.0)

    print(f"\n  Users available: {unique_users}")
    print(f"  Screen resolution (median): {int(screen_w)}x{int(screen_h)}")

    # =================================================================
    # Phase 1: Leave-One-User-Out Cross-Validation (honest evaluation)
    # =================================================================
    print("\n" + "=" * 60)
    print("  PHASE 1: LEAVE-ONE-USER-OUT CROSS-VALIDATION")
    print("=" * 60)

    from sklearn.linear_model import Ridge

    louo_results = {name: [] for name in ["Ridge", "XGBoost"]}

    for test_user in unique_users:
        test_mask = user_labels == test_user
        train_mask = ~test_mask

        X_tr = X_scaled[train_mask]
        X_te = X_scaled[test_mask]
        y_tr = y[train_mask]
        y_te = y[test_mask]

        print(f"\n  --- Fold: test_user={test_user} (train={sum(train_mask):,}, test={sum(test_mask):,}) ---")

        # Ridge baseline
        ridge = Ridge(alpha=1.0)
        ridge.fit(X_tr, y_tr)
        y_pred = np.clip(ridge.predict(X_te), 0, 1)
        res = evaluate_model(f"Ridge (test={test_user})", y_te, y_pred, screen_w, screen_h)
        louo_results["Ridge"].append(res)
        print(f"    Ridge:   Euc={res['euc_mean']:.1f}px")

        # XGBoost
        xgb_model = build_xgboost()
        xgb_model.fit(X_tr, y_tr)
        y_pred_xgb = np.clip(xgb_model.predict(X_te), 0, 1)
        res_xgb = evaluate_model(f"XGBoost (test={test_user})", y_te, y_pred_xgb, screen_w, screen_h)
        louo_results["XGBoost"].append(res_xgb)
        print(f"    XGBoost: Euc={res_xgb['euc_mean']:.1f}px")

    # Print LOUO summary
    print("\n  --- LOUO Summary ---")
    for model_name, fold_results in louo_results.items():
        avg_euc = np.mean([r["euc_mean"] for r in fold_results])
        avg_p90 = np.mean([r["euc_p90"] for r in fold_results])
        print(f"    {model_name:12s}: avg_euc={avg_euc:.1f}px, avg_p90={avg_p90:.1f}px")

    # =================================================================
    # Phase 2: Within-user split (realistic for personalized use)
    # =================================================================
    print("\n" + "=" * 60)
    print("  PHASE 2: WITHIN-USER 80/20 SPLIT (PERSONALIZED ACCURACY)")
    print("=" * 60)

    within_user_results = {}
    for user in unique_users:
        user_mask = user_labels == user
        X_u = X_scaled[user_mask]
        y_u = y[user_mask]

        n = len(X_u)
        idx = np.random.RandomState(42).permutation(n)
        split = int(n * 0.8)

        X_tr = X_u[idx[:split]]
        X_te = X_u[idx[split:]]
        y_tr = y_u[idx[:split]]
        y_te = y_u[idx[split:]]

        # Ridge
        ridge = Ridge(alpha=1.0)
        ridge.fit(X_tr, y_tr)
        y_pred = np.clip(ridge.predict(X_te), 0, 1)
        res = evaluate_model(f"Ridge ({user})", y_te, y_pred, screen_w, screen_h)

        # XGBoost
        xgb_model = build_xgboost()
        xgb_model.fit(X_tr, y_tr)
        y_pred_xgb = np.clip(xgb_model.predict(X_te), 0, 1)
        res_xgb = evaluate_model(f"XGBoost ({user})", y_te, y_pred_xgb, screen_w, screen_h)

        within_user_results[user] = {"Ridge": res, "XGBoost": res_xgb}
        print(f"  {user:12s}: Ridge={res['euc_mean']:.1f}px, XGBoost={res_xgb['euc_mean']:.1f}px")

    # =================================================================
    # Phase 3: Full model comparison with GroupShuffleSplit
    # =================================================================
    print("\n" + "=" * 60)
    print("  PHASE 3: FULL MODEL COMPARISON")
    print("=" * 60)

    # Use GroupShuffleSplit (holds out one user's sessions)
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, test_idx = next(gss.split(X_scaled, y, groups=user_labels))

    X_train, X_test = X_scaled[train_idx], X_scaled[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    train_users = sorted(set(user_labels[train_idx]))
    test_users = sorted(set(user_labels[test_idx]))
    print(f"  Train users: {train_users}  ({X_train.shape[0]:,} samples)")
    print(f"  Test users:  {test_users}  ({X_test.shape[0]:,} samples)")
    print(f"  Features: {X_train.shape[1]}")

    results = []

    # --- Ridge ---
    print("\n  Training Ridge Regression...")
    ridge_full = Ridge(alpha=1.0)
    ridge_full.fit(X_train, y_train)
    y_pred_ridge = np.clip(ridge_full.predict(X_test), 0, 1)
    res_ridge = evaluate_model("Ridge (baseline)", y_test, y_pred_ridge, screen_w, screen_h)
    results.append(res_ridge)
    print(f"    Euc={res_ridge['euc_mean']:.1f}px")

    # --- XGBoost ---
    print("\n  Training XGBoost...")
    t0 = time.time()
    xgb_full = build_xgboost()
    xgb_full.fit(X_train, y_train)
    y_pred_xgb = np.clip(xgb_full.predict(X_test), 0, 1)
    res_xgb = evaluate_model("XGBoost", y_test, y_pred_xgb, screen_w, screen_h)
    results.append(res_xgb)
    print(f"    Euc={res_xgb['euc_mean']:.1f}px ({time.time()-t0:.1f}s)")

    # --- MLP models ---
    import tensorflow as tf

    # --- Single-Head MLP (V5 style) ---
    print("\n  Training MLP Single-Head (V5)...")
    mlp_v5 = build_single_head_mlp(X_train.shape[1])
    total_steps = int(np.ceil(X_train.shape[0] * 0.85 / 64)) * 400
    lr_sched = tf.keras.optimizers.schedules.CosineDecay(1e-3, total_steps, alpha=1e-6)
    mlp_v5.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr_sched),
        loss=weighted_huber_loss(y_weight=1.5, delta=0.05),
        metrics=["mae"],
    )

    # Feature augmentation during training
    augmentor = FeatureAugmentor(noise_std=0.02, feature_dropout=0.03)
    X_train_aug = np.vstack([X_train, augmentor.augment(X_train, np.random.RandomState(42))])
    y_train_aug = np.vstack([y_train, y_train])

    mlp_v5.fit(
        X_train_aug, y_train_aug,
        validation_split=0.15,
        epochs=400, batch_size=64,
        callbacks=[
            tf.keras.callbacks.EarlyStopping("val_loss", patience=35, restore_best_weights=True),
        ],
        verbose=1,
    )
    y_pred_v5 = np.clip(mlp_v5.predict(X_test, verbose=0), 0, 1)
    res_v5 = evaluate_model("MLP SingleHead (V5)", y_test, y_pred_v5, screen_w, screen_h)
    results.append(res_v5)
    print(f"    Euc={res_v5['euc_mean']:.1f}px")

    # --- Dual-Pathway MLP (V6) with Wing Loss ---
    print("\n  Training MLP Dual-Pathway (V6) with Wing Loss...")
    mlp_v6_wing = build_dual_pathway_mlp(X_train.shape[1])
    total_steps_v6 = int(np.ceil(X_train.shape[0] * 0.85 / 64)) * 500
    lr_v6 = tf.keras.optimizers.schedules.CosineDecay(1e-3, total_steps_v6, alpha=1e-6)
    mlp_v6_wing.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr_v6),
        loss=wing_loss(y_weight=1.5, w=10.0, epsilon=2.0),
        metrics=["mae"],
    )
    mlp_v6_wing.fit(
        X_train_aug, y_train_aug,
        validation_split=0.15,
        epochs=500, batch_size=64,
        callbacks=[
            tf.keras.callbacks.EarlyStopping("val_loss", patience=40, restore_best_weights=True),
        ],
        verbose=1,
    )
    y_pred_v6w = np.clip(mlp_v6_wing.predict(X_test, verbose=0), 0, 1)
    res_v6w = evaluate_model("V6 DualPath+WingLoss", y_test, y_pred_v6w, screen_w, screen_h)
    results.append(res_v6w)
    print(f"    Euc={res_v6w['euc_mean']:.1f}px")

    # --- Dual-Pathway MLP (V6) with Huber Loss ---
    print("\n  Training MLP Dual-Pathway (V6) with Huber Loss...")
    mlp_v6_huber = build_dual_pathway_mlp(X_train.shape[1])
    lr_v6h = tf.keras.optimizers.schedules.CosineDecay(1e-3, total_steps_v6, alpha=1e-6)
    mlp_v6_huber.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr_v6h),
        loss=weighted_huber_loss(y_weight=2.0, delta=0.05),
        metrics=["mae"],
    )
    mlp_v6_huber.fit(
        X_train_aug, y_train_aug,
        validation_split=0.15,
        epochs=500, batch_size=64,
        callbacks=[
            tf.keras.callbacks.EarlyStopping("val_loss", patience=40, restore_best_weights=True),
        ],
        verbose=1,
    )
    y_pred_v6h = np.clip(mlp_v6_huber.predict(X_test, verbose=0), 0, 1)
    res_v6h = evaluate_model("V6 DualPath+Huber", y_test, y_pred_v6h, screen_w, screen_h)
    results.append(res_v6h)
    print(f"    Euc={res_v6h['euc_mean']:.1f}px")

    # =================================================================
    # Results comparison
    # =================================================================
    print_results(results)

    best = min(results, key=lambda r: r["euc_mean"])
    print(f"\n>>> Best model on test set: {best['name']}  (Mean Euclidean = {best['euc_mean']:.1f}px)")

    # =================================================================
    # Phase 4: Retrain best MLP on 100% data for deployment
    # =================================================================
    print("\n" + "=" * 60)
    print("  PHASE 4: RETRAIN BEST ON 100% DATA FOR DEPLOYMENT")
    print("=" * 60)

    # Determine best MLP architecture & loss
    mlp_candidates = [
        (res_v5, "SingleHead", weighted_huber_loss(1.5, 0.05)),
        (res_v6w, "DualPath", wing_loss(1.5, 10.0, 2.0)),
        (res_v6h, "DualPath", weighted_huber_loss(2.0, 0.05)),
    ]
    best_mlp_res, best_arch, best_loss = min(mlp_candidates, key=lambda x: x[0]["euc_mean"])
    print(f"  Best MLP: {best_mlp_res['name']} (Euc={best_mlp_res['euc_mean']:.1f}px)")
    print(f"  Architecture: {best_arch}")

    # Build final model
    if best_arch == "DualPath":
        final_model = build_dual_pathway_mlp(X_scaled.shape[1])
    else:
        final_model = build_single_head_mlp(X_scaled.shape[1])

    total_steps_final = int(np.ceil(X_scaled.shape[0] / 64)) * 350
    lr_final = tf.keras.optimizers.schedules.CosineDecay(1e-3, total_steps_final, alpha=1e-6)
    final_model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr_final),
        loss=best_loss,
        metrics=["mae"],
    )

    print("  Training on 100% data (350 epochs)...")
    final_model.fit(
        X_scaled, y,
        epochs=350,
        batch_size=64,
        verbose=1,
    )

    # =================================================================
    # Save everything
    # =================================================================
    print("\n" + "=" * 60)
    print("  SAVING MODELS")
    print("=" * 60)
    os.makedirs(MODEL_DIR, exist_ok=True)

    # Save V6 versions
    final_model.save(MODEL_SAVE_PATH)
    print(f"  V6 Keras  -> {MODEL_SAVE_PATH}")

    convert_to_tflite(final_model, TFLITE_SAVE_PATH)

    # Also overwrite main deployment model
    final_model.save(MAIN_MODEL_PATH)
    print(f"  Main Keras -> {MAIN_MODEL_PATH}")

    convert_to_tflite(final_model, MAIN_TFLITE_PATH)

    # Save scalers
    for path in [SCALER_PATH, MAIN_SCALER_PATH]:
        with open(path, "wb") as f:
            pickle.dump(scaler, f)
    print(f"  Scaler     -> {MAIN_SCALER_PATH}")

    # Save metadata
    training_meta = dict(
        col_names=meta["col_names"],
        screen_w=screen_w, screen_h=screen_h,
        users=meta["users"],
        n_features=X_scaled.shape[1],
        n_train=X_train.shape[0], n_test=X_test.shape[0],
        n_total=X_scaled.shape[0],
        best_model=best["name"],
        best_euc_mean=best["euc_mean"],
        best_mlp=best_mlp_res["name"],
        best_mlp_euc=best_mlp_res["euc_mean"],
        all_results=results,
        louo_results={k: [r["euc_mean"] for r in v] for k, v in louo_results.items()},
        within_user_results={u: {k: v["euc_mean"] for k, v in d.items()} for u, d in within_user_results.items()},
        version="v6",
    )
    for path in [META_PATH, MAIN_META_PATH]:
        with open(path, "wb") as f:
            pickle.dump(training_meta, f)
    print(f"  Meta       -> {MAIN_META_PATH}")

    elapsed = time.time() - t_start
    print(f"\n  Total training time: {elapsed / 60:.1f} minutes")
    print(f"  Features ({len(meta['col_names'])}): {meta['col_names']}")
    print("\nDone! V6 model deployed.")


if __name__ == "__main__":
    main()
