"""
Deep verification of collected gaze data — checks every aspect before
the user commits to collecting 5,000 frames.
"""
import pandas as pd
import numpy as np
import glob
import os
import sys

files = glob.glob('dataset/*.csv')
if not files:
    print("ERROR: No CSV files found in dataset/")
    sys.exit(1)

df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
print(f"Files: {len(files)}")
for f in files:
    print(f"  {os.path.basename(f)} ({len(pd.read_csv(f))} rows)")
print(f"Total samples: {len(df)}\n")

issues = []
warnings = []
passes = []

# ============================================================
# 1. GAZE RATIO SANITY — the critical fix
# ============================================================
print("=" * 60)
print("1. GAZE RATIO FIX VERIFICATION")
print("=" * 60)

for col in ['l_gaze_ratio_x', 'r_gaze_ratio_x', 'l_gaze_ratio_y', 'r_gaze_ratio_y']:
    nuniq = df[col].nunique()
    std = df[col].std()
    mn, mx = df[col].min(), df[col].max()
    spread = mx - mn
    print(f"  {col}: min={mn:.4f} max={mx:.4f} std={std:.4f} unique={nuniq} spread={spread:.4f}")
    if nuniq <= 2:
        issues.append(f"CRITICAL: {col} has only {nuniq} unique values — still broken!")
    elif std < 0.01:
        warnings.append(f"WARNING: {col} has very low variance (std={std:.4f})")
    else:
        passes.append(f"{col} is varying properly (spread={spread:.4f})")

# ============================================================
# 2. FEATURE-TARGET CORRELATIONS
# ============================================================
print(f"\n{'=' * 60}")
print("2. FEATURE-TARGET CORRELATIONS")
print("=" * 60)

df['norm_tx'] = df['target_x'] / df['screen_w']
df['norm_ty'] = df['target_y'] / df['screen_h']

avg_gh = (df.l_gaze_ratio_x + df.r_gaze_ratio_x) / 2
avg_gv = (df.l_gaze_ratio_y + df.r_gaze_ratio_y) / 2
avg_ix = (df.l_iris_x + df.r_iris_x) / 2
avg_iy = (df.l_iris_y + df.r_iris_y) / 2

correlations = {
    'avg_gaze_h  vs target_x': avg_gh.corr(df.norm_tx),
    'avg_gaze_v  vs target_y': avg_gv.corr(df.norm_ty),
    'avg_iris_x  vs target_x': avg_ix.corr(df.norm_tx),
    'avg_iris_y  vs target_y': avg_iy.corr(df.norm_ty),
    'head_yaw    vs target_x': df.head_yaw.corr(df.norm_tx),
    'head_pitch  vs target_y': df.head_pitch.corr(df.norm_ty),
    'l_gaze_h    vs target_x': df.l_gaze_ratio_x.corr(df.norm_tx),
    'r_gaze_h    vs target_x': df.r_gaze_ratio_x.corr(df.norm_tx),
    'l_gaze_v    vs target_y': df.l_gaze_ratio_y.corr(df.norm_ty),
    'r_gaze_v    vs target_y': df.r_gaze_ratio_y.corr(df.norm_ty),
}

for name, r in correlations.items():
    flag = ""
    if abs(r) > 0.7:
        flag = " *** EXCELLENT"
        passes.append(f"{name}: r={r:+.4f} (excellent)")
    elif abs(r) > 0.3:
        flag = " ** GOOD"
        passes.append(f"{name}: r={r:+.4f} (good)")
    elif abs(r) > 0.15:
        flag = " * WEAK"
    else:
        flag = "   (no signal)"
    print(f"  {name}: r = {r:+.4f}{flag}")

# ============================================================
# 3. TARGET COVERAGE
# ============================================================
print(f"\n{'=' * 60}")
print("3. TARGET COVERAGE (screen utilisation)")
print("=" * 60)

n_targets = len(df.groupby(['target_x', 'target_y']))
tx_range = (df.norm_tx.min(), df.norm_tx.max())
ty_range = (df.norm_ty.min(), df.norm_ty.max())
print(f"  Unique target positions: {n_targets}")
print(f"  Horizontal coverage: [{tx_range[0]:.3f}, {tx_range[1]:.3f}]  (want [~0.05, ~0.95])")
print(f"  Vertical   coverage: [{ty_range[0]:.3f}, {ty_range[1]:.3f}]  (want [~0.09, ~0.91])")
print(f"  Screen resolution:   {df.screen_w.iloc[0]}x{df.screen_h.iloc[0]}")

if tx_range[0] > 0.15 or tx_range[1] < 0.85:
    warnings.append(f"Horizontal coverage is narrow: [{tx_range[0]:.2f}, {tx_range[1]:.2f}]")
else:
    passes.append(f"Horizontal coverage good: [{tx_range[0]:.2f}, {tx_range[1]:.2f}]")

if ty_range[0] > 0.2 or ty_range[1] < 0.8:
    warnings.append(f"Vertical coverage is narrow: [{ty_range[0]:.2f}, {ty_range[1]:.2f}]")
else:
    passes.append(f"Vertical coverage good: [{ty_range[0]:.2f}, {ty_range[1]:.2f}]")

# ============================================================
# 4. DATA QUALITY
# ============================================================
print(f"\n{'=' * 60}")
print("4. DATA QUALITY CHECKS")
print("=" * 60)

# NaN check
nan_counts = df.isnull().sum()
has_nans = nan_counts[nan_counts > 0]
if len(has_nans) > 0:
    print(f"  NaN values found:")
    for col, cnt in has_nans.items():
        print(f"    {col}: {cnt} NaN")
    issues.append(f"NaN values in {len(has_nans)} columns")
else:
    print(f"  No NaN values — clean!")
    passes.append("No NaN values in data")

# Infinity check
inf_cols = []
for col in df.select_dtypes(include=[np.number]).columns:
    if np.isinf(df[col]).any():
        inf_cols.append(col)
if inf_cols:
    issues.append(f"Infinity values in columns: {inf_cols}")
    print(f"  Infinity values in: {inf_cols}")
else:
    print(f"  No infinity values — clean!")
    passes.append("No infinity values")

# EAR (eye aspect ratio) — are eyes open?
l_ear_mean = df.l_ear.mean()
r_ear_mean = df.r_ear.mean()
print(f"  Left EAR mean:  {l_ear_mean:.3f}")
print(f"  Right EAR mean: {r_ear_mean:.3f}")
if l_ear_mean < 0.15 or r_ear_mean < 0.15:
    issues.append(f"EAR too low — eyes appear closed (L={l_ear_mean:.3f}, R={r_ear_mean:.3f})")
else:
    passes.append(f"EAR normal (L={l_ear_mean:.3f}, R={r_ear_mean:.3f})")

# Face area
fa_mean = df.face_area.mean()
print(f"  Face area mean: {fa_mean:.4f}  (want > 0.02)")
if fa_mean < 0.02:
    warnings.append(f"Face area low ({fa_mean:.4f}) — sit closer to camera")
else:
    passes.append(f"Face area good ({fa_mean:.4f})")

# Brightness
br_mean = df.frame_brightness.mean()
print(f"  Brightness mean: {br_mean:.1f}  (want 40-200)")
if br_mean < 30:
    warnings.append(f"Frame brightness too low ({br_mean:.1f})")
elif br_mean > 220:
    warnings.append(f"Frame brightness too high ({br_mean:.1f})")
else:
    passes.append(f"Brightness OK ({br_mean:.1f})")

# ============================================================
# 5. FEATURE VALUE RANGES
# ============================================================
print(f"\n{'=' * 60}")
print("5. FEATURE VALUE RANGES")
print("=" * 60)

feature_cols = [c for c in ['l_iris_x', 'l_iris_y', 'r_iris_x', 'r_iris_y',
                'l_gaze_ratio_x', 'l_gaze_ratio_y', 'r_gaze_ratio_x', 'r_gaze_ratio_y',
                'head_pitch', 'head_yaw', 'head_roll', 'iod', 'face_area',
                'inter_ocular_dist'] if c in df.columns]
for col in feature_cols:
    mn, mx, std = df[col].min(), df[col].max(), df[col].std()
    print(f"  {col:20s}: [{mn:+.4f}, {mx:+.4f}]  std={std:.4f}")

# ============================================================
# 6. CONSISTENCY CHECK — left vs right eye agreement
# ============================================================
print(f"\n{'=' * 60}")
print("6. LEFT-RIGHT EYE CONSISTENCY")
print("=" * 60)

lr_gaze_h_corr = df.l_gaze_ratio_x.corr(df.r_gaze_ratio_x)
lr_gaze_v_corr = df.l_gaze_ratio_y.corr(df.r_gaze_ratio_y)
lr_iris_x_corr = df.l_iris_x.corr(df.r_iris_x)
lr_iris_y_corr = df.l_iris_y.corr(df.r_iris_y)

print(f"  L vs R gaze_ratio_x correlation: {lr_gaze_h_corr:+.4f}  (want > 0.8)")
print(f"  L vs R gaze_ratio_y correlation: {lr_gaze_v_corr:+.4f}  (want > 0.8)")
print(f"  L vs R iris_x correlation:       {lr_iris_x_corr:+.4f}  (want > 0.9)")
print(f"  L vs R iris_y correlation:       {lr_iris_y_corr:+.4f}  (want > 0.9)")

if lr_gaze_h_corr > 0.8:
    passes.append(f"Left-right gaze_h strongly correlated ({lr_gaze_h_corr:.2f})")
elif lr_gaze_h_corr > 0.5:
    warnings.append(f"Left-right gaze_h moderately correlated ({lr_gaze_h_corr:.2f})")
else:
    issues.append(f"Left-right gaze_h poorly correlated ({lr_gaze_h_corr:.2f})")

# ============================================================
# 7. VERTICAL GAZE SIGNAL STRENGTH
# ============================================================
print(f"\n{'=' * 60}")
print("7. VERTICAL GAZE SIGNAL")
print("=" * 60)

vert_corr = avg_gv.corr(df.norm_ty)
print(f"  avg_gaze_v vs target_y: r = {vert_corr:+.4f}")
if abs(vert_corr) < 0.15:
    print(f"  NOTE: Vertical gaze ratio has weak signal.")
    print(f"  This is COMMON — vertical eye movement is subtler than horizontal.")
    print(f"  The model will rely more on iris_y and head_pitch for vertical prediction.")
    warnings.append(f"Vertical gaze ratio weak (r={vert_corr:.2f}) — model will use iris_y/head_pitch instead")
else:
    passes.append(f"Vertical gaze signal present (r={vert_corr:.2f})")

# ============================================================
# FINAL VERDICT
# ============================================================
print(f"\n{'=' * 60}")
print("FINAL VERDICT")
print("=" * 60)

if issues:
    print(f"\n  CRITICAL ISSUES ({len(issues)}):")
    for i, issue in enumerate(issues, 1):
        print(f"    {i}. {issue}")

if warnings:
    print(f"\n  WARNINGS ({len(warnings)}):")
    for i, w in enumerate(warnings, 1):
        print(f"    {i}. {w}")

if passes:
    print(f"\n  PASSED ({len(passes)}):")
    for i, p in enumerate(passes, 1):
        print(f"    {i}. {p}")

print()
if issues:
    print("  VERDICT: DO NOT PROCEED — critical issues must be fixed first.")
elif len(warnings) > 3:
    print("  VERDICT: PROCEED WITH CAUTION — several minor issues noted.")
else:
    print("  VERDICT: ALL CLEAR — proceed with full data collection!")
