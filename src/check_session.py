import pandas as pd, glob, os, numpy as np

files = glob.glob('dataset/*.csv')
print(f'Files found: {len(files)}')
for f in files:
    print(f'  {os.path.basename(f)}')

df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
print(f'\nTotal samples: {len(df)}')

print(f'\n=== GAZE RATIO FIX CHECK ===')
print(f'l_gaze_ratio_x: min={df.l_gaze_ratio_x.min():.4f} max={df.l_gaze_ratio_x.max():.4f} std={df.l_gaze_ratio_x.std():.4f} unique={df.l_gaze_ratio_x.nunique()}')
print(f'r_gaze_ratio_x: min={df.r_gaze_ratio_x.min():.4f} max={df.r_gaze_ratio_x.max():.4f} std={df.r_gaze_ratio_x.std():.4f} unique={df.r_gaze_ratio_x.nunique()}')
print(f'l_gaze_ratio_y: min={df.l_gaze_ratio_y.min():.4f} max={df.l_gaze_ratio_y.max():.4f} std={df.l_gaze_ratio_y.std():.4f}')
print(f'r_gaze_ratio_y: min={df.r_gaze_ratio_y.min():.4f} max={df.r_gaze_ratio_y.max():.4f} std={df.r_gaze_ratio_y.std():.4f}')

# Normalised targets
df['norm_tx'] = df['target_x'] / df['screen_w']
df['norm_ty'] = df['target_y'] / df['screen_h']
avg_gh = (df.l_gaze_ratio_x + df.r_gaze_ratio_x) / 2
avg_gv = (df.l_gaze_ratio_y + df.r_gaze_ratio_y) / 2
avg_ix = (df.l_iris_x + df.r_iris_x) / 2
avg_iy = (df.l_iris_y + df.r_iris_y) / 2

print('\n=== CORRELATION CHECK (want |r| > 0.3) ===')
print(f'avg_gaze_h vs target_x:  {avg_gh.corr(df.norm_tx):+.4f}')
print(f'avg_gaze_v vs target_y:  {avg_gv.corr(df.norm_ty):+.4f}')
print(f'avg_iris_x vs target_x:  {avg_ix.corr(df.norm_tx):+.4f}')
print(f'avg_iris_y vs target_y:  {avg_iy.corr(df.norm_ty):+.4f}')
print(f'head_yaw   vs target_x:  {df.head_yaw.corr(df.norm_tx):+.4f}')
print(f'head_pitch vs target_y:  {df.head_pitch.corr(df.norm_ty):+.4f}')

print('\n=== TARGET COVERAGE ===')
n_targets = len(df.groupby(['target_x', 'target_y']))
print(f'target_x range: [{df.norm_tx.min():.3f}, {df.norm_tx.max():.3f}]')
print(f'target_y range: [{df.norm_ty.min():.3f}, {df.norm_ty.max():.3f}]')
print(f'Unique target positions: {n_targets}')
print(f'Screen: {df.screen_w.iloc[0]}x{df.screen_h.iloc[0]}')

print('\n=== QUALITY CHECK ===')
print(f'l_ear mean={df.l_ear.mean():.3f} (expect 0.2-0.35 = eyes open)')
print(f'face_area mean={df.face_area.mean():.4f}')
print(f'frame_brightness mean={df.frame_brightness.mean():.1f}')
