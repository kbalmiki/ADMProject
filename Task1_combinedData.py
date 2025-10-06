# gesture_task1_crispdm.py
# ----------------------------------------
# Task 1: CRISP-DM + Preprocessing + Visualization from combined.zip (txt sequences)
#
# Requirements:
#   pip install pandas numpy scikit-learn matplotlib
#
# What it does:
#   - Unzips combined.zip
#   - Reads *.txt gesture sequences
#   - Parses floats; reshapes into frames of size 120 (or 60 if needed)
#   - Computes 240-D features per gesture: mean/std for pos(60) and angles(60)
#   - Extracts label_name / label_id / candidate from filename (best effort)
#   - Handles missing values, scales, visualizes, and saves CSVs
# ----------------------------------------

import os
import re
import zipfile
from pathlib import Path
from collections import Counter

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.decomposition import PCA

# ---------------------------
# 0) CRISP-DM (printed log)
# ---------------------------
CRISPDM = """
CRISP-DM Summary (Task 1)
1. Business Understanding:
   - Goal: Prepare Kinect-based gesture sequences for classical ML (kNN/SVM/RF).
   - Deliverables: Clean 1x240 feature vectors per gesture, EDA visuals, ready CSVs.

2. Data Understanding:
   - Source: combined.zip -> multiple .txt files; each file = 1 gesture instance.
   - Each frame has 120 features (60 positions + 60 cosine angles). Variable #frames/file.

3. Data Preparation:
   - Parse floats, segment into frames (120 per frame; fallback to 60 if needed).
   - Compute per-feature statistics across time: mean & std.
   - Concatenate: [pos_mean(60), pos_std(60), ang_mean(60), ang_std(60)] -> 240-D.
   - Extract labels from filenames (best effort). Track #frames and filename.

4. Modeling (not in this script):
   - Outputs are suitable for kNN, SVM, RandomForest, etc.

5. Evaluation (not in this script):
   - Use stratified train/test splits exported here as train-final.csv/test-final.csv.

6. Deployment (not in this script):
   - Save processed features; version data & code for reproducibility.
"""
print(CRISPDM)

# ---------------------------
# 1) Config & Gesture List
# ---------------------------
ROOT = Path(__file__).resolve().parent
ZIP_PATH = ROOT / "combined.zip"
EXTRACT_DIR = ROOT / "combined_extracted"
EXTRACT_DIR.mkdir(exist_ok=True)

# Canonical class list from your description (lowercased)
GESTURE_LIST = [
    "afternoon","baby","big","born","bye","calendar",
    "child","cloud","come","daily","dance","dark",
    "day","enjoy","go","hello","home","love",
    "my","name","no","rain","sorry","strong",
    "study","thankyou","welcome","wind","yes","you"
]
GESTURE_SET = set(GESTURE_LIST)

# ---------------------------
# 2) Helpers
# ---------------------------
float_pattern = re.compile(r'[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?')

def read_floats_from_txt(fp: Path):
    """Read all numeric floats from a text file (ignore any stray text)."""
    txt = fp.read_text(errors='ignore')
    nums = [float(x) for x in float_pattern.findall(txt)]
    return np.array(nums, dtype=np.float64)

def infer_frame_size(nums: np.ndarray):
    """Return 120 if divisible by 120; else 60 if divisible by 60; else None."""
    if len(nums) >= 120 and len(nums) % 120 == 0:
        return 120
    if len(nums) >= 60 and len(nums) % 60 == 0:
        return 60
    # Try a heuristic: prefer 120 if at least 2 full frames exist after truncation
    if len(nums) >= 240:
        return 120
    elif len(nums) >= 120:
        return 60
    return None

def reshape_frames(nums: np.ndarray, frame_size: int):
    n_frames = len(nums) // frame_size
    usable = nums[: n_frames * frame_size]
    return usable.reshape(n_frames, frame_size), n_frames

def split_pos_angle(features_frame: np.ndarray):
    """
    Given a single frame vector:
    - If length == 120: first 60 = positions, next 60 = angles
    - If length == 60: treat all as 'positions' and create dummy angles=None
    """
    D = features_frame.shape[-1]
    if D == 120:
        return slice(0,60), slice(60,120)
    elif D == 60:
        return slice(0,60), None
    else:
        return None, None

def parse_metadata_from_name(stem: str):
    """
    Parse (label_name, label_id, candidate) from filename stem.
    Examples it can handle (best effort):
      wind_28_C3.txt, 28-wind-person1.txt, C2-wind-28.txt, wind.txt
    """
    tokens = re.split(r'[\s_\-\.]+', stem.lower())
    label_name = None
    label_id = None
    candidate = None

    # Find label name by matching known gesture list
    for t in tokens:
        if t in GESTURE_SET:
            label_name = t
            break

    # Find first integer token as a candidate id
    for t in tokens:
        if t.isdigit():
            label_id = int(t)
            break

    # Candidate heuristic: something like c1, p2, subj3, personA, userX
    for t in tokens:
        if re.match(r'^(c\d+|p\d+|subj\d+|person\d+|user\d+)$', t):
            candidate = t
            break
    # Fallback: any leftover text not label/id -> candidate
    if candidate is None:
        for t in tokens:
            if not t.isdigit() and t != label_name and t not in {'txt'}:
                candidate = t
                break

    return label_name, label_id, candidate

def label_id_from_name(name: str):
    """Map label name to a stable numeric ID using the canonical list."""
    if name is None:
        return None
    if name in GESTURE_SET:
        return GESTURE_LIST.index(name) + 1
    return None

# ---------------------------
# 3) Unzip & Collect Files
# ---------------------------
if not ZIP_PATH.exists():
    raise FileNotFoundError(f"Could not find {ZIP_PATH}. Place combined.zip next to this script.")

print(f"📦 Extracting {ZIP_PATH.name} ...")
with zipfile.ZipFile(ZIP_PATH, 'r') as z:
    z.extractall(EXTRACT_DIR)

txt_files = sorted([p for p in EXTRACT_DIR.rglob("*.txt")])
print(f"📝 Found {len(txt_files)} txt files.")

# ---------------------------
# 4) Build dataset: 240-D per file
# ---------------------------
rows = []
bad_files = []

for fp in txt_files:
    try:
        nums = read_floats_from_txt(fp)
        fs = infer_frame_size(nums)
        if fs is None:
            bad_files.append((fp, f"Not enough numbers ({len(nums)})."))
            continue

        frames, n_frames = reshape_frames(nums, fs)
        pos_slice, ang_slice = split_pos_angle(frames[0])

        # Compute per-feature mean/std along time
        feat_means = frames.mean(axis=0)
        feat_stds  = frames.std(axis=0, ddof=0)

        # Assemble 240-D as [pos_mean(60), pos_std(60), ang_mean(60), ang_std(60)]
        if pos_slice is None:
            bad_files.append((fp, f"Unexpected frame size {frames.shape[1]}"))
            continue

        pos_mean = feat_means[pos_slice]
        pos_std  = feat_stds[pos_slice]

        if ang_slice is not None:
            ang_mean = feat_means[ang_slice]
            ang_std  = feat_stds[ang_slice]
            feats = np.concatenate([pos_mean, pos_std, ang_mean, ang_std], axis=0)  # 240
            col_names = (
                [f"pos_mean_{i+1}" for i in range(60)]
                + [f"pos_std_{i+1}" for i in range(60)]
                + [f"ang_mean_{i+1}" for i in range(60)]
                + [f"ang_std_{i+1}" for i in range(60)]
            )
        else:
            # Only 60-D frames available -> 120-D (positions only)
            feats = np.concatenate([pos_mean, pos_std], axis=0)
            col_names = (
                [f"pos_mean_{i+1}" for i in range(60)]
                + [f"pos_std_{i+1}" for i in range(60)]
            )

        # Metadata from filename
        stem = fp.stem
        label_name, label_id, candidate = parse_metadata_from_name(stem)
        if label_id is None:
            label_id = label_id_from_name(label_name)

        row = dict(zip(col_names, feats))
        row.update({
            "label_name": label_name,
            "label_id": label_id,
            "candidate": candidate,
            "file": str(fp.relative_to(EXTRACT_DIR)),
            "n_frames": n_frames,
            "frame_size": frames.shape[1],
        })
        rows.append(row)

    except Exception as e:
        bad_files.append((fp, str(e)))

df = pd.DataFrame(rows)
print(f"✅ Parsed {len(df)} files. Skipped {len(bad_files)} problematic files.")
if bad_files:
    print("Examples of skipped files:")
    for fp, msg in bad_files[:5]:
        print(" -", fp.name, "->", msg)

# ---------------------------
# 5) Handle Missing Values
# ---------------------------
# Report label coverage
print("\n🎯 Label coverage (from filenames):")
print(df["label_name"].value_counts(dropna=False).head(40))

# Separate features/labels
feature_cols = [c for c in df.columns if c.startswith(("pos_mean_", "pos_std_", "ang_mean_", "ang_std_"))]
X_raw = df[feature_cols].copy()
y = df["label_name"].copy() if "label_name" in df.columns else None

# Impute numeric features
imputer = SimpleImputer(strategy="median")
X_imputed = imputer.fit_transform(X_raw)

# Scale for PCA/visualization (and later models)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_imputed)

# Keep a scaled DataFrame for plots
X_scaled_df = pd.DataFrame(X_scaled, columns=feature_cols)

# ---------------------------
# 6) Save processed CSV(s)
# ---------------------------
processed = df.drop(columns=feature_cols).copy()
processed_features = pd.DataFrame(X_imputed, columns=feature_cols)
processed_scaled   = pd.DataFrame(X_scaled, columns=[f"scaled::{c}" for c in feature_cols])

full = pd.concat([processed_features, processed, processed_scaled], axis=1)
full.to_csv(ROOT / "processed.csv", index=False)
print(f"💾 Saved processed.csv with shape {full.shape}")

# : train/test split if we have labels
if y.notna().sum() >= 2 and y.nunique() > 1:
    # filter rows with known labels
    mask_labeled = y.notna()
    X_labeled = processed_features.loc[mask_labeled]
    y_labeled = y.loc[mask_labeled]
    meta_labeled = processed.loc[mask_labeled]

    X_train, X_test, y_train, y_test, meta_train, meta_test = train_test_split(
        X_labeled, y_labeled, meta_labeled, test_size=0.2, random_state=42, stratify=y_labeled
    )

    train_out = pd.concat([X_train.reset_index(drop=True),
                           meta_train.reset_index(drop=True)], axis=1)
    test_out  = pd.concat([X_test.reset_index(drop=True),
                           meta_test.reset_index(drop=True)], axis=1)

    train_out.to_csv(ROOT / "train-final.csv", index=False)
    test_out.to_csv(ROOT / "test-final.csv", index=False)
    print(f"💾 Saved train-final.csv ({train_out.shape}) and test-final.csv ({test_out.shape})")
else:
    print("ℹ️ Not enough labeled data inferred from filenames to make train/test splits.")

# ---------------------------
# 7) Descriptive Analysis & Visualizations
# ---------------------------
plt.figure(figsize=(7,4))
df["n_frames"].plot(kind="hist", bins=20)
plt.title("Frames per Gesture Instance")
plt.xlabel("#frames")
plt.ylabel("count")
plt.tight_layout()
plt.show()

if "label_name" in df.columns and df["label_name"].notna().any():
    plt.figure(figsize=(10,5))
    df["label_name"].value_counts().plot(kind="bar")
    plt.title("Class Distribution (inferred from filenames)")
    plt.xlabel("gesture")
    plt.ylabel("count")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.show()

# Feature histograms (first 8 features)
plot_cols = feature_cols[:8]
for c in plot_cols:
    plt.figure(figsize=(6,3))
    plt.hist(X_scaled_df[c], bins=30)
    plt.title(f"Feature Distribution: {c}")
    plt.xlabel("scaled value")
    plt.ylabel("count")
    plt.tight_layout()
    plt.show()

# Correlation heatmap (first 40 features to keep readable)
subset = feature_cols[:40]
if len(subset) >= 2:
    corr = X_scaled_df[subset].corr()
    plt.figure(figsize=(10,8))
    plt.imshow(corr, aspect='auto')
    plt.title("Correlation Heatmap (first 40 features)")
    plt.colorbar()
    plt.tight_layout()
    plt.show()

# PCA 2D scatter (if we have >= 2 components)
try:
    pca = PCA(n_components=2, random_state=42)
    X_pca = pca.fit_transform(X_scaled)
    plt.figure(figsize=(7,5))
    if "label_name" in df.columns and df["label_name"].notna().any():
        # color by label (limited legend)
        labels = df["label_name"].fillna("unknown")
        # simple color cycling
        uniq = labels.unique().tolist()
        colors = {lab: i for i, lab in enumerate(uniq)}
        scatter = plt.scatter(X_pca[:,0], X_pca[:,1], c=[colors[l] for l in labels], s=20)
        plt.title("PCA (2D) of Gestures")
        plt.xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% var)")
        plt.ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}% var)")
    else:
        plt.scatter(X_pca[:,0], X_pca[:,1], s=20)
        plt.title("PCA (2D) of Gestures (unlabeled)")
        plt.xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% var)")
        plt.ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}% var)")
    plt.tight_layout()
    plt.show()
except Exception as e:
    print("PCA plotting skipped:", e)

# ---------------------------
# 8) Printed Descriptives
# ---------------------------
print("\n===== Descriptive Stats (Scaled Features) =====")
print(pd.DataFrame(X_scaled, columns=feature_cols).describe().round(3))

print("\nTop 10 candidates (from filenames):")
print(pd.Series(df["candidate"]).value_counts().head(10))

print("\nFrame size counts (60 vs 120):")
print(df["frame_size"].value_counts())

print("\nDone.")
