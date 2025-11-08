# task1_crispdm_from_csv.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Tuple
from joblib import dump

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# ---------- Paths (use Path objects!) ----------
ROOT = Path(".").resolve()
TRAIN_CSV = ROOT / "train-final.csv"
TEST_CSV  = ROOT / "test-final.csv"
ARTIFACT_DIR = ROOT / "artifacts"
ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

GESTURE_LIST = [
    "afternoon","baby","big","born","bye","calendar",
    "child","cloud","come","daily","dance","dark",
    "day","enjoy","go","hello","home","love",
    "my","name","no","rain","sorry","strong",
    "study","thankyou","welcome","wind","yes","you"
]
GESTURE_TO_ID = {g: i+1 for i, g in enumerate(GESTURE_LIST)}

# ---------- Load CSVs (no headers) ----------
def load_raw(path) -> pd.DataFrame:
    path = Path(path)  # <-- coerce to Path to be safe
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")
    df = pd.read_csv(path, header=None)
    if df.shape[1] < 241:
        raise ValueError(f"Expected at least 241 columns (240 features + label). Found {df.shape[1]} in {path}.")
    return df

train_raw = load_raw(TRAIN_CSV)
test_raw  = load_raw(TEST_CSV)
print(f"Train shape: {train_raw.shape} | Test shape: {test_raw.shape}")

# ---------- Split features & labels ----------
def split_features_labels(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    feat = df.iloc[:, :240].copy()
    for c in feat.columns:
        feat[c] = pd.to_numeric(feat[c], errors="coerce")

    meta = df.iloc[:, 240:].copy()
    meta.columns = [f"col_{i}" for i in range(meta.shape[1])]

    labels = pd.DataFrame(index=df.index)

    # label_name: first object-like column
    obj_cols = [c for c in meta.columns if meta[c].dtype == "object"]
    if obj_cols:
        labels["label_name"] = meta[obj_cols[0]].astype(str)

    # label_id: first column that looks numeric after coercion
    num_candidate = None
    for c in meta.columns:
        coerced = pd.to_numeric(meta[c], errors="coerce")
        if coerced.notna().mean() > 0.9:  # mostly numeric
            num_candidate = c
            break
    if num_candidate is not None:
        labels["label_id"] = pd.to_numeric(meta[num_candidate], errors="coerce").astype("Int64")

    # candidate: second object col if present
    if len(obj_cols) >= 2:
        labels["candidate"] = meta[obj_cols[1]].astype(str)

    # Map label id from name if needed
    if "label_name" in labels.columns:
        labels["label_id_mapped"] = labels["label_name"].str.lower().map(GESTURE_TO_ID)
        labels["label_id_final"] = labels.get("label_id", labels["label_id_mapped"]).fillna(labels["label_id_mapped"])
    elif "label_id" in labels.columns:
        labels["label_id_final"] = labels["label_id"]

    return feat, labels

X_train_raw, y_train_df = split_features_labels(train_raw)
X_test_raw,  y_test_df  = split_features_labels(test_raw)
print("Columns in labels (train):", list(y_train_df.columns))
print("Columns in labels (test): ", list(y_test_df.columns))

# ---------- Impute & Scale (fit on train only) ----------
imputer = SimpleImputer(strategy="median")
scaler  = StandardScaler()

X_train_imputed = imputer.fit_transform(X_train_raw)
X_test_imputed  = imputer.transform(X_test_raw)

X_train_scaled = scaler.fit_transform(X_train_imputed)
X_test_scaled  = scaler.transform(X_test_imputed)

dump(imputer, ARTIFACT_DIR / "imputer.pkl")
dump(scaler,  ARTIFACT_DIR / "scaler.pkl")
print("Saved artifacts: artifacts/imputer.pkl, artifacts/scaler.pkl")

# ---------- Save processed CSVs ----------
feat_cols = [f"f{i+1}" for i in range(240)]
train_proc = pd.DataFrame(X_train_scaled, columns=feat_cols)
test_proc  = pd.DataFrame(X_test_scaled,  columns=feat_cols)

train_out = pd.concat([train_proc, y_train_df.reset_index(drop=True)], axis=1)
test_out  = pd.concat([test_proc,  y_test_df.reset_index(drop=True)],  axis=1)

train_out.to_csv(ROOT / "processed_train.csv", index=False)
test_out.to_csv(ROOT / "processed_test.csv", index=False)
print(f"Saved processed_train.csv {train_out.shape} and processed_test.csv {test_out.shape}")

# ---------- Descriptive analysis & visuals ----------
print("\n=== Descriptive stats (train, scaled features) ===")
print(train_proc.describe().round(3))

missing_rate = X_train_raw.isna().mean().mean()
print(f"\nAverage missing rate in raw train features: {missing_rate:.4f}")

label_series = None
if "label_name" in y_train_df.columns and y_train_df["label_name"].notna().any():
    label_series = y_train_df["label_name"].str.lower()
elif "label_id_final" in y_train_df.columns and y_train_df["label_id_final"].notna().any():
    label_series = y_train_df["label_id_final"].astype(str)

if label_series is not None:
    plt.figure(figsize=(10,4))
    label_series.value_counts().plot(kind="bar")
    plt.title("Class Distribution (Train)")
    plt.xlabel("Class")
    plt.ylabel("Count")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.show()

for c in feat_cols[:6]:
    plt.figure(figsize=(5,3))
    plt.hist(train_proc[c], bins=30)
    plt.title(f"Distribution of {c} (scaled)")
    plt.xlabel("Value")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.show()

subset = feat_cols[:40]
corr = pd.DataFrame(train_proc[subset]).corr()
plt.figure(figsize=(9,7))
sns.heatmap(corr, cmap="coolwarm", center=0)
plt.title("Correlation Heatmap (first 40 features)")
plt.tight_layout()
plt.show()

try:
    pca = PCA(n_components=2, random_state=42)
    Z = pca.fit_transform(train_proc.values)
    plt.figure(figsize=(7,5))
    if label_series is not None:
        labs = label_series.fillna("unknown")
        uniq = labs.unique().tolist()
        color_idx = {lab: i for i, lab in enumerate(uniq)}
        plt.scatter(Z[:,0], Z[:,1], c=[color_idx[x] for x in labs], s=14)
        plt.title("PCA (Train) colored by label")
    else:
        plt.scatter(Z[:,0], Z[:,1], s=14)
        plt.title("PCA (Train)")
    plt.xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% var)")
    plt.ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}% var)")
    plt.tight_layout()
    plt.show()
except Exception as e:
    print("PCA plot skipped:", e)

print("\nDone.")
