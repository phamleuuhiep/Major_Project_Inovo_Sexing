

import pandas as pd
import numpy as np

from pyod.models.ecod import ECOD
# You can also use:
# from pyod.models.iforest import IForest
# from pyod.models.copod import COPOD

# =========================
# Load CSV
# =========================
df = pd.read_csv("all_features_label_2.csv")

# =========================
# Define columns
# =========================

label_col = "label"

# Columns NOT used for outlier detection
exclude_cols = [
    "egg_id_x", "batch", "day", "side_x", "img_path", "img_key",
    "dot", "ngay", "egg_id_y", "side_y", "image_path",
    "QA_Failed", label_col
]

# Keep only numeric feature columns
feature_cols = [
    col for col in df.columns
    if col not in exclude_cols
]

feature_cols = df[feature_cols].select_dtypes(include=[np.number]).columns.tolist()

print("Using features:")
print(feature_cols)

# =========================
# Initialize output column
# =========================
df["IsOutlier"] = False

# Optional anomaly score column
df["OutlierScore"] = np.nan

# =========================
# Detect outliers separately per label
# =========================

for current_label in ['T', 'M']:

    print(f"\nProcessing label: {current_label}")

    # Get indices for this label
    idx = df[df[label_col] == current_label].index

    # Feature matrix
    X = df.loc[idx, feature_cols].copy()

    # Fill missing values
    X = X.fillna(X.median())

    # Initialize model
    clf = ECOD()
    
    # Alternative:
    # clf = IForest(contamination=0.05, random_state=42)

    # Train
    clf.fit(X)

    # Predict
    # 0 = normal
    # 1 = outlier
    predictions = clf.labels_

    # Scores
    scores = clf.decision_scores_

    # Store results
    df.loc[idx, "IsOutlier"] = predictions.astype(bool)
    df.loc[idx, "OutlierScore"] = scores

    print("Samples:", len(idx))
    print("Outliers found:", predictions.sum())

# =========================
# Save
# =========================
df.to_csv("dataset_with_outlier_flags.csv", index=False)

print("\nDone.")
print(df["IsOutlier"].value_counts())