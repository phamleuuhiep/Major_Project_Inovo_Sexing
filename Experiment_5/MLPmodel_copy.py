# MLP Model Phân Biệt Trứng Gà Trống / Gà Mái
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)
import joblib


feature_path = "dataset_with_outlier_flags.csv"
features_df = pd.read_csv(feature_path)

# =========================================================
# REMOVE OUTLIERS
# Keep only rows where IsOutlier == False
# =========================================================

features_df = features_df[features_df["IsOutlier"] == False]

print("Number of remaining data: ", features_df.count())


label_col = "label"

y_raw = features_df[label_col]


# feature_cols = ['Seh', 'Lx', 'Wt85', 'Set', 'Ly', 'Wh50', 'Wh90', 'Wt90', 'Sxd']
feature_cols = [
    "Lx","Ly","Lxt","Lxh","Wh50","Wh85","Wh90","Wt50","Wt85","Wt90","Sx","Sxt","Sxh","Sxd","Seh","Set","No303",
    "YR90",
    "R85",
    "L85",
    "GPT",
    "GYR90",
    "Gr",
    "R50BX",
    "DFH",
    "DSY",
    "Gym"
]

X = features_df[feature_cols]

# =========================================================
# Encode label
# =========================================================

y = y_raw.astype(str).str.upper().map({
    "T": 1,
    "M": 0
})

print("Data distribution", y.value_counts())

# remove rows with invalid labels
valid_idx = y.notna()

X = X[valid_idx]
y = y[valid_idx]

# =========================================================
# TRAIN / TEST SPLIT
# =========================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# =========================================================
# PIPELINE
# =========================================================

model = Pipeline([
    (
        "imputer",
        SimpleImputer(strategy="median")
    ),
    (
        "scaler",
        StandardScaler()
    ),
    (
        "mlp",
        MLPClassifier(
                # # For less number of feature feature (10)
                # hidden_layer_sizes=(64, 32),
                # activation="tanh",
                # solver="lbfgs",
                # alpha=0.001,
                # max_iter=2000,
                # random_state=42

                # For large number of features (27)
                hidden_layer_sizes=(512, 256,128, 64, 32),
                activation="relu",
                solver="lbfgs",
                alpha=0.001,
                batch_size=32,
                learning_rate_init=0.0001,
                max_iter=2000,
                early_stopping=True,
                validation_fraction=0.1,
                n_iter_no_change=30,
                random_state=42
    
        )
    )
])

# =========================================================
# TRAIN
# =========================================================

print("\nTraining model...")
model.fit(X_train, y_train)

# =========================================================
# EVALUATION
# =========================================================

pred = model.predict(X_test)

acc = accuracy_score(y_test, pred)

print("\n==============================")
print("Accuracy:", acc)
print("==============================")

print("\nClassification Report:\n")
print(classification_report(y_test, pred))

print("\nConfusion Matrix:\n")
print(confusion_matrix(y_test, pred))

# =========================================================
# SAVE MODEL
# =========================================================

joblib.dump(model, "egg_mlp_model.pkl")

print("\nModel saved to: egg_mlp_model.pkl")