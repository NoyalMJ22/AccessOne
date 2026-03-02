"""
train_model.py — Train and evaluate the gaze/blink classifier.
Run this whenever you collect new data (e.g. after switching to IR camera).
"""

import sys
import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.metrics import classification_report, confusion_matrix
from features import FEATURE_NAMES

CSV_PATH   = "dataset.csv"
MODEL_PATH = "model.pkl"
LABEL_PATH = "label_encoder.pkl"

# ── Load data ──────────────────────────────────────────────────────────────────
try:
    data = pd.read_csv(CSV_PATH, header=None)
except Exception as e:
    print(f"ERROR reading {CSV_PATH}: {e}")
    sys.exit(1)

# Drop any rows that don't have exactly the right number of columns
expected_cols = len(FEATURE_NAMES) + 1
data = data.dropna()
data = data[data.apply(lambda r: len(r) == expected_cols, axis=1)]

if len(data) == 0:
    print("ERROR: dataset.csv is empty or has no valid rows.")
    print("Run record_data.py first to collect training samples.")
    sys.exit(1)

data.columns = FEATURE_NAMES + ["label"]

print(f"\nDataset: {len(data)} rows")
print(data["label"].value_counts().to_string())

# Warn if any class has very few samples
min_samples = data["label"].value_counts().min()
if min_samples < 10:
    print(f"\nWARNING: Some labels have fewer than 10 samples. Collect more data for better accuracy.")

X = data[FEATURE_NAMES].values
y_raw = data["label"].values

le = LabelEncoder()
y  = le.fit_transform(y_raw)

# ── Model pipeline ─────────────────────────────────────────────────────────────
pipe = Pipeline([
    ("scaler", StandardScaler()),
    ("clf", GradientBoostingClassifier(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        min_samples_leaf=2,
        random_state=42,
    )),
])

# ── Cross-validation (only if enough samples) ─────────────────────────────────
n_splits = min(5, min_samples)

if n_splits >= 2:
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    cv_res = cross_validate(pipe, X, y, cv=cv,
                            scoring=["accuracy", "f1_weighted"],
                            return_train_score=True)
    print(f"\nCV accuracy : {cv_res['test_accuracy'].mean():.3f} +/- {cv_res['test_accuracy'].std():.3f}")
    print(f"CV F1 (wtd) : {cv_res['test_f1_weighted'].mean():.3f} +/- {cv_res['test_f1_weighted'].std():.3f}")
else:
    print("\nSkipping cross-validation (need at least 2 samples per label).")

# ── Fit on all data ────────────────────────────────────────────────────────────
pipe.fit(X, y)
y_pred = pipe.predict(X)

print("\n-- Classification report --")
print(classification_report(y, y_pred, target_names=le.classes_))

print("Confusion matrix (rows=actual, cols=predicted):")
print(pd.DataFrame(
    confusion_matrix(y, y_pred),
    index=le.classes_, columns=le.classes_
))

# ── Feature importance ─────────────────────────────────────────────────────────
clf = pipe.named_steps["clf"]
imp = sorted(zip(FEATURE_NAMES, clf.feature_importances_), key=lambda x: -x[1])
print("\n-- Feature importances --")
for name, score in imp:
    bar = "#" * int(score * 40)
    print(f"  {name:<15} {score:.4f}  {bar}")

# ── Save ───────────────────────────────────────────────────────────────────────
joblib.dump(pipe, MODEL_PATH)
joblib.dump(le,   LABEL_PATH)
print(f"\nSaved model   -> {MODEL_PATH}")
print(f"Saved encoder -> {LABEL_PATH}")
print("\nDone! Run main.py to start the live detector.")