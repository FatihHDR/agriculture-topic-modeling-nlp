"""
train_fasttext_svm.py
──────────────
Melatih model SVM dari doc_embeddings_fasttext.csv
dan menyimpan pipeline ke output/svm_fasttext.pkl
"""

import sys, os
import joblib
import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

# ── Paths ──────────────────────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
DATA_PATH  = os.path.join(BASE_DIR, "output", "doc_embeddings_fasttext.csv")
MODEL_PATH = os.path.join(BASE_DIR, "output", "svm_fasttext.pkl")

print("=" * 60)
print("  ICAR Agriculture — Training FastText + SVM Classifier")
print("=" * 60)

# ── Load data ──────────────────────────────────────────────────────────
print(f"\n[1/5] Loading data dari {DATA_PATH} ...")
df = pd.read_csv(DATA_PATH)
print(f"      Rows   : {len(df)}")
print(f"      Columns: {len(df.columns)}")

# Drop rows with empty category
df = df.dropna(subset=["category"])
df["category"] = df["category"].str.strip().str.replace(r"^[^\w]+", "", regex=True)

y = df["category"].values
# Features are all columns except 'category'
feature_cols = [c for c in df.columns if c != "category"]
X = df[feature_cols].values

# ── Train/test split ────────────────────────────────────────────────────
print("\n[2/5] Splitting 80/20 ...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"      Train: {len(X_train)}  |  Test: {len(X_test)}")

# ── Build model ─────────────────────────────────────────────────────
print("\n[3/5] Building SVM model ...")
svm = SVC(
    kernel="rbf",
    C=10,
    gamma="scale",
    probability=True,
    random_state=42,
)

# ── Train ───────────────────────────────────────────────────────────────
print("\n[4/5] Training ...")
svm.fit(X_train, y_train)

# ── Evaluate ────────────────────────────────────────────────────────────
print("\n[5/5] Evaluating ...")
y_pred = svm.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"\n      Test Accuracy : {acc*100:.2f}%")
print("\n      Classification Report:")
print(classification_report(y_test, y_pred))

# ── Save model ──────────────────────────────────────────────────────────
joblib.dump(svm, MODEL_PATH)
print(f"\n✅  Model saved → {MODEL_PATH}")
print(f"    Classes : {list(svm.classes_)}")
print("=" * 60)
