"""
train_model.py
──────────────
Melatih model TF-IDF + SVM dari icar_augmented.csv
dan menyimpan pipeline ke output/svm_classifier.pkl
"""

import sys, csv, os, re
csv.field_size_limit(sys.maxsize)

import joblib
import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import LabelEncoder

# ── Paths ──────────────────────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
DATA_PATH  = os.path.join(BASE_DIR, "output", "icar_augmented.csv")
MODEL_PATH = os.path.join(BASE_DIR, "output", "svm_classifier.pkl")

print("=" * 60)
print("  ICAR Agriculture — Training TF-IDF + SVM Classifier")
print("=" * 60)

# ── Load data ──────────────────────────────────────────────────────────
print(f"\n[1/5] Loading data dari {DATA_PATH} ...")
df = pd.read_csv(DATA_PATH)
print(f"      Rows   : {len(df)}")
print(f"      Columns: {list(df.columns)}")

# Normalise category names (strip BOM / leading garbage chars)
df["category"] = df["category"].str.strip().str.replace(r"^[^\w]+", "", regex=True)
print(f"      Classes: {sorted(df['category'].unique())}")

# Drop rows with empty text
df = df.dropna(subset=["text_clean"])
df = df[df["text_clean"].str.strip() != ""]
print(f"      After cleanup: {len(df)} rows")

X = df["text_clean"].values
y = df["category"].values

# ── Train/test split ────────────────────────────────────────────────────
print("\n[2/5] Splitting 80/20 ...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"      Train: {len(X_train)}  |  Test: {len(X_test)}")

# ── Build pipeline ─────────────────────────────────────────────────────
print("\n[3/5] Building TF-IDF + SVM pipeline ...")
pipeline = Pipeline([
    ("tfidf", TfidfVectorizer(
        max_features=1000,
        ngram_range=(1, 2),
        sublinear_tf=True,
        strip_accents="unicode",
        analyzer="word",
        token_pattern=r"\b[a-zA-Z]{2,}\b",
        min_df=1,
    )),
    ("svm", SVC(
        kernel="rbf",
        C=10,
        gamma="scale",
        probability=True,   # ← dibutuhkan untuk confidence scores
        random_state=42,
    )),
])

# ── Train ───────────────────────────────────────────────────────────────
print("\n[4/5] Training ...")
pipeline.fit(X_train, y_train)

# ── Evaluate ────────────────────────────────────────────────────────────
print("\n[5/5] Evaluating ...")
y_pred = pipeline.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"\n      Test Accuracy : {acc*100:.2f}%")
print("\n      Classification Report:")
print(classification_report(y_test, y_pred))

# ── Save model ──────────────────────────────────────────────────────────
joblib.dump(pipeline, MODEL_PATH)
print(f"\n✅  Model saved → {MODEL_PATH}")
print(f"    Classes : {list(pipeline.classes_)}")
print("=" * 60)
