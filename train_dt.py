"""
train_dt.py
───────────
Melatih model Decision Tree dengan berbagai ekstraksi fitur:
1. TF-IDF + DT
2. BoW (Unigram) + DT
3. N-gram (Unigram+Bigram) + DT
"""

import sys, csv, os
csv.field_size_limit(sys.maxsize)

import joblib
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# ── Paths ──────────────────────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
DATA_PATH  = os.path.join(BASE_DIR, "output", "icar_augmented.csv")
OUT_TFIDF  = os.path.join(BASE_DIR, "output", "dt_tfidf.pkl")
OUT_BOW    = os.path.join(BASE_DIR, "output", "dt_bow.pkl")
OUT_NGRAM  = os.path.join(BASE_DIR, "output", "dt_ngram.pkl")

print("=" * 60)
print("  ICAR Agriculture — Training Decision Tree Classifiers")
print("=" * 60)

# ── Load data ──────────────────────────────────────────────────────────
df = pd.read_csv(DATA_PATH)
df["category"] = df["category"].str.strip().str.replace(r"^[^\w]+", "", regex=True)
df = df.dropna(subset=["text_clean"])
df = df[df["text_clean"].str.strip() != ""]

X = df["text_clean"].values
y = df["category"].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ── 1. TF-IDF + DT ──────────────────────────────────────────────────────
print("\n[1/3] Training TF-IDF + DT ...")
pipe_tfidf = Pipeline([
    ("tfidf", TfidfVectorizer(max_features=1000, ngram_range=(1, 2))),
    ("dt", DecisionTreeClassifier(random_state=42))
])
pipe_tfidf.fit(X_train, y_train)
acc_tfidf = accuracy_score(y_test, pipe_tfidf.predict(X_test))
print(f"      Accuracy: {acc_tfidf*100:.2f}%")
joblib.dump(pipe_tfidf, OUT_TFIDF)

# ── 2. BoW + DT ─────────────────────────────────────────────────────────
print("\n[2/3] Training BoW (Unigram) + DT ...")
pipe_bow = Pipeline([
    ("bow", CountVectorizer(max_features=1000, ngram_range=(1, 1))),
    ("dt", DecisionTreeClassifier(random_state=42))
])
pipe_bow.fit(X_train, y_train)
acc_bow = accuracy_score(y_test, pipe_bow.predict(X_test))
print(f"      Accuracy: {acc_bow*100:.2f}%")
joblib.dump(pipe_bow, OUT_BOW)

# ── 3. N-gram + DT ──────────────────────────────────────────────────────
print("\n[3/3] Training N-gram (Unigram+Bigram) + DT ...")
pipe_ngram = Pipeline([
    ("ngram", CountVectorizer(max_features=1000, ngram_range=(1, 2))),
    ("dt", DecisionTreeClassifier(random_state=42))
])
pipe_ngram.fit(X_train, y_train)
acc_ngram = accuracy_score(y_test, pipe_ngram.predict(X_test))
print(f"      Accuracy: {acc_ngram*100:.2f}%")
joblib.dump(pipe_ngram, OUT_NGRAM)

print("\n✅ All Decision Tree models saved.")
print("=" * 60)
