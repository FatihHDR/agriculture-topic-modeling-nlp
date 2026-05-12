"""
train_all_chunked.py
────────────────────
Memperbaiki masalah "Distribution Shift" pada teks pendek.
Script ini memecah dokumen panjang menjadi potongan pendek (chunk) 
sekitar 20 kata per chunk, lalu melatih ulang semua model.
"""

import sys, os
import pandas as pd
import numpy as np
import joblib
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from gensim.models import FastText
import nltk
from nltk.tokenize import word_tokenize

# Pastikan punkt tersedia
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

# ── Paths ──────────────────────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
DATA_PATH  = os.path.join(BASE_DIR, "output", "icar_augmented.csv")

TFIDF_MODEL_PATH = os.path.join(BASE_DIR, "output", "svm_classifier.pkl")
FASTTEXT_MODEL_PATH = os.path.join(BASE_DIR, "output", "fasttext.model")
SVM_FT_MODEL_PATH = os.path.join(BASE_DIR, "output", "svm_fasttext.pkl")
DT_TFIDF_PATH = os.path.join(BASE_DIR, "output", "dt_tfidf.pkl")
DT_BOW_PATH = os.path.join(BASE_DIR, "output", "dt_bow.pkl")
DT_NGRAM_PATH = os.path.join(BASE_DIR, "output", "dt_ngram.pkl")

print("=" * 60)
print("  ICAR Agriculture — Retraining with Text Chunks")
print("=" * 60)

# 1. Chunking Data
print("[1/5] Loading and chunking dataset...")
df = pd.read_csv(DATA_PATH)
df["category"] = df["category"].str.strip().str.replace(r"^[^\w]+", "", regex=True)
df = df.dropna(subset=["text_clean"])

X_chunks = []
y_chunks = []

CHUNK_SIZE = 25 # Potong per 25 kata agar mirip kalimat dashboard

for _, row in df.iterrows():
    text = str(row["text_clean"])
    cat = row["category"]
    words = text.split()
    
    # Lewati dokumen yg kosong
    if not words: continue
        
    chunks_for_doc = []
    for i in range(0, len(words), CHUNK_SIZE):
        chunk = " ".join(words[i:i+CHUNK_SIZE])
        if len(chunk.split()) >= 5: # minimal 5 kata
            chunks_for_doc.append(chunk)
    
    # Ambil maksimal 15 chunk acak dari tiap dokumen (agar tidak O(N^2) lambat di SVM)
    import random
    random.seed(42)
    if len(chunks_for_doc) > 15:
        chunks_for_doc = random.sample(chunks_for_doc, 15)
        
    for chunk in chunks_for_doc:
        X_chunks.append(chunk)
        y_chunks.append(cat)

print(f"      Original Docs: {len(df)}")
print(f"      Total Chunks created: {len(X_chunks)}")

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X_chunks, y_chunks, test_size=0.2, random_state=42, stratify=y_chunks
)

# 2. Retrain TF-IDF + SVM
print("\n[2/5] Training TF-IDF + SVM...")
pipe_tfidf_svm = Pipeline([
    ("tfidf", TfidfVectorizer(max_features=1000, ngram_range=(1, 2))),
    ("svm", SVC(kernel="rbf", C=10, gamma="scale", probability=True, random_state=42))
])
pipe_tfidf_svm.fit(X_train, y_train)
joblib.dump(pipe_tfidf_svm, TFIDF_MODEL_PATH)
print(f"      Accuracy: {accuracy_score(y_test, pipe_tfidf_svm.predict(X_test))*100:.2f}%")

# 3. Retrain FastText + SVM
print("\n[3/5] Computing FastText embeddings for chunks...")
ft_model = FastText.load(FASTTEXT_MODEL_PATH)

def get_ft_embedding(text):
    words = word_tokenize(text.lower())
    words = [w for w in words if w.isalpha()]
    if not words: return np.zeros(ft_model.vector_size)
    return np.mean([ft_model.wv[w] for w in words], axis=0)

X_train_ft = [get_ft_embedding(t) for t in X_train]
X_test_ft = [get_ft_embedding(t) for t in X_test]

print("      Training FastText + SVM...")
svm_ft = SVC(kernel="rbf", C=10, gamma="scale", probability=True, random_state=42)
svm_ft.fit(X_train_ft, y_train)
joblib.dump(svm_ft, SVM_FT_MODEL_PATH)
print(f"      Accuracy: {accuracy_score(y_test, svm_ft.predict(X_test_ft))*100:.2f}%")

# 4. Retrain Decision Trees
print("\n[4/5] Training Decision Tree variations...")
pipe_dt_tfidf = Pipeline([
    ("tfidf", TfidfVectorizer(max_features=1000, ngram_range=(1, 2))),
    ("dt", DecisionTreeClassifier(random_state=42, min_samples_leaf=3)) # Pruned sedikit
])
pipe_dt_tfidf.fit(X_train, y_train)
joblib.dump(pipe_dt_tfidf, DT_TFIDF_PATH)

pipe_dt_bow = Pipeline([
    ("bow", CountVectorizer(max_features=1000, ngram_range=(1, 1))),
    ("dt", DecisionTreeClassifier(random_state=42, min_samples_leaf=3))
])
pipe_dt_bow.fit(X_train, y_train)
joblib.dump(pipe_dt_bow, DT_BOW_PATH)

pipe_dt_ngram = Pipeline([
    ("ngram", CountVectorizer(max_features=1000, ngram_range=(1, 2))),
    ("dt", DecisionTreeClassifier(random_state=42, min_samples_leaf=3))
])
pipe_dt_ngram.fit(X_train, y_train)
joblib.dump(pipe_dt_ngram, DT_NGRAM_PATH)

print("\n[5/5] All models successfully retrained and saved!")
print("=" * 60)
