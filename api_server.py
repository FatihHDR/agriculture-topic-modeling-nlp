import os
import re
import joblib
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from gensim.models import FastText
import nltk
from nltk.tokenize import word_tokenize

# ── Paths ──────────────────────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
TFIDF_MODEL_PATH = os.path.join(BASE_DIR, "output", "svm_classifier.pkl")
FASTTEXT_MODEL_PATH = os.path.join(BASE_DIR, "output", "fasttext.model")
SVM_FT_MODEL_PATH = os.path.join(BASE_DIR, "output", "svm_fasttext.pkl")
DT_TFIDF_PATH = os.path.join(BASE_DIR, "output", "dt_tfidf.pkl")
DT_BOW_PATH = os.path.join(BASE_DIR, "output", "dt_bow.pkl")
DT_NGRAM_PATH = os.path.join(BASE_DIR, "output", "dt_ngram.pkl")

# ── Load model at startup ───────────────────────────────────────────────
print(f"Loading TF-IDF+SVM model from {TFIDF_MODEL_PATH} ...")
try:
    pipeline_tfidf = joblib.load(TFIDF_MODEL_PATH)
    CLASSES: list[str] = list(pipeline_tfidf.classes_)
    print(f"✅ TF-IDF Model loaded. Classes: {CLASSES}")
except FileNotFoundError:
    print(f"⚠️ TF-IDF Model tidak ditemukan di {TFIDF_MODEL_PATH}.")
    pipeline_tfidf = None

print(f"Loading FastText model from {FASTTEXT_MODEL_PATH} ...")
try:
    fasttext_model = FastText.load(FASTTEXT_MODEL_PATH)
    svm_fasttext = joblib.load(SVM_FT_MODEL_PATH)
    print(f"✅ FastText + SVM Model loaded.")
except FileNotFoundError:
    print(f"⚠️ FastText / SVM Model tidak ditemukan.")
    fasttext_model = None
    svm_fasttext = None

print("Loading Decision Tree models ...")
try:
    dt_tfidf = joblib.load(DT_TFIDF_PATH)
    dt_bow = joblib.load(DT_BOW_PATH)
    dt_ngram = joblib.load(DT_NGRAM_PATH)
    print("✅ Decision Tree models loaded.")
except FileNotFoundError:
    print("⚠️ Decision Tree models tidak ditemukan.")
    dt_tfidf = None
    dt_bow = None
    dt_ngram = None

def get_fasttext_embedding(text: str):
    words = word_tokenize(text.lower())
    words = [w for w in words if w.isalpha()]
    if not words:
        return np.zeros(fasttext_model.vector_size)
    return np.mean([fasttext_model.wv[w] for w in words], axis=0)

# ── Color map untuk setiap kelas ────────────────────────────────────────
CLASS_COLORS: dict[str, str] = {
    "Annual Reports":                       "#3b82f6", # Blue 500
    "Indian Farming":                       "#475569", # Slate 600
    "Indian Horticulture":                  "#64748b", # Slate 500
    "Traditional Knowledge in Agriculture": "#334155", # Slate 700
    "Books":                                "#2563eb", # Blue 600
    "Reports":                              "#1d4ed8", # Blue 700
}

def get_color(label: str) -> str:
    """Cari warna berdasarkan label (partial match juga OK)."""
    clean = re.sub(r"^[^\w]+", "", label).strip()
    if clean in CLASS_COLORS:
        return CLASS_COLORS[clean]
    for key, color in CLASS_COLORS.items():
        if key.lower() in clean.lower() or clean.lower() in key.lower():
            return color
    return "#6366f1"

def clean_label(label: str) -> str:
    """Hapus BOM / karakter aneh di awal label."""
    return re.sub(r"[^a-zA-Z\s]", "", label).strip()

# ── FastAPI app ─────────────────────────────────────────────────────────
app = FastAPI(
    title="ICAR Agriculture Text Classifier API",
    description="Classify text into one of 6 ICAR Agriculture categories using TF-IDF + SVM.",
    version="1.0.0",
)

# Allow requests from Next.js dev server
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000", "https://research.vyg.re"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Schemas ─────────────────────────────────────────────────────────────
class ClassifyRequest(BaseModel):
    text: str = Field(..., min_length=3, description="Teks yang akan diklasifikasikan")
    method: str = Field(default="tfidf", description="Metode klasifikasi (tfidf atau fasttext)")

class ClassScore(BaseModel):
    label: str
    probability: float
    color: str

class ClassifyResponse(BaseModel):
    prediction: str
    confidence: float
    color: str
    scores: list[ClassScore]

# ── Endpoints ────────────────────────────────────────────────────────────
@app.get("/", summary="Health check")
def root():
    return {
        "status": "ok",
        "model": "TF-IDF + SVM",
        "classes": [clean_label(c) for c in CLASSES],
    }

@app.post("/classify", response_model=ClassifyResponse, summary="Klasifikasikan teks")
def classify(req: ClassifyRequest):
    text = req.text.strip()
    if not text:
        raise HTTPException(status_code=400, detail="Teks tidak boleh kosong")

    if req.method == "fasttext" and fasttext_model and svm_fasttext:
        emb = get_fasttext_embedding(text)
        proba = svm_fasttext.predict_proba([emb])[0]
        classes_used = list(svm_fasttext.classes_)
    elif req.method == "dt-tfidf" and dt_tfidf:
        proba = dt_tfidf.predict_proba([text])[0]
        classes_used = list(dt_tfidf.classes_)
    elif req.method == "dt-bow" and dt_bow:
        proba = dt_bow.predict_proba([text])[0]
        classes_used = list(dt_bow.classes_)
    elif req.method == "dt-ngram" and dt_ngram:
        proba = dt_ngram.predict_proba([text])[0]
        classes_used = list(dt_ngram.classes_)
    elif pipeline_tfidf:
        # Default to TF-IDF SVM
        proba = pipeline_tfidf.predict_proba([text])[0]
        classes_used = CLASSES
    else:
        raise HTTPException(status_code=500, detail="Model belum dilatih.")

    pred_idx = int(np.argmax(proba))
    pred_label = clean_label(classes_used[pred_idx])
    confidence = float(proba[pred_idx])

    # Build per-class scores (sorted by probability desc)
    scores = sorted(
        [
            ClassScore(
                label=clean_label(classes_used[i]),
                probability=float(proba[i]),
                color=get_color(classes_used[i]),
            )
            for i in range(len(classes_used))
        ],
        key=lambda s: s.probability,
        reverse=True,
    )

    return ClassifyResponse(
        prediction=pred_label,
        confidence=confidence,
        color=get_color(CLASSES[pred_idx]),
        scores=scores,
    )
