"""
api_server.py
─────────────
FastAPI server yang meng-serve model TF-IDF + SVM untuk klasifikasi teks.
Jalankan dengan:
  .venv/bin/uvicorn api_server:app --reload --port 8000
"""

import os
import re
import joblib
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# ── Paths ──────────────────────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "output", "svm_classifier.pkl")

# ── Load model at startup ───────────────────────────────────────────────
print(f"Loading model from {MODEL_PATH} ...")
try:
    pipeline = joblib.load(MODEL_PATH)
    CLASSES: list[str] = list(pipeline.classes_)
    print(f"✅  Model loaded. Classes: {CLASSES}")
except FileNotFoundError:
    raise RuntimeError(
        f"Model tidak ditemukan di {MODEL_PATH}. "
        "Jalankan 'python train_model.py' terlebih dahulu."
    )

# ── Color map untuk setiap kelas ────────────────────────────────────────
CLASS_COLORS: dict[str, str] = {
    "Annual Reports":                       "#f59e0b",
    "Indian Farming":                       "#06b6d4",
    "Indian Horticulture":                  "#10b981",
    "Traditional Knowledge in Agriculture": "#ef4444",
    "Books":                                "#6366f1",
    "Reports":                              "#8b5cf6",
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
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Schemas ─────────────────────────────────────────────────────────────
class ClassifyRequest(BaseModel):
    text: str = Field(..., min_length=3, description="Teks yang akan diklasifikasikan")

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

    # Predict
    proba = pipeline.predict_proba([text])[0]        # shape: (n_classes,)
    pred_idx = int(np.argmax(proba))
    pred_label = clean_label(CLASSES[pred_idx])
    confidence = float(proba[pred_idx])

    # Build per-class scores (sorted by probability desc)
    scores = sorted(
        [
            ClassScore(
                label=clean_label(CLASSES[i]),
                probability=float(proba[i]),
                color=get_color(CLASSES[i]),
            )
            for i in range(len(CLASSES))
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
