import os, joblib
import pandas as pd
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "output", "icar_augmented.csv")
NB_TFIDF_PATH = os.path.join(BASE_DIR, "output", "nb_tfidf.pkl")
NB_BOW_PATH = os.path.join(BASE_DIR, "output", "nb_bow.pkl")

print("Loading data...")
df = pd.read_csv(DATA_PATH)
df["category"] = df["category"].str.strip().str.replace(r"^[^\w]+", "", regex=True)
df = df.dropna(subset=["text_clean"])

X_chunks = []
y_chunks = []
for _, row in df.iterrows():
    words = str(row["text_clean"]).split()
    if not words: continue
    chunks = [" ".join(words[i:i+25]) for i in range(0, len(words), 25) if len(" ".join(words[i:i+25]).split()) >= 5]
    import random
    random.seed(42)
    if len(chunks) > 15: chunks = random.sample(chunks, 15)
    for c in chunks:
        X_chunks.append(c)
        y_chunks.append(row["category"])

print("Training NB TFIDF...")
pipe_nb_tfidf = Pipeline([
    ("tfidf", TfidfVectorizer(max_features=1000, ngram_range=(1, 2))),
    ("nb", MultinomialNB())
])
pipe_nb_tfidf.fit(X_chunks, y_chunks)
joblib.dump(pipe_nb_tfidf, NB_TFIDF_PATH)

print("Training NB BoW...")
pipe_nb_bow = Pipeline([
    ("bow", CountVectorizer(max_features=1000, ngram_range=(1, 1))),
    ("nb", MultinomialNB())
])
pipe_nb_bow.fit(X_chunks, y_chunks)
joblib.dump(pipe_nb_bow, NB_BOW_PATH)

print("Saved NB models!")
