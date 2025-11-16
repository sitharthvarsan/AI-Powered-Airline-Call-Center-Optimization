import re
import os
import torch
import numpy as np
import pandas as pd
import torch.nn.functional as F

from typing import Any, List, Tuple
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


# RoBERTa Model Initialization

roberta_tokenizer = AutoTokenizer.from_pretrained(
    "cardiffnlp/twitter-roberta-base-sentiment"
)
roberta_model = AutoModelForSequenceClassification.from_pretrained(
    "cardiffnlp/twitter-roberta-base-sentiment"
)
def get_roberta_label(text: str) -> str:
    tokens = roberta_tokenizer(text, return_tensors="pt", truncation=True)
    with torch.no_grad():
        outputs = roberta_model(**tokens)
        probs = F.softmax(outputs.logits, dim=1)[0].numpy()
    neg, neu, pos = probs
    return "positive" if pos > neg else "negative"


# Text Preprocessing

def preprocess_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = text.encode("ascii", "ignore").decode()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\bnot\s+(\w+)", r"not_\1", text)
    return re.sub(r"\s+", " ", text).strip()


# Model Training

def build_and_train_model(data: List[Tuple[str, str]]) -> Any:
    x_clean = [preprocess_text(t[0]) for t in data]
    y = [t[1] for t in data]

    X_train, X_test, y_train, y_test = train_test_split(
        x_clean, y, test_size=0.2, random_state=42, stratify=y
    )
    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(
            ngram_range=(1, 2),
            max_features=5000,
            sublinear_tf=True,
            stop_words="english"
        )),
        ("clf", LogisticRegression(
            max_iter=2000,
            class_weight="balanced"
        ))
    ])

    pipeline.fit(X_train, y_train)
    print("\n===== MODEL PERFORMANCE =====")
    print("Train Accuracy:", accuracy_score(y_train, pipeline.predict(X_train)))
    print("Test Accuracy :", accuracy_score(y_test, pipeline.predict(X_test)))
    print("\nClassification Report:\n", classification_report(y_test, pipeline.predict(X_test)))
    print("\nConfusion Matrix:\n", confusion_matrix(y_test, pipeline.predict(X_test)))
    print("=====================================\n")

    return pipeline

def classify_review(model: Any, text: str) -> str:
    return model.predict([preprocess_text(text)])[0]


# Load Dataset with Caching

CACHE_FILE = "cached_labeled_reviews.xlsx"
if os.path.exists(CACHE_FILE):
    print("Using cached labeled file...")
    df = pd.read_excel(CACHE_FILE)
else:
    print("⚙ Running RoBERTa labeling (first time only)...")
    df = pd.read_excel("2026_dataset.xlsx")[["Title", "OverallScore"]].dropna()
    df["sentiment"] = df["Title"].apply(get_roberta_label)
    df.to_excel(CACHE_FILE, index=False)
    print("Labels saved to:", CACHE_FILE)

training_pairs = list(zip(df["Title"], df["sentiment"]))
trained_model = build_and_train_model(training_pairs)


# Demo Predictions

samples = [
    "The seats were comfortable and service was great!",
    "They lost my baggage and were very unhelpful!",
    "Nothing special, just an average flight."
]
print("\n===== SENTIMENT PREDICTIONS =====\n")
for s in samples:
    pred = classify_review(trained_model, s)
    print(f"Review: {s}")
    print(f"Sentiment: {pred}\n")
