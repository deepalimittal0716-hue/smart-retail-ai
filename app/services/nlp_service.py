"""
nlp_service.py
================
Module B (part 1) - Natural Language Processing Engine: text preprocessing
and sentiment analysis.

Pipeline: raw text -> cleaning -> TF-IDF vectorization -> Logistic
Regression classifier -> (label, confidence, class probabilities).

If a pre-trained `sentiment_model.pkl` / `vectorizer.pkl` pair is found in
app/models/ (produced by notebooks/03_sentiment_model_training.ipynb on a
real dataset such as the Kaggle "Women's E-Commerce Clothing Reviews"), it
is loaded directly. Otherwise the service bootstraps a small but sane
baseline model from an embedded seed corpus so that /analyze-sentiment is
functional out of the box in a fresh checkout.
"""

from __future__ import annotations

import re
import string
from pathlib import Path
from typing import Dict, List, Tuple

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

SENTIMENT_LABELS = ["Negative", "Neutral", "Positive"]

# Small embedded seed corpus used ONLY as a cold-start fallback when no
# trained model artifact is present. Replace with the full Kaggle reviews
# dataset for production-grade accuracy (see report, Section 3).
_SEED_CORPUS: List[Tuple[str, str]] = [
    ("absolutely love this product, best purchase ever", "Positive"),
    ("great quality and fast shipping, highly recommend", "Positive"),
    ("exceeded my expectations, will buy again", "Positive"),
    ("amazing customer service and beautiful packaging", "Positive"),
    ("this is fantastic value for money", "Positive"),
    ("works perfectly and looks even better in person", "Positive"),
    ("terrible quality, broke after one use", "Negative"),
    ("worst purchase i have made, complete waste of money", "Negative"),
    ("very disappointed, item arrived damaged", "Negative"),
    ("customer support was rude and unhelpful", "Negative"),
    ("does not match the description at all, misleading", "Negative"),
    ("stopped working within a week, refund requested", "Negative"),
    ("it's okay, nothing special about it", "Neutral"),
    ("average product, does what it says", "Neutral"),
    ("delivery was on time, product is as expected", "Neutral"),
    ("neither impressed nor disappointed, just fine", "Neutral"),
    ("packaging was standard, product works as described", "Neutral"),
    ("reasonable for the price, no complaints", "Neutral"),
]


class NLPService:
    """Text preprocessing + sentiment analysis service."""

    def __init__(
        self,
        model_path: Path = MODELS_DIR / "sentiment_model.pkl",
        vectorizer_path: Path = MODELS_DIR / "vectorizer.pkl",
    ) -> None:
        self.model_path = model_path
        self.vectorizer_path = vectorizer_path
        self._model, self._vectorizer = self._load_or_train()

    # ----------------------------------------------------------------- #
    # 1. Cleaning functions
    # ----------------------------------------------------------------- #
    @staticmethod
    def to_lowercase(text: str) -> str:
        return text.lower()

    @staticmethod
    def remove_punctuation(text: str) -> str:
        return re.sub(f"[{re.escape(string.punctuation)}]", " ", text)

    @staticmethod
    def normalize_whitespace(text: str) -> str:
        return re.sub(r"\s+", " ", text).strip()

    @staticmethod
    def remove_numbers(text: str) -> str:
        return re.sub(r"\d+", " ", text)

    def clean_text(self, text: str) -> str:
        """Full cleaning pipeline: lowercase -> strip punctuation/numbers ->
        collapse whitespace. Order matters: punctuation stripped before
        whitespace normalization avoids leaving stray double spaces."""
        text = self.to_lowercase(text)
        text = self.remove_punctuation(text)
        text = self.remove_numbers(text)
        text = self.normalize_whitespace(text)
        return text

    # ----------------------------------------------------------------- #
    # 2. Model loading / cold-start training
    # ----------------------------------------------------------------- #
    def _load_or_train(self):
        if self.model_path.exists() and self.vectorizer_path.exists():
            model = joblib.load(self.model_path)
            vectorizer = joblib.load(self.vectorizer_path)
            return model, vectorizer

        texts = [self.clean_text(t) for t, _ in _SEED_CORPUS]
        labels = [label for _, label in _SEED_CORPUS]

        vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=1)
        X = vectorizer.fit_transform(texts)

        model = LogisticRegression(max_iter=1000, class_weight="balanced")
        model.fit(X, labels)

        joblib.dump(model, self.model_path)
        joblib.dump(vectorizer, self.vectorizer_path)
        return model, vectorizer

    # ----------------------------------------------------------------- #
    # 3. Inference
    # ----------------------------------------------------------------- #
    def analyze_sentiment(self, text: str) -> Dict:
        cleaned = self.clean_text(text)
        X = self._vectorizer.transform([cleaned])

        probs = self._model.predict_proba(X)[0]
        classes = list(self._model.classes_)

        best_idx = int(np.argmax(probs))
        label = classes[best_idx]
        confidence = round(float(probs[best_idx]) * 100, 2)

        class_probabilities = {
            cls: round(float(p) * 100, 2) for cls, p in zip(classes, probs)
        }
        # Ensure all three canonical labels are always present in the response.
        for lbl in SENTIMENT_LABELS:
            class_probabilities.setdefault(lbl, 0.0)

        return {
            "label": label,
            "confidence": confidence,
            "class_probabilities": class_probabilities,
            "cleaned_text": cleaned,
        }
