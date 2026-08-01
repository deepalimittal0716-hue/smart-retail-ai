"""
chatbot_service.py
====================
Module B (part 2) - Hybrid FAQ / support chatbot.

Two-tier engine, per Module B3 of the syllabus mapping:

  1. Rule-based engine: exact/fuzzy keyword & regex pattern matching
     directly against the intent `patterns` defined in data/intents.json.
     Fast, deterministic, and interpretable -- used whenever a confident
     direct match exists.

  2. ML fallback engine: when no rule fires with sufficient confidence,
     a TF-IDF vectorizer + cosine-similarity search over ALL intent
     patterns is used to find the closest matching intent, giving the
     bot graceful coverage of paraphrased / unseen phrasings.
"""

from __future__ import annotations

import difflib
import json
import random
import re
from pathlib import Path
from typing import Dict, List, Tuple

import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data"
MODELS_DIR = Path(__file__).resolve().parent.parent / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

RULE_MATCH_MIN_RATIO = 0.80   # difflib SequenceMatcher ratio to accept a fuzzy rule match
ML_FALLBACK_MIN_SIM = 0.20    # minimum cosine similarity to accept the ML fallback's best guess
FALLBACK_REPLY = (
    "I'm not fully sure I understood that. Could you rephrase, or would you "
    "like me to connect you with a live support agent?"
)


class ChatbotService:
    """Hybrid rule-based + ML-fallback FAQ chatbot."""

    def __init__(
        self,
        intents_path: Path = DATA_DIR / "intents.json",
        model_path: Path = MODELS_DIR / "chatbot_model.pkl",
    ) -> None:
        self.intents_path = intents_path
        self.model_path = model_path

        with open(self.intents_path, "r", encoding="utf-8") as f:
            self._intents_raw = json.load(f)["intents"]

        # Flatten (pattern -> tag) pairs for both engines.
        self._pattern_tag_pairs: List[Tuple[str, str]] = [
            (pattern.lower(), intent["tag"])
            for intent in self._intents_raw
            for pattern in intent["patterns"]
        ]
        self._responses_by_tag: Dict[str, List[str]] = {
            intent["tag"]: intent["responses"] for intent in self._intents_raw
        }

        self._vectorizer, self._pattern_matrix = self._load_or_build_ml_fallback()

    # ----------------------------------------------------------------- #
    # ML fallback: TF-IDF + cosine similarity index over all patterns
    # ----------------------------------------------------------------- #
    def _load_or_build_ml_fallback(self):
        if self.model_path.exists():
            bundle = joblib.load(self.model_path)
            return bundle["vectorizer"], bundle["matrix"]

        patterns = [p for p, _ in self._pattern_tag_pairs]
        vectorizer = TfidfVectorizer(ngram_range=(1, 2))
        matrix = vectorizer.fit_transform(patterns)

        joblib.dump({"vectorizer": vectorizer, "matrix": matrix}, self.model_path)
        return vectorizer, matrix

    # ----------------------------------------------------------------- #
    # 1. Rule-based engine
    # ----------------------------------------------------------------- #
    def _normalize(self, text: str) -> str:
        text = text.lower().strip()
        text = re.sub(r"[^a-z0-9\s]", " ", text)
        return re.sub(r"\s+", " ", text).strip()

    def _rule_based_match(self, message: str) -> Tuple[str, float] | None:
        """
        Exact substring match first (highest confidence), then fuzzy
        difflib ratio matching against every known pattern. Returns
        (tag, confidence_percent) or None if nothing clears the bar.
        """
        norm_msg = self._normalize(message)

        # Exact / substring match.
        for pattern, tag in self._pattern_tag_pairs:
            norm_pattern = self._normalize(pattern)
            if norm_pattern == norm_msg or norm_pattern in norm_msg or norm_msg in norm_pattern:
                return tag, 97.5

        # Fuzzy match.
        best_tag, best_ratio = None, 0.0
        for pattern, tag in self._pattern_tag_pairs:
            ratio = difflib.SequenceMatcher(None, norm_msg, self._normalize(pattern)).ratio()
            if ratio > best_ratio:
                best_tag, best_ratio = tag, ratio

        if best_ratio >= RULE_MATCH_MIN_RATIO:
            return best_tag, round(best_ratio * 100, 2)
        return None

    # ----------------------------------------------------------------- #
    # 2. ML fallback engine
    # ----------------------------------------------------------------- #
    def _ml_fallback_match(self, message: str) -> Tuple[str, float] | None:
        vec = self._vectorizer.transform([message.lower()])
        sims = cosine_similarity(vec, self._pattern_matrix)[0]

        best_idx = int(np.argmax(sims))
        best_sim = float(sims[best_idx])

        if best_sim >= ML_FALLBACK_MIN_SIM:
            _, tag = self._pattern_tag_pairs[best_idx]
            return tag, round(best_sim * 100, 2)
        return None

    # ----------------------------------------------------------------- #
    # Public API
    # ----------------------------------------------------------------- #
    def get_reply(self, message: str) -> Dict:
        rule_result = self._rule_based_match(message)
        if rule_result is not None:
            tag, confidence = rule_result
            engine = "rule_based"
        else:
            ml_result = self._ml_fallback_match(message)
            if ml_result is not None:
                tag, confidence = ml_result
                engine = "ml_fallback"
            else:
                return {
                    "reply": FALLBACK_REPLY,
                    "matched_intent": "unknown",
                    "engine_used": "none",
                    "confidence": 0.0,
                }

        reply = random.choice(self._responses_by_tag[tag])
        return {
            "reply": reply,
            "matched_intent": tag,
            "engine_used": engine,
            "confidence": confidence,
        }
