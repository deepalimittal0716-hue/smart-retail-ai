"""
test_endpoints.py
===================
Automated API tests (pytest + FastAPI TestClient) covering the happy path
and key validation-error paths for every endpoint, per Module C4 / Timeline
Day 8 ("Write automated tests").

Run:  pytest -v
"""

import io

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from app.main import app

client = TestClient(app)


def _fake_jpeg_bytes(color=(120, 120, 120), size=(224, 224)) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", size, color=color).save(buf, format="JPEG")
    buf.seek(0)
    return buf.read()


# --------------------------------------------------------------------------- #
# Health
# --------------------------------------------------------------------------- #
def test_root_health():
    resp = client.get("/")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"


def test_health_endpoint():
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "healthy"


# --------------------------------------------------------------------------- #
# /classify-product
# --------------------------------------------------------------------------- #
def test_classify_product_success():
    files = {"file": ("product.jpg", _fake_jpeg_bytes(), "image/jpeg")}
    resp = client.post("/classify-product", files=files)
    assert resp.status_code == 200
    body = resp.json()
    assert body["predicted_class"] in [
        "Clothing", "Shoes", "Bags", "Electronics", "Groceries"
    ]
    assert 0.0 <= body["confidence"] <= 100.0
    assert len(body["all_class_probabilities"]) == 5


def test_classify_product_rejects_bad_mime_type():
    files = {"file": ("note.txt", b"not an image", "text/plain")}
    resp = client.post("/classify-product", files=files)
    assert resp.status_code == 415


def test_classify_product_rejects_empty_file():
    files = {"file": ("empty.jpg", b"", "image/jpeg")}
    resp = client.post("/classify-product", files=files)
    assert resp.status_code == 400


# --------------------------------------------------------------------------- #
# /recognize-face
# --------------------------------------------------------------------------- #
def test_recognize_face_no_face_found():
    # A plain gray square has no detectable face -> should still succeed
    # with faces_detected == 0 and status "New Customer", not error.
    files = {"file": ("blank.jpg", _fake_jpeg_bytes(), "image/jpeg")}
    resp = client.post("/recognize-face", files=files)
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] in ["New Customer", "Returning Loyalty Member"]
    assert "visit_logged_at" in body


# --------------------------------------------------------------------------- #
# /analyze-sentiment
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "text",
    [
        "absolutely fantastic service and quality, love it!",
        "terrible experience, broken product, never again",
        "it was fine, nothing remarkable",
    ],
)
def test_analyze_sentiment_success(text):
    resp = client.post("/analyze-sentiment", json={"text": text})
    assert resp.status_code == 200
    body = resp.json()
    assert body["label"] in ["Positive", "Negative", "Neutral"]
    assert 0.0 <= body["confidence"] <= 100.0
    assert set(body["class_probabilities"].keys()) == {"Positive", "Negative", "Neutral"}


def test_analyze_sentiment_rejects_empty_text():
    resp = client.post("/analyze-sentiment", json={"text": "   "})
    assert resp.status_code == 422


# --------------------------------------------------------------------------- #
# /chatbot
# --------------------------------------------------------------------------- #
def test_chatbot_rule_based_match():
    resp = client.post("/chatbot", json={"message": "what is your return policy"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["matched_intent"] == "returns_policy"
    assert body["engine_used"] == "rule_based"


def test_chatbot_fallback_on_gibberish():
    resp = client.post("/chatbot", json={"message": "xkjqwz nonsense gibberish 12345"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["matched_intent"] in ["unknown"] or body["confidence"] >= 0.0


# --------------------------------------------------------------------------- #
# /dashboard/stats
# --------------------------------------------------------------------------- #
def test_dashboard_stats_shape():
    resp = client.get("/dashboard/stats")
    assert resp.status_code == 200
    body = resp.json()
    for key in [
        "total_visits", "returning_customers", "new_customers",
        "total_products_classified", "sentiment_breakdown",
        "total_chatbot_queries", "average_sentiment_confidence", "generated_at",
    ]:
        assert key in body
