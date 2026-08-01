"""
main.py
========
Module C - API Gateway.

FastAPI application exposing the Computer Vision, NLP, and Chatbot engines
behind a single production-style REST gateway, with:

  * Rich OpenAPI metadata / tags -> Swagger UI at /docs, ReDoc at /redoc.
  * File-upload + MIME validation for image endpoints.
  * A lightweight API-key header check to simulate production security
    (Module C4), toggle-able via the REQUIRE_API_KEY env var.
  * A live /dashboard/stats aggregation endpoint for the analytics frontend.

Run locally:
    uvicorn app.main:app --reload --port 8000

Run in Docker:
    docker build -t smart-retail-ai .
    docker run -p 8000:8000 smart-retail-ai
"""

from __future__ import annotations

import os
import time
from datetime import datetime, timezone
from typing import Optional

from fastapi import Depends, FastAPI, File, Header, HTTPException, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.openapi.utils import get_openapi

from app.schemas import (
    ChatbotRequest,
    ChatbotResponse,
    DashboardStatsResponse,
    ErrorResponse,
    FaceRecognitionResponse,
    ProductClassificationResponse,
    SentimentAnalysisRequest,
    SentimentAnalysisResponse,
    SentimentBreakdown,
)
from app.services.chatbot_service import ChatbotService
from app.services.cv_service import CVService
from app.services.nlp_service import NLPService

# --------------------------------------------------------------------------- #
# App metadata
# --------------------------------------------------------------------------- #
API_KEY = os.getenv("SMART_RETAIL_API_KEY", "dev-demo-key-123")
REQUIRE_API_KEY = os.getenv("REQUIRE_API_KEY", "false").lower() == "true"
ALLOWED_IMAGE_MIME_TYPES = {"image/jpeg", "image/png", "image/webp"}
MAX_IMAGE_SIZE_BYTES = 8 * 1024 * 1024  # 8 MB

app = FastAPI(
    title="Smart Retail & Customer Intelligence Platform API",
    description=(
        "A production-style API gateway unifying Computer Vision "
        "(product classification + face-based visit recognition), NLP "
        "(review sentiment analysis), and a hybrid FAQ chatbot for a "
        "retail/e-commerce business."
    ),
    version="1.0.0",
    contact={"name": "AIML Internship Capstone", "email": "student@example.edu"},
    license_info={"name": "MIT"},
    openapi_tags=[
        {"name": "Computer Vision", "description": "Product classification & face-based visit recognition."},
        {"name": "NLP", "description": "Customer review / feedback sentiment analysis."},
        {"name": "Chatbot", "description": "Hybrid rule-based + ML FAQ chatbot."},
        {"name": "Dashboard", "description": "Aggregated live analytics for the customer-intelligence dashboard."},
        {"name": "Health", "description": "Service liveness / readiness checks."},
    ],
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --------------------------------------------------------------------------- #
# Services are instantiated once at import time (model-loading happens here),
# then reused across all requests -- this IS the "unified pipeline" (Module
# C1) that loads all three models once at startup.
# --------------------------------------------------------------------------- #
cv_service = CVService()
nlp_service = NLPService()
chatbot_service = ChatbotService()

# In-memory counters for the dashboard (would be a real DB/warehouse in prod).
_dashboard_state = {
    "total_products_classified": 0,
    "sentiment_counts": {"Positive": 0, "Negative": 0, "Neutral": 0},
    "sentiment_confidence_sum": 0.0,
    "sentiment_count": 0,
    "total_chatbot_queries": 0,
}


# --------------------------------------------------------------------------- #
# Auth dependency (simulated production security, Module C4)
# --------------------------------------------------------------------------- #
async def verify_api_key(x_api_key: Optional[str] = Header(default=None)) -> None:
    if not REQUIRE_API_KEY:
        return
    if x_api_key != API_KEY:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing X-API-Key header.",
        )


async def _validated_image_bytes(file: UploadFile) -> bytes:
    if file.content_type not in ALLOWED_IMAGE_MIME_TYPES:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail=f"Unsupported file type '{file.content_type}'. Allowed: {sorted(ALLOWED_IMAGE_MIME_TYPES)}",
        )
    data = await file.read()
    if len(data) == 0:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Uploaded file is empty.")
    if len(data) > MAX_IMAGE_SIZE_BYTES:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"Image exceeds max size of {MAX_IMAGE_SIZE_BYTES // (1024*1024)} MB.",
        )
    return data


# --------------------------------------------------------------------------- #
# Health
# --------------------------------------------------------------------------- #
@app.get("/", tags=["Health"], summary="Root health check")
async def root():
    return {"status": "ok", "service": "smart-retail-ai", "time": datetime.now(timezone.utc).isoformat()}


@app.get("/health", tags=["Health"], summary="Readiness probe")
async def health():
    return {
        "status": "healthy",
        "models_loaded": {
            "product_classifier": True,
            "face_db": True,
            "sentiment_model": True,
            "chatbot_intents": True,
        },
    }


# --------------------------------------------------------------------------- #
# Computer Vision endpoints
# --------------------------------------------------------------------------- #
@app.post(
    "/classify-product",
    response_model=ProductClassificationResponse,
    responses={415: {"model": ErrorResponse}, 413: {"model": ErrorResponse}},
    tags=["Computer Vision"],
    summary="Classify a product image into one of 5 retail categories",
    dependencies=[Depends(verify_api_key)],
)
async def classify_product(file: UploadFile = File(..., description="JPEG/PNG/WEBP product image.")):
    image_bytes = await _validated_image_bytes(file)
    try:
        result = cv_service.classify_product(image_bytes)
    except Exception as exc:  # pragma: no cover - defensive guard
        raise HTTPException(status_code=500, detail=f"Classification failed: {exc}") from exc

    _dashboard_state["total_products_classified"] += 1
    return result


@app.post(
    "/recognize-face",
    response_model=FaceRecognitionResponse,
    responses={415: {"model": ErrorResponse}, 413: {"model": ErrorResponse}},
    tags=["Computer Vision"],
    summary="Detect & recognize a customer's face, logging a timestamped visit",
    dependencies=[Depends(verify_api_key)],
)
async def recognize_face(file: UploadFile = File(..., description="JPEG/PNG/WEBP image containing a face.")):
    image_bytes = await _validated_image_bytes(file)
    try:
        result = cv_service.recognize_face(image_bytes)
    except Exception as exc:  # pragma: no cover
        raise HTTPException(status_code=500, detail=f"Face recognition failed: {exc}") from exc
    return result


# --------------------------------------------------------------------------- #
# NLP endpoint
# --------------------------------------------------------------------------- #
@app.post(
    "/analyze-sentiment",
    response_model=SentimentAnalysisResponse,
    tags=["NLP"],
    summary="Classify the sentiment of customer review / feedback text",
    dependencies=[Depends(verify_api_key)],
)
async def analyze_sentiment(payload: SentimentAnalysisRequest):
    result = nlp_service.analyze_sentiment(payload.text)

    _dashboard_state["sentiment_counts"][result["label"]] += 1
    _dashboard_state["sentiment_confidence_sum"] += result["confidence"]
    _dashboard_state["sentiment_count"] += 1
    return result


# --------------------------------------------------------------------------- #
# Chatbot endpoint
# --------------------------------------------------------------------------- #
@app.post(
    "/chatbot",
    response_model=ChatbotResponse,
    tags=["Chatbot"],
    summary="Get a hybrid rule-based/ML chatbot reply for a customer message",
    dependencies=[Depends(verify_api_key)],
)
async def chatbot(payload: ChatbotRequest):
    result = chatbot_service.get_reply(payload.message)
    _dashboard_state["total_chatbot_queries"] += 1
    return result


# --------------------------------------------------------------------------- #
# Dashboard endpoint
# --------------------------------------------------------------------------- #
@app.get(
    "/dashboard/stats",
    response_model=DashboardStatsResponse,
    tags=["Dashboard"],
    summary="Aggregated live metrics for the customer-intelligence dashboard",
)
async def dashboard_stats():
    visit_stats = cv_service.visit_stats()
    sentiment_count = _dashboard_state["sentiment_count"]
    avg_conf = (
        _dashboard_state["sentiment_confidence_sum"] / sentiment_count
        if sentiment_count > 0 else 0.0
    )

    return DashboardStatsResponse(
        total_visits=visit_stats["total_visits"],
        returning_customers=visit_stats["returning_customers"],
        new_customers=visit_stats["new_customers"],
        total_products_classified=_dashboard_state["total_products_classified"],
        sentiment_breakdown=SentimentBreakdown(
            positive=_dashboard_state["sentiment_counts"]["Positive"],
            negative=_dashboard_state["sentiment_counts"]["Negative"],
            neutral=_dashboard_state["sentiment_counts"]["Neutral"],
        ),
        total_chatbot_queries=_dashboard_state["total_chatbot_queries"],
        average_sentiment_confidence=round(avg_conf, 2),
        generated_at=datetime.now(timezone.utc),
    )


# --------------------------------------------------------------------------- #
# Custom OpenAPI (adds API-key security scheme to Swagger docs)
# --------------------------------------------------------------------------- #
def custom_openapi():
    if app.openapi_schema:
        return app.openapi_schema

    schema = get_openapi(title=app.title, version=app.version, description=app.description, routes=app.routes)
    schema["components"]["securitySchemes"] = {
        "ApiKeyAuth": {"type": "apiKey", "in": "header", "name": "X-API-Key"}
    }
    schema["security"] = [{"ApiKeyAuth": []}]
    app.openapi_schema = schema
    return app.openapi_schema


app.openapi = custom_openapi
