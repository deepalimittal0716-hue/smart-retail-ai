# Smart Retail & Customer Intelligence Platform

An end-to-end AI/ML capstone that unifies **Computer Vision**, **NLP**, and a
**hybrid chatbot** behind a single production-style **FastAPI** gateway for a
retail/e-commerce business.

## Features

| Capability | Endpoint | Technique |
|---|---|---|
| Product image classification | `POST /classify-product` | MobileNetV2 transfer learning (5 classes) |
| Returning-customer recognition | `POST /recognize-face` | Haar Cascade detection + encoding match |
| Review/feedback sentiment | `POST /analyze-sentiment` | TF-IDF + Logistic Regression |
| FAQ support chatbot | `POST /chatbot` | Rule-based intent match + TF-IDF cosine fallback |
| Live analytics | `GET /dashboard/stats` | Aggregated in-memory metrics |

Interactive API docs are auto-generated at **`/docs`** (Swagger) and **`/redoc`**.

## Quickstart (local)

```bash
python -m venv venv && source venv/bin/activate     # Windows: venv\Scripts\activate
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000
```

Visit `http://localhost:8000/docs` to try every endpoint interactively.

## Quickstart (Docker)

```bash
docker build -t smart-retail-ai .
docker run -p 8000:8000 -e REQUIRE_API_KEY=true -e SMART_RETAIL_API_KEY=my-secret smart-retail-ai
```

## Running tests

```bash
pytest -v
```

## Project structure

```
smart-retail-ai/
├── app/
│   ├── main.py                # FastAPI entrypoint / gateway
│   ├── schemas.py              # Pydantic V2 request/response models
│   ├── services/
│   │   ├── cv_service.py       # Module A - Computer Vision engine
│   │   ├── nlp_service.py      # Module B - Sentiment analysis engine
│   │   └── chatbot_service.py  # Module B - Hybrid chatbot engine
│   └── models/                 # Serialized model artifacts (.h5 / .pkl)
├── data/
│   └── intents.json            # Chatbot FAQ intents
├── notebooks/                  # Training / experimentation notebooks
├── tests/
│   └── test_endpoints.py       # pytest + FastAPI TestClient suite
├── .github/workflows/deploy.yml
├── Dockerfile
└── requirements.txt
```

## Security

Set `REQUIRE_API_KEY=true` and `SMART_RETAIL_API_KEY=<your-key>` to require an
`X-API-Key` header on every model-serving endpoint, simulating production
access control.

## Ethics & Privacy

Facial recognition is enrollment/consent-based and intended strictly for
demo/loyalty-analytics purposes. See `REPORT.md` (or the compiled PDF) for
the full ethical, legal, and privacy analysis, including GDPR/CCPA
considerations and bias mitigation notes.

## License

MIT
