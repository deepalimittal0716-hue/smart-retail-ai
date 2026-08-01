# Model Artifacts

This directory is intentionally empty in version control. Serialized model
files are generated here automatically:

- `product_classifier.h5` — MobileNetV2 transfer-learning classifier
  (built fresh on first run if absent; retrain via
  `notebooks/01_image_classifier_training.ipynb` on the full 5-class
  product image dataset for production accuracy).
- `face_db.pkl` — customer face encodings + visit log (seeded empty,
  populated as customers are recognized/enrolled).
- `sentiment_model.pkl` / `vectorizer.pkl` — TF-IDF + Logistic Regression
  sentiment classifier (bootstrapped from a small seed corpus if absent;
  retrain via `notebooks/03_sentiment_model_training.ipynb` on the full
  reviews dataset).
- `chatbot_model.pkl` — TF-IDF cosine-similarity index over `data/intents.json`.

Add `*.h5` and `*.pkl` to `.gitignore` in real deployments and instead
version large artifacts via DVC, Git LFS, or a model registry.
