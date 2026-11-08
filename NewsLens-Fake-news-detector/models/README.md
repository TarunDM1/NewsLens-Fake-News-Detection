# Model artifacts

Run `python -m src.train` after adding the dataset files.

The training script creates:

- `best_model.joblib` — selected TF-IDF + classifier pipeline
- `metrics.json` — held-out evaluation metrics

Generated model files are excluded from Git by default. If you decide to publish a trained model, first verify its size, provenance, reproducibility, and any applicable dataset/model licensing conditions.
