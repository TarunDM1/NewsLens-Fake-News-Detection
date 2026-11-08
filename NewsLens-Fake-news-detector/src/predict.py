"""Prediction helpers shared by the Streamlit interface."""
from src.preprocessing import combine_title_and_body


def predict_text(model, title: str, body: str) -> dict:
    text = combine_title_and_body(title, body)
    if not text:
        raise ValueError("The supplied article does not contain readable text.")

    predicted = int(model.predict([text])[0])
    probabilities = model.predict_proba([text])[0]
    # Training convention: 0 = FAKE, 1 = REAL.
    fake_score = float(probabilities[0])
    real_score = float(probabilities[1])
    label = "REAL" if predicted == 1 else "FAKE"

    return {
        "label": label,
        "confidence": max(fake_score, real_score),
        "fake_score": fake_score,
        "real_score": real_score,
    }
