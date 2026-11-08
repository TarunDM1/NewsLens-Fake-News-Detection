from pathlib import Path
import json
import sys

import joblib
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from src.preprocessing import clean_article
from src.predict import predict_text

MODEL_PATH = ROOT / "models" / "best_model.joblib"
METRICS_PATH = ROOT / "models" / "metrics.json"

st.set_page_config(
    page_title="NewsLens | News Credibility Analyzer",
    page_icon="📰",
    layout="wide",
)

st.markdown("""
<style>
.block-container {padding-top: 2rem; max-width: 1180px;}
.hero {padding: 1.4rem 1.6rem; border: 1px solid rgba(128,128,128,.25);
       border-radius: 18px; margin-bottom: 1.2rem;}
.hero h1 {margin: 0; font-size: 2.15rem;}
.muted {opacity: .72;}
.result {padding: 1.2rem; border-radius: 14px;
         border: 1px solid rgba(128,128,128,.3);}
.small-note {font-size: .88rem; opacity: .75;}
</style>
""", unsafe_allow_html=True)

with st.sidebar:
    st.title("📰 NewsLens")
    st.caption("Machine-learning news text analyzer")
    page = st.radio("Workspace", ["Analyze article", "Batch analysis", "Model insights"])
    st.divider()
    st.caption("Educational NLP project • Not a fact-checking service")

st.markdown("""
<div class="hero">
  <h1>News Credibility Analyzer</h1>
  <p class="muted">Explore how machine learning models classify news text using linguistic patterns.</p>
</div>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        return None
    return joblib.load(MODEL_PATH)

@st.cache_data
def load_metrics():
    if not METRICS_PATH.exists():
        return None
    with open(METRICS_PATH, "r", encoding="utf-8") as f:
        return json.load(f)

model = load_model()

if model is None:
    st.warning("The trained model is not included yet.")
    st.markdown(
        "To create it, add the two ISOT dataset files as "
        "`dataset/Fake.csv` and `dataset/True.csv`, install the requirements, "
        "then run `python -m src.train`. The generated model will be saved in `models/`."
    )
    st.stop()

if page == "Analyze article":
    st.subheader("Analyze a news article")
    col1, col2 = st.columns([2, 1])
    with col1:
        title = st.text_input("Article headline", placeholder="Enter the news headline")
        article = st.text_area("Article text", height=260, placeholder="Paste the article text here...")
        analyze = st.button("Analyze article", type="primary", use_container_width=True)
    with col2:
        st.info(
            "**How it works**\n\n"
            "The application cleans the text, transforms it into TF-IDF features, "
            "and passes those features to the selected trained classifier."
        )
        st.caption("Longer, substantive article text generally gives the model more information than a headline alone.")

    if analyze:
        if len((title + " " + article).strip()) < 40:
            st.error("Please provide a headline or article text containing at least 40 characters.")
        else:
            result = predict_text(model, title, article)
            is_fake = result["label"] == "FAKE"
            st.divider()
            left, right = st.columns([1, 1])
            with left:
                st.markdown('<div class="result">', unsafe_allow_html=True)
                st.metric("Model classification", result["label"])
                st.progress(result["confidence"])
                st.write(f"Model score: **{result['confidence']:.1%}**")
                st.markdown("</div>", unsafe_allow_html=True)
            with right:
                st.metric("Fake class score", f"{result['fake_score']:.1%}")
                st.metric("Real class score", f"{result['real_score']:.1%}")
            st.caption(
                "This score reflects the model's output, not the probability that a claim is factually true. "
                "Verify important claims using reliable independent sources."
            )

elif page == "Batch analysis":
    st.subheader("Analyze multiple articles")
    st.write("Upload a CSV file containing a `text` column. A `title` column is optional.")
    uploaded = st.file_uploader("Choose CSV file", type=["csv"])
    if uploaded:
        try:
            data = pd.read_csv(uploaded)
            if "text" not in data.columns:
                st.error("The CSV must contain a column named `text`.")
            else:
                if "title" not in data.columns:
                    data["title"] = ""
                st.write(f"Loaded **{len(data):,}** rows.")
                if st.button("Run batch analysis", type="primary"):
                    predictions = [
                        predict_text(model, str(row.get("title", "")), str(row.get("text", "")))
                        for _, row in data.iterrows()
                    ]
                    output = data.copy()
                    output["model_prediction"] = [p["label"] for p in predictions]
                    output["model_score"] = [round(p["confidence"], 4) for p in predictions]
                    output["fake_score"] = [round(p["fake_score"], 4) for p in predictions]
                    output["real_score"] = [round(p["real_score"], 4) for p in predictions]
                    st.dataframe(output, use_container_width=True)
                    st.download_button(
                        "Download analyzed CSV",
                        output.to_csv(index=False).encode("utf-8"),
                        file_name="newslens_predictions.csv",
                        mime="text/csv",
                    )
        except Exception as exc:
            st.error(f"Could not read the CSV file: {exc}")

else:
    st.subheader("Model insights")
    metrics = load_metrics()
    if not metrics:
        st.info("Evaluation results are not available. Train the models first.")
    else:
        st.caption(f"Evaluation dataset size: {metrics.get('test_size', 'N/A'):,}")
        st.write(f"**Selected model:** {metrics.get('best_model', 'N/A')}")
        table = pd.DataFrame(metrics.get("model_results", {})).T
        if not table.empty:
            st.dataframe(table.style.format("{:.3f}"), use_container_width=True)
            st.bar_chart(table[["accuracy", "precision", "recall", "f1"]])
        st.caption(
            "Metrics describe performance on the held-out portion of the supplied dataset. "
            "They do not guarantee performance on current events or unseen publishers."
        )

st.divider()
st.markdown(
    '<p class="small-note">NewsLens is an educational text-classification demonstration. '
    'It does not verify sources, investigate claims, or establish factual truth.</p>',
    unsafe_allow_html=True,
)
