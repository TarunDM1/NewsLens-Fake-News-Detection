# 📰 NewsLens — Fake News Detection & News Credibility Analyzer

NewsLens is an educational Natural Language Processing (NLP) and Machine Learning project that explores how text-classification models distinguish between news articles labelled as **FAKE** and **REAL** in a training dataset.

It combines text cleaning, TF-IDF feature extraction, model comparison, evaluation metrics, and an interactive Streamlit application for single-article and CSV-based predictions.

> **Important:** NewsLens is not a fact-checking engine. It predicts patterns learned from a labelled dataset. A model score does not prove that an article or claim is true or false.

## ✨ Features

- Clean and normalize article headlines and body text.
- Convert text into unigram and bigram TF-IDF features.
- Train and compare Logistic Regression, Multinomial Naive Bayes, and calibrated Linear SVM.
- Select a model using held-out F1-score.
- Analyze one article through a Streamlit interface.
- Upload a CSV file for batch predictions.
- Download prediction results.
- View accuracy, precision, recall, F1-score, and ROC-AUC.
- Keep preprocessing, training, and prediction code in separate modules.

## 🧰 Technology Stack

| Tool | Usage |
|---|---|
| Python | Application and model development |
| Pandas | Dataset handling |
| Scikit-learn | TF-IDF, classifiers, train/test split and metrics |
| Streamlit | Interactive web interface |
| Joblib | Saving and loading the trained pipeline |

## 🗂️ Project Structure

```text
NewsLens/
├── dataset/
│   └── README.md
├── models/
│   └── README.md
├── src/
│   ├── __init__.py
│   ├── preprocessing.py
│   ├── predict.py
│   └── train.py
├── app.py
├── requirements.txt
├── .gitignore
└── README.md
```

## 📚 Dataset

The training workflow is designed for the ISOT Fake News Dataset format:

- `Fake.csv`
- `True.csv`

Each file must contain a `text` column. A `title` column is recommended.

The raw data is not included in this repository. Obtain it from an authorized source, review its terms, and place the two CSV files inside `dataset/`. See [`dataset/README.md`](dataset/README.md).

## ⚙️ Setup

```bash
git clone https://github.com/YOUR-USERNAME/NewsLens.git
cd NewsLens

python -m venv .venv
```

Activate the environment:

**Windows**
```bash
.venv\Scripts\activate
```

**macOS / Linux**
```bash
source .venv/bin/activate
```

Install packages:

```bash
pip install -r requirements.txt
```

## 🧠 Train the Models

After adding `dataset/Fake.csv` and `dataset/True.csv`, run:

```bash
python -m src.train
```

The training script:

1. Loads and cleans the article text.
2. Removes empty and duplicate text records.
3. Creates a stratified training and test split.
4. Fits TF-IDF only on the training portion through a Scikit-learn Pipeline.
5. Trains three classifiers.
6. Calculates evaluation metrics on the held-out test set.
7. Saves the best pipeline and metrics in `models/`.

The model is selected by F1-score, not by a hard-coded accuracy claim.

## ▶️ Launch the App

```bash
streamlit run app.py
```

The app provides three sections:

- **Analyze article:** classify one headline/article and inspect model scores.
- **Batch analysis:** upload a CSV containing a `text` column and optionally a `title` column.
- **Model insights:** compare evaluation metrics from the training run.

## 📏 Evaluation

The project reports:

- **Accuracy:** overall proportion of correct predictions.
- **Precision:** proportion of predicted positive-class examples that are correct.
- **Recall:** proportion of actual positive-class examples identified.
- **F1-score:** harmonic mean of precision and recall.
- **ROC-AUC:** ranking quality across classification thresholds.

Metrics depend on the dataset, its labels, and the test split. High scores on one benchmark do not guarantee generalization to current news or unfamiliar publishers. Dataset duplicates, publisher-specific writing patterns, and temporal leakage can make results overly optimistic; a future extension should include publisher-aware and time-based evaluation.

## 🔭 Future Enhancements

- Add publisher-aware and time-based evaluation.
- Compare against transformer models such as DistilBERT.
- Add explainability using LIME or SHAP.
- Add source citations and links to independent fact-checking resources.
- Add automated tests and GitHub Actions.
- Deploy the application using Streamlit Community Cloud after generating and managing model artifacts appropriately.

## 🎓 Learning Outcomes

This project demonstrates practical understanding of text preprocessing, TF-IDF vectorization, supervised classification, model comparison, evaluation metrics, model serialization, and Streamlit application development.

## ⚠️ Responsible Use

Do not use the prediction as the sole basis for sharing, suppressing, or judging news. The classifier learns correlations in a specific dataset and can be wrong, biased, or outdated. Always verify important claims through multiple credible sources.

## 👤 Author

**Tarun D M**

GitHub: https://github.com/YOUR-USERNAME

## 🙏 Dataset and Reference Acknowledgment

This implementation is a fresh educational project inspired by the general problem and workflow demonstrated in the public reference repository:

https://github.com/milanaht05/Fake-News-Detection

The source repository is a reference for the project concept and feature scope; this repository uses its own modular implementation. Review and comply with the dataset's terms and cite the dataset authors when using their data.
