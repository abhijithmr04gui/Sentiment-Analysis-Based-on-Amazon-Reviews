import os
import re
import pickle
from pathlib import Path

import pandas as pd
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import GridSearchCV, train_test_split


def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+|www\.\S+", " ", text)
    text = re.sub(r"[^a-z0-9\s']", " ", text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


ROOT_DIR = Path(__file__).resolve().parent
MODEL_PATH = ROOT_DIR / "model.pkl"
VECTORIZER_PATH = ROOT_DIR / "vectorizer.pkl"
DEFAULT_DATASET_CANDIDATES = ["Amazon_Reviews.csv", "amazon_alexa.tsv"]


def pick_dataset_path():
    for name in DEFAULT_DATASET_CANDIDATES:
        candidate = ROOT_DIR / name
        if candidate.exists():
            return candidate
    return ROOT_DIR / "amazon_alexa.tsv"


def safe_read_reviews(path: Path):
    path = Path(path)
    if path.suffix.lower() == ".tsv":
        return pd.read_csv(path, sep="\t")
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.read_csv(path, engine="python", on_bad_lines="skip")


def prepare_text_and_labels(df: pd.DataFrame):
    columns = {c.lower(): c for c in df.columns}

    text_col = None
    for candidate in ["verified_reviews", "review text", "review_text", "review", "text"]:
        if candidate in columns:
            text_col = columns[candidate]
            break
    if text_col is None:
        raise ValueError("No review text column found.")

    label_col = None
    if "feedback" in columns:
        label_col = columns["feedback"]
        y = pd.to_numeric(df[label_col], errors="coerce")
    elif "rating" in columns:
        rating_col = columns["rating"]
        ratings = (
            df[rating_col]
            .astype(str)
            .str.extract(r"(\d+(\.\d+)?)")[0]
            .astype(float)
        )
        y = ratings.apply(lambda r: 1 if r >= 4 else (0 if r <= 2 else None))
    else:
        raise ValueError("No label column found. Need 'feedback' or 'rating'.")

    data = pd.DataFrame({"text": df[text_col], "label": y})
    data = data.dropna(subset=["text", "label"]).copy()
    data["label"] = data["label"].astype(int)
    data["cleaned"] = data["text"].apply(clean_text)
    data = data[data["cleaned"].str.len() > 2].copy()

    if data["label"].nunique() < 2:
        raise ValueError("Need at least two classes for training.")

    return data


@st.cache_resource
def train_model(dataset_path=None):
    dataset_path = Path(dataset_path) if dataset_path else pick_dataset_path()
    df = safe_read_reviews(dataset_path)
    data = prepare_text_and_labels(df)

    X_train_text, X_test_text, y_train, y_test = train_test_split(
        data["cleaned"],
        data["label"],
        test_size=0.2,
        random_state=42,
        stratify=data["label"],
    )

    vectorizer = TfidfVectorizer(
        max_features=20000,
        ngram_range=(1, 2),
        min_df=2,
        sublinear_tf=True,
    )

    X_train = vectorizer.fit_transform(X_train_text)
    X_test = vectorizer.transform(X_test_text)

    grid = GridSearchCV(
        estimator=LogisticRegression(max_iter=3000, class_weight="balanced"),
        param_grid={
            "C": [0.5, 1.0, 2.0, 4.0],
            "solver": ["liblinear", "saga"],
        },
        scoring="f1_macro",
        cv=5,
        n_jobs=1,
        verbose=0,
    )
    grid.fit(X_train, y_train)
    model = grid.best_estimator_

    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)

    with open(MODEL_PATH, "wb") as f:
        pickle.dump(model, f)
    with open(VECTORIZER_PATH, "wb") as f:
        pickle.dump(vectorizer, f)

    metrics = {
        "dataset": str(dataset_path.name),
        "rows_used": int(len(data)),
        "best_params": grid.best_params_,
        "accuracy": float(acc),
        "classification_report": report,
    }
    return model, vectorizer, metrics



def load_model():
    try:
        with open(MODEL_PATH, "rb") as f:
            model = pickle.load(f)
        with open(VECTORIZER_PATH, "rb") as f:
            vectorizer = pickle.load(f)
        return model, vectorizer, None
    except Exception:
        return train_model()


def infer_text_column(df: pd.DataFrame):
    lower = {c.lower(): c for c in df.columns}
    for candidate in ["verified_reviews", "review text", "review_text", "review", "text"]:
        if candidate in lower:
            return lower[candidate]
    return None


if os.getenv("TRAIN_ONLY") == "1":
    _, _, metrics = train_model()
    print("Model trained successfully")
    print(f"Dataset: {metrics['dataset']}")
    print(f"Rows used: {metrics['rows_used']}")
    print(f"Best params: {metrics['best_params']}")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
else:
    st.set_page_config(page_title="Sentiment Analyzer", layout="centered")

    st.title("Amazon Review Sentiment Analysis")
    st.write("Predict whether a review is Positive or Negative")

    model, vectorizer, metrics = load_model()

    if metrics:
        st.success(f"Model trained successfully. Accuracy: {metrics['accuracy']:.4f}")
        st.caption(
            f"Dataset: {metrics['dataset']} | Rows used: {metrics['rows_used']} | "
            f"Best params: {metrics['best_params']}"
        )
        st.text("Classification Report:")
        st.text(metrics["classification_report"])

    user_input = st.text_area("Enter a review")

    if st.button("Predict Sentiment"):
        if user_input.strip() == "":
            st.warning("Please enter some text")
        else:
            cleaned = clean_text(user_input)
            vector = vectorizer.transform([cleaned])
            prediction = model.predict(vector)[0]
            if prediction == 1:
                st.success("Positive Review")
            else:
                st.error("Negative Review")

    st.write("---")
    st.subheader("Bulk Prediction")

    file = st.file_uploader("Upload CSV or TSV file", type=["csv", "tsv"])

    if file:
        if file.name.lower().endswith(".tsv"):
            data = pd.read_csv(file, sep="\t")
        else:
            data = pd.read_csv(file)

        text_col = infer_text_column(data)
        if text_col:
            data["cleaned"] = data[text_col].apply(clean_text)
            vectors = vectorizer.transform(data["cleaned"])
            preds = model.predict(vectors)
            data["Predicted Sentiment"] = pd.Series(preds).map({1: "Positive", 0: "Negative"})
            st.write(data.head())
            st.download_button(
                "Download Results",
                data.to_csv(index=False),
                file_name="predictions.csv",
            )
        else:
            st.error("No review text column found in uploaded file.")