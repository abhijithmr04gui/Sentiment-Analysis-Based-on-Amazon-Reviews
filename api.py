import os
import re
import pickle
from pathlib import Path

import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import GridSearchCV, train_test_split


# =========================
# TEXT CLEANING + NEGATION HANDLING
# =========================
def clean_text(text):
    text = str(text).lower()

    # remove urls
    text = re.sub(r"http\S+|www\.\S+", " ", text)

    # keep only valid chars
    text = re.sub(r"[^a-z0-9\s']", " ", text)

    words = text.split()

    # 🔥 NEGATION HANDLING
    negation_words = {"not", "no", "never", "n't"}
    new_words = []
    negate = False

    for word in words:
        if word in negation_words:
            negate = True
            new_words.append(word)
            continue

        if negate:
            new_words.append("NOT_" + word)
            negate = False
        else:
            new_words.append(word)

    return " ".join(new_words)


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

    if "feedback" in columns:
        y = pd.to_numeric(df[columns["feedback"]], errors="coerce")
    elif "rating" in columns:
        ratings = (
            df[columns["rating"]]
            .astype(str)
            .str.extract(r"(\d+(\.\d+)?)")[0]
            .astype(float)
        )
        y = ratings.apply(lambda r: 1 if r >= 4 else (0 if r <= 2 else None))
    else:
        raise ValueError("No label column found.")

    data = pd.DataFrame({"text": df[text_col], "label": y})
    data = data.dropna(subset=["text", "label"]).copy()
    data["label"] = data["label"].astype(int)
    data["cleaned"] = data["text"].apply(clean_text)
    data = data[data["cleaned"].str.len() > 2].copy()

    return data


# =========================
# TRAIN MODEL
# =========================
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

    # 🔥 IMPROVED TF-IDF
    vectorizer = TfidfVectorizer(
        max_features=30000,
        ngram_range=(1, 3),
        min_df=2,
        max_df=0.9,
        sublinear_tf=True,
    )

    X_train = vectorizer.fit_transform(X_train_text)
    X_test = vectorizer.transform(X_test_text)

    # 🔥 SVM MODEL
    grid = GridSearchCV(
        estimator=LinearSVC(),
        param_grid={
            "C": [0.1, 1, 2, 4, 8]
        },
        scoring="f1_macro",
        cv=5,
    )

    grid.fit(X_train, y_train)
    model = grid.best_estimator_

    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)

    # save
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
    except:
        return train_model()


def infer_text_column(df: pd.DataFrame):
    lower = {c.lower(): c for c in df.columns}
    for candidate in ["verified_reviews", "review text", "review_text", "review", "text"]:
        if candidate in lower:
            return lower[candidate]
    return None


# =========================
# TRAIN MODE
# =========================
if os.getenv("TRAIN_ONLY") == "1":
    _, _, metrics = train_model()
    print("Model trained successfully")
    print(f"Dataset: {metrics['dataset']}")
    print(f"Rows used: {metrics['rows_used']}")
    print(f"Best params: {metrics['best_params']}")
    print(f"Accuracy: {metrics['accuracy']:.4f}")

# =========================
# APP MODE
# =========================
else:
    st.set_page_config(page_title="Sentiment Analyzer", layout="centered")

    st.title("Amazon Review Sentiment Analysis")

    model, vectorizer, metrics = load_model()

    if metrics:
        st.success(f"Accuracy: {metrics['accuracy']:.4f}")

    # SINGLE
    user_input = st.text_area("Enter a review")

    if st.button("Predict Sentiment"):
        if user_input.strip():
            cleaned = clean_text(user_input)
            vector = vectorizer.transform([cleaned])
            pred = model.predict(vector)[0]

            if pred == 1:
                st.success("Positive")
            else:
                st.error("Negative")

    # BULK
    st.write("---")
    file = st.file_uploader("Upload CSV/TSV", type=["csv", "tsv"])

    if file:
        data = pd.read_csv(file, sep="\t") if file.name.endswith(".tsv") else pd.read_csv(file)

        col = infer_text_column(data)

        if col:
            data["cleaned"] = data[col].apply(clean_text)
            preds = model.predict(vectorizer.transform(data["cleaned"]))

            data["Predicted Sentiment"] = pd.Series(preds).map({1: "Positive", 0: "Negative"})

            st.write(data.head())

            # 📊 VISUALIZATION
            counts = data["Predicted Sentiment"].value_counts()

            fig1, ax1 = plt.subplots()
            ax1.pie(counts, labels=counts.index, autopct='%1.1f%%')
            st.pyplot(fig1)

            fig2, ax2 = plt.subplots()
            counts.plot(kind='bar', ax=ax2)
            st.pyplot(fig2)

            st.download_button("Download", data.to_csv(index=False), "predictions.csv")

        else:
            st.error("No valid text column found.")