# Amazon Review Sentiment Analysis

A production-style NLP project that classifies Amazon customer reviews as **Positive** or **Negative** using a tuned machine learning pipeline and an interactive Streamlit app.

This project is built to handle both small clean datasets and larger real-world review exports (including malformed CSV rows), then serve predictions through a simple UI for individual and bulk inference.

## Features

- Sentiment prediction for a **single review**
- **Bulk prediction** from uploaded CSV/TSV files
- Robust training pipeline for larger datasets
- Automatic dataset schema adaptation:
  - Supports `feedback` labels directly
  - Can derive labels from `rating` values
- Hyperparameter tuning with cross-validation
- Class imbalance handling (`class_weight='balanced'`)
- Exports trained artifacts (`model.pkl`, `vectorizer.pkl`)

## Project Structure

```text
sentiment_analysis amazon /
├── api.py                 # Main Streamlit app + training pipeline
├── Amazon_Reviews.csv     # Large dataset (new)
├── model.pkl              # Trained model artifact
├── vectorizer.pkl         # Trained TF-IDF vectorizer artifact
├── imbal.py               # Small utility script for class distribution check
└── templates/             # Optional template assets
```

## How the Model Works

1. Loads dataset (`Amazon_Reviews.csv` preferred, falls back to `amazon_alexa.tsv`)
2. Detects text and target columns automatically
3. Cleans review text (normalization + URL/symbol cleanup)
4. Converts text to TF-IDF features (uni+bi grams)
5. Trains `LogisticRegression` with grid search (`GridSearchCV`)
6. Evaluates with `accuracy_score` and `classification_report`
7. Saves the best model and vectorizer for inference

## Label Strategy

- If `feedback` exists: use it directly (`0/1`)
- If only `rating` exists:
  - `4` or `5` -> Positive (`1`)
  - `1` or `2` -> Negative (`0`)
  - `3` -> dropped as neutral

## Tech Stack

- Python
- Pandas
- Scikit-learn
- Streamlit

## Setup

> The folder name has a trailing space in this environment, so keep quotes exactly as shown.

```bash
cd "/Users/abhijithmr/sentiment_analysis amazon "
```

If you use the existing virtual environment:

```bash
source /Users/abhijithmr/venv/bin/activate
pip install pandas scikit-learn streamlit
```

If your setup uses local project dependencies:

```bash
PYTHONPATH="$PWD/.pydeps" /Users/abhijithmr/venv/bin/python -m pip install --target ".pydeps" pandas scikit-learn streamlit
```

## Train the Model

Run in train-only mode:

```bash
cd "/Users/abhijithmr/sentiment_analysis amazon "
PYTHONPATH="$PWD/.pydeps" TRAIN_ONLY=1 /Users/abhijithmr/venv/bin/python api.py
```

Expected output includes:

- dataset used
- rows used
- best hyperparameters
- final accuracy score

## Run the App

```bash
cd "/Users/abhijithmr/sentiment_analysis amazon "
PYTHONPATH="$PWD/.pydeps" /Users/abhijithmr/venv/bin/python -m streamlit run api.py --server.port 8504 --global.developmentMode=false
```

Then open:

- [http://localhost:8504](http://localhost:8504)

## Usage

### Single Prediction

1. Enter a review in the text area
2. Click **Predict Sentiment**
3. App returns Positive/Negative result

### Bulk Prediction

1. Upload a `.csv` or `.tsv`
2. App auto-detects review text column (`verified_reviews`, `Review Text`, `review`, `text`, etc.)
3. Predictions are shown in table
4. Download results as `predictions.csv`

## Current Performance

On the current larger dataset run:

- Rows used: ~20k
- Accuracy: ~0.95

> Accuracy can vary slightly by data quality and randomization.

## Troubleshooting

### Port already in use

Run on another port:

```bash
PYTHONPATH="$PWD/.pydeps" /Users/abhijithmr/venv/bin/python -m streamlit run api.py --server.port 8505 --global.developmentMode=false
```

Check who is using a port:

```bash
lsof -nP -iTCP:8504 -sTCP:LISTEN
```

### CSV parser errors

The app already has a fallback parser for malformed rows. If needed, clean/validate source CSV before training for best consistency.

## Future Improvements

- Add confusion matrix and ROC-AUC visualization
- Persist experiment metadata (timestamp, params, metrics)
- Add unit tests for preprocessing and schema detection
- Try transformer-based models (DistilBERT) for quality comparison
- Dockerize for easy deployment

---

If this project helps you, consider starring the repository and contributing improvements.
