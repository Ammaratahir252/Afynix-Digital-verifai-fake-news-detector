# VerifAI — Fake News Detector

A real-time fake news detection web app built as **NLP Project**.

![Python](https://img.shields.io/badge/Python-3.8+-blue)
![Flask](https://img.shields.io/badge/Flask-2.x-lightgrey)
![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-orange)

---

## Features

- Real-time article analysis via Flask API
- Dual ML models: **Naive Bayes** + **Logistic Regression**
- **TF-IDF vectorization** with 15,000 features and bigrams
- Linguistic signal detection (caps, exclamations, sensationalism score)
- Separate **Analysis** and **History** pages
- Browser opens automatically on startup
- Session-based history with filter and clear support

---

## Project Structure

```
verifai_app/
├── app.py                  # Flask backend + ML pipeline
├── requirements.txt        # Python dependencies
└── templates/
    ├── index.html          # Analysis page  ( / )
```

---

## Setup & Run

### 1. Clone the repo
```bash
git clone https://github.com/YOUR_USERNAME/verifai-fake-news-detector.git
cd verifai-fake-news-detector
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the app
```bash
python app.py
```

Browser opens automatically at `http://127.0.0.1:5000`

> Models train on startup — takes 10–20 seconds on first run.

---

## Dataset

The app loads data in this order:

| Priority | Source |
|----------|--------|
| 1st | `Fake.csv` + `True.csv` (local Kaggle files) |
| 2nd | Auto-download from GitHub URL |
| 3rd | Built-in synthetic demo data (fallback) |

For best accuracy, download the full dataset from Kaggle:  
[Fake and Real News Dataset](https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset)

Place `Fake.csv` and `True.csv` in the root of the project folder.

---

## ML Pipeline

```
Raw text
   ↓
Text Cleaning      (lowercase, strip URLs, HTML, punctuation, digits)
   ↓
TF-IDF Vectors     (15,000 features, unigrams + bigrams, stopwords removed)
   ↓
┌──────────────┐   ┌───────────────────┐
│ Naive Bayes  │   │ Logistic Regression│
│ MultinomialNB│   │ lbfgs solver       │
└──────────────┘   └───────────────────┘
   ↓
Combined confidence score + linguistic signal analysis
   ↓
Verdict: FAKE / REAL / UNCERTAIN
```

---

## Requirements

```
flask
pandas
scikit-learn
requests
numpy
```

## License

MIT
