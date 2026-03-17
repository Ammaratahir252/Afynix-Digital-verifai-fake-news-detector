from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
import re, string, io, requests, threading, webbrowser

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, precision_score,
                             recall_score, f1_score)
import warnings
warnings.filterwarnings("ignore")

app = Flask(__name__)

# ── Global state ─────────────────────────────────────────────────
models_ready  = False
nb_model      = None
lr_model      = None
vectorizer    = None
model_metrics = {}

# ── Dataset URL ──────────────────────────────────────────────────
DATASET_URL = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/fake_or_real_news.csv"

# ── Text Cleaning ─────────────────────────────────────────────────
def clean_text(text: str) -> str:
    text = str(text).lower()
    text = re.sub(r"http\S+|www\.\S+", " ", text)
    text = re.sub(r"<[^>]+>", " ", text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\d+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

# ── Synthetic fallback ────────────────────────────────────────────
def make_synthetic():
    np.random.seed(42)
    fk = [
        "shocking government secret hiding truth conspiracy exposed hoax viral false deep state plot lies propaganda manipulation agenda corrupt bribed leaked exclusive breaking bombshell explosive scandal suppressed censored forced wake sheeple elites cabal miracle cure wont believe",
        "BREAKING shocking truth exposed secret hidden agenda conspiracy government hiding suppressing information miracle cure doctors hate explosive scandal leaked documents deep state manipulation propaganda lies media censored wake sheeple bombshell unverified rumored allegedly",
    ]
    rl = [
        "study research published journal official confirmed data statistics analysis report scientists university government evidence peer reviewed according spokesperson said stated findings results survey trial clinical regulatory commission authority department agency institute",
        "researchers university published peer reviewed journal findings results data analysis official statement confirmed government spokesperson said evidence study shows statistics survey clinical trial regulatory agency experts analysts officials announced",
    ]
    rows = []
    rnd = lambda a: a[np.random.randint(len(a))]
    for _ in range(800): rows.append((" ".join(rnd(fk) for _ in range(3)), "FAKE"))
    for _ in range(800): rows.append((" ".join(rnd(rl)  for _ in range(3)), "REAL"))
    df = pd.DataFrame(rows, columns=["text", "label"])
    return df.sample(frac=1, random_state=42).reset_index(drop=True)

# ── Load Dataset ──────────────────────────────────────────────────
def load_dataset():
    # 1. Local Kaggle files
    try:
        fake = pd.read_csv("Fake.csv")
        true = pd.read_csv("True.csv")
        fake["label"] = "FAKE"; true["label"] = "REAL"
        df = pd.concat([fake, true], ignore_index=True)
        tc = "text"  if "text"  in df.columns else df.columns[0]
        tl = "title" if "title" in df.columns else None
        df["text"] = (df[tl].fillna("") + " " if tl else "") + df[tc].fillna("")
        df = df[["text","label"]].dropna()
        print(f"[OK] Local files: {len(df):,} articles")
        return df
    except FileNotFoundError:
        pass

    # 2. Online URL
    try:
        print("[..] Fetching dataset from URL...")
        r = requests.get(DATASET_URL, headers={"User-Agent":"Mozilla/5.0"}, timeout=15)
        r.raise_for_status()
        df = pd.read_csv(io.StringIO(r.text))
        if "label" in df.columns:
            df["label"] = df["label"].str.strip().str.upper()
            df = df[df["label"].isin(["FAKE","REAL"])]
            tc = "text"  if "text"  in df.columns else df.columns[1]
            tl = "title" if "title" in df.columns else None
            df["text"] = (df[tl].fillna("") + " " if tl else "") + df[tc].fillna("")
            df = df[["text","label"]].dropna()
            print(f"[OK] URL dataset: {len(df):,} articles")
            return df
    except Exception as e:
        print(f"[!!] URL failed: {e}")

    # 3. Synthetic fallback
    print("[..] Using synthetic demo data")
    return make_synthetic()

# ── Train ─────────────────────────────────────────────────────────
def train_models():
    global nb_model, lr_model, vectorizer, model_metrics, models_ready
    print("\n" + "="*50)
    print("  VerifAI — Training models...")
    print("="*50)

    df = load_dataset()
    df["clean"]     = df["text"].apply(clean_text)
    df["label_int"] = (df["label"] == "REAL").astype(int)

    X_tr, X_te, y_tr, y_te = train_test_split(
        df["clean"], df["label_int"],
        test_size=0.2, random_state=42, stratify=df["label_int"]
    )

    vectorizer = TfidfVectorizer(
        max_features=15000, ngram_range=(1,2),
        stop_words="english", sublinear_tf=True, min_df=2
    )
    Xtr = vectorizer.fit_transform(X_tr)
    Xte = vectorizer.transform(X_te)

    nb_model = MultinomialNB(alpha=0.1)
    nb_model.fit(Xtr, y_tr)

    lr_model = LogisticRegression(max_iter=1000, C=1.0, solver="lbfgs", n_jobs=-1)
    lr_model.fit(Xtr, y_tr)

    nbp = nb_model.predict(Xte)
    lrp = lr_model.predict(Xte)

    def m(yt, yp):
        return {
            "accuracy":  round(accuracy_score(yt, yp)*100, 1),
            "precision": round(precision_score(yt, yp, zero_division=0)*100, 1),
            "recall":    round(recall_score(yt, yp, zero_division=0)*100, 1),
            "f1":        round(f1_score(yt, yp, zero_division=0)*100, 1),
        }

    model_metrics = {
        "nb": m(y_te, nbp), "lr": m(y_te, lrp),
        "train_size": len(X_tr), "test_size": len(X_te),
        "vocab_size": len(vectorizer.vocabulary_),
    }
    models_ready = True
    print(f"[OK] Naive Bayes      Acc: {model_metrics['nb']['accuracy']}%  F1: {model_metrics['nb']['f1']}%")
    print(f"[OK] Logistic Reg.    Acc: {model_metrics['lr']['accuracy']}%  F1: {model_metrics['lr']['f1']}%")
    print(f"[OK] Vocab: {model_metrics['vocab_size']:,} features\n")
    print("="*50)
    print(f"  Open: http://127.0.0.1:5000")
    print("="*50 + "\n")

# ── Signal Detection ──────────────────────────────────────────────
FAKE_WORDS = ['shocking','secret','exposed','conspiracy','hidden','wont','believe',
              'miracle','cure','hoax','viral','false','deep','state','plot','lies',
              'propaganda','manipulation','agenda','corrupt','bribed','leaked',
              'exclusive','breaking','bombshell','explosive','scandal','suppressed',
              'censored','forced','wake','sheeple','elites','cabal','coverup']
REAL_WORDS = ['study','research','published','journal','official','confirmed','data',
              'statistics','analysis','report','scientists','university','government',
              'evidence','peer','reviewed','according','spokesperson','said','stated',
              'findings','results','survey','trial','clinical','regulatory','authority',
              'department','agency','institute','experts','analysts','officials']
HEDGE_WORDS = ['reportedly','allegedly','claimed','rumored','sources say',
               'anonymous','unconfirmed','some say','many believe','insiders']

def detect_signals(text):
    signals, lower = [], text.lower()
    caps = [m for m in re.findall(r'[A-Z]{3,}', text)
            if m not in ('URL','HTTP','HTTPS','WWW','NLP','AI','USA','UK')]
    if len(caps) > 2:
        signals.append({"level":"high","text":f"Excessive capitalization ({len(caps)} instances)","weight":"−"})
    excl = text.count('!')
    if excl > 1:
        signals.append({"level":"high","text":f"Multiple exclamation marks ({excl})","weight":"−"})
    fk = [w for w in FAKE_WORDS if w in lower]
    if len(fk) > 3:
        signals.append({"level":"high","text":f"High sensationalism score ({len(fk)} markers)","weight":"−"})
    elif len(fk) > 1:
        signals.append({"level":"medium","text":"Some sensationalist language detected","weight":"~"})
    rl = [w for w in REAL_WORDS if w in lower]
    if len(rl) > 3:
        signals.append({"level":"low","text":f"Strong credibility indicators ({len(rl)} markers)","weight":"+"})
    hg = [w for w in HEDGE_WORDS if w in lower]
    if hg:
        signals.append({"level":"medium","text":f'Hedging language: "{hg[0]}"',"weight":"~"})
    words = text.split()
    avg = sum(len(w) for w in words) / max(len(words), 1)
    if avg < 4.0:
        signals.append({"level":"high","text":"Very short word length (simple vocabulary)","weight":"−"})
    elif avg > 5.5:
        signals.append({"level":"low","text":"Complex vocabulary — suggests informative writing","weight":"+"})
    if not signals:
        signals.append({"level":"medium","text":"No strong linguistic signals detected","weight":"~"})
    return signals[:5]

# ── Routes ────────────────────────────────────────────────────────
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/api/status")
def status():
    return jsonify({"ready": models_ready, "metrics": model_metrics})

@app.route("/api/predict", methods=["POST"])
def predict():
    if not models_ready:
        return jsonify({"error": "Models not ready yet"}), 503
    data = request.get_json()
    text = data.get("text","").strip()
    if len(text) < 10:
        return jsonify({"error": "Text too short"}), 400

    cleaned  = clean_text(text)
    features = vectorizer.transform([cleaned])
    nb_pred  = int(nb_model.predict(features)[0])
    nb_proba = nb_model.predict_proba(features)[0].tolist()
    lr_pred  = int(lr_model.predict(features)[0])
    lr_proba = lr_model.predict_proba(features)[0].tolist()
    signals  = detect_signals(text)

    penalty   = sum(1 for s in signals if s["level"]=="high")*0.06
    boost     = sum(1 for s in signals if s["level"]=="low")*0.04
    avg_fake  = max(0, min(1, (nb_proba[0]+lr_proba[0])/2 + penalty - boost))

    if   avg_fake > 0.60: verdict = "fake"
    elif avg_fake < 0.40: verdict = "real"
    else:                  verdict = "unsure"

    conf = 50 if verdict=="unsure" else round(abs(avg_fake-0.5)*2*100)

    return jsonify({
        "verdict": verdict,
        "confidence": conf,
        "nb": {"pred":"FAKE" if nb_pred==0 else "REAL","conf":round(max(nb_proba)*100),"proba":nb_proba},
        "lr": {"pred":"FAKE" if lr_pred==0 else "REAL","conf":round(max(lr_proba)*100),"proba":lr_proba},
        "signals": signals,
        "avg_fake_prob": round(avg_fake*100, 1)
    })

# ── Launch ────────────────────────────────────────────────────────
def open_browser():
    webbrowser.open("http://127.0.0.1:5000")

if __name__ == "__main__":
    # Train models first
    train_models()
    # Open browser after 1.2s (gives Flask time to start)
    threading.Timer(1.2, open_browser).start()
    # Start server
    app.run(debug=False, port=5000, use_reloader=False)