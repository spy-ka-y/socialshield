"""Train the post/message alert model.

Reads the text of a post (or a DM) and decides which alert to raise:
Safe, Phishing, Money Scam, Spam, Misinformation, Impersonation or Adult Scam.

The split is done BY ACCOUNT, so no account's posts appear in both train and test.

Run from the project root:  python src/train_post_model.py
"""
import json
import sys
from pathlib import Path

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, f1_score, accuracy_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.naive_bayes import ComplementNB
from sklearn.pipeline import FeatureUnion, make_pipeline

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from features import alert_label, clean_text  # noqa: E402

SEED = 42
CLASSES = ["Safe", "Phishing", "Money Scam", "Spam", "Misinformation", "Impersonation", "Adult Scam"]


def vectorizer():
    # Word phrases catch scam wording ("dm me", "verify now"); character pieces catch odd spellings.
    return FeatureUnion([
        ("words", TfidfVectorizer(preprocessor=clean_text, ngram_range=(1, 2), min_df=2, sublinear_tf=True)),
        ("chars", TfidfVectorizer(preprocessor=clean_text, analyzer="char_wb", ngram_range=(3, 5), min_df=3, sublinear_tf=True)),
    ])


def main():
    a = pd.read_csv(ROOT / "data/raw_user_activities.csv")
    a["alert"] = a["content_type"].map(alert_label)

    split = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=SEED)
    tr, te = next(split.split(a, groups=a["user_id"]))
    train, test = a.iloc[tr], a.iloc[te]

    # Templates repeat a lot. Train on unique (text, label) pairs so repeats do not dominate.
    train_u = train.drop_duplicates(["content", "alert"])
    print(f"train posts {len(train)} -> {len(train_u)} unique | test posts {len(test)}")

    models = {
        "Naive Bayes": make_pipeline(vectorizer(), ComplementNB()),
        "Logistic Regression": make_pipeline(vectorizer(), LogisticRegression(max_iter=3000, class_weight="balanced", C=5)),
    }
    results = {}
    for name, m in models.items():
        m.fit(train_u["content"], train_u["alert"])
        pred = m.predict(test["content"])
        results[name] = {
            "accuracy": accuracy_score(test["alert"], pred),
            "macro_f1": f1_score(test["alert"], pred, average="macro"),
            "report": classification_report(test["alert"], pred, output_dict=True, zero_division=0),
            "confusion_matrix": confusion_matrix(test["alert"], pred, labels=CLASSES).tolist(),
        }
        print(f"\n{name}: acc={results[name]['accuracy']:.4f} macroF1={results[name]['macro_f1']:.4f}")
        print(classification_report(test["alert"], pred, labels=CLASSES, digits=3, zero_division=0))

    # Logistic Regression is deployed: its word weights let the app highlight WHY a message was flagged.
    best = models["Logistic Regression"]
    joblib.dump({"model": best, "classes": CLASSES}, ROOT / "models/post_model.joblib")
    (ROOT / "reports/post_results.json").write_text(json.dumps({"labels": CLASSES, "models": results}, indent=2), encoding="utf-8")

    for msg in [
        "Your Instagram account will be suspended. Verify your details here: insta-help-center.net",
        "Hey bro send me 2000 rs on gpay urgently, I will return tomorrow",
        "Had a great time at the beach with family today!",
        "BREAKING: government hiding the truth about vaccines, share before they delete this!",
    ]:
        probs = best.predict_proba([msg])[0]
        top = sorted(zip(best.classes_, probs), key=lambda t: -t[1])[:2]
        print(f"{msg[:70]:72s} -> {[(c, round(p, 2)) for c, p in top]}")


if __name__ == "__main__":
    main()
