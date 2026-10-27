"""SocialShield v3: learn from real messages, and learn how to weigh the layers.

v2 adds hand-picked weights (0.35, 0.45, ...) for rules, links and the ML model, and its text model has only seen
the synthetic Kaggle posts. v3 fixes both:
  1. A second text model ("real-text model") is trained on synthetic posts AND real SMS, so it has seen real wording.
  2. A small logistic-regression "combiner" learns how much each layer should count, from labelled examples.

Training data (never any test data):
  * Kaggle posts from the training accounts (same account-level split as the v2 text model), a sample of 3,000
  * Real SMS: the 80% training split (same seed as src/real_data_test.py)
  * Test 1 (62 team-written DMs). Test 1 was already used to improve v2, so it is not a fair test anyway.
To avoid the combiner seeing over-confident scores, the real-text model's score for every training row is an
out-of-fold prediction (5-fold cross-validation).

Held-out evaluation (never trained on): real SMS test split (1,115), Test 2 (30 DMs), Kaggle posts from test accounts.

Run from the project root:  python src/train_v3.py
Writes models/v3.joblib and reports/v3_results.json
"""
import json
import re
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupShuffleSplit, StratifiedKFold, cross_val_predict, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from detector import ATTACK_INFO, BENIGN_COMPILED, SAFE, MessageDetector  # noqa: E402
from features import alert_label  # noqa: E402
from train_post_model import vectorizer  # noqa: E402

SEED = 42
CATS = list(ATTACK_INFO)
FEATURES = [f"rules_{c}" for c in CATS] + ["amplifiers", "benign", "evasion", "link_strong", "link_weak",
                                           "ml_synthetic", "ml_real", "log_len"]


def load_sets():
    k = pd.read_csv(ROOT / "data/raw_user_activities.csv")
    k["y"] = (k["content_type"].map(alert_label) != SAFE).astype(int)
    tr, te = next(GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=SEED).split(k, groups=k["user_id"]))
    k_tr = k.iloc[tr].drop_duplicates("content")
    k_te = k.iloc[te]
    sms = pd.read_csv(ROOT / "data/external/sms.tsv", sep="\t", header=None, names=["label", "content"], quoting=3)
    sms["y"] = (sms["label"] == "spam").astype(int)
    s_tr, s_te = train_test_split(sms, test_size=0.2, stratify=sms["y"], random_state=SEED)
    t1 = pd.read_csv(ROOT / "data/dm_messages.csv").rename(columns={"text": "content"})
    t2 = pd.read_csv(ROOT / "data/dm_test2.csv").rename(columns={"text": "content"})
    for t in (t1, t2):
        t["y"] = (t["label"] != SAFE).astype(int)
    return k_tr, k_te, s_tr, s_te, t1, t2


def layer_features(det, texts, ml_real, channel="dm"):
    """One row per message: what each v2 layer found, plus the two text models' scores."""
    rows = []
    for t, v, pr in zip(texts, det.analyze_many(texts, channel=channel), ml_real):
        text = v.decoded or str(t)
        r = {f"rules_{c}": sum(h["weight"] for h in v.rule_hits if h["category"] == c and not h["id"].startswith("LK")) for c in CATS}
        r["amplifiers"] = sum(h["weight"] for h in v.rule_hits if h["category"] is None and not h["id"].startswith(("LK", "EV")))
        r["benign"] = sum(w for _, w, rx, _ in BENIGN_COMPILED if rx.search(text))
        r["evasion"] = float(any(h["id"] == "EV-01" for h in v.rule_hits))
        r["link_strong"] = max([h["weight"] for h in v.rule_hits if h["id"] in ("LK-01", "LK-02", "LK-03", "LK-04")] or [0])
        r["link_weak"] = sum(h["weight"] for h in v.rule_hits if h["id"] in ("LK-05", "LK-06", "LK-07", "LK-08"))
        r["ml_synthetic"] = 1 - v.ml_probs.get(SAFE, 0)
        r["ml_real"] = float(pr)
        r["log_len"] = np.log1p(len(str(t)))
        rows.append(r)
    return pd.DataFrame(rows)[FEATURES], det.analyze_many(texts, channel=channel)


def scores(y, p):
    y, p = np.asarray(y), np.asarray(p)
    return {"caught": round(float(p[y == 1].mean()), 4) if (y == 1).any() else None,
            "false_alarms": round(float(p[y == 0].mean()), 4) if (y == 0).any() else None, "n": int(len(y))}


def main():
    k_tr, k_te, s_tr, s_te, t1, t2 = load_sets()
    k_sample = k_tr.sample(3000, random_state=SEED)
    train = pd.concat([k_sample.assign(src="kaggle"), s_tr.assign(src="sms"), t1.assign(src="test1")], ignore_index=True)
    print("training rows:", train["src"].value_counts().to_dict())

    # 1) Real-text model: same TF-IDF + Logistic Regression design, binary (scam or not), trained on all sources.
    #    Training texts for this model: ALL Kaggle training posts + SMS train + Test 1.
    text_train = pd.concat([k_tr[["content", "y"]], s_tr[["content", "y"]], t1[["content", "y"]]], ignore_index=True)
    real_text = make_pipeline(vectorizer(), LogisticRegression(max_iter=3000, class_weight="balanced", C=5))
    # out-of-fold scores for the combiner's training rows (each row scored by a model that did not see it)
    oof_model = make_pipeline(vectorizer(), LogisticRegression(max_iter=3000, class_weight="balanced", C=5))
    oof = cross_val_predict(oof_model, train["content"], train["y"], cv=StratifiedKFold(5, shuffle=True, random_state=SEED),
                            method="predict_proba")[:, 1]
    real_text.fit(text_train["content"], text_train["y"])

    det = MessageDetector()
    X_tr, _ = layer_features(det, train["content"].tolist(), oof)
    w = train["src"].map({"kaggle": 1.0, "sms": 1.0, "test1": 10.0}).values   # the few Indian DMs count more
    comb = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, class_weight="balanced", C=1.0))
    comb.fit(X_tr, train["y"], logisticregression__sample_weight=w)

    # Threshold: the lowest that keeps out-of-fold false alarms on genuine training messages at or below 2%
    p_tr = cross_val_predict(make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, class_weight="balanced")),
                             X_tr, train["y"], cv=StratifiedKFold(5, shuffle=True, random_state=SEED), method="predict_proba")[:, 1]
    genuine = p_tr[train["y"].values == 0]
    threshold = float(max(0.5, np.quantile(genuine, 0.98)))
    print("threshold", round(threshold, 3))

    # Held-out evaluation
    out = {"threshold": threshold, "features": FEATURES,
           "weights": dict(zip(FEATURES, np.round(comb[-1].coef_[0], 3).tolist())), "eval": {}}
    sets = {"Real SMS test split": (s_te, "dm"), "Test 2 (unseen DMs)": (t2, "dm"),
            "Kaggle test accounts (posts)": (k_te.drop_duplicates("content").sample(5000, random_state=SEED), "post"),
            "Test 1 (used in training)": (t1, "dm")}
    for name, (df, ch) in sets.items():
        texts = df["content"].tolist()
        X, vs = layer_features(det, texts, real_text.predict_proba(texts)[:, 1], channel=ch)
        p3 = comb.predict_proba(X)[:, 1] >= threshold
        p2 = np.array([v.label != SAFE for v in vs])
        out["eval"][name] = {"v2": scores(df["y"], p2), "v3": scores(df["y"], p3)}
        print(f"{name:32s} v2 {out['eval'][name]['v2']}  v3 {out['eval'][name]['v3']}")
    joblib.dump({"text_model": real_text, "combiner": comb, "threshold": threshold, "features": FEATURES}, ROOT / "models/v3.joblib")
    (ROOT / "reports/v3_results.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps(out["weights"], indent=1))


if __name__ == "__main__":
    main()
