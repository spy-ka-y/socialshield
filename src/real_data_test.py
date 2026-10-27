"""Real-world check: how does SocialShield do on REAL messages it was never built for?

Dataset: UCI SMS Spam Collection (Almeida, Gomez Hidalgo and Yamakami, 2011).
5,574 real SMS messages, each labelled "ham" (genuine) or "spam".
Download (one file, tab separated, no header):
    https://raw.githubusercontent.com/justmarkham/pycon-2016-tutorial/master/data/sms.tsv
Save it as data/external/sms.tsv, then run from the project root:
    python src/real_data_test.py

Three experiments:
  A. Transfer: the frozen detector (trained only on the synthetic Kaggle data) on all 5,574 real messages.
  B. Retrain: the SAME pipeline design trained on real messages (80/20 split, stratified).
  C. Learning curve: how many real labelled messages are needed to get close to B.
Writes reports/real_data_results.json and reports/figures/real_data_*.png
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from detector import SAFE, MessageDetector  # noqa: E402
from train_post_model import vectorizer  # noqa: E402

SEED = 42
DATA = ROOT / "data/external/sms.tsv"


def scores(y_true, y_pred):
    """y = 1 for spam. Detection = recall on spam, false alarms = share of genuine messages flagged."""
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    return {
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
        "spam_caught": round(float(recall_score(y_true, y_pred)), 4),
        "precision": round(float(precision_score(y_true, y_pred, zero_division=0)), 4),
        "f1": round(float(f1_score(y_true, y_pred)), 4),
        "false_alarms": round(float(y_pred[y_true == 0].mean()), 4),
    }


def main():
    if not DATA.exists():
        print(f"Missing {DATA}. See the download link at the top of this file.")
        sys.exit(1)
    d = pd.read_csv(DATA, sep="\t", header=None, names=["label", "text"], quoting=3)
    d["y"] = (d["label"] == "spam").astype(int)
    print(f"{len(d)} real messages: {d.y.sum()} spam, {(1 - d.y).sum()} genuine")
    out = {"dataset": "UCI SMS Spam Collection", "n": len(d), "n_spam": int(d.y.sum())}

    # ---- A. Transfer: frozen detector, no retraining
    det = MessageDetector()
    v = det.analyze_many(d["text"].tolist(), channel="dm")
    d["hybrid"] = [int(x.label != SAFE) for x in v]
    d["ml_only"] = [int(x.ml_label != SAFE) for x in v]
    d["rules"] = [",".join(h["id"] for h in x.rule_hits) for x in v]
    d["pred_label"] = [x.label for x in v]
    out["A_transfer"] = {"ML model only": scores(d.y, d.ml_only), "Rules + ML (frozen)": scores(d.y, d.hybrid)}
    print("A transfer:", json.dumps(out["A_transfer"], indent=1))

    # Which rules fire most on GENUINE messages (false-alarm sources) and on missed spam
    ham_hits = d[(d.y == 0) & (d.hybrid == 1)]
    out["A_false_alarm_rules"] = ham_hits["rules"].str.split(",").explode().value_counts().head(8).to_dict()
    out["A_false_alarm_examples"] = ham_hits["text"].head(6).tolist()
    out["A_missed_spam_examples"] = d[(d.y == 1) & (d.hybrid == 0)]["text"].head(6).tolist()
    out["A_category_of_caught_spam"] = d[(d.y == 1) & (d.hybrid == 1)]["pred_label"].value_counts().to_dict()

    # ---- B. Retrain the same design on real data
    tr, te = train_test_split(d, test_size=0.2, stratify=d.y, random_state=SEED)
    m = make_pipeline(vectorizer(), LogisticRegression(max_iter=3000, class_weight="balanced", C=5))
    m.fit(tr["text"], tr["y"])
    out["B_retrained"] = {"trained_on": len(tr), "tested_on": len(te), **scores(te.y, m.predict(te["text"]))}
    # The frozen detector on the same 20% test split, for a like-for-like comparison
    out["B_frozen_same_split"] = scores(te.y, te["hybrid"])
    print("B retrained:", out["B_retrained"])

    # ---- C. Learning curve: average of 5 random draws per size
    curve = []
    for n in [20, 50, 100, 200, 500, 1000, 2000, len(tr)]:
        f1s, fas, caught = [], [], []
        for s in range(5 if n < len(tr) else 1):
            sub, _ = (tr, None) if n >= len(tr) else train_test_split(tr, train_size=n, stratify=tr.y, random_state=s)
            mm = make_pipeline(vectorizer(), LogisticRegression(max_iter=3000, class_weight="balanced", C=5))
            mm.fit(sub["text"], sub["y"])
            sc = scores(te.y, mm.predict(te["text"]))
            f1s.append(sc["f1"]); fas.append(sc["false_alarms"]); caught.append(sc["spam_caught"])
        curve.append({"n_train": int(n), "f1": round(float(np.mean(f1s)), 4),
                      "spam_caught": round(float(np.mean(caught)), 4), "false_alarms": round(float(np.mean(fas)), 4)})
        print("  curve", curve[-1])
    out["C_learning_curve"] = curve

    # ---- D. Cross-domain matrix: does the real-SMS model work on OUR team-written DM tests?
    out["D_cross_domain"] = {}
    for name, f in [("Test 1 (62 DMs)", "data/dm_messages.csv"), ("Test 2 (30 DMs, unseen)", "data/dm_test2.csv")]:
        t = pd.read_csv(ROOT / f)
        yy = (t["label"] != SAFE).astype(int)
        vv = det.analyze_many(t["text"].tolist(), channel="dm")
        out["D_cross_domain"][name] = {
            "Rules + ML (synthetic-trained)": scores(yy, [int(x.label != SAFE) for x in vv]),
            "ML (synthetic-trained)": scores(yy, [int(x.ml_label != SAFE) for x in vv]),
            "ML (real-SMS-trained)": scores(yy, m.predict(t["text"])),
        }
    out["D_cross_domain"]["Real SMS test split (1,115)"] = {
        "Rules + ML (synthetic-trained)": out["B_frozen_same_split"],
        "ML (synthetic-trained)": scores(te.y, te["ml_only"]),
        "ML (real-SMS-trained)": {k: v for k, v in out["B_retrained"].items() if k not in ("trained_on", "tested_on")},
    }
    print("D cross-domain:", json.dumps(out["D_cross_domain"], indent=1))

    (ROOT / "reports").mkdir(exist_ok=True)
    (ROOT / "reports/real_data_results.json").write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print("Saved reports/real_data_results.json")


if __name__ == "__main__":
    main()
