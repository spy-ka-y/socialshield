"""Real messages from the team's own phones: an honest leave-one-person-out test, then the final model.

Input: a CSV with columns  text, label (scam or genuine), person (P, Y or S)
For each person: train on everything else (synthetic posts, real SMS, Test 1, Test 2 and the OTHER two people's
messages), then test on that person's messages, which the model has never seen. This shows how SocialShield does on
real Indian messages from someone it never learned from.

Run from the project root:  python src/team_test.py data/team_messages.csv
Writes reports/team_results.json and models/v3_final.joblib (trained on everything, for the app).
"""
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from detector import SAFE, MessageDetector  # noqa: E402
from train_post_model import vectorizer  # noqa: E402
from train_v3 import FEATURES, SEED, layer_features, load_sets, scores  # noqa: E402

LOCAL_WEIGHT = 10.0   # Indian DMs and the team's messages count more than synthetic posts and UK SMS
SCAM_WORDS = {"scam", "spam", "fraud", "phishing", "fake", "1", "yes", "malicious"}


def load_team(path):
    t = pd.read_csv(path)
    t.columns = [c.strip().lower() for c in t.columns]
    t = t.rename(columns={"text": "content"}).dropna(subset=["content", "label"])
    t["content"] = t["content"].astype(str).str.strip()
    t["y"] = t["label"].astype(str).str.strip().str.lower().isin(SCAM_WORDS).astype(int)
    t["person"] = t.get("person", pd.Series(["?"] * len(t))).astype(str).str.strip().str.upper().str[:1]
    return t[t["content"].str.len() > 0].drop_duplicates("content")


def fit(train, det):
    """train: content, y, local (bool). Returns the real-text model and the combiner."""
    text_model = make_pipeline(vectorizer(), LogisticRegression(max_iter=3000, class_weight="balanced", C=5))
    oof = cross_val_predict(make_pipeline(vectorizer(), LogisticRegression(max_iter=3000, class_weight="balanced", C=5)),
                            train["content"], train["y"], cv=StratifiedKFold(5, shuffle=True, random_state=SEED),
                            method="predict_proba")[:, 1]
    text_model.fit(train["content"], train["y"])
    X, _ = layer_features(det, train["content"].tolist(), oof)
    comb = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, class_weight="balanced", C=1.0))
    comb.fit(X, train["y"], logisticregression__sample_weight=np.where(train["local"], LOCAL_WEIGHT, 1.0))
    return text_model, comb


def predict(det, text_model, comb, texts, threshold=0.5):
    X, vs = layer_features(det, texts, text_model.predict_proba(texts)[:, 1])
    return comb.predict_proba(X)[:, 1] >= threshold, np.array([v.label != SAFE for v in vs])


def main():
    team = load_team(sys.argv[1] if len(sys.argv) > 1 else ROOT / "data/team_messages.csv")
    print(f"team messages: {len(team)} ({team['y'].sum()} scam) from {sorted(team['person'].unique())}")
    k_tr, _, s_tr, _, t1, t2 = load_sets()
    base = pd.concat([k_tr.sample(3000, random_state=SEED).assign(local=False), s_tr.assign(local=False),
                      t1.assign(local=True), t2.assign(local=True)], ignore_index=True)[["content", "y", "local"]]
    det = MessageDetector()
    v3_old = joblib.load(ROOT / "models/v3.joblib")

    out = {"n": int(len(team)), "n_scam": int(team["y"].sum()), "per_person": {}}
    all_y, all_v2, all_v3old, all_v3team = [], [], [], []
    for person in sorted(team["person"].unique()):
        test = team[team["person"] == person]
        train = pd.concat([base, team[team["person"] != person][["content", "y"]].assign(local=True)], ignore_index=True)
        tm, comb = fit(train, det)
        texts = test["content"].tolist()
        p_team, p_v2 = predict(det, tm, comb, texts)
        p_old, _ = predict(det, v3_old["text_model"], v3_old["combiner"], texts, v3_old["threshold"])
        out["per_person"][person] = {"n": int(len(test)), "v2": scores(test["y"], p_v2), "v3 (no team data)": scores(test["y"], p_old),
                                     "v3 + other people's messages": scores(test["y"], p_team)}
        print(person, json.dumps(out["per_person"][person]))
        all_y += test["y"].tolist(); all_v2 += p_v2.tolist(); all_v3old += p_old.tolist(); all_v3team += p_team.tolist()
        out.setdefault("misses", []).extend(
            {"person": person, "text": t[:100], "label": int(y), "v3_team": bool(p)} for t, y, p in zip(texts, test["y"], p_team) if bool(p) != bool(y))
    out["overall"] = {"v2": scores(all_y, all_v2), "v3 (no team data)": scores(all_y, all_v3old),
                      "v3 + other people's messages": scores(all_y, all_v3team)}
    print("OVERALL", json.dumps(out["overall"], indent=1))

    # Final model for the app: trained on everything, including all team messages
    final = pd.concat([base, team[["content", "y"]].assign(local=True)], ignore_index=True)
    tm, comb = fit(final, det)
    joblib.dump({"text_model": tm, "combiner": comb, "threshold": 0.5, "features": FEATURES}, ROOT / "models/v3_final.joblib")
    (ROOT / "reports/team_results.json").write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print("saved reports/team_results.json and models/v3_final.joblib")


if __name__ == "__main__":
    main()
