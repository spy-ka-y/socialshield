"""Test 3: 25 realistic Indian messages written by the team after every model was built (none were used for tuning).

They were written to look like real messages (bank alerts, OTPs, delivery updates, college groups, Hinglish chats,
job and UPI scams); they are not copies of real messages, and the file says so in its "source" column.
Run from the project root:  python src/test3_eval.py   ->  reports/test3_results.json
"""
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from detector import SAFE, MessageDetector  # noqa: E402
from train_v3 import layer_features, scores  # noqa: E402


def main():
    t = pd.read_csv(ROOT / "data/test3_realistic.csv")
    y = (t["label"].str.lower() == "scam").astype(int).values
    det = MessageDetector()
    v3 = joblib.load(ROOT / "models/v3.joblib")
    texts = t["text"].tolist()
    X, vs = layer_features(det, texts, v3["text_model"].predict_proba(texts)[:, 1])
    p2 = np.array([v.label != SAFE for v in vs])
    p3 = v3["combiner"].predict_proba(X)[:, 1] >= v3["threshold"]
    out = {"n": int(len(t)), "n_scam": int(y.sum()), "v2": scores(y, p2), "v3": scores(y, p3),
           "rows": [{"text": a, "label": int(b), "v2": bool(c), "v3": bool(d)} for a, b, c, d in zip(texts, y, p2, p3)]}
    (ROOT / "reports/test3_results.json").write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print("v2", out["v2"], "\nv3", out["v3"])


if __name__ == "__main__":
    main()
