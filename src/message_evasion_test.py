"""Message evasion test: attack the detector with obfuscation tricks, then measure the normaliser defence.

Attacked messages = attacks the detector CATCHES on clean text (so every drop is caused by the trick):
  * team-written DMs (Test 1 + Test 2)
  * 300 Kaggle attack posts (random, stratified by type)
  * real SMS spam (UCI SMS Spam Collection), if data/external/sms.tsv exists
Each attack is applied with 3 random seeds and averaged.
Also checks that the defence adds no false alarms on clean genuine messages.

Run from the project root:  python src/message_evasion_test.py
Writes reports/message_evasion_results.json
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from detector import SAFE, MessageDetector  # noqa: E402
from features import alert_label  # noqa: E402
from obfuscate import ATTACKS  # noqa: E402

SEEDS = (0, 1, 2)


def caught(det, texts, channel="dm"):
    return np.array([v.label != SAFE for v in det.analyze_many(texts, channel=channel)])


def main():
    defended, undefended = MessageDetector(), MessageDetector(normalise_input=False)

    corpora, genuine = {}, {}
    dms = pd.concat([pd.read_csv(ROOT / "data/dm_messages.csv"), pd.read_csv(ROOT / "data/dm_test2.csv")])
    corpora["Team-written DMs"] = dms.loc[dms["label"] != SAFE, "text"].tolist()
    genuine["Team-written DMs"] = dms.loc[dms["label"] == SAFE, "text"].tolist()
    k = pd.read_csv(ROOT / "data/raw_user_activities.csv")
    k["alert"] = k["content_type"].map(alert_label)
    k = k.drop_duplicates("content")
    atk = k[k["alert"] != SAFE]
    corpora["Kaggle attack posts"] = atk.groupby("alert", group_keys=False).apply(
        lambda g: g.sample(min(len(g), 60), random_state=42))["content"].tolist()
    genuine["Kaggle attack posts"] = k[k["alert"] == SAFE].sample(1000, random_state=42)["content"].tolist()
    sms = ROOT / "data/external/sms.tsv"
    if sms.exists():
        s = pd.read_csv(sms, sep="\t", header=None, names=["label", "text"], quoting=3)
        corpora["Real SMS spam"] = s.loc[s["label"] == "spam", "text"].tolist()
        genuine["Real SMS spam"] = s.loc[s["label"] == "ham", "text"].tolist()

    # Kaggle items are public posts (promotional wording is normal there); the others are private messages
    channel = {"Kaggle attack posts": "post"}
    out = {"attacks": list(ATTACKS), "corpora": {}, "examples": []}
    for name, texts in corpora.items():
        ch = channel.get(name, "dm")
        base = caught(defended, texts, ch)
        texts = [t for t, c in zip(texts, base) if c]          # only messages caught on clean text
        res = {"n_attacked": len(texts)}
        for an, fn in ATTACKS.items():
            off, on = [], []
            for sd in SEEDS:
                ev = [fn(t, seed=sd) for t in texts]
                off.append(caught(undefended, ev, ch).mean())
                on.append(caught(defended, ev, ch).mean())
            res[an] = {"no_defence": round(float(np.mean(off)), 4), "with_defence": round(float(np.mean(on)), 4)}
            print(f"{name:22s} {an:32s} no defence {np.mean(off):6.1%}  with defence {np.mean(on):6.1%}")
        # false alarms on clean genuine messages, with and without the defence
        g = genuine[name]
        res["genuine_false_alarms"] = {"no_defence": round(float(caught(undefended, g, ch).mean()), 4),
                                       "with_defence": round(float(caught(defended, g, ch).mean()), 4), "n": len(g)}
        print(f"{name:22s} false alarms on {len(g)} genuine: {res['genuine_false_alarms']}")
        out["corpora"][name] = res

    # A few concrete examples for the report and the app
    for t in corpora["Team-written DMs"][:40]:
        for an in ("Look-alike letters (Cyrillic)", "Invisible characters", "All tricks combined"):
            ev = ATTACKS[an](t, seed=0)
            a, b = undefended.analyze(ev), defended.analyze(ev)
            if a.label == SAFE and b.label != SAFE and len(out["examples"]) < 6:
                out["examples"].append({"attack": an, "original": t, "evasive": ev, "no_defence": a.label,
                                        "with_defence": b.label, "tricks_found": b.evasion})
    (ROOT / "reports/message_evasion_results.json").write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print("Saved reports/message_evasion_results.json")


if __name__ == "__main__":
    main()
