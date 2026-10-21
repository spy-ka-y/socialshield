"""Measure the three detectors on the team-written DM test set.

Compares: ML model only, Hybrid (rules + ML), Hybrid + local AI second opinion.
Needs Ollama running with the model pulled (ollama pull qwen2.5:3b).

Run from the project root:  python src/evaluate_ai.py [path/to/messages.csv]
Writes reports/ai_eval_results.json and prints a summary.
"""
import json
import sys
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import ai_layer  # noqa: E402
from detector import SAFE, MessageDetector  # noqa: E402


def main():
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "data/dm_messages.csv"
    d = pd.read_csv(path)
    ready, model = ai_layer.status()
    if not ready:
        print("AI layer not available:", model)
        sys.exit(1)
    det = MessageDetector()
    rows = []
    print(f"Testing {len(d)} messages with {model} (a few seconds each) ...")
    ai_layer.check("warm up", model)
    for k, r in d.iterrows():
        v = det.analyze(r["text"])
        t0 = time.time()
        ai = ai_layer.check(r["text"], model)
        secs = time.time() - t0
        v2, by = ai_layer.combine(v, ai, SAFE)
        rows.append({"text": r["text"], "label": r["label"], "ml": v.ml_label, "hybrid": v.label, "with_ai": v2.label,
                     "ai_verdict": ai.verdict, "ai_category": ai.category, "ai_conf": ai.confidence,
                     "ai_reason": ai.reason, "seconds": round(secs, 2)})
        print(f"  {k + 1:3d}/{len(d)}  {r['label']:14s} -> {v2.label:14s} ({secs:.1f}s)")
    df = pd.DataFrame(rows)
    atk = df["label"] != SAFE
    summary = {}
    for name, col in [("ML model only", "ml"), ("Hybrid (rules + ML)", "hybrid"), ("Hybrid + local AI", "with_ai")]:
        summary[name] = {
            "attacks_detected": round(float((df.loc[atk, col] != SAFE).mean()), 3),
            "correct_type": round(float((df.loc[atk, col] == df.loc[atk, "label"]).mean()), 3),
            "false_alarms": round(float((df.loc[~atk, col] != SAFE).mean()), 3),
        }
    per_class = df[atk].groupby("label").apply(
        lambda g: {c: round(float((g[c] != SAFE).mean()), 3) for c in ("ml", "hybrid", "with_ai")}).to_dict()
    out = {"model": model, "file": path.name, "n_messages": len(df), "n_attacks": int(atk.sum()),
           "avg_seconds_per_message": round(float(df["seconds"].mean()), 2),
           "summary": summary, "per_class_detection": per_class, "rows": rows}
    (ROOT / "reports").mkdir(exist_ok=True)
    (ROOT / "reports/ai_eval_results.json").write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print("\nRESULTS")
    for name, s in summary.items():
        print(f"  {name:22s} attacks caught {s['attacks_detected']:.0%} | correct type {s['correct_type']:.0%} | false alarms {s['false_alarms']:.0%}")
    print(f"  Average AI time per message: {out['avg_seconds_per_message']} s")
    print("Saved to reports/ai_eval_results.json")


if __name__ == "__main__":
    main()
