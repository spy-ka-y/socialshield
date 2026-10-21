"""Robustness tests and report charts.

1. Feature-removal test: how accurate is the account model without its strongest clues?
2. Evasion test: if fake accounts gradually imitate real users, when does detection fail?
3. Hybrid vs ML-only message detection on the dataset's held-out posts.
4. Charts saved to reports/figures/.

Run from the project root (after both training scripts):  python src/evaluate.py
"""
import json
import sys
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, f1_score, recall_score
from sklearn.model_selection import GroupShuffleSplit

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from detector import MessageDetector, SAFE  # noqa: E402
from features import alert_label  # noqa: E402

FIG = ROOT / "reports/figures"
FIG.mkdir(parents=True, exist_ok=True)
SEED = 42

# Validated reference palette (light mode) and text tokens.
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK2, GRID, SURF = "#0b0b0b", "#52514e", "#e6e5e0", "#fcfcfb"
plt.rcParams.update({
    "figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF,
    "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "text.color": INK, "font.size": 11, "axes.titlesize": 13, "axes.titleweight": "bold",
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
    "grid.color": GRID, "grid.linewidth": 0.8, "axes.axisbelow": True,
})

# Features a real attacker can change cheaply when trying to look genuine.
MIMICABLE = ["account_age_days", "followers_count", "following_count", "avg_likes", "avg_comments",
             "avg_shares", "avg_chars", "media_ratio", "night_ratio", "hour_std", "gap_mean_h",
             "gap_std_h", "repeat_ratio", "url_ratio", "has_bio", "has_location", "profile_picture",
             "follower_following_ratio"]


def save(fig, name):
    fig.tight_layout()
    fig.savefig(FIG / name, dpi=200)
    plt.close(fig)


def account_tests(results):
    X = pd.read_csv(ROOT / "models/account_features.csv", index_col=0)
    prof = pd.read_csv(ROOT / "data/raw_user_profiles.csv").set_index("user_id").loc[X.index]
    y = prof["is_fake"].astype(int)
    test_ids = pd.read_csv(ROOT / "models/test_user_ids.csv")["user_id"]
    train_ids = X.index.difference(test_ids)
    importance = pd.Series(results["feature_importance"])

    # 1. Remove the strongest features one group at a time and retrain.
    removal = []
    for k in [0, 1, 3, 5, 8, 12]:
        dropped = list(importance.index[:k])
        cols = [c for c in X.columns if c not in dropped]
        rf = RandomForestClassifier(n_estimators=300, class_weight="balanced", random_state=SEED, n_jobs=-1)
        rf.fit(X.loc[train_ids, cols], y.loc[train_ids])
        pred = rf.predict(X.loc[test_ids, cols])
        removal.append({"removed": k, "dropped": dropped, "f1": f1_score(y.loc[test_ids], pred),
                        "recall": recall_score(y.loc[test_ids], pred)})
    print("Feature removal:", [(r["removed"], round(r["f1"], 3)) for r in removal])

    # 2. Evasion: move test fake accounts toward the typical real account, step by step.
    model = joblib.load(ROOT / "models/account_model.joblib")["model"]
    real_median = X.loc[train_ids][y.loc[train_ids] == 0].median()
    fakes = X.loc[test_ids][y.loc[test_ids] == 1]
    atk = prof.loc[fakes.index, "account_type"]
    evasion = []
    for alpha in np.round(np.linspace(0, 1, 11), 2):
        Xe = fakes.copy()
        Xe[MIMICABLE] = fakes[MIMICABLE] + alpha * (real_median[MIMICABLE] - fakes[MIMICABLE])
        caught = model.predict(Xe) == 1
        row = {"alpha": float(alpha), "detection_rate": float(caught.mean())}
        row.update({f"rate_{t}": float(caught[(atk == t).values].mean()) for t in sorted(atk.unique())})
        evasion.append(row)
    print("Evasion:", [(r["alpha"], round(r["detection_rate"], 3)) for r in evasion])
    return removal, evasion, X, y, test_ids


def message_tests():
    a = pd.read_csv(ROOT / "data/raw_user_activities.csv")
    a["alert"] = a["content_type"].map(alert_label)
    _, te = next(GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=SEED).split(a, groups=a["user_id"]))
    test = a.iloc[te]
    det = MessageDetector()
    uniq = test["content"].drop_duplicates()
    verdicts = {t: det.analyze(t) for t in uniq}
    hybrid = test["content"].map(lambda t: verdicts[t].label)
    ml = test["content"].map(lambda t: verdicts[t].ml_label)
    labels = ["Safe", "Phishing", "Money Scam", "Spam", "Misinformation", "Impersonation", "Adult Scam"]
    out = {}
    for name, pred in [("ML only", ml), ("Hybrid (rules + ML)", hybrid)]:
        safe = test["alert"] == SAFE
        out[name] = {
            "accuracy": float((pred == test["alert"]).mean()),
            "macro_f1": float(f1_score(test["alert"], pred, average="macro")),
            "false_alarm_rate": float((pred[safe] != SAFE).mean()),
            "attack_detection_rate": float((pred[~safe] != SAFE).mean()),
            "confusion": confusion_matrix(test["alert"], pred, labels=labels).tolist(),
        }
        print(name, {k: round(v, 4) for k, v in out[name].items() if k != "confusion"})
    return out, labels


def charts(results, removal, evasion, msg, labels):
    # Feature importance
    nice = {"avg_chars": "Average post length", "account_age_days": "Account age (days)", "avg_shares": "Average shares per post",
            "avg_comments": "Average comments per post", "avg_likes": "Average likes per post", "night_ratio": "Share of posts at night",
            "repeat_ratio": "Repeated (copy-paste) posts", "gap_mean_h": "Average gap between posts", "hour_std": "Spread of posting hours",
            "media_ratio": "Posts with media", "following_count": "Following count", "followers_count": "Follower count",
            "follower_following_ratio": "Follower/following ratio", "url_ratio": "Posts containing links", "has_location": "Has location"}
    imp = pd.Series(results["feature_importance"]).head(12)[::-1]
    imp.index = [nice.get(i, i) for i in imp.index]
    fig, ax = plt.subplots(figsize=(8, 5.2))
    ax.barh(imp.index, imp.values, color=BLUE, height=0.6)
    ax.set_title("What the account model relies on most")
    ax.set_xlabel("Feature importance (Random Forest)")
    ax.grid(axis="y", visible=False)
    for i, v in enumerate(imp.values):
        ax.text(v + 0.003, i, f"{v:.3f}", va="center", fontsize=9, color=INK2)
    save(fig, "01_feature_importance.png")

    # Account confusion matrix (Random Forest)
    cm = np.array(results["binary"]["Random Forest"]["confusion_matrix"])
    fig, ax = plt.subplots(figsize=(5, 4.4))
    ax.imshow(cm, cmap="Blues")
    ax.grid(False)
    for (i, j), v in np.ndenumerate(cm):
        ax.text(j, i, str(v), ha="center", va="center", fontsize=14, fontweight="bold",
                color="white" if v > cm.max() / 2 else INK)
    ax.set_xticks([0, 1], ["Predicted genuine", "Predicted fake"])
    ax.set_yticks([0, 1], ["Actually genuine", "Actually fake"])
    ax.set_title("Account model: test set (1,000 accounts)")
    save(fig, "02_account_confusion_matrix.png")

    # Feature removal robustness
    r = pd.DataFrame(removal)
    fig, ax = plt.subplots(figsize=(8, 4.6))
    ax.plot(r["removed"], r["f1"], color=BLUE, lw=2, marker="o", ms=8)
    for x, v in zip(r["removed"], r["f1"]):
        ax.annotate(f"{v:.3f}", (x, v), textcoords="offset points", xytext=(0, 10), ha="center", fontsize=9, color=INK2)
    ax.set_title("Accuracy holds up when the strongest clues are removed")
    ax.set_xlabel("Number of top features removed before retraining")
    ax.set_ylabel("F1-score on test set")
    ax.set_ylim(min(0.8, r["f1"].min() - 0.05), 1.03)
    save(fig, "03_feature_removal.png")

    # Evasion curve
    e = pd.DataFrame(evasion)
    fig, ax = plt.subplots(figsize=(8, 4.8))
    ax.plot(e["alpha"] * 100, e["detection_rate"] * 100, color=BLUE, lw=2, marker="o", ms=8, label="All fake accounts")
    ax.set_title("Evasion test: detection rate as fake accounts imitate real users")
    ax.set_xlabel("How closely fake accounts imitate a typical real account (%)")
    ax.set_ylabel("Fake accounts still detected (%)")
    ax.set_ylim(-3, 105)
    for x, v in zip(e["alpha"] * 100, e["detection_rate"] * 100):
        if x % 20 == 0:
            ax.annotate(f"{v:.0f}%", (x, v), textcoords="offset points", xytext=(12, 8), ha="center", fontsize=9, color=INK2)
    save(fig, "04_evasion_curve.png")

    # Message model: ML only vs hybrid on held-out posts
    names = list(msg)
    metrics = [("attack_detection_rate", "Attacks detected"), ("false_alarm_rate", "False alarms on safe posts"), ("macro_f1", "Macro F1")]
    fig, ax = plt.subplots(figsize=(8, 4.6))
    w = 0.36
    xs = np.arange(len(metrics))
    for i, (n, c) in enumerate(zip(names, [BLUE, ORANGE])):
        vals = [msg[n][m] * 100 for m, _ in metrics]
        bars = ax.bar(xs + (i - 0.5) * w, vals, w - 0.03, color=c, label=n)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + 1.5, f"{v:.1f}%", ha="center", fontsize=9, color=INK2)
    ax.set_xticks(xs, [l for _, l in metrics])
    ax.set_ylim(0, 112)
    ax.set_ylabel("Percent")
    ax.set_title("Message detection on held-out dataset posts")
    ax.legend(frameon=False, loc="upper right")
    ax.grid(axis="x", visible=False)
    save(fig, "05_message_ml_vs_hybrid.png")

    # Message confusion matrix (hybrid)
    cm = np.array(msg["Hybrid (rules + ML)"]["confusion"])
    norm = cm / cm.sum(axis=1, keepdims=True)
    fig, ax = plt.subplots(figsize=(8, 6.4))
    ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
    ax.grid(False)
    for (i, j), v in np.ndenumerate(cm):
        if v:
            ax.text(j, i, str(v), ha="center", va="center", fontsize=9,
                    color="white" if norm[i, j] > 0.5 else INK)
    ax.set_xticks(range(len(labels)), labels, rotation=35, ha="right")
    ax.set_yticks(range(len(labels)), labels)
    ax.set_xlabel("Predicted alert")
    ax.set_ylabel("Actual type")
    ax.set_title("Hybrid detector: post counts per alert type")
    save(fig, "06_message_confusion_matrix.png")


def main():
    results = json.loads((ROOT / "reports/account_results.json").read_text(encoding="utf-8"))
    removal, evasion, *_ = account_tests(results)
    msg, labels = message_tests()
    charts(results, removal, evasion, msg, labels)
    (ROOT / "reports/robustness_results.json").write_text(json.dumps(
        {"feature_removal": removal, "evasion": evasion, "message": msg}, indent=2), encoding="utf-8")
    print("Charts saved to", FIG)


if __name__ == "__main__":
    main()
