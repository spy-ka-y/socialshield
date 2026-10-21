"""Score every account and every post once, so the app loads instantly.

Outputs (in models/):
  scored_accounts.csv  one row per account: fake probability, predicted attack type, top reasons
  scored_posts.csv     one row per post: hybrid alert, risk, rules fired
  summary.json         headline numbers for the dashboard

Run from the project root (after training):  python src/precompute.py
"""
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from detector import MessageDetector, SAFE  # noqa: E402
from features import alert_label  # noqa: E402

# Human-readable descriptions of the account features used in explanations.
READABLE = {
    "account_age_days": ("Account age", "{:.0f} days"),
    "avg_chars": ("Average post length", "{:.0f} characters"),
    "avg_shares": ("Average shares per post", "{:.1f}"),
    "avg_comments": ("Average comments per post", "{:.1f}"),
    "avg_likes": ("Average likes per post", "{:.1f}"),
    "night_ratio": ("Posts made between midnight and 6 AM", "{:.0%}"),
    "repeat_ratio": ("Copy-paste (repeated) posts", "{:.0%}"),
    "gap_mean_h": ("Average gap between posts", "{:.1f} hours"),
    "hour_std": ("Spread of posting hours", "{:.1f} hours"),
    "media_ratio": ("Posts with photos or videos", "{:.0%}"),
    "url_ratio": ("Posts containing links", "{:.0%}"),
    "followers_count": ("Followers", "{:.0f}"),
    "following_count": ("Following", "{:.0f}"),
    "follower_following_ratio": ("Follower to following ratio", "{:.2f}"),
    "has_location": ("Has a location on profile", "{:.0f}"),
    "profile_picture": ("Has a profile picture", "{:.0f}"),
    "has_bio": ("Has a bio", "{:.0f}"),
    "is_verified": ("Verified badge", "{:.0f}"),
}


def account_reasons(row, real_med, fake_med, importance, k=3):
    """Pick the important features where this account looks like a typical fake, not a typical real one."""
    reasons = []
    for f in importance.index:
        if f not in READABLE:
            continue
        v, r, fk = row[f], real_med[f], fake_med[f]
        if r == fk:
            continue
        if abs(v - fk) < abs(v - r):
            name, fmt = READABLE[f]
            reasons.append(f"{name}: {fmt.format(v)} (typical genuine account: {fmt.format(r)})")
        if len(reasons) == k:
            break
    return " | ".join(reasons)


def main():
    profiles = pd.read_csv(ROOT / "data/raw_user_profiles.csv")
    acts = pd.read_csv(ROOT / "data/raw_user_activities.csv")
    X = pd.read_csv(ROOT / "models/account_features.csv", index_col=0)
    acc = joblib.load(ROOT / "models/account_model.joblib")["model"]
    atk = joblib.load(ROOT / "models/attack_type_model.joblib")["model"]
    importance = pd.Series(json.loads((ROOT / "reports/account_results.json").read_text(encoding="utf-8"))["feature_importance"])

    p = profiles.set_index("user_id").loc[X.index]
    fake_prob = acc.predict_proba(X)[:, 1]
    pred_fake = fake_prob >= 0.5
    attack_pred = np.where(pred_fake, atk.predict(X), "genuine")
    real_med = X[p["is_fake"] == False].median()  # noqa: E712
    fake_med = X[p["is_fake"] == True].median()  # noqa: E712
    test_ids = set(pd.read_csv(ROOT / "models/test_user_ids.csv")["user_id"])

    accounts = pd.DataFrame({
        "user_id": X.index,
        "username": p["username"].values,
        "full_name": p["full_name"].values,
        "followers_count": p["followers_count"].values,
        "following_count": p["following_count"].values,
        "posts_count": p["posts_count"].values,
        "account_age_days": p["account_age_days"].values,
        "is_verified": p["is_verified"].values,
        "home_country": p["home_country"].values,
        "fake_prob": fake_prob.round(4),
        "predicted_fake": pred_fake,
        "predicted_attack": attack_pred,
        "true_is_fake": p["is_fake"].values,
        "true_type": p["account_type"].values,
        "in_test_set": [u in test_ids for u in X.index],
    })
    accounts["reasons"] = [
        account_reasons(X.loc[u], real_med, fake_med, importance) if f else ""
        for u, f in zip(X.index, pred_fake)
    ]

    # Score every post with the hybrid detector (unique texts only, then map back).
    det = MessageDetector()
    uniq = acts["content"].drop_duplicates().tolist()
    print(f"scoring {len(uniq)} unique post texts ...")
    verdicts = dict(zip(uniq, det.analyze_many(uniq)))
    acts["alert"] = acts["content"].map(lambda t: verdicts[t].label)
    acts["risk"] = acts["content"].map(lambda t: verdicts[t].risk)
    acts["rules"] = acts["content"].map(lambda t: ",".join(h["id"] for h in verdicts[t].rule_hits))
    acts["true_alert"] = acts["content_type"].map(alert_label)
    posts = acts[["activity_id", "user_id", "timestamp", "content", "likes", "comments", "shares",
                  "device", "platform", "post_country", "alert", "risk", "rules", "true_alert"]]

    # Counters the teacher asked for: who is spamming, asking for money, sending phishing forms.
    flagged = posts[posts["alert"] != SAFE]
    by_alert_accounts = flagged.groupby("alert")["user_id"].nunique().to_dict()
    by_alert_posts = flagged["alert"].value_counts().to_dict()

    # Real evidence for the spread view: shares relative to the author's audience, per post type.
    m = posts.merge(accounts[["user_id", "followers_count"]], on="user_id")
    m["true_type"] = acts["content_type"].values
    spread = m.groupby("true_type").agg(avg_shares=("shares", "mean"), median_followers=("followers_count", "median"))
    spread["shares_per_100_followers"] = 100 * spread["avg_shares"] / spread["median_followers"].clip(lower=1)

    summary = {
        "accounts_total": int(len(accounts)),
        "accounts_flagged_fake": int(pred_fake.sum()),
        "posts_total": int(len(posts)),
        "posts_flagged": int(len(flagged)),
        "accounts_by_alert": {k: int(v) for k, v in by_alert_accounts.items()},
        "posts_by_alert": {k: int(v) for k, v in by_alert_posts.items()},
        "attack_type_counts": accounts.loc[pred_fake, "predicted_attack"].value_counts().to_dict(),
        "spread_by_type": spread.round(3).reset_index().to_dict(orient="records"),
    }

    accounts.to_csv(ROOT / "models/scored_accounts.csv", index=False)
    posts.to_csv(ROOT / "models/scored_posts.csv", index=False)
    (ROOT / "models/summary.json").write_text(json.dumps(summary, indent=2, default=int), encoding="utf-8")
    print(json.dumps({k: v for k, v in summary.items() if k != "spread_by_type"}, indent=2, default=int))
    print(spread.round(2).sort_values("shares_per_100_followers", ascending=False))


if __name__ == "__main__":
    main()
