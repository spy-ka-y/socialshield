"""Real bot accounts: test and train the account model on Cresci-2017 (real Twitter accounts).

Cresci et al. (2017) labelled 3,474 genuine accounts and 9,263 bot accounts of five kinds:
  social spambots 1-3 (bots built to imitate humans), traditional spambots, fake followers.
The files are the users.csv tables of each group (profile data only), saved in data/external/cresci2017/.

Experiments (profile features only, because the tweets are not used):
  A. Transfer: the profile part of our Kaggle-trained account model, tested on real Twitter accounts.
  B. Same design trained on real accounts: Random Forest, 80/20 split stratified by group.
  C. Unseen bot type: train without one bot group, test on that group (can it catch a NEW kind of bot?).
Writes reports/real_bots_results.json and models/real_account_model.joblib (used by the app's profile checker).

Run from the project root:  python src/real_bots_test.py
"""
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from features import FREE_EMAIL  # noqa: E402,F401

DATA = ROOT / "data/external/cresci2017"
GROUPS = {"genuine_accounts": 0, "social_spambots_1": 1, "social_spambots_2": 1, "social_spambots_3": 1,
          "traditional_spambots_1": 1, "fake_followers": 1}
NAMES = {"social_spambots_1": "Social spambots 1 (political retweeters)", "social_spambots_2": "Social spambots 2 (app promoters)",
         "social_spambots_3": "Social spambots 3 (product spammers)", "traditional_spambots_1": "Traditional spambots",
         "fake_followers": "Fake followers"}
REF_DATE = pd.Timestamp("2015-12-31", tz="UTC")   # accounts were collected between 2011 and 2015

# Profile features that exist on both platforms (same meaning as in src/features.py)
SHARED = ["account_age_days", "followers_count", "following_count", "posts_count", "follower_following_ratio",
          "is_private", "is_verified", "profile_picture", "profile_banner", "has_bio", "has_website", "has_location",
          "username_len", "username_digit_ratio", "username_has_official"]
# Extra Twitter profile signals used only by the real-data model
TWITTER_EXTRA = ["favourites_count", "listed_count", "default_profile", "geo_enabled", "name_len"]
REAL_FEATURES = SHARED + TWITTER_EXTRA


def flag(s):
    return pd.to_numeric(s, errors="coerce").fillna(0).astype(float).clip(0, 1)


def has(s):
    return s.notna() & (s.astype(str).str.strip() != "")


def twitter_features(u: pd.DataFrame) -> pd.DataFrame:
    sn = u["screen_name"].astype(str)
    raw = u["created_at"].astype(str)
    created = pd.to_datetime(raw, format="%a %b %d %H:%M:%S %z %Y", errors="coerce", utc=True)
    # one group stores the date as milliseconds since 1970, written like "1183552203000L"
    ms = pd.to_numeric(raw.str.rstrip("L"), errors="coerce")
    created = created.fillna(pd.to_datetime(ms, unit="ms", errors="coerce", utc=True))
    f = pd.DataFrame(index=u.index)
    f["account_age_days"] = (REF_DATE - created).dt.days.clip(lower=0).fillna(0)
    f["followers_count"] = pd.to_numeric(u["followers_count"], errors="coerce").fillna(0)
    f["following_count"] = pd.to_numeric(u["friends_count"], errors="coerce").fillna(0)
    f["posts_count"] = pd.to_numeric(u["statuses_count"], errors="coerce").fillna(0)
    f["follower_following_ratio"] = f["followers_count"] / (f["following_count"] + 1)
    f["is_private"] = flag(u["protected"])
    f["is_verified"] = flag(u["verified"])
    f["profile_picture"] = 1 - flag(u["default_profile_image"])
    f["profile_banner"] = has(u["profile_banner_url"]).astype(float)
    f["has_bio"] = has(u["description"]).astype(float)
    f["has_website"] = has(u["url"]).astype(float)
    f["has_location"] = has(u["location"]).astype(float)
    f["username_len"] = sn.str.len()
    f["username_digit_ratio"] = sn.apply(lambda s: sum(c.isdigit() for c in s) / max(len(s), 1))
    f["username_has_official"] = sn.str.contains(r"official|real|verified", case=False).astype(float)
    f["favourites_count"] = pd.to_numeric(u["favourites_count"], errors="coerce").fillna(0)
    f["listed_count"] = pd.to_numeric(u["listed_count"], errors="coerce").fillna(0)
    f["default_profile"] = flag(u["default_profile"])
    f["geo_enabled"] = flag(u["geo_enabled"])
    f["name_len"] = u["name"].astype(str).str.len()
    return f


def load():
    parts = []
    for g, y in GROUPS.items():
        u = pd.read_csv(DATA / g / "users.csv", low_memory=False)
        f = twitter_features(u)
        f["y"], f["group"] = y, g
        parts.append(f)
    return pd.concat(parts, ignore_index=True)


def metrics(y, p):
    return {"accuracy": round(accuracy_score(y, p), 4), "precision": round(precision_score(y, p, zero_division=0), 4),
            "recall": round(recall_score(y, p, zero_division=0), 4), "f1": round(f1_score(y, p, zero_division=0), 4),
            "false_alarms": round(float(np.asarray(p)[np.asarray(y) == 0].mean()), 4) if (np.asarray(y) == 0).any() else None}


def rf():
    return RandomForestClassifier(n_estimators=300, class_weight="balanced", random_state=42, n_jobs=-1)


def main():
    d = load()
    print(d.groupby("group").size().to_dict())
    out = {"n": int(len(d)), "groups": d.groupby("group").size().to_dict(), "features": REAL_FEATURES}

    # A. Transfer from the synthetic Kaggle data (profile features only)
    profiles = pd.read_csv(ROOT / "data/raw_user_profiles.csv")
    from features import username_features
    kf = profiles[["account_age_days", "followers_count", "following_count", "posts_count", "is_private", "is_verified",
                   "profile_picture", "profile_banner", "has_bio", "has_website", "has_location"]].astype(float)
    kf = kf.join(username_features(profiles)[["follower_following_ratio", "username_len", "username_digit_ratio", "username_has_official"]])
    k_model = rf().fit(kf[SHARED], profiles["is_fake"].astype(int))
    out["A_transfer_from_kaggle"] = metrics(d["y"], k_model.predict(d[SHARED]))
    print("A transfer", out["A_transfer_from_kaggle"])

    # B. Trained on real accounts
    tr, te = train_test_split(d, test_size=0.2, random_state=42, stratify=d["group"])
    m = rf().fit(tr[REAL_FEATURES], tr["y"])
    p = m.predict(te[REAL_FEATURES])
    out["B_trained_on_real"] = metrics(te["y"], p)
    out["B_by_group"] = {g: round(float((p[te["group"].values == g] == GROUPS[g]).mean()), 4) for g in GROUPS}
    out["B_feature_importance"] = pd.Series(m.feature_importances_, REAL_FEATURES).sort_values(ascending=False).round(4).head(10).to_dict()
    m_shared = rf().fit(tr[SHARED], tr["y"])
    out["B_shared_features_only"] = metrics(te["y"], m_shared.predict(te[SHARED]))
    print("B real", out["B_trained_on_real"], out["B_by_group"])

    # C. A new kind of bot: leave one bot group out of training
    out["C_unseen_bot_type"] = {}
    for g in [k for k, v in GROUPS.items() if v == 1]:
        tr_c = tr[tr["group"] != g]
        mc = rf().fit(tr_c[REAL_FEATURES], tr_c["y"])
        caught = float(mc.predict(d.loc[d["group"] == g, REAL_FEATURES]).mean())
        fa = float(mc.predict(te.loc[te["y"] == 0, REAL_FEATURES]).mean())
        out["C_unseen_bot_type"][g] = {"caught": round(caught, 4), "false_alarms_on_genuine_test": round(fa, 4)}
        print("C", g, out["C_unseen_bot_type"][g])

    # Final real-data model for the app (trained on all real accounts)
    final = rf().fit(d[REAL_FEATURES], d["y"])
    (ROOT / "models").mkdir(exist_ok=True)
    joblib.dump({"model": final, "features": REAL_FEATURES, "medians_genuine": d[d["y"] == 0][REAL_FEATURES].median().to_dict(),
                 "medians_bot": d[d["y"] == 1][REAL_FEATURES].median().to_dict()}, ROOT / "models/real_account_model.joblib")
    (ROOT / "reports/real_bots_results.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    print("saved")


if __name__ == "__main__":
    main()
