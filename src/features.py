"""Shared feature engineering for SocialShield.

Used by both the training scripts and the Streamlit app, so the app
computes features exactly the same way the models were trained.
"""
import re

import numpy as np
import pandas as pd

# Columns that directly reveal the label. They are NEVER used as features.
LEAKY_COLUMNS = ["account_type", "content_type", "is_fake"]

PROFILE_FEATURES = [
    "account_age_days", "profile_completeness", "followers_count",
    "following_count", "posts_count", "is_private", "is_verified",
    "profile_picture", "profile_banner", "has_bio", "has_website",
    "has_location",
]

USERNAME_FEATURES = [
    "follower_following_ratio", "username_len", "username_digit_ratio",
    "username_has_official", "email_free_provider",
]

BEHAVIOR_FEATURES = [
    "avg_likes", "avg_comments", "avg_shares", "url_ratio", "avg_hashtags",
    "avg_mentions", "avg_chars", "media_ratio", "gap_mean_h", "gap_std_h",
    "hour_std", "night_ratio", "uniq_devices", "uniq_countries",
    "repeat_ratio",
]

ACCOUNT_FEATURES = PROFILE_FEATURES + USERNAME_FEATURES + BEHAVIOR_FEATURES

FREE_EMAIL = ("gmail.com", "yahoo.com", "hotmail.com", "outlook.com", "icloud.com")

# Map the dataset's fine-grained post types to the alert categories the app shows.
ALERT_MAP = {
    "phishing": "Phishing",
    "crypto_scam": "Money Scam",
    "bot": "Spam",
    "porn_scam": "Adult Scam",
    "impersonation": "Impersonation",
    "hate_speech": "Misinformation",
}
SAFE = "Safe"


def alert_label(content_type: str) -> str:
    return ALERT_MAP.get(content_type, SAFE)


def username_features(df: pd.DataFrame) -> pd.DataFrame:
    """Simple signals from username and email (no names are stored as features)."""
    u = df["username"].astype(str)
    out = pd.DataFrame(index=df.index)
    out["follower_following_ratio"] = df["followers_count"] / (df["following_count"] + 1)
    out["username_len"] = u.str.len()
    out["username_digit_ratio"] = u.apply(lambda s: sum(c.isdigit() for c in s) / max(len(s), 1))
    out["username_has_official"] = u.str.contains(r"official|real|verified", case=False).astype(int)
    out["email_free_provider"] = df["email"].astype(str).str.split("@").str[-1].isin(FREE_EMAIL).astype(int)
    return out


def behavior_features(activities: pd.DataFrame) -> pd.DataFrame:
    """Aggregate each user's posts into behavior features, indexed by user_id."""
    a = activities.copy()
    a["timestamp"] = pd.to_datetime(a["timestamp"])
    a = a.sort_values(["user_id", "timestamp"])
    a["gap_h"] = a.groupby("user_id")["timestamp"].diff().dt.total_seconds() / 3600
    a["is_night"] = (a["hour_of_day"] < 6).astype(int)
    g = a.groupby("user_id")
    out = pd.DataFrame({
        "avg_likes": g["likes"].mean(),
        "avg_comments": g["comments"].mean(),
        "avg_shares": g["shares"].mean(),
        "url_ratio": g["contains_url"].mean(),
        "avg_hashtags": g["hashtag_count"].mean(),
        "avg_mentions": g["mention_count"].mean(),
        "avg_chars": g["character_count"].mean(),
        "media_ratio": g["has_media"].mean(),
        "gap_mean_h": g["gap_h"].mean(),
        "gap_std_h": g["gap_h"].std(),
        "hour_std": g["hour_of_day"].std(),
        "night_ratio": g["is_night"].mean(),
        "uniq_devices": g["device"].nunique(),
        "uniq_countries": g["post_country"].nunique(),
        # Share of posts that repeat an earlier message word for word (spam signal).
        "repeat_ratio": 1 - g["content"].nunique() / g.size(),
    })
    return out.fillna(0)


def build_account_table(profiles: pd.DataFrame, activities: pd.DataFrame) -> pd.DataFrame:
    """One row per account with all features plus labels kept separately."""
    p = profiles.set_index("user_id")
    feats = p[PROFILE_FEATURES].astype(float)
    feats = feats.join(username_features(p))
    feats = feats.join(behavior_features(activities)).fillna(0)
    return feats[ACCOUNT_FEATURES]


URL_RE = re.compile(r"(https?://|www\.|bit\.ly|\b[\w-]+\.(com|net|info|org|io|ru|xyz)\b)", re.I)


def clean_text(s: str) -> str:
    """Normalise text for the post model; mark links so the model sees them as one token."""
    s = str(s)
    s = URL_RE.sub(" urltoken ", s)
    s = re.sub(r"\$\s?\d[\d,]*k?", " moneytoken ", s, flags=re.I)
    s = re.sub(r"\d+", " numtoken ", s)
    return s.lower()
