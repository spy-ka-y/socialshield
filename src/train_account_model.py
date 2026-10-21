"""Train the account-level models.

1. Binary model: is this account genuine or fake/bot?
2. Attack-type model: if fake, what kind (bot, phishing, crypto scam, ...)?

Run from the project root:  python src/train_account_model.py
"""
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix,
                             f1_score, precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from features import ACCOUNT_FEATURES, build_account_table  # noqa: E402

SEED = 42


def main():
    profiles = pd.read_csv(ROOT / "data/raw_user_profiles.csv")
    activities = pd.read_csv(ROOT / "data/raw_user_activities.csv")

    X = build_account_table(profiles, activities)
    labels = profiles.set_index("user_id").loc[X.index]
    y = labels["is_fake"].astype(int)
    attack = labels["account_type"]

    # 80/20 split by account, stratified on the fine-grained type so every attack type is in both sets.
    # Plain numpy array of IDs: works whether or not pandas uses Arrow-backed strings.
    tr_idx, te_idx = train_test_split(np.asarray(X.index, dtype=object), test_size=0.2, random_state=SEED, stratify=attack)
    Xtr, Xte, ytr, yte = X.loc[tr_idx], X.loc[te_idx], y.loc[tr_idx], y.loc[te_idx]

    models = {
        "Logistic Regression": make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, class_weight="balanced")),
        "SVM (RBF)": make_pipeline(StandardScaler(), SVC(kernel="rbf", class_weight="balanced", random_state=SEED)),
        "Random Forest": RandomForestClassifier(n_estimators=300, class_weight="balanced", random_state=SEED, n_jobs=-1),
    }

    results = {}
    for name, model in models.items():
        model.fit(Xtr, ytr)
        pred = model.predict(Xte)
        # SVM has no predict_proba by default, so use its decision score for ROC-AUC.
        prob = model.predict_proba(Xte)[:, 1] if hasattr(model, "predict_proba") else model.decision_function(Xte)
        results[name] = {
            "accuracy": accuracy_score(yte, pred),
            "precision": precision_score(yte, pred),
            "recall": recall_score(yte, pred),
            "f1": f1_score(yte, pred),
            "roc_auc": roc_auc_score(yte, prob),
            "confusion_matrix": confusion_matrix(yte, pred).tolist(),
        }
        print(f"{name:20s}", {k: round(v, 4) for k, v in results[name].items() if k != "confusion_matrix"})

    # Random Forest is the deployed model: accurate and explainable through feature importance.
    rf = models["Random Forest"]
    importance = pd.Series(rf.feature_importances_, index=ACCOUNT_FEATURES).sort_values(ascending=False)

    # Attack-type model, trained only on fake accounts.
    fake_tr = tr_idx[y.loc[tr_idx].values == 1]
    fake_te = te_idx[y.loc[te_idx].values == 1]
    atk = RandomForestClassifier(n_estimators=300, random_state=SEED, n_jobs=-1)
    atk.fit(X.loc[fake_tr], attack.loc[fake_tr])
    atk_pred = atk.predict(X.loc[fake_te])
    atk_report = classification_report(attack.loc[fake_te], atk_pred, output_dict=True)
    print("\nAttack type model:\n", classification_report(attack.loc[fake_te], atk_pred, digits=3))

    (ROOT / "models").mkdir(exist_ok=True)
    joblib.dump({"model": rf, "features": ACCOUNT_FEATURES}, ROOT / "models/account_model.joblib")
    joblib.dump({"model": atk, "features": ACCOUNT_FEATURES}, ROOT / "models/attack_type_model.joblib")
    X.to_csv(ROOT / "models/account_features.csv")
    pd.Series(te_idx, name="user_id").to_csv(ROOT / "models/test_user_ids.csv", index=False)

    out = {
        "binary": results,
        "feature_importance": importance.round(5).to_dict(),
        "attack_type": {k: v for k, v in atk_report.items()},
        "attack_type_confusion": {
            "labels": sorted(attack.loc[fake_te].unique()),
            "matrix": confusion_matrix(attack.loc[fake_te], atk_pred, labels=sorted(attack.loc[fake_te].unique())).tolist(),
        },
        "n_train": int(len(tr_idx)), "n_test": int(len(te_idx)),
    }
    (ROOT / "reports").mkdir(exist_ok=True)
    (ROOT / "reports/account_results.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    print("\nTop 10 features:\n", importance.head(10).round(4))


if __name__ == "__main__":
    main()
