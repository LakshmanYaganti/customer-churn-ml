"""Train the churn models and write metrics to ``reports/metrics.json``.

Models:
- Logistic Regression baseline (class_weight='balanced')
- XGBoost (scale_pos_weight set from the class ratio)

Protocol: single stratified 80/20 train/test split with a fixed seed. All
reported metrics are computed on the held-out test set. The trained XGBoost
pipeline is also persisted to ``models/churn_model.joblib`` (git-ignored —
the API retrains on first use if it is absent).
"""

from __future__ import annotations

import json
import os

import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

from .data import load_data
from .evaluate import classification_metrics, decile_lift_table
from .features import build_preprocessor, split_xy
from .retention import campaign_impact

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(ROOT, "models", "churn_model.joblib")
METRICS_PATH = os.path.join(ROOT, "reports", "metrics.json")
RANDOM_STATE = 42


def build_models(y_train) -> dict[str, Pipeline]:
    neg, pos = (y_train == 0).sum(), (y_train == 1).sum()
    scale_pos_weight = neg / pos
    return {
        "logistic_regression": Pipeline(
            steps=[
                ("preprocessor", build_preprocessor()),
                (
                    "model",
                    LogisticRegression(max_iter=1000, class_weight="balanced"),
                ),
            ]
        ),
        "xgboost": Pipeline(
            steps=[
                ("preprocessor", build_preprocessor()),
                (
                    "model",
                    XGBClassifier(
                        n_estimators=300,
                        max_depth=4,
                        learning_rate=0.05,
                        subsample=0.9,
                        colsample_bytree=0.9,
                        scale_pos_weight=scale_pos_weight,
                        eval_metric="logloss",
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        ),
    }


def train_and_evaluate(df=None) -> dict:
    if df is None:
        df = load_data()
    X, y = split_xy(df)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
    )

    results: dict = {
        "dataset": {
            "rows": int(len(df)),
            "churners": int(y.sum()),
            "churn_rate": float(y.mean()),
            "train_rows": int(len(X_train)),
            "test_rows": int(len(X_test)),
        },
        "models": {},
    }

    best_name, best_pr_auc = None, -1.0
    for name, pipeline in build_models(y_train).items():
        pipeline.fit(X_train, y_train)
        proba = pipeline.predict_proba(X_test)[:, 1]
        metrics = classification_metrics(y_test, proba, threshold=0.5)
        lift = decile_lift_table(y_test, proba)
        metrics["top_decile_capture"] = float(lift.iloc[0]["capture_rate"])
        metrics["top3_decile_capture"] = float(lift.iloc[2]["cumulative_capture"])
        results["models"][name] = metrics
        results[f"decile_lift_{name}"] = lift.to_dict(orient="records")
        if metrics["pr_auc"] > best_pr_auc:
            best_name, best_pr_auc = name, metrics["pr_auc"]
            best_pipeline, best_proba = pipeline, proba

    results["best_model"] = best_name
    results["retention_impact_test_set"] = campaign_impact(y_test, best_proba)

    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    joblib.dump(
        {"pipeline": best_pipeline, "model_name": best_name}, MODEL_PATH
    )

    os.makedirs(os.path.dirname(METRICS_PATH), exist_ok=True)
    with open(METRICS_PATH, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    return results


if __name__ == "__main__":
    out = train_and_evaluate()
    print(json.dumps(out["dataset"], indent=2))
    for name, m in out["models"].items():
        print(
            f"{name}: ROC-AUC={m['roc_auc']:.4f} PR-AUC={m['pr_auc']:.4f} "
            f"P={m['precision']:.3f} R={m['recall']:.3f} "
            f"top-decile capture={m['top_decile_capture']:.1%} "
            f"top-3 capture={m['top3_decile_capture']:.1%}"
        )
    impact = out["retention_impact_test_set"]
    print("break-even threshold:", round(impact["assumptions"]["breakeven_threshold"], 4))
    for key in ("target_top_decile", "target_ev_optimal", "blanket"):
        s = impact[key]
        print(
            f"{key}: contacted={s['customers_contacted']} "
            f"saved≈{s['est_customers_saved']:.1f} "
            f"net≈${s['est_net_benefit']:,.0f}"
        )
