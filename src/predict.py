"""Single-customer scoring: churn probability, risk band, and action.

Risk bands are operational, tied to the retention economics in
``src/retention.py``:

- ``high``   — at or above 2x the break-even threshold: contact proactively
  with the retention offer.
- ``medium`` — above the break-even threshold: contact (positive expected
  value) with a lighter-touch offer.
- ``low``    — below break-even: a paid offer loses money in expectation;
  monitor only.
"""

from __future__ import annotations

import os

import joblib
import pandas as pd

from .features import add_features, FEATURE_COLUMNS_EXT
from .retention import breakeven_threshold

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(ROOT, "models", "churn_model.joblib")

_model_cache: dict | None = None


def load_model() -> dict:
    """Load the persisted model bundle; train and persist on first use."""
    global _model_cache
    if _model_cache is None:
        if os.path.exists(MODEL_PATH):
            _model_cache = joblib.load(MODEL_PATH)
        else:
            from .train import train_and_evaluate

            train_and_evaluate()
            _model_cache = joblib.load(MODEL_PATH)
    return _model_cache


def risk_band(probability: float, threshold: float | None = None) -> str:
    threshold = threshold if threshold is not None else breakeven_threshold()
    if probability >= 2 * threshold:
        return "high"
    if probability >= threshold:
        return "medium"
    return "low"


_ACTIONS = {
    "high": "Proactive outreach with retention offer",
    "medium": "Light-touch retention offer (email/app)",
    "low": "No paid offer — monitor only",
}


def score_customer(customer: dict) -> dict:
    """Score one customer dict of raw Telco-style fields.

    Returns churn probability, risk band, the break-even threshold used, and
    the recommended retention action."""
    bundle = load_model()
    pipeline = bundle["pipeline"]

    df = pd.DataFrame([customer])
    # Derive tenure_group exactly as training did.
    from .data import TENURE_BINS, TENURE_LABELS

    df["tenure_group"] = pd.cut(
        df["tenure"], bins=TENURE_BINS, labels=TENURE_LABELS, include_lowest=True
    ).astype(str)
    df["SeniorCitizen"] = df["SeniorCitizen"].map({0: "No", 1: "Yes", "No": "No", "Yes": "Yes"})
    df = add_features(df)
    X = df[FEATURE_COLUMNS_EXT]

    proba = float(pipeline.predict_proba(X)[0, 1])
    threshold = breakeven_threshold()
    band = risk_band(proba, threshold)
    return {
        "churn_probability": round(proba, 4),
        "risk_band": band,
        "breakeven_threshold": round(threshold, 4),
        "recommended_action": _ACTIONS[band],
        "model": bundle.get("model_name", "unknown"),
    }
