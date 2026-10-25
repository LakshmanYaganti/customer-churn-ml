"""Model evaluation: ranking quality, threshold metrics, and decile lift.

For churn, ranking quality matters more than raw accuracy: the retention team
can only contact a limited number of customers, so what counts is how many
churners are concentrated in the top risk deciles (the "lift" table).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def classification_metrics(y_true, y_proba, threshold: float = 0.5) -> dict:
    y_pred = (np.asarray(y_proba) >= threshold).astype(int)
    return {
        "threshold": threshold,
        "roc_auc": float(roc_auc_score(y_true, y_proba)),
        "pr_auc": float(average_precision_score(y_true, y_proba)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
    }


def decile_lift_table(y_true, y_proba, n_bins: int = 10) -> pd.DataFrame:
    """Decile lift table: customers sorted by predicted risk, split into
    equal deciles. ``capture_rate`` is the share of ALL churners that fall in
    that decile; ``cumulative_capture`` is the running total — the number the
    retention team cares about."""
    df = pd.DataFrame({"y": np.asarray(y_true), "p": np.asarray(y_proba)})
    df = df.sort_values("p", ascending=False).reset_index(drop=True)
    df["decile"] = pd.qcut(df.index, n_bins, labels=False) + 1

    total_churners = df["y"].sum()
    total_customers = len(df)
    rows = []
    cum_churners = 0
    cum_customers = 0
    for decile, group in df.groupby("decile"):
        cum_churners += int(group["y"].sum())
        cum_customers += len(group)
        rows.append(
            {
                "decile": int(decile),
                "customers": int(len(group)),
                "churners": int(group["y"].sum()),
                "churn_rate": float(group["y"].mean()),
                "avg_predicted_risk": float(group["p"].mean()),
                "capture_rate": float(group["y"].sum() / total_churners),
                "cumulative_capture": float(cum_churners / total_churners),
                "cumulative_customers_pct": float(cum_customers / total_customers),
                "lift": float(group["y"].mean() / df["y"].mean()),
            }
        )
    return pd.DataFrame(rows)
