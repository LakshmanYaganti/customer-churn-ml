"""Retention economics: from churn probabilities to a profit-based targeting
decision.

A churn model alone does not make money — the retention campaign does. This
module prices the campaign with EXPLICIT assumptions (named constants below;
change them and every conclusion updates). No number here is measured data:
they are planning assumptions, stated in the open, exactly as a data
scientist would present them to a business stakeholder.

Expected value logic
--------------------
For a customer with churn probability ``p``:

- If we do nothing, expected retained value = (1 - p) * CLV, where
  CLV = MONTHLY_VALUE * EXPECTED_REMAINING_MONTHS.
- If we make a retention offer costing OFFER_COST (incurred for everyone we
  contact), a would-be churner is saved with probability SAVE_PROBABILITY, so
  expected retained value = (1 - p * (1 - SAVE_PROBABILITY)) * CLV - OFFER_COST.

Contacting is profitable in expectation when::

    p * SAVE_PROBABILITY * CLV > OFFER_COST
    =>  p > OFFER_COST / (SAVE_PROBABILITY * CLV)   = break-even threshold
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Planning assumptions (NOT measured data — edit these and rerun).
# ---------------------------------------------------------------------------
OFFER_COST = 20.0  # $ cost of one retention offer (discount + outreach) per contacted customer
SAVE_PROBABILITY = 0.30  # probability an offer saves a customer who would have churned
MONTHLY_VALUE = 65.0  # $ average monthly revenue per customer (dataset mean MonthlyCharges ≈ $64.76)
EXPECTED_REMAINING_MONTHS = 12  # months of revenue retained if a churner is saved
CUSTOMER_LIFETIME_VALUE = MONTHLY_VALUE * EXPECTED_REMAINING_MONTHS  # $780


def breakeven_threshold(
    offer_cost: float = OFFER_COST,
    save_probability: float = SAVE_PROBABILITY,
    clv: float = CUSTOMER_LIFETIME_VALUE,
) -> float:
    """Minimum churn probability at which contacting a customer has positive
    expected value."""
    return offer_cost / (save_probability * clv)


def expected_profit_per_contact(
    churn_proba: float | np.ndarray,
    offer_cost: float = OFFER_COST,
    save_probability: float = SAVE_PROBABILITY,
    clv: float = CUSTOMER_LIFETIME_VALUE,
) -> float | np.ndarray:
    """Expected incremental profit of contacting one customer with the given
    churn probability, versus doing nothing."""
    return np.asarray(churn_proba) * save_probability * clv - offer_cost


def optimal_threshold(
    y_proba,
    offer_cost: float = OFFER_COST,
    save_probability: float = SAVE_PROBABILITY,
    clv: float = CUSTOMER_LIFETIME_VALUE,
) -> float:
    """EV-optimal risk threshold: the break-even probability above. Customers
    above it are profitable to contact in expectation; below it the offer
    costs more than the expected saved revenue."""
    return float(breakeven_threshold(offer_cost, save_probability, clv))


def campaign_impact(
    y_true,
    y_proba,
    n_customers_total: int | None = None,
    offer_cost: float = OFFER_COST,
    save_probability: float = SAVE_PROBABILITY,
    monthly_value: float = MONTHLY_VALUE,
    expected_remaining_months: int = EXPECTED_REMAINING_MONTHS,
) -> dict:
    """Estimated impact of three strategies, evaluated on a labelled set.

    ``estimated_net_benefit`` uses TRUE labels only to count how many actual
    churners were in the targeted group; the saved-customers estimate applies
    SAVE_PROBABILITY to those actual churners. This is an ex-post estimate of
    campaign value under the stated assumptions — not a guarantee.

    Strategies compared:
    - ``target_top_decile``: contact the riskiest 10%.
    - ``target_ev_optimal``: contact everyone above the break-even threshold.
    - ``blanket``: contact everybody (the naive alternative).
    """
    y_true = np.asarray(y_true)
    y_proba = np.asarray(y_proba)
    clv = monthly_value * expected_remaining_months
    threshold = breakeven_threshold(offer_cost, save_probability, clv)
    n_total = n_customers_total or len(y_true)

    def _evaluate(mask: np.ndarray, name: str) -> dict:
        contacted = int(mask.sum())
        churners_reached = int(y_true[mask].sum())
        est_saved = churners_reached * save_probability
        gross_saved_value = est_saved * clv
        cost = contacted * offer_cost
        return {
            "strategy": name,
            "customers_contacted": contacted,
            "pct_of_base_contacted": float(contacted / n_total * 100),
            "actual_churners_reached": churners_reached,
            "est_customers_saved": float(est_saved),
            "est_gross_saved_value": float(gross_saved_value),
            "campaign_cost": float(cost),
            "est_net_benefit": float(gross_saved_value - cost),
        }

    order = np.argsort(-y_proba)
    top_k = max(1, int(round(0.10 * len(y_proba))))
    top_mask = np.zeros(len(y_proba), dtype=bool)
    top_mask[order[:top_k]] = True

    return {
        "assumptions": {
            "offer_cost": offer_cost,
            "save_probability": save_probability,
            "monthly_value": monthly_value,
            "expected_remaining_months": expected_remaining_months,
            "customer_lifetime_value": clv,
            "breakeven_threshold": float(threshold),
        },
        "evaluated_on_customers": int(len(y_true)),
        "total_actual_churners": int(y_true.sum()),
        "target_top_decile": _evaluate(top_mask, "target_top_decile"),
        "target_ev_optimal": _evaluate(y_proba >= threshold, "target_ev_optimal"),
        "blanket": _evaluate(np.ones(len(y_proba), dtype=bool), "blanket"),
    }


def impact_table(impact: dict) -> pd.DataFrame:
    """Flatten ``campaign_impact`` output into a comparison DataFrame."""
    return pd.DataFrame(
        [impact["target_top_decile"], impact["target_ev_optimal"], impact["blanket"]]
    )
