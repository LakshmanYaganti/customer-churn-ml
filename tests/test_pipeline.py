"""Tests for data cleaning, feature engineering, evaluation math, the
retention-economics layer, and the scoring API."""

import numpy as np
import pandas as pd
import pytest

from src.data import clean_data
from src.evaluate import decile_lift_table
from src.features import add_features
from src.retention import (
    breakeven_threshold,
    campaign_impact,
    expected_profit_per_contact,
)


def _tiny_raw() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "customerID": ["A", "B", "C"],
            "gender": ["Female", "Male", "Female"],
            "SeniorCitizen": [0, 1, 0],
            "Partner": ["Yes", "No", "No"],
            "Dependents": ["No", "No", "Yes"],
            "tenure": [0, 24, 60],
            "PhoneService": ["No", "Yes", "Yes"],
            "MultipleLines": ["No phone service", "No", "Yes"],
            "InternetService": ["DSL", "Fiber optic", "DSL"],
            "OnlineSecurity": ["No", "No", "Yes"],
            "OnlineBackup": ["Yes", "No", "No"],
            "DeviceProtection": ["No", "No", "Yes"],
            "TechSupport": ["No", "No", "Yes"],
            "StreamingTV": ["No", "Yes", "No"],
            "StreamingMovies": ["No", "Yes", "No"],
            "Contract": ["Month-to-month", "Month-to-month", "Two year"],
            "PaperlessBilling": ["Yes", "Yes", "No"],
            "PaymentMethod": ["Electronic check", "Electronic check", "Mailed check"],
            "MonthlyCharges": [29.85, 89.10, 54.20],
            "TotalCharges": [" ", "2138.40", "3252.00"],  # blank for tenure=0
            "Churn": ["No", "Yes", "No"],
        }
    )


def test_cleaning_total_charges_blank_becomes_zero():
    cleaned = clean_data(_tiny_raw())
    assert cleaned.loc[0, "TotalCharges"] == 0.0
    assert cleaned.loc[1, "TotalCharges"] == pytest.approx(2138.40)
    assert cleaned["TotalCharges"].dtype.kind == "f"


def test_cleaning_target_and_tenure_group():
    cleaned = clean_data(_tiny_raw())
    assert cleaned["Churn"].tolist() == [0, 1, 0]
    assert cleaned.loc[0, "tenure_group"] == "0-12m"
    assert cleaned.loc[1, "tenure_group"] == "13-24m"
    assert cleaned.loc[2, "tenure_group"] == "49-72m"
    assert cleaned["SeniorCitizen"].tolist() == ["No", "Yes", "No"]


def test_add_features():
    feats = add_features(clean_data(_tiny_raw()))
    # Customer C has OnlineSecurity + DeviceProtection + TechSupport = 3
    assert feats.loc[2, "num_services"] == 3
    assert feats.loc[2, "has_internet_addons"] == "Yes"
    # tenure=0 is clipped to 1 to avoid division by zero
    assert feats.loc[0, "charges_per_tenure_month"] == 0.0


def test_decile_lift_perfect_ranking():
    # 100 customers, 10 churners, perfectly ranked -> decile 1 captures all.
    y = np.array([1] * 10 + [0] * 90)
    p = np.linspace(0.99, 0.01, 100)
    table = decile_lift_table(y, p)
    assert table.iloc[0]["capture_rate"] == pytest.approx(1.0)
    assert table.iloc[0]["cumulative_capture"] == pytest.approx(1.0)
    assert table["customers"].sum() == 100


def test_decile_lift_random_ranking_captures_about_ten_percent():
    rng = np.random.default_rng(0)
    y = np.array([1] * 100 + [0] * 900)
    p = rng.random(1000)
    table = decile_lift_table(y, p)
    assert 0.03 < table.iloc[0]["capture_rate"] < 0.20


def test_breakeven_threshold_math():
    # 20 / (0.30 * 780) = 0.08547...
    assert breakeven_threshold(20.0, 0.30, 780.0) == pytest.approx(0.08547, abs=1e-4)


def test_expected_profit_per_contact():
    assert expected_profit_per_contact(0.5, 20.0, 0.30, 780.0) == pytest.approx(97.0)
    assert expected_profit_per_contact(0.0, 20.0, 0.30, 780.0) == pytest.approx(-20.0)


def test_campaign_impact_blanket_costs_most():
    rng = np.random.default_rng(1)
    y = (rng.random(500) < 0.25).astype(int)
    p = np.clip(y * 0.4 + rng.random(500) * 0.5, 0, 1)
    impact = campaign_impact(y, p)
    assert impact["blanket"]["customers_contacted"] == 500
    assert impact["blanket"]["campaign_cost"] == pytest.approx(500 * 20.0)
    assert impact["target_top_decile"]["customers_contacted"] == 50
    assert impact["assumptions"]["customer_lifetime_value"] == pytest.approx(780.0)


SAMPLE_CUSTOMER = {
    "gender": "Female",
    "SeniorCitizen": 0,
    "Partner": "No",
    "Dependents": "No",
    "tenure": 2,
    "PhoneService": "Yes",
    "MultipleLines": "No",
    "InternetService": "Fiber optic",
    "OnlineSecurity": "No",
    "OnlineBackup": "No",
    "DeviceProtection": "No",
    "TechSupport": "No",
    "StreamingTV": "No",
    "StreamingMovies": "No",
    "Contract": "Month-to-month",
    "PaperlessBilling": "Yes",
    "PaymentMethod": "Electronic check",
    "MonthlyCharges": 85.0,
    "TotalCharges": 170.0,
}


def test_api_health_and_score():
    from fastapi.testclient import TestClient

    from api.app import app

    client = TestClient(app)
    assert client.get("/health").json() == {"status": "ok"}

    resp = client.post("/score", json=SAMPLE_CUSTOMER)
    assert resp.status_code == 200
    body = resp.json()
    assert 0.0 <= body["churn_probability"] <= 1.0
    assert body["risk_band"] in {"low", "medium", "high"}
    assert body["recommended_action"]
    # A month-to-month fiber customer with no add-ons at tenure 2 is a
    # classic high-risk profile; probability should not be near zero.
    assert body["churn_probability"] > 0.2
