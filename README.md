# Customer Churn Prediction — with a Retention Strategy Built In

**A churn model alone doesn't make money. A retention campaign does.**

This project predicts which telecom customers are about to churn
(XGBoost vs a logistic-regression baseline) — and then goes the step most
portfolio projects skip: it turns the model's risk scores into a
**profit-based targeting decision**. Given explicit campaign assumptions, it
derives the expected-value-optimal risk threshold and estimates the impact of
targeting the riskiest customers versus blanket outreach.

Built by **Lakshman Sai Yaganti — Data Scientist @ Square**.

---

## Problem

Acquiring a new customer costs far more than keeping one, but retention
budgets are finite: a retention team can only contact a fraction of the base.
The business questions are therefore not just *"who will churn?"* but:

1. How concentrated is churn risk? (Can we reach most churners by contacting
   a small slice of the base?)
2. Above what risk level is a paid retention offer actually profitable?

## Dataset

**IBM Telco Customer Churn** — 7,043 customers, 21 columns (demographics,
tenure, services, contract, billing), binary `Churn` label, overall churn
rate ≈ 26.5%.

- Source: IBM sample data mirrored at
  `IBM/telco-customer-churn-on-icp4d` on GitHub (see `data/README.md`).
- The CSV is **not committed**; `src.data.load_data()` downloads it on first
  use. Cleaning handles the known quirks: `TotalCharges` stored as text with
  blanks for tenure-0 customers, `SeniorCitizen` as 0/1, plus tenure bins.

## Approach

- **Features** — one shared scikit-learn `ColumnTransformer` (median/scaled
  numerics, one-hot categoricals) used identically in training, the notebook,
  and the API, plus engineered features: number of add-on services,
  historical charges per tenure month, internet add-on flag.
- **Models** — Logistic Regression baseline (`class_weight="balanced"`) vs
  XGBoost (`scale_pos_weight` from the class ratio). Stratified 80/20 split,
  seed 42. Metrics below are on the held-out test set.
- **Decision layer** — decile lift analysis (churner capture by risk decile)
  and an expected-value model of the retention campaign
  (`src/retention.py`).

## Results (held-out test set, 1,409 customers, 374 churners)

| Model | ROC-AUC | PR-AUC | Precision @0.5 | Recall @0.5 |
|---|---|---|---|---|
| Logistic Regression (baseline) | 0.842 | 0.636 | 0.503 | 0.781 |
| **XGBoost** | **0.843** | **0.660** | **0.528** | **0.762** |

Ranking quality is what matters here: the top risk decile alone captures
**29.1%** of all churners (decile churn rate 77.3%, lift 2.9×), and the top
three deciles capture **65.2%** — the retention team reaches almost
two-thirds of all churners while contacting only 30% of the base.

## Retention strategy

Stated planning assumptions (explicit constants in `src/retention.py` —
**assumptions, not measured data**; change them and every conclusion
updates):

| Assumption | Value |
|---|---|
| Retention offer cost per contacted customer | $20 |
| Probability an offer saves a would-be churner | 30% |
| Average monthly customer value | $65 |
| Months of revenue retained if saved | 12 |
| → Customer lifetime value (CLV) | $780 |

Expected value of contacting a customer with churn probability *p*:
`p × 0.30 × $780 − $20`. Break-even at **p ≈ 8.5%** — contact everyone above
that, nobody below.

Estimated impact of the three strategies, evaluated on the 1,409-customer
test set (374 actual churners). Saved customers = actual churners reached ×
30% save probability; values are estimates under the assumptions above, not
guarantees:

| Strategy | Contacted | Churners reached | Est. saved | Campaign cost | Est. net benefit |
|---|---|---|---|---|---|
| Top decile only (riskiest 10%) | 141 | 109 | ≈ 33 | $2,820 | ≈ $22,686 |
| **EV-optimal (p ≥ 8.5%)** | **1,028** | **363** | **≈ 109** | **$20,560** | **≈ $64,382** |
| Blanket (contact everyone) | 1,409 | 374 | ≈ 112 | $28,180 | ≈ $59,336 |

Under these assumptions the EV-optimal targeting (everyone above the
break-even threshold) delivers the best estimated net benefit — blanket
outreach wastes most of its extra spend on customers who were never going
to leave, and top-decile-only targeting leaves profitable saves on the
table.

## Project structure

```
customer-churn-ml/
├── api/app.py               # FastAPI: GET /health, POST /score
├── data/README.md           # dataset provenance + download notes
├── notebooks/
│   └── 01_churn_modeling.ipynb   # executed EDA → model → economics walkthrough
├── reports/metrics.json     # metrics from the real training run
├── src/
│   ├── data.py              # download / load / clean
│   ├── features.py          # shared preprocessing pipeline + engineered features
│   ├── train.py             # train baseline + XGBoost, write metrics.json
│   ├── evaluate.py          # ROC/PR-AUC, threshold metrics, decile lift
│   ├── retention.py         # EV-based targeting + campaign impact estimates
│   └── predict.py           # single-customer score → risk band + action
└── tests/test_pipeline.py   # cleaning, lift math, EV math, API tests
```

## How to run

```bash
pip install -r requirements.txt

# Train (downloads the dataset on first run) and write reports/metrics.json
python -m src.train

# Run the tests
pytest

# Score customers with the API
uvicorn api.app:app --reload
curl -X POST localhost:8000/score -H 'Content-Type: application/json' -d '{
  "gender": "Female", "SeniorCitizen": 0, "Partner": "No", "Dependents": "No",
  "tenure": 2, "PhoneService": "Yes", "MultipleLines": "No",
  "InternetService": "Fiber optic", "OnlineSecurity": "No", "OnlineBackup": "No",
  "DeviceProtection": "No", "TechSupport": "No", "StreamingTV": "No",
  "StreamingMovies": "No", "Contract": "Month-to-month",
  "PaperlessBilling": "Yes", "PaymentMethod": "Electronic check",
  "MonthlyCharges": 85.0, "TotalCharges": 170.0}'
# → {"churn_probability": 0.8964, "risk_band": "high",
#    "breakeven_threshold": 0.0855,
#    "recommended_action": "Proactive outreach with retention offer",
#    "model": "xgboost"}
```

The API trains on first use if no model artifact exists
(`models/churn_model.joblib`, git-ignored), so a fresh clone works end to
end. `/score` returns a risk band derived from the retention break-even
threshold: **high** (≥ 2× break-even) → proactive offer, **medium** (≥
break-even) → light-touch offer, **low** → monitor only.

## Tech stack

Python · pandas · scikit-learn · XGBoost · SHAP · FastAPI · matplotlib /
seaborn · pytest

## License

MIT — see `LICENSE`. Copyright (c) 2026 Lakshman Sai Yaganti.
