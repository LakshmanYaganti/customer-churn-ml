"""Churn scoring API.

Run locally::

    uvicorn api.app:app --reload

Endpoints:
- ``GET /health`` — service status.
- ``POST /score`` — one customer's raw attributes in, churn probability,
  risk band and recommended retention action out.

Model artifact: if ``models/churn_model.joblib`` does not exist (e.g. fresh
clone), the model is trained on first use from the dataset downloaded by
``src.data`` and then cached — the first /score call simply takes longer.
"""

from __future__ import annotations

import os
import sys
from typing import Literal

from fastapi import FastAPI
from pydantic import BaseModel, Field

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.predict import score_customer  # noqa: E402

app = FastAPI(title="Customer Churn Scoring API", version="1.0.0")


class Customer(BaseModel):
    gender: Literal["Female", "Male"]
    SeniorCitizen: int = Field(ge=0, le=1)
    Partner: Literal["Yes", "No"]
    Dependents: Literal["Yes", "No"]
    tenure: int = Field(ge=0)
    PhoneService: Literal["Yes", "No"]
    MultipleLines: Literal["Yes", "No", "No phone service"]
    InternetService: Literal["DSL", "Fiber optic", "No"]
    OnlineSecurity: Literal["Yes", "No", "No internet service"]
    OnlineBackup: Literal["Yes", "No", "No internet service"]
    DeviceProtection: Literal["Yes", "No", "No internet service"]
    TechSupport: Literal["Yes", "No", "No internet service"]
    StreamingTV: Literal["Yes", "No", "No internet service"]
    StreamingMovies: Literal["Yes", "No", "No internet service"]
    Contract: Literal["Month-to-month", "One year", "Two year"]
    PaperlessBilling: Literal["Yes", "No"]
    PaymentMethod: Literal[
        "Electronic check",
        "Mailed check",
        "Bank transfer (automatic)",
        "Credit card (automatic)",
    ]
    MonthlyCharges: float = Field(ge=0)
    TotalCharges: float = Field(ge=0)


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/score")
def score(customer: Customer) -> dict:
    return score_customer(customer.model_dump())
