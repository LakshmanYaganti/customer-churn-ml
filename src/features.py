"""Feature engineering and preprocessing for churn modeling.

A single scikit-learn ``ColumnTransformer`` is used everywhere (training,
evaluation, API) so that preprocessing can never drift between the notebook,
the batch scorer and the live API:

- numeric features -> median imputation + standard scaling
- categorical features -> most-frequent imputation + one-hot encoding
"""

from __future__ import annotations

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .data import TARGET

ID_COLUMN = "customerID"

NUMERIC_FEATURES = ["tenure", "MonthlyCharges", "TotalCharges"]

CATEGORICAL_FEATURES = [
    "gender",
    "SeniorCitizen",
    "Partner",
    "Dependents",
    "PhoneService",
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
    "PaperlessBilling",
    "PaymentMethod",
    "tenure_group",
]

FEATURE_COLUMNS = NUMERIC_FEATURES + CATEGORICAL_FEATURES


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add simple, business-interpretable engineered features.

    - ``charges_per_tenure_month``: average revenue per month of relationship
      (TotalCharges / tenure), a proxy for historical customer value.
    - ``num_services``: how many add-on services the customer subscribes to —
      more services usually means more stickiness.
    - ``has_internet_addons``: whether any internet add-on is active.
    """
    df = df.copy()
    service_cols = [
        "OnlineSecurity",
        "OnlineBackup",
        "DeviceProtection",
        "TechSupport",
        "StreamingTV",
        "StreamingMovies",
    ]
    df["num_services"] = (df[service_cols] == "Yes").sum(axis=1)
    df["has_internet_addons"] = (df[["OnlineSecurity", "OnlineBackup", "TechSupport"]] == "Yes").any(
        axis=1
    )
    df["has_internet_addons"] = df["has_internet_addons"].map({True: "Yes", False: "No"})
    df["charges_per_tenure_month"] = df["TotalCharges"] / df["tenure"].clip(lower=1)
    return df


# Engineered features join the feature lists.
NUMERIC_FEATURES_EXT = NUMERIC_FEATURES + ["num_services", "charges_per_tenure_month"]
CATEGORICAL_FEATURES_EXT = CATEGORICAL_FEATURES + ["has_internet_addons"]
FEATURE_COLUMNS_EXT = NUMERIC_FEATURES_EXT + CATEGORICAL_FEATURES_EXT


def build_preprocessor() -> ColumnTransformer:
    numeric_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipe, NUMERIC_FEATURES_EXT),
            ("cat", categorical_pipe, CATEGORICAL_FEATURES_EXT),
        ]
    )


def split_xy(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Add engineered features and split into (X, y)."""
    df = add_features(df)
    X = df[FEATURE_COLUMNS_EXT].copy()
    y = df[TARGET].astype(int)
    return X, y
