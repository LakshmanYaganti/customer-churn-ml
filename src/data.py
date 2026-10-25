"""Dataset loading and cleaning for the IBM Telco Customer Churn dataset.

The dataset is NOT committed to this repository. ``load_data`` downloads it
from the public IBM mirror on first use and caches it under ``data/`` (which is
git-ignored). If the file already exists locally it is used as-is.

Cleaning steps (well-known quirks of this dataset):
- ``TotalCharges`` arrives as text with blank strings for 11 brand-new
  customers (tenure == 0); it is coerced to numeric and the blanks become 0.0
  (these customers have genuinely paid nothing yet).
- The target ``Churn`` (Yes/No) is mapped to 1/0.
- ``tenure`` is binned into business-meaningful groups.
"""

from __future__ import annotations

import os
import urllib.request

import pandas as pd

DATA_URL = (
    "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/"
    "master/data/Telco-Customer-Churn.csv"
)
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
DATA_PATH = os.path.join(DATA_DIR, "Telco-Customer-Churn.csv")

TARGET = "Churn"

TENURE_BINS = [0, 12, 24, 48, 72, 10_000]
TENURE_LABELS = ["0-12m", "13-24m", "25-48m", "49-72m", "72m+"]


def download_data(url: str = DATA_URL, dest: str = DATA_PATH) -> str:
    """Download the Telco churn CSV to ``dest`` (skips if already present)."""
    if not os.path.exists(dest):
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        urllib.request.urlretrieve(url, dest)
    return dest


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Return a cleaned copy of the raw Telco frame."""
    df = df.copy()

    # TotalCharges is stored as text in the raw file; blanks -> NaN -> 0.0
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df["TotalCharges"] = df["TotalCharges"].fillna(0.0)

    # Target to binary (raw values are "Yes"/"No" strings; the dtype check
    # must not rely on `object`, newer pandas uses Arrow string dtype)
    if not pd.api.types.is_integer_dtype(df[TARGET]):
        df[TARGET] = df[TARGET].map({"Yes": 1, "No": 0}).astype(int)

    # SeniorCitizen is already 0/1 ints in the raw file; normalise to Yes/No
    # labels so it is handled as a categorical feature downstream.
    df["SeniorCitizen"] = df["SeniorCitizen"].map({0: "No", 1: "Yes"})

    # Tenure groups used by the business team
    df["tenure_group"] = pd.cut(
        df["tenure"], bins=TENURE_BINS, labels=TENURE_LABELS, include_lowest=True
    ).astype(str)

    return df


def load_data(path: str | None = None) -> pd.DataFrame:
    """Load (downloading if needed) and clean the Telco churn dataset."""
    path = path or download_data()
    raw = pd.read_csv(path)
    return clean_data(raw)


if __name__ == "__main__":
    frame = load_data()
    print(frame.shape)
    print(frame[TARGET].value_counts(normalize=True).rename("churn_rate"))
