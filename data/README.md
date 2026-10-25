# Dataset

**IBM Telco Customer Churn** — 7,043 customers of a fictional telecom
company, 21 columns (demographics, tenure, services, contract, billing) with
a binary `Churn` label. Overall churn rate ≈ 26.5%.

- Source: IBM sample data, mirrored in the `IBM/telco-customer-churn-on-icp4d`
  GitHub repository —
  https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv
- The CSV is **not committed** to this repository (see `.gitignore`).
  `src.data.load_data()` downloads it on first use and caches it here as
  `data/Telco-Customer-Churn.csv`.

Known quirks handled in `src/data.py`:

- `TotalCharges` is text, with blank values for the 11 customers whose
  tenure is 0 — coerced to numeric, blanks treated as 0.0.
- `SeniorCitizen` is 0/1 in the raw file — normalised to Yes/No categorical.
