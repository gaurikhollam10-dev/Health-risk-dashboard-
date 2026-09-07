"""
Data validation, cleaning and feature preparation for the Health Risk Dashboard.
"""

from __future__ import annotations

import re
from typing import Dict, Iterable, Tuple

import numpy as np
import pandas as pd


# Canonical project columns and common dataset aliases.
COLUMN_ALIASES: Dict[str, Iterable[str]] = {
    "Patient ID": ["patient id", "patient_id", "patientid", "id", "patient number"],
    "Age": ["age", "patient age"],
    "Gender": ["gender", "sex"],
    "BMI": ["bmi", "body mass index"],
    "Blood Pressure": [
        "blood pressure",
        "blood_pressure",
        "bp",
        "systolic blood pressure",
        "systolic bp",
    ],
    "Glucose Level": [
        "glucose level",
        "glucose",
        "blood glucose",
        "blood sugar",
        "blood sugar level",
        "blood glucose level",
    ],
    "Cholesterol": ["cholesterol", "cholesterol level", "total cholesterol"],
    "Smoking Status": ["smoking status", "smoking", "smoker", "smoking_status"],
    "Physical Activity": [
        "physical activity",
        "activity",
        "activity level",
        "physical_activity",
    ],
    "Medical History": [
        "medical history",
        "medical_history",
        "medical condition",
        "medical conditions",
        "history",
    ],
    "Health Risk": [
        "health risk",
        "health_risk",
        "risk",
        "risk category",
        "risk_category",
        "health risk category",
        "outcome",
        "target",
    ],
}

NUMERIC_COLUMNS = [
    "Age",
    "BMI",
    "Blood Pressure",
    "Glucose Level",
    "Cholesterol",
]

CATEGORICAL_COLUMNS = [
    "Gender",
    "Smoking Status",
    "Physical Activity",
    "Medical History",
]

MODEL_FEATURES = NUMERIC_COLUMNS + CATEGORICAL_COLUMNS
REQUIRED_COLUMNS = MODEL_FEATURES + ["Health Risk"]


def normalize_name(name: str) -> str:
    """Normalize a column name for alias matching."""
    return re.sub(r"[^a-z0-9]+", " ", str(name).strip().lower()).strip()


def canonicalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Rename recognized aliases to the project canonical names."""
    rename_map = {}
    normalized_existing = {normalize_name(c): c for c in df.columns}

    for canonical, aliases in COLUMN_ALIASES.items():
        if canonical in df.columns:
            continue

        candidates = [canonical, *aliases]
        for candidate in candidates:
            actual = normalized_existing.get(normalize_name(candidate))
            if actual is not None:
                rename_map[actual] = canonical
                break

    return df.rename(columns=rename_map).copy()


def validate_columns(df: pd.DataFrame, require_target: bool = True) -> Tuple[bool, list[str]]:
    """Return whether required columns exist and a list of missing columns."""
    required = MODEL_FEATURES + (["Health Risk"] if require_target else [])
    missing = [col for col in required if col not in df.columns]
    return len(missing) == 0, missing


def _clean_text(series: pd.Series) -> pd.Series:
    return series.astype("string").str.strip().replace({"": pd.NA, "nan": pd.NA})


def clean_dataframe(df: pd.DataFrame, require_target: bool = True) -> pd.DataFrame:
    """
    Canonicalize, remove duplicates, clean numeric/text fields, and fill missing values.

    The function deliberately uses conservative validation rather than inventing values.
    """
    if df is None or df.empty:
        raise ValueError("The uploaded dataset is empty.")

    df = canonicalize_columns(df)
    valid, missing = validate_columns(df, require_target=require_target)
    if not valid:
        raise ValueError(
            "Missing required columns: " + ", ".join(missing)
        )

    df = df.copy()
    df = df.drop_duplicates().reset_index(drop=True)

    # Convert numeric fields. For values such as "120/80" or "105 mg/dL",
    # use the first numeric component so common public-dataset formats remain usable.
    for col in NUMERIC_COLUMNS:
        numeric = pd.to_numeric(df[col], errors="coerce")
        if numeric.notna().sum() < len(df):
            extracted = (
                df[col].astype("string")
                .str.extract(r"([-+]?\d+(?:\.\d+)?)", expand=False)
            )
            numeric = numeric.fillna(pd.to_numeric(extracted, errors="coerce"))
        df[col] = numeric

    # Remove impossible obvious values by converting them to missing.
    valid_ranges = {
        "Age": (0, 120),
        "BMI": (5, 80),
        "Blood Pressure": (50, 250),
        "Glucose Level": (30, 500),
        "Cholesterol": (50, 500),
    }
    for col, (low, high) in valid_ranges.items():
        mask = df[col].notna() & ~df[col].between(low, high)
        df.loc[mask, col] = np.nan

    # Numeric imputation with medians.
    for col in NUMERIC_COLUMNS:
        if df[col].notna().any():
            df[col] = df[col].fillna(df[col].median())
        else:
            df[col] = df[col].fillna(0)

    # Text cleanup and mode imputation.
    for col in CATEGORICAL_COLUMNS:
        df[col] = _clean_text(df[col])
        if df[col].notna().any():
            mode = df[col].mode(dropna=True)
            fill_value = str(mode.iloc[0]) if not mode.empty else "Unknown"
        else:
            fill_value = "Unknown"
        df[col] = df[col].fillna(fill_value)

    if "Patient ID" in df.columns:
        df["Patient ID"] = df["Patient ID"].astype("string").fillna(
            pd.Series([f"P{i+1:05d}" for i in range(len(df))], index=df.index)
        )

    if require_target:
        df["Health Risk"] = _clean_text(df["Health Risk"])
        df = df[df["Health Risk"].notna()].copy()
        if df["Health Risk"].nunique() < 2:
            raise ValueError("Health Risk must contain at least two classes.")

    return df.reset_index(drop=True)


def make_age_group(age: pd.Series) -> pd.Series:
    """Create the age groups used by the dashboard."""
    bins = [-np.inf, 29, 44, 59, np.inf]
    labels = ["Under 30", "30–44", "45–59", "60+"]
    return pd.cut(age, bins=bins, labels=labels, right=True)


def prepare_model_data(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    """Return model features X and target y."""
    cleaned = clean_dataframe(df, require_target=True)
    X = cleaned[MODEL_FEATURES].copy()
    y = cleaned["Health Risk"].astype(str)
    return X, y
