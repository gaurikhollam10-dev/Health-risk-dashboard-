"""
Dashboard analytics and insight helpers.
"""

from __future__ import annotations

import pandas as pd
from .preprocessing import make_age_group


def add_dashboard_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Add derived columns used by dashboard charts and filters."""
    result = df.copy()
    if "Age" in result.columns:
        result["Age Group"] = make_age_group(result["Age"])
    return result


def kpi_summary(df: pd.DataFrame) -> dict:
    """Calculate core dashboard KPIs."""
    return {
        "total_patients": int(len(df)),
        "average_age": float(df["Age"].mean()),
        "average_bmi": float(df["BMI"].mean()),
        "average_bp": float(df["Blood Pressure"].mean()),
        "average_glucose": float(df["Glucose Level"].mean()),
        "average_cholesterol": float(df["Cholesterol"].mean()),
    }


def risk_counts(df: pd.DataFrame) -> pd.DataFrame:
    """Return risk counts."""
    return (
        df["Health Risk"]
        .value_counts()
        .rename_axis("Health Risk")
        .reset_index(name="Patients")
    )


def top_risk_factors(df: pd.DataFrame) -> pd.DataFrame:
    """
    Provide simple group-level association summaries.

    These are descriptive relationships, not causal medical conclusions.
    """
    rows = []
    for feature in ["Age", "BMI", "Blood Pressure", "Glucose Level", "Cholesterol"]:
        if feature in df.columns:
            group_means = df.groupby("Health Risk", observed=True)[feature].mean()
            if len(group_means) > 1:
                rows.append(
                    {
                        "Factor": feature,
                        "Range across risk groups": float(group_means.max() - group_means.min()),
                    }
                )
    return pd.DataFrame(rows).sort_values(
        "Range across risk groups", ascending=False
    )


def generate_insights(df: pd.DataFrame) -> list[str]:
    """Generate simple automatically computed dashboard insights."""
    insights = []
    if df.empty:
        return insights

    risk = df["Health Risk"].value_counts()
    if not risk.empty:
        top = risk.index[0]
        pct = risk.iloc[0] / len(df) * 100
        insights.append(f"{top} is the largest risk category in the current filtered data ({pct:.1f}% of records).")

    if "Age Group" in df.columns:
        age_risk = pd.crosstab(df["Age Group"], df["Health Risk"], normalize="index") * 100
        if not age_risk.empty:
            high_col = next((c for c in age_risk.columns if str(c).lower() == "high"), None)
            if high_col is not None:
                group = age_risk[high_col].idxmax()
                insights.append(f"The {group} age group has the highest share of the High category among displayed groups.")

    if "Smoking Status" in df.columns:
        smoking = df.groupby("Smoking Status", observed=True)["Health Risk"].apply(
            lambda s: (s.astype(str).str.lower() == "high").mean() * 100
        )
        if len(smoking) > 1:
            group = smoking.idxmax()
            insights.append(f"Among smoking-status groups, {group} has the highest observed share of High-risk labels.")

    return insights[:5]
