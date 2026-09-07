"""
SHAP-based explainability for the Random Forest pipeline.
"""

from __future__ import annotations

from typing import Any, Tuple

import numpy as np
import pandas as pd
import shap

from .model import feature_names_from_pipeline


def _extract_class_shap_values(values: Any, class_index: int) -> np.ndarray:
    """Normalize SHAP output across common SHAP versions and multiclass formats."""
    if isinstance(values, list):
        # One array per class: (n_samples, n_features)
        return np.asarray(values[class_index])[0]

    arr = np.asarray(values)

    if arr.ndim == 2:
        return arr[0]

    if arr.ndim == 3:
        # Common multiclass format: (n_samples, n_features, n_classes)
        if arr.shape[2] > class_index:
            return arr[0, :, class_index]
        # Fallback for alternate ordering.
        if arr.shape[1] > class_index:
            return arr[0, class_index, :]

    raise ValueError(f"Unsupported SHAP output shape: {arr.shape}")


def explain_prediction(
    pipeline: Any,
    patient_df: pd.DataFrame,
    predicted_class_index: int,
) -> pd.DataFrame:
    """
    Explain one prediction using TreeExplainer on the transformed Random Forest input.
    """
    preprocessor = pipeline.named_steps["preprocessor"]
    classifier = pipeline.named_steps["classifier"]

    transformed = preprocessor.transform(patient_df)
    feature_names = feature_names_from_pipeline(pipeline)

    explainer = shap.TreeExplainer(classifier)
    shap_values = explainer.shap_values(transformed)

    values = _extract_class_shap_values(shap_values, predicted_class_index)

    result = pd.DataFrame(
        {
            "Feature": feature_names,
            "SHAP Value": values,
            "Absolute SHAP": np.abs(values),
        }
    ).sort_values("Absolute SHAP", ascending=False)

    return result.reset_index(drop=True)


def global_feature_importance(pipeline: Any) -> pd.DataFrame:
    """Return Random Forest global feature importance."""
    classifier = pipeline.named_steps["classifier"]
    names = feature_names_from_pipeline(pipeline)

    return (
        pd.DataFrame(
            {
                "Feature": names,
                "Importance": classifier.feature_importances_,
            }
        )
        .sort_values("Importance", ascending=False)
        .reset_index(drop=True)
    )


def human_feature_name(feature: str) -> str:
    """Make transformed feature names easier to read."""
    return (
        feature.replace("num__", "")
        .replace("cat__", "")
        .replace("_", " ")
        .strip()
    )


def top_explanation_text(explanation_df: pd.DataFrame, n: int = 5) -> list[str]:
    """Convert SHAP values into short, non-diagnostic explanation statements."""
    items = []
    for _, row in explanation_df.head(n).iterrows():
        direction = "increased" if row["SHAP Value"] > 0 else "decreased"
        items.append(
            f"{human_feature_name(row['Feature'])} {direction} the model score for the predicted class."
        )
    return items
