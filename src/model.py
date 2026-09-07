"""
Random Forest training, prediction and evaluation utilities.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

from .preprocessing import CATEGORICAL_COLUMNS, MODEL_FEATURES, NUMERIC_COLUMNS


DEFAULT_MODEL_PATH = Path("models/random_forest.pkl")
DEFAULT_ENCODER_PATH = Path("models/label_encoder.pkl")


def build_pipeline(random_state: int = 42) -> Any:
    """Build the preprocessing + Random Forest pipeline."""
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", "passthrough", NUMERIC_COLUMNS),
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                CATEGORICAL_COLUMNS,
            ),
        ],
        remainder="drop",
    )

    classifier = RandomForestClassifier(
        n_estimators=250,
        random_state=random_state,
        class_weight="balanced",
        min_samples_leaf=2,
        n_jobs=-1,
    )

    from sklearn.pipeline import Pipeline

    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", classifier),
        ]
    )


def train_random_forest(
    X: pd.DataFrame,
    y: pd.Series,
    random_state: int = 42,
) -> Tuple[Any, Any, Dict[str, Any], pd.DataFrame, pd.DataFrame]:
    """
    Train Random Forest and return pipeline, LabelEncoder, metrics, predictions and test data.
    """
    from sklearn.preprocessing import LabelEncoder

    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y.astype(str))

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y_encoded,
        test_size=0.20,
        random_state=random_state,
        stratify=y_encoded,
    )

    pipeline = build_pipeline(random_state=random_state)
    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    labels = np.arange(len(label_encoder.classes_))

    metrics = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision_weighted": float(
            precision_score(y_test, y_pred, average="weighted", zero_division=0)
        ),
        "recall_weighted": float(
            recall_score(y_test, y_pred, average="weighted", zero_division=0)
        ),
        "f1_weighted": float(
            f1_score(y_test, y_pred, average="weighted", zero_division=0)
        ),
        "classification_report": classification_report(
            y_test,
            y_pred,
            labels=labels,
            target_names=label_encoder.classes_,
            output_dict=True,
            zero_division=0,
        ),
        "classes": list(label_encoder.classes_),
        "test_size": int(len(y_test)),
        "train_size": int(len(y_train)),
    }

    cm = confusion_matrix(y_test, y_pred, labels=labels)
    cm_df = pd.DataFrame(
        cm,
        index=label_encoder.classes_,
        columns=label_encoder.classes_,
    )
    cm_df.index.name = "Actual"
    cm_df.columns.name = "Predicted"

    pred_df = X_test.copy()
    pred_df["Actual"] = label_encoder.inverse_transform(y_test)
    pred_df["Predicted"] = label_encoder.inverse_transform(y_pred)

    return pipeline, label_encoder, metrics, cm_df, pred_df


def save_artifacts(
    pipeline: Any,
    label_encoder: Any,
    model_path: Path = DEFAULT_MODEL_PATH,
    encoder_path: Path = DEFAULT_ENCODER_PATH,
) -> None:
    """Save model and label encoder."""
    model_path.parent.mkdir(parents=True, exist_ok=True)
    encoder_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, model_path)
    joblib.dump(label_encoder, encoder_path)


def load_artifacts(
    model_path: Path = DEFAULT_MODEL_PATH,
    encoder_path: Path = DEFAULT_ENCODER_PATH,
) -> Tuple[Any, Any]:
    """Load saved model artifacts."""
    if not model_path.exists() or not encoder_path.exists():
        raise FileNotFoundError(
            "Trained model artifacts were not found. Run notebook 03_model_training.ipynb first."
        )
    return joblib.load(model_path), joblib.load(encoder_path)


def predict_risk(
    pipeline: Any,
    label_encoder: Any,
    patient_df: pd.DataFrame,
) -> Tuple[str, np.ndarray]:
    """Predict risk label and class probabilities."""
    encoded_pred = pipeline.predict(patient_df[MODEL_FEATURES])[0]
    probabilities = pipeline.predict_proba(patient_df[MODEL_FEATURES])[0]
    label = label_encoder.inverse_transform([int(encoded_pred)])[0]
    return str(label), probabilities


def feature_names_from_pipeline(pipeline: Any) -> list[str]:
    """Return readable transformed feature names."""
    preprocessor = pipeline.named_steps["preprocessor"]
    return list(preprocessor.get_feature_names_out())
