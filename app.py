from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

from src.analytics import add_dashboard_columns, generate_insights, kpi_summary, risk_counts
from src.explainability import explain_prediction, global_feature_importance, human_feature_name
from src.model import load_artifacts, predict_risk
from src.preprocessing import (
    CATEGORICAL_COLUMNS,
    MODEL_FEATURES,
    NUMERIC_COLUMNS,
    canonicalize_columns,
    clean_dataframe,
    validate_columns,
)

BASE_DIR = Path(__file__).resolve().parent
DEFAULT_DATA = BASE_DIR / "data" / "healthcare_risk.csv"
MODEL_PATH = BASE_DIR / "models" / "random_forest.pkl"
ENCODER_PATH = BASE_DIR / "models" / "label_encoder.pkl"
METRICS_PATH = BASE_DIR / "outputs" / "model_results" / "metrics.json"
CM_PATH = BASE_DIR / "outputs" / "model_results" / "confusion_matrix.csv"

st.set_page_config(
    page_title="Health Risk Dashboard",
    page_icon="🩺",
    layout="wide",
)

st.markdown(
    """
    <style>
    .main-title {font-size: 2.2rem; font-weight: 700; margin-bottom: 0.2rem;}
    .subtitle {color: #666; margin-bottom: 1.2rem;}
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data
def load_default_data() -> pd.DataFrame:
    return pd.read_csv(DEFAULT_DATA)


@st.cache_data
def clean_for_dashboard(raw: pd.DataFrame) -> pd.DataFrame:
    return add_dashboard_columns(clean_dataframe(raw, require_target=True))


@st.cache_resource
def load_model():
    return load_artifacts(MODEL_PATH, ENCODER_PATH)


def show_kpis(df: pd.DataFrame) -> None:
    k = kpi_summary(df)
    cols = st.columns(6)
    cols[0].metric("Total Patients", f"{k['total_patients']:,}")
    cols[1].metric("Avg. Age", f"{k['average_age']:.1f}")
    cols[2].metric("Avg. BMI", f"{k['average_bmi']:.1f}")
    cols[3].metric("Avg. BP", f"{k['average_bp']:.1f}")
    cols[4].metric("Avg. Glucose", f"{k['average_glucose']:.1f}")
    cols[5].metric("Avg. Cholesterol", f"{k['average_cholesterol']:.1f}")


def chart_risk_distribution(df: pd.DataFrame):
    counts = risk_counts(df)
    return px.bar(
        counts,
        x="Health Risk",
        y="Patients",
        color="Health Risk",
        title="Risk Category Distribution",
        text_auto=True,
    )


def apply_filters(df: pd.DataFrame) -> pd.DataFrame:
    st.sidebar.header("Filters")

    result = df.copy()

    age_groups = [str(x) for x in df["Age Group"].dropna().unique()]
    selected_age = st.sidebar.multiselect("Age Group", age_groups, default=age_groups)
    if selected_age:
        result = result[result["Age Group"].astype(str).isin(selected_age)]

    if "Gender" in df.columns:
        values = sorted(df["Gender"].astype(str).unique())
        selected = st.sidebar.multiselect("Gender", values, default=values)
        if selected:
            result = result[result["Gender"].astype(str).isin(selected)]

    risks = sorted(df["Health Risk"].astype(str).unique())
    selected_risk = st.sidebar.multiselect("Risk Category", risks, default=risks)
    if selected_risk:
        result = result[result["Health Risk"].astype(str).isin(selected_risk)]

    for col in ["Smoking Status", "Physical Activity", "Medical History"]:
        if col in df.columns:
            values = sorted(df[col].astype(str).unique())
            if len(values) <= 20:
                selected = st.sidebar.multiselect(col, values, default=values)
                if selected:
                    result = result[result[col].astype(str).isin(selected)]

    st.sidebar.caption(f"{len(result):,} records after filtering.")
    return result


def overview_page(df: pd.DataFrame) -> None:
    st.markdown('<div class="main-title">Health Risk Dashboard</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="subtitle">Interactive healthcare analytics, risk patterns, prediction and explainability</div>',
        unsafe_allow_html=True,
    )

    show_kpis(df)
    st.divider()

    c1, c2 = st.columns(2)
    with c1:
        st.plotly_chart(chart_risk_distribution(df), use_container_width=True)
    with c2:
        age_counts = df["Age Group"].value_counts().sort_index().reset_index()
        age_counts.columns = ["Age Group", "Patients"]
        st.plotly_chart(
            px.bar(age_counts, x="Age Group", y="Patients", title="Patient Distribution by Age Group", text_auto=True),
            use_container_width=True,
        )

    c3, c4 = st.columns(2)
    with c3:
        gender = df["Gender"].value_counts().reset_index()
        gender.columns = ["Gender", "Patients"]
        st.plotly_chart(
            px.pie(gender, names="Gender", values="Patients", title="Gender Distribution"),
            use_container_width=True,
        )
    with c4:
        st.plotly_chart(
            px.box(df, x="Health Risk", y="BMI", points="outliers", title="BMI by Risk Category"),
            use_container_width=True,
        )

    st.subheader("Automatically Generated Insights")
    insights = generate_insights(df)
    if insights:
        for item in insights:
            st.info(item)
    else:
        st.write("No insights available for the current selection.")

    st.caption("These are descriptive model/data insights, not medical conclusions.")


def risk_analysis_page(df: pd.DataFrame) -> None:
    st.title("Risk Analysis")
    st.write("Explore relationships between risk labels and demographic, clinical and lifestyle variables.")

    tabs = st.tabs(
        [
            "BMI",
            "Blood Pressure",
            "Glucose",
            "Cholesterol",
            "Lifestyle",
            "Age & Gender",
        ]
    )

    with tabs[0]:
        st.plotly_chart(
            px.box(df, x="Health Risk", y="BMI", color="Health Risk", title="BMI vs Health Risk"),
            use_container_width=True,
        )
    with tabs[1]:
        st.plotly_chart(
            px.box(
                df,
                x="Health Risk",
                y="Blood Pressure",
                color="Health Risk",
                title="Blood Pressure vs Health Risk",
            ),
            use_container_width=True,
        )
    with tabs[2]:
        st.plotly_chart(
            px.box(
                df,
                x="Health Risk",
                y="Glucose Level",
                color="Health Risk",
                title="Glucose Level vs Health Risk",
            ),
            use_container_width=True,
        )
    with tabs[3]:
        st.plotly_chart(
            px.box(
                df,
                x="Health Risk",
                y="Cholesterol",
                color="Health Risk",
                title="Cholesterol vs Health Risk",
            ),
            use_container_width=True,
        )
    with tabs[4]:
        c1, c2 = st.columns(2)
        smoking = pd.crosstab(df["Smoking Status"], df["Health Risk"], normalize="index") * 100
        activity = pd.crosstab(df["Physical Activity"], df["Health Risk"], normalize="index") * 100
        with c1:
            st.plotly_chart(
                px.bar(
                    smoking.reset_index().melt(id_vars="Smoking Status"),
                    x="Smoking Status",
                    y="value",
                    color="Health Risk",
                    title="Smoking Status vs Risk (%)",
                    labels={"value": "Percentage"},
                ),
                use_container_width=True,
            )
        with c2:
            st.plotly_chart(
                px.bar(
                    activity.reset_index().melt(id_vars="Physical Activity"),
                    x="Physical Activity",
                    y="value",
                    color="Health Risk",
                    title="Physical Activity vs Risk (%)",
                    labels={"value": "Percentage"},
                ),
                use_container_width=True,
            )
    with tabs[5]:
        age_risk = pd.crosstab(df["Age Group"], df["Health Risk"], normalize="index") * 100
        gender_risk = pd.crosstab(df["Gender"], df["Health Risk"], normalize="index") * 100
        c1, c2 = st.columns(2)
        with c1:
            st.plotly_chart(
                px.bar(
                    age_risk.reset_index().melt(id_vars="Age Group"),
                    x="Age Group",
                    y="value",
                    color="Health Risk",
                    title="Risk by Age Group (%)",
                    labels={"value": "Percentage"},
                    barmode="stack",
                ),
                use_container_width=True,
            )
        with c2:
            st.plotly_chart(
                px.bar(
                    gender_risk.reset_index().melt(id_vars="Gender"),
                    x="Gender",
                    y="value",
                    color="Health Risk",
                    title="Risk by Gender (%)",
                    labels={"value": "Percentage"},
                    barmode="stack",
                ),
                use_container_width=True,
            )


def prediction_page(df: pd.DataFrame) -> None:
    st.title("Risk Prediction")
    st.write("Enter patient attributes to obtain the model's predicted risk category and SHAP explanation.")

    try:
        pipeline, label_encoder = load_model()
    except Exception as exc:
        st.error(str(exc))
        return

    with st.form("prediction_form"):
        c1, c2, c3 = st.columns(3)
        with c1:
            age = st.number_input("Age", min_value=0, max_value=120, value=int(df["Age"].median()))
            bmi = st.number_input("BMI", min_value=5.0, max_value=80.0, value=float(df["BMI"].median()))
            bp = st.number_input(
                "Blood Pressure",
                min_value=50.0,
                max_value=250.0,
                value=float(df["Blood Pressure"].median()),
            )
        with c2:
            glucose = st.number_input(
                "Glucose Level",
                min_value=30.0,
                max_value=500.0,
                value=float(df["Glucose Level"].median()),
            )
            cholesterol = st.number_input(
                "Cholesterol",
                min_value=50.0,
                max_value=500.0,
                value=float(df["Cholesterol"].median()),
            )
            gender = st.selectbox("Gender", sorted(df["Gender"].astype(str).unique()))
        with c3:
            smoking = st.selectbox("Smoking Status", sorted(df["Smoking Status"].astype(str).unique()))
            activity = st.selectbox(
                "Physical Activity",
                sorted(df["Physical Activity"].astype(str).unique()),
            )
            history = st.selectbox(
                "Medical History",
                sorted(df["Medical History"].astype(str).unique()),
            )

        submitted = st.form_submit_button("Predict Risk")

    if not submitted:
        return

    patient = pd.DataFrame(
        [
            {
                "Age": age,
                "BMI": bmi,
                "Blood Pressure": bp,
                "Glucose Level": glucose,
                "Cholesterol": cholesterol,
                "Gender": gender,
                "Smoking Status": smoking,
                "Physical Activity": activity,
                "Medical History": history,
            }
        ]
    )

    predicted_label, probabilities = predict_risk(pipeline, label_encoder, patient)

    st.success(f"Predicted Risk Category: **{predicted_label}**")

    probability_df = pd.DataFrame(
        {
            "Risk Category": label_encoder.classes_,
            "Probability": probabilities,
        }
    ).sort_values("Probability", ascending=False)

    st.plotly_chart(
        px.bar(
            probability_df,
            x="Risk Category",
            y="Probability",
            text=probability_df["Probability"].map(lambda x: f"{x:.1%}"),
            title="Prediction Probabilities",
        ),
        use_container_width=True,
    )

    st.subheader("SHAP Explanation")
    try:
        class_index = list(label_encoder.classes_).index(predicted_label)
        explanation = explain_prediction(pipeline, patient, class_index)
        explanation["Readable Feature"] = explanation["Feature"].map(human_feature_name)

        st.plotly_chart(
            px.bar(
                explanation.head(10).sort_values("SHAP Value"),
                x="SHAP Value",
                y="Readable Feature",
                orientation="h",
                title="Top SHAP Contributions for This Prediction",
            ),
            use_container_width=True,
        )

        with st.expander("Explanation details"):
            st.dataframe(
                explanation[["Readable Feature", "SHAP Value", "Absolute SHAP"]].head(10),
                use_container_width=True,
            )
            st.caption(
                "A positive SHAP value means the feature pushed the model output toward the predicted class; "
                "a negative value pushed it away. This describes model behavior, not medical causation."
            )
    except Exception as exc:
        st.warning(f"SHAP explanation could not be generated: {exc}")


def model_performance_page() -> None:
    st.title("Model Performance")

    if not METRICS_PATH.exists():
        st.warning("Evaluation results are not available. Run notebook 03_model_training.ipynb.")
        return

    metrics = json.loads(METRICS_PATH.read_text())
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Accuracy", f"{metrics['accuracy']:.2%}")
    c2.metric("Precision", f"{metrics['precision_weighted']:.2%}")
    c3.metric("Recall", f"{metrics['recall_weighted']:.2%}")
    c4.metric("F1-score", f"{metrics['f1_weighted']:.2%}")

    st.subheader("Confusion Matrix")
    if CM_PATH.exists():
        cm = pd.read_csv(CM_PATH, index_col=0)
        st.dataframe(cm, use_container_width=True)

        fig = px.imshow(
            cm.values,
            x=cm.columns,
            y=cm.index,
            text_auto=True,
            labels={"x": "Predicted", "y": "Actual", "color": "Count"},
            title="Confusion Matrix",
        )
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Global Feature Importance")
    try:
        pipeline, _ = load_model()
        importance = global_feature_importance(pipeline).head(15)
        importance["Readable Feature"] = importance["Feature"].map(human_feature_name)
        st.plotly_chart(
            px.bar(
                importance.sort_values("Importance"),
                x="Importance",
                y="Readable Feature",
                orientation="h",
                title="Random Forest Feature Importance",
            ),
            use_container_width=True,
        )
    except Exception as exc:
        st.warning(f"Feature importance unavailable: {exc}")

    st.subheader("Classification Report")
    report = pd.DataFrame(metrics["classification_report"]).T
    st.dataframe(report, use_container_width=True)


def about_page() -> None:
    st.title("About / Methodology")

    st.markdown(
        """
        ### Project objective
        Build a local interactive dashboard that analyzes anonymized healthcare data,
        visualizes risk patterns, predicts a selected health-risk category and provides
        simple explanations for model predictions.

        ### Workflow
        **Healthcare Dataset → Data Cleaning → Exploratory Analysis → Feature Engineering
        → Random Forest → SHAP Explanation → Streamlit Dashboard → Insights**

        ### Technology
        - Python
        - Streamlit
        - Pandas / NumPy
        - Plotly / Matplotlib / Seaborn
        - Scikit-learn Random Forest
        - SHAP
        - CSV as the initial data source
        - SQLite can be added later if structured storage is needed

        ### Privacy
        Use only anonymized or synthetic academic data. Do not include names,
        phone numbers, addresses, medical record numbers or other unnecessary identifiers.

        ### Important limitation
        The dashboard is for educational and analytical purposes. Model associations
        should not be interpreted as medical causation, diagnosis or treatment advice.
        """
    )


def main():
    st.sidebar.title("Health Risk Dashboard")
    page = st.sidebar.radio(
        "Navigation",
        ["Overview", "Risk Analysis", "Prediction", "Model Performance", "About / Methodology"],
    )

    st.sidebar.divider()
    uploaded = st.sidebar.file_uploader("Upload healthcare CSV", type=["csv"])

    try:
        raw = pd.read_csv(uploaded) if uploaded is not None else load_default_data()
        if uploaded is not None:
            st.sidebar.info(
                "Uploaded data is used for dashboard analysis. The saved model is used for prediction; "
                "retrain notebook 03 if the uploaded dataset is your new training dataset."
            )
        raw = canonicalize_columns(raw)

        valid, missing = validate_columns(raw, require_target=True)
        if not valid:
            st.error("Dataset validation failed. Missing columns: " + ", ".join(missing))
            st.stop()

        df = clean_for_dashboard(raw)
    except Exception as exc:
        st.error(f"Could not load dataset: {exc}")
        st.stop()

    if page in ["Overview", "Risk Analysis"]:
        filtered = apply_filters(df)
    else:
        filtered = df

    if filtered.empty and page in ["Overview", "Risk Analysis"]:
        st.warning("No records match the selected filters.")
        return

    if page == "Overview":
        overview_page(filtered)
    elif page == "Risk Analysis":
        risk_analysis_page(filtered)
    elif page == "Prediction":
        prediction_page(df)
    elif page == "Model Performance":
        model_performance_page()
    else:
        about_page()


if __name__ == "__main__":
    main()
