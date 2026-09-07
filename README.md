# Health Risk Dashboard

A TY B.Sc. Data Science mini project that analyzes anonymized healthcare data, visualizes health-risk patterns, predicts a risk category using a Random Forest classifier, and explains predictions using SHAP.

> **Academic-use notice:** This project is an educational analytics system. Predictions and explanations are not a medical diagnosis and must not be used for treatment decisions.

## Project structure

```text
Health-Risk-Dashboard/
├── app.py
├── requirements.txt
├── README.md
├── .gitignore
├── data/
│   ├── healthcare_risk.csv
│   └── cleaned_data.csv
├── models/
│   ├── random_forest.pkl
│   └── label_encoder.pkl
├── src/
│   ├── __init__.py
│   ├── preprocessing.py
│   ├── model.py
│   ├── analytics.py
│   └── explainability.py
├── notebooks/
│   ├── 01_data_cleaning.ipynb
│   ├── 02_eda.ipynb
│   └── 03_model_training.ipynb
├── outputs/
│   ├── graphs/
│   ├── reports/
│   └── model_results/
├── assets/
│   └── dashboard_screenshots/
└── docs/
    ├── PRD.pdf
    ├── FRD.pdf
    └── TRD.pdf
```

## 1. Create the environment

Python 3.11+ is recommended.

### Windows

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

### macOS/Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## 2. Run the notebooks

Open the project folder in VS Code/Jupyter and run:

1. `01_data_cleaning.ipynb`
2. `02_eda.ipynb`
3. `03_model_training.ipynb`

The training notebook creates:

- `data/cleaned_data.csv`
- `models/random_forest.pkl`
- `models/label_encoder.pkl`
- evaluation files in `outputs/model_results/`

A synthetic starter dataset is included so the project can be run immediately. Replace it with an approved anonymized academic dataset when required.

## 3. Start the dashboard

From the project root:

```bash
streamlit run app.py
```

The dashboard contains:

- Overview
- Risk Analysis
- Prediction
- Model Performance
- About / Methodology

## Dataset columns

The starter dataset uses:

- Patient ID
- Age
- Gender
- BMI
- Blood Pressure
- Glucose Level
- Cholesterol
- Smoking Status
- Physical Activity
- Medical History
- Health Risk

The application also recognizes common alternative column names such as `Blood Sugar`, `Glucose`, `Risk Category`, and `Health Risk Category`.

## Optional local LLM

The PRD/TRD describe Qwen3-4B through Ollama as an optional feature. The core dashboard does not depend on it. If Ollama is unavailable, the application continues to work.

## Workflow

Healthcare CSV → Cleaning → EDA → Feature preparation → Random Forest → Risk prediction → SHAP explanation → Streamlit dashboard → Insights

## Limitations

- The quality of the prediction depends on the dataset and target labels.
- Correlation/model contribution is not the same as medical causation.
- Only anonymized or synthetic academic data should be used.
- Do not deploy patient-level data publicly.
