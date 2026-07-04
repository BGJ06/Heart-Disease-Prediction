# 🩺 Cardiovascular Risk Prediction & Clinic Decision Support System

A modular Python data pipeline and premium interactive Streamlit dashboard that downloads/generates a messy clinical heart disease dataset, performs statistical cleaning and group-wise imputation, trains multiple classification models (Logistic Regression, Decision Trees, and Random Forests), and presents a real-time risk assessment form with tailored clinical recommendations.

### 🏷️ Repository Details
*   **Description**: A modular Python data pipeline and machine learning system that downloads/corrupts a clinical cardiovascular dataset, applies group-wise imputation, compares classification models, and serves predictions via an interactive Streamlit dashboard.
*   **Topics**: `python`, `data-science`, `machine-learning`, `eda`, `pandas`, `scikit-learn`, `streamlit`, `data-cleaning`, `data-pipeline`, `cardiology`

---

## 🎯 Project Overview
This project demonstrates applied data science and predictive modeling within the **Health Domain**. It simulates a real-world clinic database intake system by intentionally corrupting clean historical patient records (injecting missing data, duplicate entries, extreme physiological outliers, and spelling variations) and running it through a robust cleaning pipeline. The cleaned data is used to train and compare classification models, which are served through a premium dark-mode decision support interface.

---

## ⚙️ Core Pipeline Components

1.  **Clinical Data Generator** ([generator.py](file:///d:/Internship%20project/Project%20on%20real%20world%20application/src/generator.py)): Downloads the UCI Cleveland dataset and injects clinical flaws (missing values in cholesterol/blood pressure, string casing inconsistencies in gender/chest-pain, negative blood pressure, and conflicting patient duplicates).
2.  **Clinical Data Cleaner** ([cleaner.py](file:///d:/Internship%20project/Project%20on%20real%20world%20application/src/cleaner.py)): Resolves patient ID conflicts, standardizes categorical text, caps physiological outliers, and performs group-wise imputation (e.g. median cholesterol grouped by gender). It also engineers `Age_Group` and `BP_Category` columns.
3.  **Exploratory Data Analysis** ([analyzer.py](file:///d:/Internship%20project/Project%20on%20real%20world%20application/src/analyzer.py)): Generates static charts (such as age vs. maximum heart rate achieved, correlation heatmap, and clinical distributions).
4.  **Machine Learning Pipeline** ([model.py](file:///d:/Internship%20project/Project%20on%20real%20world%20application/src/model.py)): Fits a `ColumnTransformer` with `StandardScaler` (numerical features) and `OneHotEncoder` (categorical features) followed by the estimator. Evaluates models using Accuracy, Precision, Recall (Sensitivity), and ROC-AUC.
5.  **Interactive Web Dashboard** ([app.py](file:///d:/Internship%20project/Project%20on%20real%20world%20application/app.py)): A premium dark-mode Streamlit dashboard with 5 tabs for clinic metrics, preprocessors, interactive filtered EDA, classifier training, and a patient risk intake tool.

---

## 🤖 Model Comparison & Clinical Viability

In heart disease screening, **Recall (Sensitivity)** is prioritized to minimize False Negatives (missing patients who actually have heart disease). 

On our stratified test split, the classifiers achieved:

| Classifier | Accuracy | Recall (Sensitivity) | Precision (PPV) | F1-Score | ROC-AUC |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression** | **78.69%** | **93.75%** | 73.17% | **82.19%** | **0.8912** |
| **Random Forest** | 75.41% | 84.38% | 73.00% | 78.26% | 0.8707 |
| **Decision Tree** | 72.13% | 78.12% | 71.43% | 74.63% | 0.7188 |

*Logistic Regression is recommended for production deployment due to its high recall of **93.75%** (minimizing critical false negatives) and mathematical transparency.*

---

## 📂 Project Structure
```
d:\Internship project\Project on real world application\
├── data/                      # Data storage
│   ├── raw_heart_disease.csv  # Messy dataset
│   └── cleaned_heart_disease.csv # Preprocessed dataset
├── plots/                     # Static EDA visualizations
├── src/                       # Source pipeline packages
│   ├── __init__.py
│   ├── generator.py           # Ingestion & corruption script
│   ├── cleaner.py             # Imputation & deduplication
│   ├── analyzer.py            # Static chart builder
│   └── model.py               # Preprocessing & ML trainer
├── app.py                     # Streamlit web application
├── heart_disease_analysis.ipynb # Step-by-step storytelling notebook
├── requirements.txt           # Project dependencies
└── README.md                  # Project documentation
```

---

## 🚀 Quickstart Guide

### 1. Setup Virtual Environment & Install Dependencies
Ensure you are in the project folder, activate the parent virtual environment, or create a local one:
```bash
# Create local virtual environment (optional)
python -m venv .venv
.venv\Scripts\Activate.ps1

# Install requirements
pip install -r requirements.txt
```

### 2. Run Pipeline Commands
Execute each pipeline stage independently from the command line:
```bash
# Generate messy data
python src/generator.py

# Run cleaning script
python src/cleaner.py

# Render static charts
python src/analyzer.py

# Test ML pipeline
python src/model.py
```

### 3. Start the Web Dashboard
Launch the interactive decision support dashboard:
```bash
streamlit run app.py
```
Open **`http://localhost:8501`** in your browser to view the clinical dashboard.
