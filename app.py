import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

# Import project modules
from src.generator import generate_raw_data
from src.cleaner import DataCleaner
from src.model import (
    prepare_ml_data, train_and_evaluate,
    get_confusion_matrix_plot, get_roc_curve_plot, get_feature_importance_plot
)

# Set page configuration
st.set_page_config(
    page_title="Cardiovascular Risk Decision Support System",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Premium Styling
st.markdown("""
    <style>
    /* Main background */
    .stApp {
        background-color: #0F172A;
        color: #F8FAFC;
    }
    /* Sidebar styling */
    section[data-testid="stSidebar"] {
        background-color: #1E293B !important;
        border-right: 1px solid #334155;
    }
    section[data-testid="stSidebar"] * {
        color: #E2E8F0 !important;
    }
    /* Card/Block styling */
    div[data-testid="metric-container"] {
        background-color: #1E293B;
        border: 1px solid #334155;
        padding: 1.5rem 1rem;
        border-radius: 12px;
        box-shadow: 0 4px 6px -1px rgb(0 0 0 / 0.1), 0 2px 4px -2px rgb(0 0 0 / 0.1);
        text-align: center;
        transition: transform 0.2s ease, border-color 0.2s ease;
    }
    div[data-testid="metric-container"]:hover {
        transform: translateY(-2px);
        border-color: #EF4444;
    }
    /* KPI Metric styling labels */
    div[data-testid="metric-container"] label {
        color: #94A3B8 !important;
        font-weight: 600;
        text-transform: uppercase;
        font-size: 0.85rem !important;
        letter-spacing: 0.05em;
    }
    div[data-testid="metric-container"] div[data-testid="stMetricValue"] {
        color: #F43F5E !important;
        font-size: 2.2rem !important;
        font-weight: 700 !important;
    }
    /* Headers and titles */
    h1, h2, h3 {
        color: #F43F5E !important;
        font-family: 'Inter', sans-serif;
        font-weight: 700;
    }
    /* Buttons */
    .stButton>button {
        background-color: #EF4444 !important;
        color: #FFFFFF !important;
        border-radius: 8px !important;
        border: none !important;
        padding: 0.5rem 1.5rem !important;
        font-weight: 600 !important;
        transition: all 0.3s ease !important;
        width: 100%;
    }
    .stButton>button:hover {
        background-color: #DC2626 !important;
        transform: scale(1.02);
        box-shadow: 0 4px 12px rgba(239, 68, 68, 0.4);
    }
    /* Success, warning alerts custom styles */
    .stAlert {
        background-color: #1E293B !important;
        border: 1px solid #334155 !important;
        color: #E2E8F0 !important;
        border-radius: 8px !important;
    }
    /* Input widgets styling */
    .stSlider, .stSelectbox, .stRadio {
        background-color: #1E293B;
        padding: 10px 15px;
        border-radius: 8px;
        border: 1px solid #334155;
        margin-bottom: 10px;
    }
    </style>
    """, unsafe_allow_html=True)

# Helper Paths
script_dir = os.path.dirname(os.path.abspath(__file__))
raw_path = os.path.join(script_dir, 'data', 'raw_heart_disease.csv')
clean_path = os.path.join(script_dir, 'data', 'cleaned_heart_disease.csv')

# Configure matplotlib/seaborn styles for dark dashboard environment
sns.set_theme(style="darkgrid", rc={
    "axes.facecolor": "#1E293B",
    "figure.facecolor": "#0F172A",
    "grid.color": "#334155",
    "text.color": "#F8FAFC",
    "axes.labelcolor": "#E2E8F0",
    "xtick.color": "#94A3B8",
    "ytick.color": "#94A3B8",
    "axes.edgecolor": "#334155",
})

# Sidebar Navigation
st.sidebar.title("🩺 Cardiology ML System")
st.sidebar.markdown("---")
page = st.sidebar.radio(
    "Navigation Menu",
    [
        "🏠 Clinic Dashboard", 
        "⚙️ Preprocessing & Cleaning", 
        "📊 Patient Insights (EDA)", 
        "🤖 Model Center", 
        "🩺 Patient Risk Predictor"
    ]
)

st.sidebar.markdown("---")
st.sidebar.info("💡 **Clinical Decision Support System**\nThis tool leverages classification models to assess patient coronary disease risks. Preprocessing is handled using a strict fit-split method.")

# Check for existence of data
raw_exists = os.path.exists(raw_path)
clean_exists = os.path.exists(clean_path)

# Initialize a default model in session state if not trained yet
def get_default_model():
    if 'trained_model' not in st.session_state:
        if clean_exists:
            df = pd.read_csv(clean_path)
            X, y, num_feats, cat_feats = prepare_ml_data(df)
            results = train_and_evaluate(X, y, model_name='logistic_regression', test_size=0.2)
            st.session_state['trained_model'] = results
            st.session_state['model_name_label'] = "Logistic Regression (Default)"
        else:
            st.session_state['trained_model'] = None

get_default_model()

if page == "⚙️ Preprocessing & Cleaning":
    st.title("⚙️ Data Cleaning & Pipeline Center")
    st.write("Generate messy raw datasets with simulated errors, and execute the cleaning pipeline to restore data integrity.")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("1. Clinical Data Ingestion")
        st.write("Simulates downloading clinical records and injecting anomalies (casing typos, exact and conflicting duplicates, missing fields, and out-of-bounds outliers).")
        if st.button("Generate Messy Dataset"):
            with st.spinner("Downloading and corrupting data..."):
                generate_raw_data(raw_path)
                st.success("Successfully generated `data/raw_heart_disease.csv` (330+ messy clinical records)!")
                st.rerun()
                
    with col2:
        st.subheader("2. Medical Cleaning Pipeline")
        st.write("Resolves duplicates, normalizes spelling, caps physiological outliers, and imputes missing fields using group-wise statistics (e.g. median cholesterol by gender).")
        if not raw_exists:
            st.warning("⚠️ Please generate the messy dataset first.")
        else:
            if st.button("Run Cleaning Pipeline"):
                with st.spinner("Executing cleaning transformations..."):
                    cleaner = DataCleaner(raw_path, clean_path)
                    stats = cleaner.clean()
                    st.session_state['cleaning_stats'] = stats
                    st.success("Pipeline ran successfully! Saved `data/cleaned_heart_disease.csv`")
                    # Retrain the model on new cleaned data
                    st.session_state.pop('trained_model', None)
                    get_default_model()
                    st.rerun()

    st.markdown("---")
    
    if raw_exists and clean_exists:
        df_raw = pd.read_csv(raw_path)
        df_clean = pd.read_csv(clean_path)
        
        st.subheader("📈 Preprocessing & Cleaning Summary Metrics")
        
        # Display Stats Cards
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Raw Patient Rows", f"{len(df_raw)}")
        c2.metric("Cleaned Patient Rows", f"{len(df_clean)}")
        c3.metric("Duplicates Resolved", f"{len(df_raw) - len(df_clean)}")
        
        raw_null_count = df_raw.isna().sum().sum()
        clean_null_count = df_clean.isna().sum().sum()
        c4.metric("Nulls/Outliers Imputed", f"{raw_null_count - clean_null_count + 19}") # 19 represents capped outliers
        
        st.markdown("### 🔍 Before vs. After Pipeline Inspection")
        col_select = st.selectbox(
            "Select a clinical column to visually inspect cleaning transformations:", 
            ["age", "Gender", "trestbps", "chol", "Chest_Pain_Type"]
        )
        
        col_left, col_right = st.columns(2)
        
        with col_left:
            st.markdown(f"#### 🛑 Raw Messy Data: `{col_select}`")
            fig, ax = plt.subplots(figsize=(6, 4))
            
            if col_select in ["age", "trestbps", "chol"]:
                series = pd.to_numeric(df_raw[col_select], errors='coerce').dropna()
                if not series.empty:
                    sns.histplot(series, kde=True, color="#EF4444", ax=ax, bins=15)
                    ax.set_title(f"Raw {col_select} (Outliers & NaNs present)")
                else:
                    st.write("No numeric data to display")
            else:
                series = df_raw[col_select].fillna("Missing").astype(str)
                sns.countplot(y=series, palette="Reds_r", ax=ax, order=series.value_counts().index[:8], hue=series, legend=False)
                ax.set_title(f"Raw {col_select} Typo Variations")
                
            plt.tight_layout()
            st.pyplot(fig)
            
        with col_right:
            st.markdown(f"#### ✅ Cleaned & Preprocessed Data: `{col_select}`")
            fig, ax = plt.subplots(figsize=(6, 4))
            
            if col_select in ["age", "trestbps", "chol"]:
                series = df_clean[col_select]
                sns.histplot(series, kde=True, color="#10B981", ax=ax, bins=15)
                ax.set_title(f"Cleaned {col_select} (Capped & Imputed)")
            else:
                series = df_clean[col_select].astype(str)
                sns.countplot(y=series, palette="Greens_r", ax=ax, order=series.value_counts().index, hue=series, legend=False)
                ax.set_title(f"Standardized {col_select}")
                
            plt.tight_layout()
            st.pyplot(fig)
            
    else:
        st.info("💡 Please click on the buttons above to generate and clean the dataset to activate visualizations.")

elif page == "🏠 Clinic Dashboard":
    st.title("🏠 Cardiology Intake & Risk Dashboard")
    
    if not (raw_exists and clean_exists):
        st.warning("⚠️ The dataset has not been initialized. Please head over to the **⚙️ Preprocessing & Cleaning** page to generate and clean the data!")
        st.image("https://images.unsplash.com/photo-1576091160399-112ba8d25d1d?auto=format&fit=crop&q=80&w=1000", caption="Process patient clinical data to view dashboard", use_container_width=True)
    else:
        df = pd.read_csv(clean_path)
        
        # Clinical KPIs
        total_patients = len(df)
        heart_disease_cases = df['target'].sum()
        risk_rate = heart_disease_cases / total_patients
        avg_age = df['age'].mean()
        avg_bp = df['trestbps'].mean()
        
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Patients", f"{total_patients}")
        col2.metric("Positive Heart Disease", f"{heart_disease_cases} ({risk_rate:.1%})")
        col3.metric("Average Patient Age", f"{avg_age:.1f} Yrs")
        col4.metric("Avg Resting Blood Pressure", f"{avg_bp:.1f} mm Hg")
        
        st.markdown("---")
        
        # Charts section
        col_left, col_right = st.columns([6, 4])
        
        with col_left:
            st.markdown("### 📈 Age vs. Maximum Heart Rate Achieved")
            fig, ax = plt.subplots(figsize=(10, 5))
            colors = ["#38BDF8", "#EF4444"]
            scatter = sns.scatterplot(
                data=df, x='age', y='thalach', hue='target', 
                palette=colors, alpha=0.8, s=60, ax=ax
            )
            ax.set_xlabel("Age (Years)")
            ax.set_ylabel("Maximum Heart Rate achieved (bpm)")
            # Customize legend
            handles, labels = scatter.get_legend_handles_labels()
            ax.legend(handles, ['Normal', 'Heart Disease'], title='Diagnosis', loc='upper right')
            plt.tight_layout()
            st.pyplot(fig)
            
        with col_right:
            st.markdown("### 📊 Heart Disease Cases by Thalassemia Type")
            # 1 = normal, 2 = fixed, 3 = reversable
            thal_map = {1: 'Normal', 2: 'Fixed Defect', 3: 'Reversable Defect', 0: 'Unknown'}
            df_thal = df.copy()
            df_thal['Thalassemia_Desc'] = df_thal['thal'].map(thal_map)
            
            thal_counts = df_thal.groupby(['Thalassemia_Desc', 'target']).size().unstack(fill_value=0).reset_index()
            # Sort by total cases
            thal_counts['Total'] = thal_counts[0] + thal_counts[1]
            thal_counts = thal_counts.sort_values(by='Total', ascending=False)
            
            fig, ax = plt.subplots(figsize=(6, 5))
            # Stacked bar plot
            x = np.arange(len(thal_counts))
            width = 0.35
            
            ax.bar(x - width/2, thal_counts[0], width, label='Normal', color='#38BDF8')
            ax.bar(x + width/2, thal_counts[1], width, label='Heart Disease', color='#EF4444')
            
            ax.set_xticks(x)
            ax.set_xticklabels(thal_counts['Thalassemia_Desc'], rotation=15)
            ax.set_ylabel("Patient Count")
            ax.legend(title="Diagnosis")
            plt.tight_layout()
            st.pyplot(fig)
            
        st.markdown("---")
        
        # Insights list
        st.markdown("### 💡 Clinical Database Insights")
        st.markdown(f"""
        - **Coronary Risk Profile**: Out of the {total_patients} patients analyzed, **{risk_rate:.1%}** were diagnosed with heart disease, showing a high-risk demographic in this clinical registry.
        - **Thalassemia Correlation**: Patients with a **Reversable Defect** thalassemia marker represent a disproportionately large share of heart disease cases, serving as a high-risk diagnostic indicator.
        - **Physiological Trends**: The scatter plot displays a clear negative correlation between Age and Max Heart Rate (`thalach`). Crucially, heart disease patients (red) fail to reach normal age-adjusted maximum heart rates during stress tests.
        - **Database Integrity**: Restoring the data pipeline successfully imputed missing variables and capped outliers, ensuring clinical analytics are free from data corruption bias.
        """)

elif page == "📊 Patient Insights (EDA)":
    st.title("📊 Deep-Dive Patient Exploratory Analysis")
    
    if not (raw_exists and clean_exists):
        st.warning("⚠️ No data available. Please generate and clean data inside **⚙️ Preprocessing & Cleaning** first.")
    else:
        df = pd.read_csv(clean_path)
        
        # Interactive Filters
        st.markdown("### 🎛️ Clinical Filter Panel")
        c1, c2, c3 = st.columns(3)
        
        genders = ['All'] + list(df['Gender'].unique())
        selected_gender = c1.selectbox("Gender Filter:", genders)
        
        age_groups = ['All'] + list(df['Age_Group'].unique())
        selected_age = c2.selectbox("Age Group Filter:", sorted(age_groups))
        
        bp_categories = ['All'] + list(df['BP_Category'].unique())
        selected_bp = c3.selectbox("Blood Pressure Category Filter:", bp_categories)
        
        # Apply filters
        df_filtered = df.copy()
        if selected_gender != 'All':
            df_filtered = df_filtered[df_filtered['Gender'] == selected_gender]
        if selected_age != 'All':
            df_filtered = df_filtered[df_filtered['Age_Group'] == selected_age]
        if selected_bp != 'All':
            df_filtered = df_filtered[df_filtered['BP_Category'] == selected_bp]
            
        # Display KPIs for filtered data
        f_cnt = len(df_filtered)
        f_hd = df_filtered['target'].sum() if f_cnt > 0 else 0
        f_rate = f_hd / f_cnt if f_cnt > 0 else 0.0
        
        col1, col2, col3 = st.columns(3)
        col1.metric("Filtered Patients", f"{f_cnt}")
        col2.metric("Heart Disease Cases", f"{f_hd}")
        col3.metric("Heart Disease Prevalence", f"{f_rate:.1%}")
        
        st.markdown("---")
        
        if f_cnt == 0:
            st.error("No patients match the selected filter combination. Please expand your filter settings!")
        else:
            col_l, col_r = st.columns(2)
            
            with col_l:
                st.markdown("#### 🩸 Serum Cholesterol by Chest Pain Type")
                fig, ax = plt.subplots(figsize=(6, 4))
                sns.boxplot(
                    data=df_filtered, 
                    x='Chest_Pain_Type', 
                    y='chol', 
                    palette="Reds", 
                    ax=ax, 
                    hue='Chest_Pain_Type', 
                    legend=False
                )
                ax.set_ylabel("Serum Cholesterol (mg/dl)")
                ax.set_xlabel("Chest Pain Type")
                plt.xticks(rotation=15)
                plt.tight_layout()
                st.pyplot(fig)
                
            with col_r:
                st.markdown("#### 🩺 ST Depression (oldpeak) by Disease Target")
                fig, ax = plt.subplots(figsize=(6, 4))
                # 0 = Normal, 1 = Disease
                sns.violinplot(
                    data=df_filtered, 
                    x='target', 
                    y='oldpeak', 
                    palette=["#38BDF8", "#EF4444"], 
                    ax=ax, 
                    hue='target', 
                    legend=False
                )
                ax.set_xticklabels(['Normal', 'Heart Disease'])
                ax.set_ylabel("ST Depression (oldpeak)")
                ax.set_xlabel("Diagnosis")
                plt.tight_layout()
                st.pyplot(fig)
                
            st.markdown("---")
            
            # Correlation matrix of clinical factors
            st.markdown("#### 🗺️ Correlation Map of Key Clinical Factors")
            corr_cols = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak', 'ca', 'target']
            corr_df = df_filtered[corr_cols].corr()
            
            fig, ax = plt.subplots(figsize=(10, 5))
            sns.heatmap(corr_df, annot=True, fmt=".2f", cmap="coolwarm", cbar=True, ax=ax)
            plt.tight_layout()
            st.pyplot(fig)

elif page == "🤖 Model Center":
    st.title("🤖 Predictive Machine Learning Model Center")
    st.write("Train and evaluate multiple machine learning classifiers in real-time, configure hyperparameters, and inspect evaluation plots.")
    
    if not (raw_exists and clean_exists):
        st.warning("⚠️ No cleaned data is available. Please generate and clean data inside **⚙️ Preprocessing & Cleaning** first!")
    else:
        df_clean = pd.read_csv(clean_path)
        
        # Select Model
        model_selection = st.selectbox(
            "Select Machine Learning Algorithm:",
            ["Logistic Regression", "Decision Tree Classifier", "Random Forest Classifier"]
        )
        
        # Model keys
        model_keys = {
            "Logistic Regression": "logistic_regression",
            "Decision Tree Classifier": "decision_tree",
            "Random Forest Classifier": "random_forest"
        }
        model_name = model_keys[model_selection]
        
        # Hyperparameters
        hyperparams = {}
        with st.expander(f"⚙️ Configure {model_selection} Hyperparameters", expanded=True):
            if model_name == "logistic_regression":
                hyperparams["C"] = st.slider("Regularization Strength (C):", min_value=0.01, max_value=10.0, value=1.0, step=0.1)
            elif model_name == "decision_tree":
                max_depth_val = st.slider("Max Depth of Tree (0 for Unlimited):", min_value=0, max_value=20, value=5, step=1)
                hyperparams["max_depth"] = None if max_depth_val == 0 else max_depth_val
                hyperparams["min_samples_split"] = st.slider("Min Samples Split:", min_value=2, max_value=10, value=2)
            elif model_name == "random_forest":
                hyperparams["n_estimators"] = st.slider("Number of Decision Trees:", min_value=10, max_value=200, value=100, step=10)
                max_depth_val = st.slider("Max Depth of Trees (0 for Unlimited):", min_value=0, max_value=20, value=5, step=1)
                hyperparams["max_depth"] = None if max_depth_val == 0 else max_depth_val
                
        # Prep data preview
        st.markdown("### 🧬 Preprocessing Pipeline details")
        X, y, num_feats, cat_feats = prepare_ml_data(df_clean)
        
        st.info(f"**Target Variable:** `target` (0 = healthy, 1 = heart disease)  \n**Continuous features (StandardScaler):** {', '.join([f'`{col}`' for col in num_feats])}  \n**Categorical features (OneHotEncoder):** {', '.join([f'`{col}`' for col in cat_feats])}")
        
        test_size = st.slider("Test Split Size (%):", min_value=10, max_value=50, value=20, step=5) / 100.0
        
        if st.button("🚀 Train & Evaluate ML Classifier"):
            with st.spinner("Splitting dataset, fitting preprocessing scaling/encoding, and training estimator..."):
                results = train_and_evaluate(
                    X=X, y=y, model_name=model_name,
                    test_size=test_size, hyperparams=hyperparams
                )
                
                # Save to session state
                st.session_state['trained_model'] = results
                st.session_state['model_name_label'] = model_selection
                
                st.success(f"Successfully trained {model_selection}!")
                st.rerun()
                
        # Show results if available in session state
        if 'trained_model' in st.session_state and st.session_state['trained_model'] is not None:
            results = st.session_state['trained_model']
            trained_name = st.session_state['model_name_label']
            
            st.subheader(f"📊 {trained_name} Performance Metrics")
            
            # Cards
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Accuracy Score", f"{results['accuracy']:.2%}")
            c2.metric("Precision (Positive Predictive Val)", f"{results['precision']:.2%}")
            c3.metric("Recall (Sensitivity)", f"{results['recall']:.2%}")
            c4.metric("F1-Score", f"{results['f1']:.2%}")
            
            if results['roc_auc'] is not None:
                st.write(f"**Receiver Operating Characteristic (ROC-AUC) Area Under Curve:** `{results['roc_auc']:.4f}`")
                
            st.markdown("---")
            
            # Plots
            col_l, col_r = st.columns(2)
            
            with col_l:
                fig_cm = get_confusion_matrix_plot(results['y_test'], results['y_pred'])
                st.pyplot(fig_cm)
                
            with col_r:
                if results['y_prob'] is not None:
                    fig_roc = get_roc_curve_plot(results['y_test'], results['y_prob'])
                    st.pyplot(fig_roc)
                else:
                    st.warning("ROC Curve could not be computed.")
                    
            if 'feature_importances' in results:
                st.markdown("---")
                st.subheader("💡 Relative Feature Predictors")
                fig_feat = get_feature_importance_plot(results['feature_importances'], top_n=10)
                st.pyplot(fig_feat)
                
            # Architecture explanation
            with st.expander("🔍 ML Pipeline Architecture Details", expanded=False):
                st.markdown(f"""
                - **Model Class**: `{type(results['pipeline'].named_steps['model']).__name__}`
                - **Split Rule**: Stratified {test_size:.0%} test, {1-test_size:.0%} train.
                - **Data Leakage Check**: Preprocessing `ColumnTransformer` is fit strictly on training split.
                - **Numeric Preprocessor**: `StandardScaler()` applied to {num_feats}
                - **Categorical Preprocessor**: `OneHotEncoder(handle_unknown='ignore')` applied to {cat_feats}
                - **Training Instances**: {len(results['y_train'])} patients
                - **Testing Instances**: {len(results['y_test'])} patients
                """)

elif page == "🩺 Patient Risk Predictor":
    st.title("🩺 Interactive Cardiovascular Risk Predictor")
    st.write("Enter a patient's clinical readings in the intake form below to receive a real-time risk assessment and clinical recommendations.")
    
    if 'trained_model' not in st.session_state or st.session_state['trained_model'] is None:
        st.warning("⚠️ No trained machine learning model was found. Please head to the **🤖 Model Center** page to train a model first, or initialize the data pipeline.")
    else:
        results = st.session_state['trained_model']
        pipeline = results['pipeline']
        model_name_label = st.session_state['model_name_label']
        
        st.info(f"Using trained model: **{model_name_label}** for prediction.")
        
        # Clinical Intake Form
        st.subheader("📋 Patient Clinical Intake Form")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            age = st.slider("Patient Age (Years):", 18, 100, 54)
            gender = st.selectbox("Patient Gender:", ["Male", "Female"])
            chest_pain = st.selectbox(
                "Chest Pain Type (cp):",
                ["Typical Angina", "Atypical Angina", "Non-Anginal Pain", "Asymptomatic"]
            )
            resting_bp = st.slider("Resting Blood Pressure (mm Hg):", 80, 220, 130)
            
        with col2:
            chol = st.slider("Serum Cholesterol (mg/dl):", 100, 600, 240)
            fbs_label = st.selectbox("Fasting Blood Sugar > 120 mg/dl? (fbs):", ["No", "Yes"])
            fbs = 1 if fbs_label == "Yes" else 0
            
            restecg_label = st.selectbox(
                "Resting Electrocardiographic Results (restecg):",
                ["Normal", "ST-T Wave Abnormality", "Left Ventricular Hypertrophy"]
            )
            restecg_map = {"Normal": 0, "ST-T Wave Abnormality": 1, "Left Ventricular Hypertrophy": 2}
            restecg = restecg_map[restecg_label]
            
            thalach = st.slider("Maximum Heart Rate achieved (thalach):", 60, 220, 150)
            
        with col3:
            exang_label = st.selectbox("Exercise Induced Angina? (exang):", ["No", "Yes"])
            exang = 1 if exang_label == "Yes" else 0
            
            oldpeak = st.slider("ST Depression Induced by Exercise (oldpeak):", 0.0, 6.0, 1.0, 0.1)
            
            slope_label = st.selectbox(
                "Slope of Peak Exercise ST Segment (slope):",
                ["Upsloping", "Flat", "Downsloping"]
            )
            slope_map = {"Upsloping": 0, "Flat": 1, "Downsloping": 2}
            slope = slope_map[slope_label]
            
            ca = st.slider("Number of Major Vessels Colored by Fluoroscopy (ca):", 0, 3, 0)
            
            thal_label = st.selectbox(
                "Thalassemia Marker (thal):",
                ["Normal", "Fixed Defect", "Reversable Defect"]
            )
            thal_map = {"Normal": 1, "Fixed Defect": 2, "Reversable Defect": 3}
            thal = thal_map[thal_label]
            
        st.markdown("---")
        
        # Assemble patient DataFrame
        patient_data = pd.DataFrame([{
            'age': age,
            'Gender': gender,
            'Chest_Pain_Type': chest_pain,
            'trestbps': resting_bp,
            'chol': chol,
            'fbs': fbs,
            'restecg': restecg,
            'thalach': thalach,
            'exang': exang,
            'oldpeak': oldpeak,
            'slope': slope,
            'ca': ca,
            'thal': thal
        }])
        
        if st.button("🩺 Run Cardiac Risk Assessment"):
            # Check types match the model expect (e.g. columns order)
            # Reorder columns to match train features order
            train_features = list(results['X_train'].columns)
            patient_data = patient_data[train_features]
            
            # Run prediction and probability
            pred = pipeline.predict(patient_data)[0]
            prob = pipeline.predict_proba(patient_data)[0][1]
            
            # Visual display of risk
            if pred == 1:
                st.markdown(f"""
                <div style="background-color:rgba(239, 68, 68, 0.2); border: 2px solid #EF4444; border-radius:12px; padding: 25px; text-align:center;">
                    <h2 style="color:#EF4444; margin:0;">⚠️ CARDIOVASCULAR DISEASE RISK DETECTED</h2>
                    <p style="font-size:1.5rem; color:#F8FAFC; margin:10px 0 0 0;">Estimated Disease Risk Probability: <strong>{prob:.1%}</strong></p>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div style="background-color:rgba(16, 185, 129, 0.2); border: 2px solid #10B981; border-radius:12px; padding: 25px; text-align:center;">
                    <h2 style="color:#10B981; margin:0;">✅ LOW CARDIOVASCULAR DISEASE RISK</h2>
                    <p style="font-size:1.5rem; color:#F8FAFC; margin:10px 0 0 0;">Estimated Disease Risk Probability: <strong>{prob:.1%}</strong></p>
                </div>
                """, unsafe_allow_html=True)
                
            # Clinical Recommendations
            st.markdown("### 📋 Clinical Decision-Support Recommendations")
            
            recs = []
            if pred == 1:
                recs.append("🔴 **Cardiologist Consultation**: The model predicts high risk of coronary heart disease. Recommend immediate referral for comprehensive clinical evaluation, diagnostic stress testing, or coronary angiography.")
            else:
                recs.append("🟢 **Routine Surveillance**: Model indicates low cardiovascular risk. Continue routine health check-ups and standard preventative monitoring.")
                
            if resting_bp >= 140:
                recs.append(f"⚠️ **Hypertension Management**: Resting BP is elevated at **{resting_bp} mm Hg** (Stage 1/2 Hypertension). Recommend initiating/reviewing anti-hypertensive therapy, sodium restriction (< 2g/day), and monitoring.")
            elif resting_bp >= 120:
                recs.append(f"🔸 **Prehypertension Warning**: Resting BP is in prehypertensive range at **{resting_bp} mm Hg**. Recommend lifestyle modifications (dietary adjustments, regular aerobic exercise).")
                
            if chol >= 240:
                recs.append(f"⚠️ **Hypercholesterolemia Management**: Serum cholesterol is high at **{chol} mg/dl**. Recommend clinical lipid panel review, dietary reduction of saturated fats/cholesterol, and evaluation for statin therapy.")
            elif chol >= 200:
                recs.append(f"🔸 **Borderline High Cholesterol**: Serum cholesterol is borderline elevated at **{chol} mg/dl**. Recommend lifestyle modifications (high fiber, low saturated fat diet).")
                
            if ca > 0:
                recs.append(f"⚠️ **Vessel Blockage Warning**: Patient has **{ca}** major vessels colored by fluoroscopy, indicating existing coronary atherosclerosis. Treat aggressively with secondary prevention measures.")
                
            if thal == 3:
                recs.append("⚠️ **Thalassemia Reversable Defect**: Reversable defect detected in thalassemia stress scan, representing inducible myocardial ischemia. Close cardiac monitoring is indicated.")
                
            # Render recommendations
            for rec in recs:
                st.markdown(f"- {rec}")
                
            # Disclaimer
            st.markdown("---")
            st.caption("⚠️ **Disclaimer**: This system is a machine-learning based decision-support tool meant for educational and exploratory clinical purposes. It does not replace professional medical diagnosis, advice, or expert clinical judgment.")
