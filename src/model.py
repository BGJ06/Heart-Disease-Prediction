import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, roc_curve
)
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

def prepare_ml_data(df, target_col='target'):
    """Prepares features and target from the cleaned DataFrame."""
    # Drop helper/identifier columns
    cols_to_drop = ['Patient_ID', 'Age_Group', 'BP_Category', target_col]
    cols_to_drop = [c for c in cols_to_drop if c in df.columns]
    
    X = df.drop(columns=cols_to_drop)
    y = df[target_col]
    
    # Identify feature types
    numeric_features = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']
    # All other columns are categorical
    categorical_features = [col for col in X.columns if col not in numeric_features]
    
    return X, y, numeric_features, categorical_features

def build_preprocessing_pipeline(numeric_features, categorical_features):
    """Creates a ColumnTransformer for preprocessing features."""
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numeric_features),
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_features)
        ]
    )
    return preprocessor

def train_and_evaluate(X, y, model_name='random_forest', test_size=0.2, hyperparams=None):
    """Splits data, trains the model pipeline, and evaluates on the test set."""
    if hyperparams is None:
        hyperparams = {}
        
    # Split the dataset BEFORE fitting pipelines to prevent data leakage
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=y
    )
    
    # Identify columns
    numeric_features = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']
    categorical_features = [col for col in X.columns if col not in numeric_features]
    
    # Initialize estimator
    if model_name == 'logistic_regression':
        # Default max_iter to 1000 for convergence
        C = hyperparams.get('C', 1.0)
        estimator = LogisticRegression(C=C, max_iter=1000, random_state=42)
    elif model_name == 'decision_tree':
        max_depth = hyperparams.get('max_depth', None)
        min_samples_split = hyperparams.get('min_samples_split', 2)
        estimator = DecisionTreeClassifier(max_depth=max_depth, min_samples_split=min_samples_split, random_state=42)
    elif model_name == 'random_forest':
        n_estimators = hyperparams.get('n_estimators', 100)
        max_depth = hyperparams.get('max_depth', None)
        estimator = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=42)
    else:
        raise ValueError(f"Unknown model name: {model_name}")
        
    # Construct the full pipeline
    preprocessor = build_preprocessing_pipeline(numeric_features, categorical_features)
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('model', estimator)
    ])
    
    # Fit the pipeline on training data
    pipeline.fit(X_train, y_train)
    
    # Predict
    y_pred = pipeline.predict(X_test)
    y_train_pred = pipeline.predict(X_train)
    
    # Get probabilities
    if hasattr(pipeline, "predict_proba"):
        y_prob = pipeline.predict_proba(X_test)[:, 1]
        y_train_prob = pipeline.predict_proba(X_train)[:, 1]
    else:
        y_prob = None
        y_train_prob = None
        
    # Calculate metrics
    results = {
        'pipeline': pipeline,
        'X_train': X_train,
        'X_test': X_test,
        'y_train': y_train,
        'y_test': y_test,
        'y_pred': y_pred,
        'y_prob': y_prob,
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred, zero_division=0),
        'recall': recall_score(y_test, y_pred, zero_division=0),
        'f1': f1_score(y_test, y_pred, zero_division=0),
        'roc_auc': roc_auc_score(y_test, y_prob) if y_prob is not None else None,
        'numeric_features': numeric_features,
        'categorical_features': categorical_features
    }
    
    # Extract feature importance if available
    try:
        # Get one-hot encoded category names
        cat_encoder = pipeline.named_steps['preprocessor'].named_transformers_['cat']
        encoded_cat_names = cat_encoder.get_feature_names_out(categorical_features)
        all_feature_names = np.concatenate([numeric_features, encoded_cat_names])
        results['feature_names'] = all_feature_names
        
        if model_name in ['decision_tree', 'random_forest']:
            importances = pipeline.named_steps['model'].feature_importances_
            results['feature_importances'] = dict(zip(all_feature_names, importances))
        elif model_name == 'logistic_regression':
            coefs = pipeline.named_steps['model'].coef_[0]
            results['feature_importances'] = dict(zip(all_feature_names, np.abs(coefs))) # Absolute coefficient size as proxy for importance
    except Exception as e:
        print(f"Could not extract feature importances: {e}")
        
    return results

def get_confusion_matrix_plot(y_true, y_pred):
    """Generates a Confusion Matrix plot."""
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
                xticklabels=['Normal', 'Heart Disease'],
                yticklabels=['Normal', 'Heart Disease'], ax=ax)
    ax.set_ylabel('Actual Label')
    ax.set_xlabel('Predicted Label')
    ax.set_title('Confusion Matrix')
    plt.tight_layout()
    return fig

def get_roc_curve_plot(y_true, y_prob):
    """Generates a ROC Curve plot."""
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    auc_score = roc_auc_score(y_true, y_prob)
    fig, ax = plt.subplots(figsize=(5, 4.2))
    ax.plot(fpr, tpr, color='#EF4444', lw=2, label=f'ROC Curve (AUC = {auc_score:.3f})')
    ax.plot([0, 1], [0, 1], color='#64748B', lw=1.5, linestyle='--')
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('Receiver Operating Characteristic (ROC)')
    ax.legend(loc="lower right")
    plt.tight_layout()
    return fig

def get_feature_importance_plot(feature_importances_dict, top_n=10):
    """Plots top N feature importances."""
    sorted_importances = sorted(feature_importances_dict.items(), key=lambda x: x[1], reverse=True)[:top_n]
    features, importances = zip(*sorted_importances)
    
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.barplot(x=list(importances), y=list(features), palette='viridis', ax=ax, hue=list(features), legend=False)
    ax.set_xlabel('Relative Importance / Coefficient Magnitude')
    ax.set_ylabel('Features')
    ax.set_title(f'Top {top_n} Predictor Features')
    plt.tight_layout()
    return fig

if __name__ == "__main__":
    # Test script locally
    script_dir = os.path.dirname(os.path.abspath(__file__))
    clean_data_path = os.path.join(script_dir, '..', 'data', 'cleaned_heart_disease.csv')
    
    if os.path.exists(clean_data_path):
        df = pd.read_csv(clean_data_path)
        X, y, num_feats, cat_feats = prepare_ml_data(df)
        print(f"Data prepared. X shape: {X.shape}, y shape: {y.shape}")
        
        # Test Random Forest
        rf_results = train_and_evaluate(X, y, 'random_forest')
        print(f"Random Forest - Accuracy: {rf_results['accuracy']:.4f}, Recall: {rf_results['recall']:.4f}, ROC-AUC: {rf_results['roc_auc']:.4f}")
        
        # Test Logistic Regression
        lr_results = train_and_evaluate(X, y, 'logistic_regression')
        print(f"Logistic Regression - Accuracy: {lr_results['accuracy']:.4f}, Recall: {lr_results['recall']:.4f}, ROC-AUC: {lr_results['roc_auc']:.4f}")
    else:
        print("Cleaned data does not exist. Please run generator.py and cleaner.py first.")
