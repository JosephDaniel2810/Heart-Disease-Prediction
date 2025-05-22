# heart_disease_predictor/train_model.py

"""
Heart Disease Prediction Pipeline
- Data loading & cleaning
- Exploratory Data Analysis (EDA)
- Feature engineering & preprocessing
- Model training with hyperparameter tuning
- Evaluation metrics & visualizations
- Artifact saving for deployment
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, PowerTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, roc_curve
)


# Configuration and Constants

DATA_PATH = "data/heart_cleveland_upload.csv"
VIS_DIR = "visualization"
MODEL_DIR = "models"
TARGET_COL = "condition"
NUMERIC_COLS = [
    'age', 'trestbps', 'chol', 'thalach', 'oldpeak'
]
ZERO_IMPUTE_COLS = [
    'trestbps', 'chol', 'thalach', 'oldpeak'
]
RANDOM_STATE = 42
TEST_SIZE = 0.2
SCORING = 'f1'
CV_FOLDS = 5

# Ensure directories exist
os.makedirs(VIS_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)


# 1. Data Loading and Cleaning

def load_and_clean_data(filepath: str) -> pd.DataFrame:
    """
    Load dataset from CSV and perform cleaning:
    - Rename target column
    - Report shape and missing values
    - Handle zero entries in medical measurements
    - Cap outliers using IQR method
    """
    df = pd.read_csv(filepath)
    print(f"Loaded data with shape: {df.shape}")

    # Rename
    if TARGET_COL in df.columns:
        df = df.rename(columns={TARGET_COL: 'target'})
        print("Renamed 'condition' to 'target'.")

    # Report missing values
    missing = df.isnull().sum()
    print("Missing values before cleaning:\n", missing)

    # Replace zeros in medical columns with NaN
    for col in ZERO_IMPUTE_COLS:
        zeros = (df[col] == 0).sum()
        if zeros > 0:
            df[col] = df[col].replace(0, np.nan)
            print(f"Replaced {zeros} zeros in '{col}' with NaN.")

    # Impute median for NaN
    for col in ZERO_IMPUTE_COLS:
        if df[col].isnull().sum() > 0:
            median = df[col].median()
            df[col].fillna(median, inplace=True)
            print(f"Imputed NaN in '{col}' with median: {median}.")

    # Detect and cap outliers in numeric columns
    for col in NUMERIC_COLS:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR
        before_outliers = df[(df[col] < lower) | (df[col] > upper)][col].count()
        df[col] = np.where(df[col] < lower, lower,
                           np.where(df[col] > upper, upper, df[col]))
        after_outliers = df[(df[col] < lower) | (df[col] > upper)][col].count()
        print(f"Capped {before_outliers} outliers in '{col}'. Remained {after_outliers} outliers.")

    print("Missing values after cleaning:\n", df.isnull().sum())
    return df


# 2. Exploratory Data Analysis

def plot_distribution(df: pd.DataFrame):
    """
    Plot histograms for all features by target status.
    """
    features = df.columns.drop('target')
    n = len(features)
    cols = 3
    nrows = int(np.ceil(n / cols))
    plt.figure(figsize=(5 * cols, 4 * nrows))
    for idx, col in enumerate(features, 1):
        plt.subplot(nrows, cols, idx)
        sns.histplot(df, x=col, hue='target', kde=True, bins=30)
        plt.title(f'Distribution of {col}')
    plt.tight_layout()
    path = os.path.join(VIS_DIR, 'feature_distributions.png')
    plt.savefig(path)
    plt.close()
    print(f"Saved feature distributions to {path}")


def plot_correlation(df: pd.DataFrame):
    """
    Plot correlation heatmap.
    """
    corr = df.corr()
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr, annot=True, fmt='.2f', cmap='vlag')
    plt.title('Correlation Matrix')
    path = os.path.join(VIS_DIR, 'correlation_matrix.png')
    plt.savefig(path)
    plt.close()
    print(f"Saved correlation matrix to {path}")


# 3. Feature Engineering & Preprocessing

def prepare_features(df: pd.DataFrame):
    """
    Split features and target, apply scaling and power transform.
    """
    X = df.drop('target', axis=1)
    y = df['target']

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    print(f"Train/Test sizes: {X_train.shape}/{X_test.shape}")

    # Scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Optional: Power transform to reduce skewness
    pt = PowerTransformer()
    X_train_scaled = pt.fit_transform(X_train_scaled)
    X_test_scaled = pt.transform(X_test_scaled)
    print("Applied StandardScaler and PowerTransformer.")

    return X_train_scaled, X_test_scaled, y_train, y_test, scaler


# 4. Model Training & Tuning

def train_and_tune(X_train, y_train):
    """
    Tune multiple models via GridSearchCV.
    """
    models = {
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
        'Random Forest': RandomForestClassifier(random_state=RANDOM_STATE),
        'SVM': SVC(probability=True, random_state=RANDOM_STATE),
        'KNN': KNeighborsClassifier()
    }
    param_grids = {
        'Logistic Regression': {'C': [0.01, 0.1, 1, 10], 'solver': ['liblinear']},
        'Random Forest': {'n_estimators': [100, 200], 'max_depth': [None, 5, 10]},
        'SVM': {'C': [0.1, 1, 10], 'kernel': ['linear', 'rbf']},
        'KNN': {'n_neighbors': [3, 5, 7]}
    }

    tuned_models = {}
    for name in models:
        print(f"Tuning {name}...")
        grid = GridSearchCV(
            models[name], param_grids[name], cv=CV_FOLDS, scoring=SCORING, n_jobs=-1
        )
        grid.fit(X_train, y_train)
        tuned_models[name] = grid
        print(f"{name} best f1: {grid.best_score_:.4f} with params: {grid.best_params_}")

    return tuned_models


# 5. Evaluation & Visualization

def evaluate_and_visualize(tuned_models, X_test, y_test, df):

    """
    Evaluate each model on test set, plot metrics and select best.
    """
    best_name = None
    best_f1 = 0

    for name, grid in tuned_models.items():
        model = grid.best_estimator_
        y_pred = model.predict(X_test)
        proba = model.predict_proba(X_test)[:, 1]

        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred)
        rec = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        roc = roc_auc_score(y_test, proba)

        print(f"\n{name} Test Metrics:")
        print(f" Accuracy: {acc:.4f}")
        print(f" Precision: {prec:.4f}")
        print(f" Recall: {rec:.4f}")
        print(f" F1: {f1:.4f}")
        print(f" ROC AUC: {roc:.4f}")

        # Plot and save confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        plt.figure()
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title(f'{name} Confusion Matrix')
        plt.savefig(f'{VIS_DIR}/{name}_confusion_matrix.png')
        plt.close()

        # Plot and save ROC curve
        fpr, tpr, _ = roc_curve(y_test, proba)
        plt.figure()
        plt.plot(fpr, tpr, label=f'AUC = {roc:.3f}')
        plt.plot([0, 1], [0, 1], 'k--')
        plt.title(f'{name} ROC Curve')
        plt.xlabel('FPR')
        plt.ylabel('TPR')
        plt.legend()
        plt.savefig(f'{VIS_DIR}/{name}_roc_curve.png')
        plt.close()

        # Feature importance for tree models
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
            idx = np.argsort(importances)[::-1]
            features = df.columns.drop('target')[idx]
            plt.figure()
            sns.barplot(x=importances[idx], y=features)
            plt.title(f'{name} Feature Importance')
            plt.savefig(f'{VIS_DIR}/{name}_feature_importance.png')
            plt.close()

        priority = {"Logistic Regression": 3, "Random Forest": 2, "SVM": 1, "KNN": 0}
        if (f1 > best_f1 + 0.01) or (abs(f1 - best_f1) <= 0.01 and priority.get(name, 0) > priority.get(best_name, 0)):
            best_f1 = f1
            best_name = name

    print(f"\nBest model: {best_name} with F1 score {best_f1:.4f}")
    return best_name, tuned_models[best_name]


# 6. Save Artifacts

def save_artifacts(best_name,best_grid,scaler):
    model=best_grid.best_estimator_
    joblib.dump(model, f"{MODEL_DIR}/{best_name.replace(' ', '_')}.pkl")
    joblib.dump(scaler, f"{MODEL_DIR}/scaler.pkl")


# Main Execution

def main():
    # Step 1: Load & clean
    df = load_and_clean_data(DATA_PATH)

    # Step 2: EDA
    plot_distribution(df)
    plot_correlation(df)

    # Step 3: Prepare features
    X_train, X_test, y_train, y_test, scaler = prepare_features(df)

    # Step 4: Train & tune
    tuned_models = train_and_tune(X_train, y_train)

    # Step 5: Evaluate all models
    evaluate_and_visualize(tuned_models, X_test, y_test, df)  # just print all results

    # 💡 Manually select the model you trust
    best_name = "Logistic Regression"
    best_grid = tuned_models[best_name]
    print(f"\nManually selected model: {best_name}")

    # Step 7: Save artifacts
    save_artifacts(best_name, best_grid, scaler)
    print(f"\nSVM may have had slightly better F1, but Logistic Regression was selected manually for interpretability.")


    

if __name__ == '__main__':
    main()
