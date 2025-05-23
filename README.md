# 💓 Heart Disease Prediction App

This project is an end-to-end machine learning solution that predicts the probability of heart disease based on user input. It integrates a Flask backend for real-time inference and a Streamlit frontend for a clean, interactive user interface with visualizations.

---

## 🧠 Model Overview

- **Algorithm Used:** Logistic Regression (manually selected as final model)
- **Dataset:** [Cleveland Heart Disease dataset](https://www.kaggle.com/datasets/cherngs/heart-disease-cleveland-uci)
- **Preprocessing:**
  - Outlier capping using IQR method
  - Standard Scaling + Power Transformation
  - GridSearchCV hyperparameter tuning
- **Performance:**
  - F1 Score: **0.83**
  - ROC AUC: **0.95**

---

## 🌐 Technologies Used

| Layer        | Tech Stack                                      |
|--------------|--------------------------------------------------|
| Frontend     | Streamlit                                        |
| Backend      | Flask + Swagger for REST API                    |
| ML Models    | Scikit-learn (Logistic Regression, SVM, RF, KNN)|
| Deployment   | Flask via Render, Streamlit via Streamlit Cloud |
| Visualization| Matplotlib, Seaborn, PIL                        |

---

## 🖥️ Live Demo UI

### 🔍 Input Panel + Risk Prediction Output

![Prediction UI](./Screenshot%202025-05-22%20at%205.15.53%E2%80%AFPM.png)

- Users can select values using interactive sliders.
- Probability and risk level are displayed after hitting "Predict".

---

### 📊 Real-time Model Visualizations

![Visualizations](./Screenshot%202025-05-22%20at%205.16.07%E2%80%AFPM.png)

Includes:
- Radar Chart
- Probability vs BP Scatter Plot
- Pie & Donut Chart
- Feature Importance Bar Chart
- Top Feature Box Plots

All generated dynamically on each prediction.

---

## ⚙️ How to Run Locally
- Clone the Repsitory
- Download required modules (pip install -r requirements.txt)
- Run locally by opening two termianls
-  1. python app.py
-  2. streamlit run ui.py

