import streamlit as st
import requests
import os
from PIL import Image

st.set_page_config(page_title="Heart Disease Predictor", layout="wide")

# Add sidebar navigation for Help page
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to:", ["Prediction", "Help"])

if page == "Help":
    st.title("Help: Input Parameter Explanations")
    st.markdown("""
    ### Heart Disease Prediction Input Parameter Explanations
    
    - **Age:** The age of the patient in years.
    - **Sex:** The biological sex of the patient.  
      - Male: 1  
      
      - Female: 0
    - **Chest Pain Type (cp):** The type of chest pain experienced by the patient.  
      - 0: Typical angina – Chest pain related to decreased blood supply to the heart  
      - 1: Atypical angina – Chest pain not related to heart  
      - 2: Non-anginal pain – Chest pain not related to angina  
      - 3: Asymptomatic – No chest pain
    - **Resting Blood Pressure (trestbps):** The patient's resting blood pressure (in mm Hg) when admitted to the hospital.
    - **Serum Cholesterol (chol):** The patient's cholesterol level in mg/dL.
    - **Fasting Blood Sugar > 120 mg/dl (fbs):** Whether the patient's fasting blood sugar is greater than 120 mg/dL.  
      - Yes: 1  
      - No: 0
    - **Resting ECG Results (restecg):** Results of the patient's resting electrocardiogram.  
      - 0: Normal  
      - 1: ST-T wave abnormality – Possible sign of heart disease  
      - 2: Left ventricular hypertrophy – Thickening of the heart muscle
    - **Max Heart Rate Achieved (thalach):** The maximum heart rate achieved during exercise.
    - **Exercise Induced Angina (exang):** Whether the patient experienced angina (chest pain) induced by exercise.  
      - Yes: 1  
      - No: 0
    - **ST Depression (oldpeak):** The amount of ST depression induced by exercise relative to rest, measured in millimeters. Indicates possible heart stress.
    - **Slope of Peak Exercise ST Segment (slope):** The slope of the peak exercise ST segment.  
      - 0: Upsloping – Better heart function  
      - 1: Flat – Possible abnormality  
      - 2: Downsloping – Higher risk of heart disease
    - **Number of Major Vessels Colored by Fluoroscopy (ca):** The number of major blood vessels (0–3) colored by a special dye during a heart scan.
    - **Thalassemia (thal):** A blood disorder detected by a test.  
      - 1: Normal  
      - 2: Fixed defect – Defect present, not changing with exercise  
      - 3: Reversible defect – Defect that changes with exercise
    """)
    st.stop()

# Main prediction page
st.title("💓 Heart Disease Prediction")
# Add a help note at the top
st.info("Need help understanding the parameters? See the Help page in the sidebar.")

# Configuration

API_URL = "http://127.0.0.1:8800/predict"
IMG_DIR = "img"

# Input Form

with st.form("prediction_form"):
    age = st.slider("Age", 18, 100, 50, help="Age in years.")
    sex_option = st.selectbox("Sex", ["Male", "Female"], help="Sex (1 = male; 0 = female)")
    cp = st.selectbox(
        "Chest Pain Type",
        [
            "0: Typical angina",
            "1: Atypical angina",
            "2: Non-anginal pain",
            "3: Asymptomatic"
        ],
        help="Chest pain type (0-3: typical, atypical, non-anginal, asymptomatic)"
    )
    trestbps = st.slider("Resting Blood Pressure (mm Hg)", 80, 200, 120, help="Resting blood pressure in mm Hg.")
    chol = st.slider("Serum Cholesterol (mg/dL)", 100, 600, 200, help="Serum cholesterol in mg/dl.")
    fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl", ["No", "Yes"], help="Fasting blood sugar > 120 mg/dl (1 = true; 0 = false)")
    restecg = st.selectbox(
        "Resting ECG Results",
        [
            "0: Normal",
            "1: ST-T wave abnormality",
            "2: Left ventricular hypertrophy"
        ],
        help="Resting electrocardiographic results (0-2)"
    )
    thalach = st.slider("Max Heart Rate Achieved", 60, 250, 150, help="Maximum heart rate achieved.")
    exang = st.selectbox("Exercise Induced Angina", ["No", "Yes"], help="Exercise induced angina (1 = yes; 0 = no)")
    oldpeak = st.slider("ST Depression (Oldpeak)", 0.0, 6.0, 1.0, step=0.1, help="ST depression induced by exercise relative to rest.")
    slope = st.selectbox(
        "Slope of Peak Exercise ST Segment",
        [
            "0: Upsloping",
            "1: Flat",
            "2: Downsloping"
        ],
        help="Slope of the peak exercise ST segment (0-2)"
    )
    ca = st.slider("Number of Major Vessels Colored by Fluoroscopy", 0, 3, 0, help="Number of major vessels (0-3) colored by fluoroscopy.")
    thal = st.selectbox(
        "Thalassemia",
        [
            "1: Normal",
            "2: Fixed defect",
            "3: Reversible defect"
        ],
        help="Thalassemia (1 = normal; 2 = fixed defect; 3 = reversible defect)"
    )
    submitted = st.form_submit_button("🔍 Predict")


# Prediction Logic

if submitted:
    payload = {
        "age": age,
        "sex": 1 if sex_option == "Male" else 0,
        "cp": int(cp.split(":")[0]),
        "trestbps": trestbps,
        "chol": chol,
        "fbs": 1 if fbs == "Yes" else 0,
        "restecg": int(restecg.split(":")[0]),
        "thalach": thalach,
        "exang": 1 if exang == "Yes" else 0,
        "oldpeak": oldpeak,
        "slope": int(slope.split(":")[0]),
        "ca": ca,
        "thal": int(thal.split(":")[0])
    }

    with st.spinner("Calling the model..."):
        try:
            response = requests.post(API_URL, json=payload)
            result = response.json()

            if "error" in result:
                st.error(f"❌ {result['error']}")
            else:
                prediction = "🚨 High Risk" if result["prediction"] == 1 else "✅ Low Risk"
                probability = round(result["probability"] * 100, 2)

                st.success(f"**Prediction:** {prediction}")
                st.metric(label="Probability of Heart Disease", value=f"{probability}%")

                st.divider()
                st.subheader("📊 Model Visualizations")

                # Explanations for each chart
                st.markdown("""
                **Chart Explanations:**
                - **Risk Gauge**: Shows your probability of heart disease as a semicircular gauge.
                - **Feature Comparison**: Compares your health metrics to healthy and disease populations.
                - **Probability Meter**: Visualizes your risk probability and interpretation.
                - **Risk Factors (Donut)**: Highlights your individual risk factors as a heatmap.
                - **Feature Importance (Bar)**: Shows which features most influenced the prediction.
                - **Patient Profile (Boxplot)**: Summarizes your health profile and key statistics.
                """)

                # chart filenames expected from the backend
                image_map = {
                    "Risk Gauge": "prediction_radar.png",
                    "Feature Comparison": "prediction_scatter.png",
                    "Probability Meter": "prediction_pie.png",
                    "Risk Factors (Donut)": "prediction_donut.png",
                    "Feature Importance (Bar)": "prediction_bar.png",
                    "Patient Profile (Boxplot)": "prediction_boxplot.png"
                }

                # First row: Risk Gauge | Feature Comparison | Probability Meter
                row1 = st.columns(3)
                for i, key in enumerate(["Risk Gauge", "Feature Comparison", "Probability Meter"]):
                    path = os.path.join(IMG_DIR, image_map[key])
                    if os.path.exists(path):
                        with row1[i]:
                            st.markdown(f"**{key}**")
                            st.image(Image.open(path), use_container_width=True)
                    else:
                        with row1[i]:
                            st.warning(f"{key} chart not found.")

                # Second row: Feature Importance (Bar) | Risk Factors (Donut) | Patient Profile (Boxplot)
                row2 = st.columns(3)
                for i, key in enumerate(["Feature Importance (Bar)", "Risk Factors (Donut)", "Patient Profile (Boxplot)"]):
                    path = os.path.join(IMG_DIR, image_map[key])
                    if os.path.exists(path):
                        with row2[i]:
                            st.markdown(f"**{key}**")
                            st.image(Image.open(path), use_container_width=True)
                    else:
                        with row2[i]:
                            st.warning(f"{key} chart not found.")

        except Exception as e:
            st.error(f"Request failed: {e}")
