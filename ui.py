import streamlit as st
import requests
import os
from PIL import Image


# Configuration

API_URL = "https://checkyourheart.streamlit.app"
IMG_DIR = "img"
st.set_page_config(page_title="Heart Disease Predictor", layout="wide")


# Title and Instructions

st.title("💓 Heart Disease Prediction")
st.markdown("Use the sliders below to enter patient data and get a risk prediction.")


# Input Form

with st.form("prediction_form"):
    sex_option = st.selectbox("Sex", ["Male", "Female"])
    age = st.slider("Age", 18, 100, 50)
    trestbps = st.slider("Resting Blood Pressure (mm Hg)", 80, 200, 120)
    chol = st.slider("Serum Cholesterol (mg/dL)", 100, 600, 200)
    thalach = st.slider("Max Heart Rate Achieved", 60, 250, 150)
    oldpeak = st.slider("ST Depression (Oldpeak)", 0.0, 6.0, 1.0, step=0.1)
    submitted = st.form_submit_button("🔍 Predict")


# Prediction Logic

if submitted:
    payload = {
        "age": age,
        "sex": 1 if sex_option == "Male" else 0,
        "cp": 0,
        "trestbps": trestbps,
        "chol": chol,
        "fbs": 0,
        "restecg": 0,
        "thalach": thalach,
        "exang": 0,
        "oldpeak": oldpeak,
        "slope": 1,
        "ca": 0,
        "thal": 2
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

                # chart filenames expected from the backend
                image_map = {
                    "Radar": "prediction_radar.png",
                    "Scatter": "prediction_scatter.png",
                    "Pie": "prediction_pie.png",
                    "Bar": "prediction_bar.png",
                    "Donut": "prediction_donut.png",
                    "Box": "prediction_boxplot.png"
                }

                # First row: Radar | Scatter | Pie
                row1 = st.columns(3)
                for i, key in enumerate(["Radar", "Scatter", "Pie"]):
                    path = os.path.join(IMG_DIR, image_map[key])
                    if os.path.exists(path):
                        with row1[i]:
                            st.markdown(f"**{key} Chart**")
                            st.image(Image.open(path), use_container_width=True)
                    else:
                        with row1[i]:
                            st.warning(f"{key} chart not found.")

                # Second row: Bar | Donut | Box
                row2 = st.columns(3)
                for i, key in enumerate(["Bar", "Donut", "Box"]):
                    path = os.path.join(IMG_DIR, image_map[key])
                    if os.path.exists(path):
                        with row2[i]:
                            st.markdown(f"**{key} Chart**")
                            st.image(Image.open(path), use_container_width=True)
                    else:
                        with row2[i]:
                            st.warning(f"{key} chart not found.")

        except Exception as e:
            st.error(f"Request failed: {e}")
