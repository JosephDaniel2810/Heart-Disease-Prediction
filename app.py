from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import numpy as np
import pandas as pd
import os
import joblib
import matplotlib
matplotlib.use('Agg')  # For headless server
import matplotlib.pyplot as plt

# Initialize app
app = Flask(__name__)
CORS(app)

# Load model, scaler, training data
model = joblib.load("models/Logistic_Regression.pkl")
scaler = joblib.load("models/scaler.pkl")
training_data = joblib.load("models/training_data.pkl")

X_train = training_data["X_train"]
y_train = training_data["y_train"]
feature_names = training_data["feature_names"]

IMG_DIR = "img"
os.makedirs(IMG_DIR, exist_ok=True)

@app.route("/")
def home():
    return """
    <h2>Heart Disease Prediction API</h2>
    <p>Send a POST request to <code>/predict</code> with patient data.</p>
    """

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json(force=True)

        required_features = feature_names
        if not all(feature in data for feature in required_features):
            return jsonify({
                "error": "Missing features. Required: " + ", ".join(required_features)
            }), 400

        # Convert input to DataFrame
        input_df = pd.DataFrame([data])[required_features]

        # Scale
        scaled_input = scaler.transform(input_df)

        # Predict
        pred = model.predict(scaled_input)[0]
        prob = model.predict_proba(scaled_input)[0][1]

        # Save visualizations
        generate_visuals(scaled_input, input_df, prob, pred)

        return jsonify({
            "prediction": int(pred),
            "probability": float(prob),
            "model": "Logistic Regression",
            "visualizations": {
                "radar_chart": "prediction_radar.png",
                "scatter_plot": "prediction_scatter.png",
                "pie_chart": "prediction_pie.png",
                "donut_chart": "prediction_donut.png",
                "bar_chart": "prediction_bar.png",
                "box_plot": "prediction_boxplot.png"
            }
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/img/<filename>")
def get_image(filename):
    return send_from_directory(IMG_DIR, filename)

# ---------------------------------------------
# Visualization Functions
# ---------------------------------------------

def generate_visuals(scaled_input, raw_input, prob, pred):
    generate_radar(prob, pred)
    generate_scatter(prob, scaled_input)
    generate_pie(prob)
    generate_donut(prob)
    generate_bar(raw_input)
    generate_box(scaled_input)

def generate_radar(prob, pred):
    labels = ['Probability', 'Prediction', 'Risk', 'Confidence']
    values = [prob, pred, 0.7, 0.9]
    baseline = [0.5, 0.5, 0.5, 0.5]

    values += values[:1]
    baseline += baseline[:1]
    angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(subplot_kw={'polar': True})
    ax.plot(angles, values, label='Current', color='r')
    ax.fill(angles, values, alpha=0.25, color='r')
    ax.plot(angles, baseline, label='Baseline', linestyle='--', color='b')
    ax.set_thetagrids(np.degrees(angles[:-1]), labels)
    ax.set_title('Radar Chart')
    ax.legend()
    plt.savefig(os.path.join(IMG_DIR, "prediction_radar.png"))
    plt.close()

def generate_scatter(prob, scaled_input):
    glucose_index = feature_names.index("trestbps")
    x = model.predict_proba(X_train)[:, 1]
    y = X_train[:, glucose_index]
    fig, ax = plt.subplots()
    ax.scatter(x, y, c=y_train, cmap='coolwarm', alpha=0.6, label='Training Data')
    ax.scatter(prob, scaled_input[0, glucose_index], c='gold', edgecolors='black', s=150, marker='*', label='Patient')
    ax.axvline(x=0.5, color='gray', linestyle='--')
    ax.set_xlabel("Predicted Probability")
    ax.set_ylabel("Blood Pressure (scaled)")
    ax.set_title("Scatter: Probability vs BP")
    ax.legend()
    plt.savefig(os.path.join(IMG_DIR, "prediction_scatter.png"))
    plt.close()

def generate_pie(prob):
    fig, ax = plt.subplots()
    ax.pie([prob, 1 - prob], labels=["Disease", "No Disease"], autopct="%1.1f%%", colors=['red', 'green'])
    ax.set_title("Prediction Pie")
    plt.savefig(os.path.join(IMG_DIR, "prediction_pie.png"))
    plt.close()

def generate_donut(prob):
    fig, ax = plt.subplots()
    wedges, texts, autotexts = ax.pie([prob, 1 - prob], labels=["Disease", "No Disease"], autopct="%1.1f%%", startangle=90)
    centre = plt.Circle((0, 0), 0.70, fc='white')
    fig.gca().add_artist(centre)
    ax.set_title("Prediction Donut")
    plt.savefig(os.path.join(IMG_DIR, "prediction_donut.png"))
    plt.close()

def generate_bar(raw_input):
    if hasattr(model, "coef_"):
        importances = np.abs(model.coef_[0])
    else:
        importances = np.ones(len(feature_names))

    sorted_idx = np.argsort(importances)[::-1]
    fig, ax = plt.subplots()
    bars = ax.barh([feature_names[i] for i in sorted_idx], importances[sorted_idx], color='skyblue')
    ax.set_title("Feature Importance")
    ax.invert_yaxis()
    plt.savefig(os.path.join(IMG_DIR, "prediction_bar.png"))
    plt.close()

def generate_box(scaled_input):
    top_features = np.argsort(np.abs(model.coef_[0]))[::-1][:4]
    fig, axs = plt.subplots(2, 2, figsize=(10, 6))
    axs = axs.flatten()
    for i, idx in enumerate(top_features):
        axs[i].boxplot([X_train[y_train==0, idx], X_train[y_train==1, idx]], labels=["No Disease", "Disease"])
        axs[i].scatter([1, 2], [scaled_input[0, idx]]*2, color='red', marker='*', label='Patient')
        axs[i].set_title(f"{feature_names[idx]}")
    plt.tight_layout()
    plt.savefig(os.path.join(IMG_DIR, "prediction_boxplot.png"))
    plt.close()

# ---------------------------------------------
# Run Server
# ---------------------------------------------
if __name__ == "__main__":
    from waitress import serve
    serve(app, host="0.0.0.0", port=8800)
