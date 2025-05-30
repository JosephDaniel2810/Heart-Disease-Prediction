from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import numpy as np
import pandas as pd
import os
import joblib
import matplotlib
matplotlib.use('Agg')  # For headless server
import matplotlib.pyplot as plt
import seaborn as sns

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

def set_plot_style():
    try:
        plt.style.use('seaborn-v0_8-darkgrid')
    except:
        plt.style.use('seaborn-darkgrid')
    plt.rcParams.update({
        'axes.titlesize': 18,
        'axes.labelsize': 14,
        'xtick.labelsize': 12,
        'ytick.labelsize': 12,
        'legend.fontsize': 12,
        'figure.titlesize': 20,
        'figure.facecolor': 'white',
        'axes.facecolor': '#f8f9fa',
        'axes.edgecolor': '#dee2e6',
        'axes.linewidth': 1.5,
        'grid.color': '#dee2e6',
        'grid.linewidth': 0.8,
        'font.family': 'DejaVu Sans',
        'font.sans-serif': 'DejaVu Sans',
    })

def generate_visuals(scaled_input, raw_input, prob, pred):
    set_plot_style()
    generate_risk_gauge(prob, pred)
    generate_feature_comparison(scaled_input, raw_input)
    generate_probability_meter(prob)
    generate_risk_factors(scaled_input, raw_input)
    generate_feature_importance_modern(raw_input)
    generate_patient_profile(scaled_input, raw_input)

def generate_risk_gauge(prob, pred):
    """Modern risk gauge showing probability as a semicircular gauge"""
    fig, ax = plt.subplots(figsize=(10, 6), subplot_kw={'projection': 'polar'})
    
    # Create semicircle gauge
    theta = np.linspace(0, np.pi, 100)
    r = np.ones_like(theta)
    
    # Background sections
    sections = [0, 0.3, 0.7, 1.0]  # Low, Medium, High risk thresholds
    colors = ['#2ecc71', '#f39c12', '#e74c3c']
    labels = ['Low Risk', 'Medium Risk', 'High Risk']
    
    for i in range(len(sections)-1):
        theta_section = np.linspace(sections[i]*np.pi, sections[i+1]*np.pi, 50)
        ax.fill_between(theta_section, 0, 1, color=colors[i], alpha=0.3)
        mid_angle = (sections[i] + sections[i+1]) * np.pi / 2
        ax.text(mid_angle, 1.15, labels[i], ha='center', va='center', fontsize=14, fontweight='bold')
    
    # Pointer for patient's probability
    pointer_angle = prob * np.pi
    ax.plot([pointer_angle, pointer_angle], [0, 0.9], color='black', linewidth=4)
    ax.scatter(pointer_angle, 0.9, s=200, color='black', zorder=5)
    
    # Clean up the plot
    ax.set_ylim(0, 1.3)
    ax.set_theta_offset(np.pi)
    ax.set_theta_direction(-1)
    ax.axis('off')
    
    # Title and probability text
    fig.suptitle(f'Heart Disease Risk Assessment', fontsize=22, fontweight='bold', y=0.98)
    ax.text(np.pi/2, -0.3, f'{prob*100:.1f}%', fontsize=48, fontweight='bold', ha='center', va='center',
            bbox=dict(boxstyle="round,pad=0.3", facecolor='white', edgecolor='black', linewidth=2), fontname='DejaVu Sans')
    ax.text(np.pi/2, -0.5, 'Probability of Heart Disease', fontsize=16, ha='center', va='center', fontname='DejaVu Sans')
    
    # Use subplots_adjust instead of tight_layout
    plt.subplots_adjust(left=0.08, right=0.98, top=0.92, bottom=0.18)
    plt.savefig(os.path.join(IMG_DIR, "prediction_radar.png"), dpi=150, bbox_inches='tight')
    plt.close()

def generate_feature_comparison(scaled_input, raw_input):
    """Compare patient's features to healthy/disease populations"""
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    # Select 6 most important features
    if hasattr(model, "coef_"):
        importance_idx = np.argsort(np.abs(model.coef_[0]))[::-1][:6]
    else:
        importance_idx = range(6)
    
    for i, idx in enumerate(importance_idx):
        ax = axes[i]
        feature = feature_names[idx]
        
        # Get distributions for healthy and disease populations
        healthy_vals = X_train[y_train == 0, idx]
        disease_vals = X_train[y_train == 1, idx]
        patient_val = scaled_input[0, idx]
        
        # Create violin plots
        parts = ax.violinplot([healthy_vals, disease_vals], positions=[1, 2], widths=0.7,
                             showmeans=True, showmedians=True)
        
        # Color the violins
        colors = ['#3498db', '#e74c3c']
        for pc, color in zip(parts['bodies'], colors):
            pc.set_facecolor(color)
            pc.set_alpha(0.6)
            pc.set_edgecolor('black')
            pc.set_linewidth(1.5)
        
        # Add patient's value
        ax.scatter([1.5], [patient_val], s=300, c='#f39c12', marker='*', 
                  edgecolors='black', linewidth=2, zorder=10, label='You')
        ax.hlines(patient_val, 0.5, 2.5, colors='#f39c12', linestyles='dashed', linewidth=2)
        
        # Styling
        ax.set_xticks([1, 2])
        ax.set_xticklabels(['Healthy', 'Disease'], fontsize=12)
        ax.set_title(f'{feature.upper()}', fontsize=14, fontweight='bold', pad=10)
        ax.set_ylabel('Scaled Value', fontsize=12)
        ax.grid(True, alpha=0.3)
        
        # Add legend only to first plot
        if i == 0:
            ax.legend(loc='upper right', fontsize=12)
    
    fig.suptitle('Your Health Metrics vs. Population Distribution', fontsize=20, fontweight='bold', y=0.98)
    # Use subplots_adjust instead of tight_layout
    plt.subplots_adjust(left=0.08, right=0.98, top=0.92, bottom=0.18)
    plt.savefig(os.path.join(IMG_DIR, "prediction_scatter.png"), dpi=150, bbox_inches='tight')
    plt.close()

def generate_probability_meter(prob):
    """Modern probability display with context"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Left: Probability bar with zones
    ax1.barh([''], [prob], color='#e74c3c' if prob > 0.5 else '#3498db', height=0.5, alpha=0.8)
    ax1.barh([''], [1-prob], left=[prob], color='#ecf0f1', height=0.5, alpha=0.8)
    
    # Add zone backgrounds
    zones = [(0, 0.3, '#2ecc71', 'Low Risk'), 
             (0.3, 0.7, '#f39c12', 'Moderate Risk'), 
             (0.7, 1.0, '#e74c3c', 'High Risk')]
    
    for start, end, color, label in zones:
        ax1.axvspan(start, end, alpha=0.2, color=color)
        ax1.text((start+end)/2, -0.15, label, ha='center', va='top', fontsize=12, fontweight='bold', fontname='DejaVu Sans')
    
    # Add percentage text
    ax1.text(prob/2, 0, f'{prob*100:.1f}%', ha='center', va='center', 
            fontsize=24, fontweight='bold', color='white', fontname='DejaVu Sans')
    ax1.text(prob + (1-prob)/2, 0, f'{(1-prob)*100:.1f}%', ha='center', va='center', 
            fontsize=24, fontweight='bold', color='#7f8c8d', fontname='DejaVu Sans')
    
    ax1.set_xlim(0, 1)
    ax1.set_ylim(-0.3, 0.3)
    ax1.set_xlabel('Probability', fontsize=14)
    ax1.set_title('Disease Risk Probability', fontsize=16, fontweight='bold', pad=20)
    ax1.set_yticks([])
    
    # Right: Interpretation guide
    ax2.axis('off')
    interpretation = "High Risk" if prob > 0.7 else "Moderate Risk" if prob > 0.3 else "Low Risk"
    color = '#e74c3c' if prob > 0.7 else '#f39c12' if prob > 0.3 else '#2ecc71'
    
    ax2.text(0.5, 0.8, 'Risk Interpretation', fontsize=20, fontweight='bold', ha='center', fontname='DejaVu Sans')
    ax2.text(0.5, 0.6, interpretation, fontsize=36, fontweight='bold', ha='center', color=color, fontname='DejaVu Sans')
    
    recommendations = {
        "High Risk": "• Immediate medical consultation recommended\n• Regular monitoring essential\n• Lifestyle modifications crucial",
        "Moderate Risk": "• Schedule a check-up soon\n• Monitor risk factors\n• Consider lifestyle improvements",
        "Low Risk": "• Maintain healthy lifestyle\n• Regular health check-ups\n• Continue preventive measures"
    }
    
    ax2.text(0.5, 0.3, recommendations[interpretation], fontsize=14, ha='center', va='center',
            bbox=dict(boxstyle="round,pad=0.5", facecolor='#ecf0f1', edgecolor='gray'), fontname='DejaVu Sans')
    
    # Use subplots_adjust instead of tight_layout
    plt.subplots_adjust(left=0.08, right=0.98, top=0.92, bottom=0.18)
    plt.savefig(os.path.join(IMG_DIR, "prediction_pie.png"), dpi=150, bbox_inches='tight')
    plt.close()

def generate_risk_factors(scaled_input, raw_input):
    """Heatmap showing patient's risk factors"""
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Normalize features to 0-1 range for visualization
    feature_values = []
    feature_labels = []
    risk_levels = []
    
    for i, feature in enumerate(feature_names):
        val = scaled_input[0, i]
        # Convert to percentile
        percentile = (val > X_train[:, i]).mean()
        feature_values.append(percentile)
        feature_labels.append(feature)
        
        # Determine risk level based on percentile and feature
        if percentile > 0.75:
            risk_levels.append(3)  # High
        elif percentile > 0.5:
            risk_levels.append(2)  # Medium
        elif percentile > 0.25:
            risk_levels.append(1)  # Low
        else:
            risk_levels.append(0)  # Very Low
    
    # Create heatmap data
    data = np.array(feature_values).reshape(-1, 1)
    
    # Create custom colormap
    colors = ['#2ecc71', '#f1c40f', '#e67e22', '#e74c3c']
    n_bins = 100
    cmap = sns.blend_palette(colors, n_colors=n_bins, as_cmap=True)
    
    # Plot heatmap
    sns.heatmap(data.T, xticklabels=feature_labels, yticklabels=['Risk Level'],
                cmap=cmap, cbar_kws={'label': 'Percentile vs. Population'},
                annot=[[f'{v*100:.0f}%' for v in feature_values]], fmt='',
                annot_kws={'fontsize': 12, 'fontweight': 'bold'},
                linewidths=2, linecolor='white')
    
    ax.set_title('Individual Risk Factor Analysis', fontsize=18, fontweight='bold', pad=20)
    ax.set_xlabel('Health Metrics', fontsize=14)
    plt.xticks(rotation=45, ha='right')
    
    # Add risk level indicators
    for i, (feature, risk) in enumerate(zip(feature_labels, risk_levels)):
        # Use a supported character instead of the check mark to avoid font warnings
        risk_text = ['✔', '!', '!!', '!!!'][risk]  # '✔' is more widely supported
        risk_color = colors[risk]
        # Ensure i + 0.5 is finite
        if np.isfinite(i + 0.5):
            ax.text(i + 0.5, -0.1, risk_text, ha='center', va='top', 
                   fontsize=16, fontweight='bold', color=risk_color, fontname='DejaVu Sans')
    # Use subplots_adjust instead of tight_layout to avoid warnings
    plt.subplots_adjust(left=0.08, right=0.98, top=0.92, bottom=0.18)
    plt.savefig(os.path.join(IMG_DIR, "prediction_donut.png"), dpi=150, bbox_inches='tight')
    plt.close()

def generate_feature_importance_modern(raw_input):
    """Modern feature importance with patient values"""
    fig, ax = plt.subplots(figsize=(10, 8))
    
    if hasattr(model, "coef_"):
        importances = np.abs(model.coef_[0])
    else:
        importances = np.ones(len(feature_names))
    
    # Sort features by importance
    sorted_idx = np.argsort(importances)
    sorted_features = [feature_names[i] for i in sorted_idx]
    sorted_importances = importances[sorted_idx]
    
    # Create horizontal bar chart
    bars = ax.barh(range(len(sorted_features)), sorted_importances)
    
    # Color bars with gradient
    norm = plt.Normalize(vmin=min(sorted_importances), vmax=max(sorted_importances))
    sm = plt.cm.ScalarMappable(cmap='viridis', norm=norm)
    sm.set_array([])
    
    for bar, importance in zip(bars, sorted_importances):
        bar.set_color(sm.to_rgba(importance))
        bar.set_edgecolor('black')
        bar.set_linewidth(1.5)
    
    # Add feature names and importance values
    for i, (feature, importance) in enumerate(zip(sorted_features, sorted_importances)):
        ax.text(importance + max(sorted_importances)*0.01, i, f'{importance:.3f}', 
               va='center', fontsize=12, fontweight='bold', fontname='DejaVu Sans')
    
    ax.set_yticks(range(len(sorted_features)))
    ax.set_yticklabels(sorted_features, fontsize=12)
    ax.set_xlabel('Importance Score', fontsize=14)
    ax.set_title('Feature Importance in Prediction Model', fontsize=18, fontweight='bold', pad=20)
    ax.grid(True, axis='x', alpha=0.3)
    
    # Add colorbar
    cbar = plt.colorbar(sm, ax=ax, pad=0.01)
    cbar.set_label('Importance Level', fontsize=12)
    
    # Use subplots_adjust instead of tight_layout
    plt.subplots_adjust(left=0.12, right=0.98, top=0.92, bottom=0.12)
    plt.savefig(os.path.join(IMG_DIR, "prediction_bar.png"), dpi=150, bbox_inches='tight')
    plt.close()

def generate_patient_profile(scaled_input, raw_input):
    """Comprehensive patient profile visualization"""
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
    
    # Main radar plot (2x2)
    ax_radar = fig.add_subplot(gs[:2, :2], projection='polar')
    
    # Feature values normalized to 0-1
    values = []
    for i in range(len(feature_names)):
        val = scaled_input[0, i]
        percentile = (val > X_train[:, i]).mean()
        values.append(percentile)
    
    # Radar plot
    angles = np.linspace(0, 2 * np.pi, len(feature_names), endpoint=False).tolist()
    values_plot = values + values[:1]
    angles += angles[:1]
    
    ax_radar.plot(angles, values_plot, 'o-', linewidth=2, color='#3498db', markersize=8)
    ax_radar.fill(angles, values_plot, alpha=0.25, color='#3498db')
    
    # Add reference lines
    for ref in [0.25, 0.5, 0.75]:
        ax_radar.plot(angles, [ref] * len(angles), '--', color='gray', alpha=0.5, linewidth=1)
    
    ax_radar.set_xticks(angles[:-1])
    ax_radar.set_xticklabels(feature_names, fontsize=11)
    ax_radar.set_ylim(0, 1)
    ax_radar.set_title('Patient Health Profile', fontsize=16, fontweight='bold', pad=20, fontname='DejaVu Sans')
    
    # Top features (top right)
    ax_top = fig.add_subplot(gs[0, 2])
    top_3_idx = np.argsort(np.abs(model.coef_[0]))[::-1][:3]
    top_features = [feature_names[i] for i in top_3_idx]
    top_values = [values[i] for i in top_3_idx]
    
    bars = ax_top.bar(range(3), top_values, color=['#e74c3c', '#f39c12', '#3498db'])
    ax_top.set_xticks(range(3))
    ax_top.set_xticklabels(top_features, rotation=45, ha='right')
    ax_top.set_ylim(0, 1)
    ax_top.set_ylabel('Percentile')
    ax_top.set_title('Key Risk Factors', fontsize=14, fontweight='bold', fontname='DejaVu Sans')
    ax_top.grid(True, axis='y', alpha=0.3)
    
    # Risk summary (middle right)
    ax_summary = fig.add_subplot(gs[1, 2])
    ax_summary.axis('off')
    
    prob = model.predict_proba(scaled_input)[0][1]
    risk_level = "High" if prob > 0.7 else "Moderate" if prob > 0.3 else "Low"
    risk_color = '#e74c3c' if prob > 0.7 else '#f39c12' if prob > 0.3 else '#2ecc71'
    
    ax_summary.text(0.5, 0.8, 'Overall Risk', fontsize=14, ha='center', fontweight='bold', fontname='DejaVu Sans')
    ax_summary.text(0.5, 0.5, risk_level, fontsize=24, ha='center', fontweight='bold', color=risk_color, fontname='DejaVu Sans')
    ax_summary.text(0.5, 0.2, f'{prob*100:.1f}%', fontsize=20, ha='center', fontname='DejaVu Sans')
    
    # Statistics (bottom)
    ax_stats = fig.add_subplot(gs[2, :])
    ax_stats.axis('off')
    
    # Create summary statistics table
    stats_text = "Patient Summary Statistics:\n"
    for i, feature in enumerate(feature_names[:6]):  # Show first 6 features
        val = scaled_input[0, i]
        percentile = values[i] * 100
        stats_text += f"{feature}: {percentile:.0f}th percentile  "
        if (i + 1) % 3 == 0:
            stats_text += "\n"
    
    ax_stats.text(0.5, 0.5, stats_text, fontsize=12, ha='center', va='center',
                 bbox=dict(boxstyle="round,pad=0.5", facecolor='#ecf0f1', edgecolor='gray'), fontname='DejaVu Sans')
    
    fig.suptitle('Comprehensive Patient Assessment Report', fontsize=20, fontweight='bold', fontname='DejaVu Sans')
    # Use subplots_adjust instead of tight_layout
    plt.subplots_adjust(left=0.06, right=0.98, top=0.92, bottom=0.10)
    plt.savefig(os.path.join(IMG_DIR, "prediction_boxplot.png"), dpi=150, bbox_inches='tight')
    plt.close()

# ---------------------------------------------
# Run Server
# ---------------------------------------------
if __name__ == "__main__":
    from waitress import serve
    serve(app, host="0.0.0.0", port=8800)
