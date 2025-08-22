#!/usr/bin/python3
# -*- coding: UTF-8 -*-
#!/usr/bin/python3
# -*- coding: UTF-8 -*-

# =======================
# Imports
# =======================
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go
import io
from pathlib import Path
import json

# =======================
# ✅ Must be the first Streamlit command
# =======================
st.set_page_config(page_title="Loan Interest Rate Prediction System", page_icon="💹")

# =======================
# Paths (adjust if needed)
# =======================
MODEL_PATH    = "loan_rate_pipeline.pkl"   # Preprocessing + model pipeline
FEATURES_PATH = "features.pkl"             # Feature order used during training
MEDIANS_PATH  = "feature_medians.pkl"      # Optional: feature medians (for "typical client" comparison)
METRICS_PATH  = "val_metrics.json"         # Optional: validation metrics

# =======================
# Global CSS style
# =======================
st.markdown("""
    <style>
        html, body, [data-testid="stApp"] {
            background: linear-gradient(to bottom, #e0f7fa, #ede7f6) !important;
        }
        .block-container { padding: 2rem; }
    </style>
""", unsafe_allow_html=True)

# =======================
# Load artifacts
# =======================
@st.cache_resource
def load_assets():
    if not Path(MODEL_PATH).exists():
        st.error(f"Model file not found: {MODEL_PATH}")
        st.stop()
    if not Path(FEATURES_PATH).exists():
        st.error(f"Feature file not found: {FEATURES_PATH}")
        st.stop()

    pipe = joblib.load(MODEL_PATH)
    feats = joblib.load(FEATURES_PATH)

    medians = None
    if Path(MEDIANS_PATH).exists():
        try:
            medians = joblib.load(MEDIANS_PATH)
            if isinstance(medians, pd.Series):
                medians = medians.to_dict()
        except Exception:
            medians = None

    metrics = {}
    if Path(METRICS_PATH).exists():
        try:
            metrics = json.loads(Path(METRICS_PATH).read_text(encoding="utf-8"))
        except Exception:
            metrics = {}

    return pipe, feats, medians, metrics

pipeline, feature_list, feature_medians, val_metrics = load_assets()

# =======================
# Feature display names (customize as needed)
# Default: pick first 5 features for sidebar inputs
# =======================
TOP5 = feature_list[:5]
display_names = {
    TOP5[0]: "Annual Income",
    TOP5[1]: "Outstanding Debt",
    TOP5[2]: "Days of Delay",
    TOP5[3]: "Monthly EMI",
    TOP5[4]: "Credit Mix (encoded)"
}
def dn(col):
    return display_names.get(col, col)

# =======================
# Title
# =======================
st.markdown(
    "<h1 style='text-align:center;color:#1e3c72;'>💹 Loan Interest Rate Prediction System</h1>"
    "<p style='text-align:center;'>Real-time prediction and visualization powered by regression models</p>",
    unsafe_allow_html=True
)

with st.expander("Validation Metrics (Train/Test Split)", expanded=False):
    if val_metrics:
        st.json(val_metrics)
    else:
        st.write("No validation metrics available.")

# =======================
# Sidebar inputs
# =======================
st.sidebar.header("📥 Enter Client Information (Top-5 Features)")

input_data = {}
for f in feature_list:
    if f in TOP5:
        default_val = 0.0
        if feature_medians and f in feature_medians:
            try:
                default_val = float(feature_medians[f])
            except Exception:
                default_val = 0.0
        input_data[f] = float(st.sidebar.number_input(dn(f), value=default_val))
    else:
        # For other features: use median if available, otherwise 0
        if feature_medians and f in feature_medians:
            try:
                input_data[f] = float(feature_medians[f])
            except Exception:
                input_data[f] = 0.0
        else:
            input_data[f] = 0.0

trigger = st.sidebar.button("🔍 Predict Loan Rate")

# =======================
# Prediction (after button click)
# =======================
if trigger:
    X = pd.DataFrame([input_data], columns=feature_list)

    # Regression prediction (interpreted as % here)
    y_pred = float(pipeline.predict(X)[0])

    # Typical client values for comparison
    if feature_medians:
        typical_vals = [float(feature_medians.get(f, 0.0)) for f in TOP5]
    else:
        typical_vals = [input_data[f] for f in TOP5]

    # Feature importance (if supported)
    def get_feature_importances(pipe, feats):
        mdl = getattr(pipe, "named_steps", {}).get("model", pipe)
        if hasattr(mdl, "feature_importances_"):
            imp = mdl.feature_importances_
            if len(imp) == len(feats):
                return pd.DataFrame({"Feature": feats, "Importance": imp}) \
                         .sort_values("Importance", ascending=False)
        return None
    importance_df = get_feature_importances(pipeline, feature_list)

    # Comparison table (Top-5)
    df_compare = pd.DataFrame({
        "Feature": [dn(f) for f in TOP5],
        "Client": [input_data[f] for f in TOP5],
        "Typical": typical_vals
    }).set_index("Feature")

    # Radar chart (Top-5)
    radar_fig = go.Figure()
    radar_fig.add_trace(go.Scatterpolar(
        r=[input_data[f] for f in TOP5],
        theta=[dn(f) for f in TOP5],
        fill='toself',
        name='Client',
    ))
    radar_fig.add_trace(go.Scatterpolar(
        r=typical_vals,
        theta=[dn(f) for f in TOP5],
        fill='toself',
        name='Typical Client',
    ))
    radar_fig.update_layout(polar=dict(radialaxis=dict(visible=True)), showlegend=True)

    # Top-10 importance
    if importance_df is not None:
        imp_top = importance_df.head(10).set_index("Feature")

    # Report text
    report_text = f"Loan Interest Rate Prediction\n"
    report_text += f"Predicted Rate: {y_pred:.2f}%\n\nTop-5 Features:\n"
    for f in TOP5:
        report_text += f"- {dn(f)}: {input_data[f]}\n"

    # =======================
    # Layout
    # =======================
    tab1, tab2 = st.tabs(["📊 Summary", "📋 Detailed Insights"])

    with tab1:
        st.markdown(f"""
        <div style="background-color:#f0f8ff;padding:20px;border-radius:10px;">
            <h3>🎯 Predicted Loan Rate: <span style="color:#1e3c72;">{y_pred:.2f}%</span></h3>
            <p>Prediction generated by the regression pipeline (including preprocessing & feature engineering).</p>
        </div>
        """, unsafe_allow_html=True)

        st.subheader("📘 Interpretation")
        st.write(
            "• The predicted loan rate is based on debt, income, delinquency, and other key indicators.\n"
            "• For more robust performance, consider saving feature medians/distributions after training for comparison."
        )

        st.subheader("📥 Export Report")
        st.download_button("Download Report",
                           io.BytesIO(report_text.encode("utf-8")),
                           file_name="loan_rate_prediction.txt")

        with st.expander("ℹ️ Model Information"):
            st.markdown(
                "- **Model Type**: Regression (e.g., RandomForestRegressor)\n"
                "- **Input**: Features consistent with `features.pkl`\n"
                "- **Files**: `loan_rate_pipeline.pkl` + `features.pkl` (optional: `feature_medians.pkl`, `val_metrics.json`)"
            )

    with tab2:
        st.subheader("📏 Client vs Typical (Top-5 Features)")
        st.bar_chart(df_compare)

        st.subheader("📊 Client Radar Chart (Top-5 Features)")
        st.plotly_chart(radar_fig, use_container_width=True)

        if importance_df is not None:
            st.subheader("🔎 Feature Importance (Top-10)")
            st.bar_chart(imp_top)
        else:
            st.info("Current model does not support `feature_importances_` or mismatch in feature count.")

else:
    st.info("Enter client information on the left and click **“🔍 Predict Loan Rate”** to see results.")
