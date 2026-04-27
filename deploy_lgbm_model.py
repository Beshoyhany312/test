import joblib
import numpy as np
import pandas as pd
import streamlit as st
import os
import io
import requests

# ── PAGE CONFIG ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Smart City Electricity Cost Predictor",
    page_icon="🏙️",
    layout="centered",
)

# ── CUSTOM CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Sans:wght@300;400;500&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
}

h1, h2, h3 {
    font-family: 'Syne', sans-serif !important;
}

/* Dark card background for sections */
[data-testid="stForm"] {
    background: #0f1117;
    border: 1px solid #2a2d3a;
    border-radius: 16px;
    padding: 1.5rem;
}

/* Metric styling */
[data-testid="stMetric"] {
    background: linear-gradient(135deg, #1a1d2e, #252840);
    border: 1px solid #3d4166;
    border-radius: 12px;
    padding: 1.2rem 1.5rem;
}

[data-testid="stMetricValue"] {
    font-family: 'Syne', sans-serif !important;
    font-size: 2.2rem !important;
    font-weight: 800 !important;
    color: #7DF9AA !important;
}

[data-testid="stMetricLabel"] {
    font-size: 0.75rem !important;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    color: #8b8fa8 !important;
}

/* Submit button */
[data-testid="stFormSubmitButton"] button {
    background: linear-gradient(135deg, #7DF9AA, #4fc3f7) !important;
    color: #0a0d1a !important;
    font-family: 'Syne', sans-serif !important;
    font-weight: 700 !important;
    font-size: 1rem !important;
    border: none !important;
    border-radius: 10px !important;
    padding: 0.75rem 2rem !important;
    width: 100% !important;
    letter-spacing: 0.05em;
    transition: opacity 0.2s !important;
}

[data-testid="stFormSubmitButton"] button:hover {
    opacity: 0.85 !important;
}

/* Number inputs and selects */
[data-testid="stNumberInput"] input,
[data-testid="stSelectbox"] select {
    background: #1a1d2e !important;
    border-color: #2a2d3a !important;
    color: #e8eaf6 !important;
    border-radius: 8px !important;
}

/* Divider */
hr { border-color: #2a2d3a; }

/* Info / success boxes */
[data-testid="stAlert"] {
    border-radius: 10px;
    border-left-width: 4px;
}

.badge {
    display: inline-block;
    background: #1a1d2e;
    border: 1px solid #3d4166;
    border-radius: 20px;
    padding: 0.2rem 0.8rem;
    font-size: 0.75rem;
    color: #8b8fa8;
    font-family: 'DM Sans', sans-serif;
    margin-right: 0.4rem;
}
</style>
""", unsafe_allow_html=True)

# ── PATHS ──────────────────────────────────────────────────────────────────────
BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
SCALER_PATH = os.path.join(BASE_DIR, "scaler.joblib")
MODEL_PATH  = os.path.join(BASE_DIR, "lightgbm_regressor_model.joblib")

FEATURE_COLUMNS = [
    'Site Area (square meters)',
    'Water Consumption (liters/day)',
    'Recycling Rate (%)',
    'Utilisation Rate (%)',
    'Air Quality Index (AQI)',
    'Issue Resolution Time (hours)',
    'Resident Count (number of people)',
    'Structure Type_Industrial',
    'Structure Type_Mixed-use',
    'Structure Type_Residential',
]

# ── MODEL LOADING ──────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading model…")
def load_artifacts(model_path: str, scaler_path: str):
    missing = [p for p in (model_path, scaler_path) if not os.path.exists(p)]
    if missing:
        st.error(f"Missing files: {missing}\n\nMake sure both `.joblib` files are in the same folder as this script.")
        st.stop()
    model  = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    return model, scaler

lgbm_model, scaler = load_artifacts(MODEL_PATH, SCALER_PATH)

# ── PREDICTION FUNCTION ────────────────────────────────────────────────────────
def predict_cost(input_dict: dict) -> float:
    df = pd.DataFrame([input_dict], columns=FEATURE_COLUMNS)
    scaled = scaler.transform(df)
    return float(lgbm_model.predict(scaled)[0])

# ── HEADER ─────────────────────────────────────────────────────────────────────
st.markdown("""
<div style="padding: 2rem 0 1rem 0;">
    <span class="badge">🏙️ Smart City</span>
    <span class="badge">⚡ LightGBM</span>
    <span class="badge">ML Prediction</span>
    <h1 style="margin-top: 1rem; font-size: 2.4rem; line-height: 1.2; color: #e8eaf6;">
        Electricity Cost<br><span style="color: #7DF9AA;">Predictor</span>
    </h1>
    <p style="color: #8b8fa8; margin-top: 0.5rem; font-size: 0.95rem;">
        Enter site and structure details below to estimate your monthly electricity cost.
    </p>
</div>
""", unsafe_allow_html=True)

st.divider()

# ── INPUT FORM ─────────────────────────────────────────────────────────────────
with st.form("lgbm_form"):

    # ── Section 1: Site Info ───────────────────────────────────────────────────
    st.markdown("#### 📐 Site Information")
    c1, c2 = st.columns(2)
    with c1:
        site_area = st.number_input(
            "Site Area (m²)", min_value=100, max_value=100_000,
            value=1360, step=50,
            help="Total footprint of the site in square meters."
        )
        resident_count = st.number_input(
            "Resident Count", min_value=1, max_value=10_000,
            value=6, step=1,
            help="Number of people living or working on-site."
        )
    with c2:
        utilisation_rate = st.slider(
            "Utilisation Rate (%)", 0, 100, 59,
            help="How actively the site is being used."
        )
        issue_resolution = st.number_input(
            "Issue Resolution Time (hrs)", min_value=0, max_value=720,
            value=34, step=1,
            help="Average time to resolve reported issues."
        )

    st.divider()

    # ── Section 2: Environment ─────────────────────────────────────────────────
    st.markdown("#### 🌿 Environmental Metrics")
    c3, c4 = st.columns(2)
    with c3:
        water_consumption = st.number_input(
            "Water Consumption (L/day)", min_value=0.0, max_value=100_000.0,
            value=2519.0, step=10.0,
            help="Daily water usage across the site."
        )
        recycling_rate = st.slider(
            "Recycling Rate (%)", 0, 100, 68,
            help="Percentage of waste that is recycled."
        )
    with c4:
        aqi = st.number_input(
            "Air Quality Index (AQI)", min_value=0, max_value=500,
            value=51, step=1,
            help="Current AQI reading for the site location."
        )

    st.divider()

    # ── Section 3: Structure Type ──────────────────────────────────────────────
    st.markdown("#### 🏗️ Structure Type")
    structure_type = st.selectbox(
        "Select the primary structure type",
        options=["Commercial", "Industrial", "Mixed-use", "Residential"],
        index=2,
        help="'Commercial' maps to all dummy columns = 0 (reference category)."
    )

    st.markdown("<br>", unsafe_allow_html=True)
    submitted = st.form_submit_button("⚡ Predict Electricity Cost", use_container_width=True)

# ── PREDICTION ─────────────────────────────────────────────────────────────────
if submitted:
    input_dict = {
        'Site Area (square meters)':          site_area,
        'Water Consumption (liters/day)':     water_consumption,
        'Recycling Rate (%)':                 recycling_rate,
        'Utilisation Rate (%)':               utilisation_rate,
        'Air Quality Index (AQI)':            aqi,
        'Issue Resolution Time (hours)':      issue_resolution,
        'Resident Count (number of people)':  resident_count,
        'Structure Type_Industrial':          int(structure_type == "Industrial"),
        'Structure Type_Mixed-use':           int(structure_type == "Mixed-use"),
        'Structure Type_Residential':         int(structure_type == "Residential"),
    }

    with st.spinner("Running model…"):
        try:
            cost = predict_cost(input_dict)

            st.success("Prediction complete!")
            st.markdown("<br>", unsafe_allow_html=True)

            m1, m2, m3 = st.columns(3)
            m1.metric("💡 Monthly Cost",    f"${cost:,.2f}",    "USD / month")
            m2.metric("📅 Annual Estimate", f"${cost*12:,.2f}", "USD / year")
            m3.metric("📊 Daily Average",   f"${cost/30:,.2f}", "USD / day")

            st.markdown("<br>", unsafe_allow_html=True)

            # Cost interpretation
            if cost < 500:
                level, color, note = "Low",    "#7DF9AA", "This site has a very efficient electricity footprint."
            elif cost < 1500:
                level, color, note = "Moderate","#FFD700", "Typical range for sites of this type and size."
            else:
                level, color, note = "High",   "#FF6B6B", "Consider energy audits or efficiency improvements."

            st.markdown(f"""
            <div style="background:#1a1d2e; border:1px solid #2a2d3a; border-left: 4px solid {color};
                        border-radius:12px; padding:1rem 1.5rem; margin-top:0.5rem;">
                <span style="font-family:Syne,sans-serif; font-weight:700; color:{color}; font-size:1rem;">
                    {level} Consumption Zone
                </span><br>
                <span style="color:#8b8fa8; font-size:0.88rem;">{note}</span>
            </div>
            """, unsafe_allow_html=True)

        except Exception as e:
            st.error(f"Prediction failed: {e}")
            st.exception(e)

st.divider()
st.markdown(
    "<p style='text-align:center; color:#3d4166; font-size:0.8rem;'>"
    "Powered by LightGBM · Smart City Analytics</p>",
    unsafe_allow_html=True
)
