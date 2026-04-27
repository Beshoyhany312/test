import os
import io
import joblib
import numpy as np
import pandas as pd
import streamlit as st

# ── PAGE CONFIG ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Electricity Predictor Suite",
    page_icon="⚡",
    layout="wide",
)

# ── CUSTOM CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=Outfit:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"] { font-family: 'Outfit', sans-serif; }
h1, h2, h3 { font-family: 'Space Mono', monospace !important; }

/* Tab styling */
[data-testid="stTabs"] [role="tab"] {
    font-family: 'Space Mono', monospace !important;
    font-size: 0.85rem !important;
    font-weight: 700 !important;
    letter-spacing: 0.05em;
    padding: 0.6rem 1.2rem !important;
}
[data-testid="stTabs"] [role="tab"][aria-selected="true"] {
    border-bottom: 3px solid #00e5ff !important;
    color: #00e5ff !important;
}

/* Form */
[data-testid="stForm"] {
    background: #0d1117;
    border: 1px solid #21262d;
    border-radius: 14px;
    padding: 1.5rem;
}

/* Metrics */
[data-testid="stMetric"] {
    background: linear-gradient(145deg, #161b22, #1c2128);
    border: 1px solid #30363d;
    border-radius: 12px;
    padding: 1.1rem 1.4rem;
}
[data-testid="stMetricValue"] {
    font-family: 'Space Mono', monospace !important;
    font-size: 1.9rem !important;
    font-weight: 700 !important;
    color: #00e5ff !important;
}
[data-testid="stMetricLabel"] {
    font-size: 0.72rem !important;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    color: #8b949e !important;
}

/* Buttons */
[data-testid="stFormSubmitButton"] button {
    background: linear-gradient(135deg, #00e5ff, #7c4dff) !important;
    color: #0d1117 !important;
    font-family: 'Space Mono', monospace !important;
    font-weight: 700 !important;
    font-size: 0.9rem !important;
    border: none !important;
    border-radius: 10px !important;
    width: 100% !important;
    letter-spacing: 0.05em;
}
[data-testid="stFormSubmitButton"] button:hover { opacity: 0.85 !important; }

/* Inputs */
[data-testid="stNumberInput"] input { border-radius: 8px !important; }

hr { border-color: #21262d; }

.model-badge {
    display: inline-block;
    background: #161b22;
    border: 1px solid #30363d;
    border-radius: 20px;
    padding: 0.25rem 0.9rem;
    font-size: 0.72rem;
    color: #8b949e;
    font-family: 'Space Mono', monospace;
    margin-right: 0.4rem;
    letter-spacing: 0.05em;
}
.result-card {
    background: #161b22;
    border: 1px solid #30363d;
    border-left: 4px solid;
    border-radius: 12px;
    padding: 1rem 1.4rem;
    margin-top: 0.8rem;
}
.default-note {
    font-size: 0.78rem;
    color: #8b949e;
    font-style: italic;
    margin-top: -0.4rem;
    margin-bottom: 0.8rem;
}
</style>
""", unsafe_allow_html=True)

# ── MEAN DEFAULTS (extracted from feature_scaler.joblib) ──────────────────────
# Order: ac_units, ac_hp, fridges, tvs, fans, pcs,
#        daily_hours, house_m2, has_heater, washing_pw,
#        season_winter, insulation_low, insulation_medium
MEANS = {
    'number_of_air_conditioners':     2.01,
    'ac_power_hp':                    2.25,
    'number_of_refrigerators':        1.51,
    'number_of_televisions':          1.49,
    'number_of_fans':                 1.99,
    'number_of_computers':            1.00,
    'average_daily_usage_hours':      6.49,
    'house_size_m2':                  132.15,
    'has_water_heater':               0.50,   # ~50% have one → default Yes
    'washing_machine_usage_per_week': 3.01,
    # season_winter mean ≈ 0.49  → default Summer (0)
    # insulation means ≈ 0.33 each → default High (both=0)
}

BASE_DIR    = os.path.dirname(os.path.abspath(__file__))

# ── FILE PATHS ─────────────────────────────────────────────────────────────────
SCALER_PATH      = os.path.join(BASE_DIR, "feature_scaler.joblib")
CATBOOST_KWH     = os.path.join(BASE_DIR, "catboost_kwh_model.joblib")
CATBOOST_BILL    = os.path.join(BASE_DIR, "catboost_bill_model.joblib")
LGBM_MODEL_PATH  = os.path.join(BASE_DIR, "lightgbm_regressor_model.joblib")
LGBM_SCALER_PATH = os.path.join(BASE_DIR, "scaler.joblib")   # LGBM uses its own scaler

# ── LOADERS ───────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading CatBoost models…")
def load_catboost():
    missing = [p for p in (CATBOOST_KWH, CATBOOST_BILL, SCALER_PATH) if not os.path.exists(p)]
    if missing:
        st.warning(f"CatBoost files missing: {[os.path.basename(p) for p in missing]}")
        return None, None, None
    kwh_m  = joblib.load(CATBOOST_KWH)
    bill_m = joblib.load(CATBOOST_BILL)
    scaler = joblib.load(SCALER_PATH)
    return kwh_m, bill_m, scaler

@st.cache_resource(show_spinner="Loading LightGBM model…")
def load_lgbm():
    missing = [p for p in (LGBM_MODEL_PATH, LGBM_SCALER_PATH) if not os.path.exists(p)]
    if missing:
        st.warning(f"LightGBM files missing: {[os.path.basename(p) for p in missing]}")
        return None, None
    model  = joblib.load(LGBM_MODEL_PATH)
    scaler = joblib.load(LGBM_SCALER_PATH)
    return model, scaler

# ── PREPROCESSING (shared for CatBoost & LGBM electricity models) ─────────────
ELECTRICITY_COLS = [
    'number_of_air_conditioners', 'ac_power_hp', 'number_of_refrigerators',
    'number_of_televisions', 'number_of_fans', 'number_of_computers',
    'average_daily_usage_hours', 'house_size_m2', 'has_water_heater',
    'washing_machine_usage_per_week', 'season_winter',
    'insulation_quality_low', 'insulation_quality_medium'
]

def preprocess_electricity(raw: dict, scaler) -> np.ndarray:
    df = pd.DataFrame([raw])
    if 'season' in df.columns:
        df['season_winter'] = (df['season'].str.lower() == 'winter').astype(int)
        df.drop(columns=['season'], inplace=True)
    else:
        df['season_winter'] = 0
    if 'insulation_quality' in df.columns:
        df['insulation_quality_low']    = (df['insulation_quality'].str.lower() == 'low').astype(int)
        df['insulation_quality_medium'] = (df['insulation_quality'].str.lower() == 'medium').astype(int)
        df.drop(columns=['insulation_quality'], inplace=True)
    else:
        df['insulation_quality_low']    = 0
        df['insulation_quality_medium'] = 0
    for col in ELECTRICITY_COLS:
        if col not in df.columns:
            df[col] = 0
    df = df[ELECTRICITY_COLS]
    return scaler.transform(df)

# ── RESULT CARD HELPER ─────────────────────────────────────────────────────────
def cost_zone(val):
    if val < 300:
        return "#00e5ff", "Low Consumption Zone", "Great efficiency — below average for this configuration."
    elif val < 700:
        return "#FFD700", "Moderate Consumption Zone", "Typical range for a household of this size."
    else:
        return "#FF6B6B", "High Consumption Zone", "Consider energy-saving measures or appliance upgrades."

# ══════════════════════════════════════════════════════════════════════════════
# ── HEADER ────────────────────────────────────────────────────────────────────
st.markdown("""
<div style="padding: 1.5rem 0 0.5rem 0;">
    <span class="model-badge">⚡ ELECTRICITY SUITE</span>
    <span class="model-badge">🇪🇬 EGP</span>
    <h1 style="margin-top:0.8rem; font-size:2rem; color:#e6edf3; line-height:1.3;">
        Electricity Cost &<br><span style="color:#00e5ff;">Energy Predictor</span>
    </h1>
    <p style="color:#8b949e; font-size:0.9rem; margin-top:0.4rem;">
        Two ML models, one interface. Inputs default to dataset averages — 
        only fill in what you know.
    </p>
</div>
""", unsafe_allow_html=True)

st.divider()

# ── TABS ───────────────────────────────────────────────────────────────────────
tab1, tab2 = st.tabs(["🐱 CatBoost Model", "💡 LightGBM Model (Smart City)"])

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — CatBoost
# ══════════════════════════════════════════════════════════════════════════════
with tab1:
    st.markdown("""
    <span class="model-badge">CatBoost Regressor</span>
    <span class="model-badge">Dual Output: kWh + EGP</span>
    <p class="default-note" style="margin-top:0.6rem;">
        ℹ️ All inputs default to dataset means. Leave unchanged if unsure.
    </p>
    """, unsafe_allow_html=True)

    kwh_model, bill_model, cb_scaler = load_catboost()

    with st.form("catboost_form"):
        st.markdown("#### 🏠 Home & Appliances")
        c1, c2, c3 = st.columns(3)
        with c1:
            cb_ac      = st.number_input("Air Conditioners",   min_value=0, value=int(round(MEANS['number_of_air_conditioners'])))
            cb_ac_hp   = st.number_input("AC Power (HP)",      min_value=0.0, step=0.5, value=round(MEANS['ac_power_hp'], 1))
            cb_fridge  = st.number_input("Refrigerators",      min_value=0, value=int(round(MEANS['number_of_refrigerators'])))
            cb_tv      = st.number_input("Televisions",        min_value=0, value=int(round(MEANS['number_of_televisions'])))
        with c2:
            cb_fans    = st.number_input("Fans",               min_value=0, value=int(round(MEANS['number_of_fans'])))
            cb_pc      = st.number_input("Computers/Laptops",  min_value=0, value=int(round(MEANS['number_of_computers'])))
            cb_washing = st.number_input("Washing (times/wk)", min_value=0, max_value=20, value=int(round(MEANS['washing_machine_usage_per_week'])))
            cb_heater  = st.selectbox("Water Heater?", ["Yes", "No"], index=0)
        with c3:
            cb_hours   = st.slider("Daily Usage Hours", 0.0, 24.0, round(MEANS['average_daily_usage_hours'], 1))
            cb_house   = st.number_input("House Size (m²)", min_value=10.0, value=round(MEANS['house_size_m2'], 0))
            cb_season  = st.selectbox("Season", ["Summer", "Winter"], index=0)
            cb_insul   = st.selectbox("Insulation Quality", ["High", "Medium", "Low"], index=0)

        cb_submit = st.form_submit_button("⚡ Predict with CatBoost", use_container_width=True)

    if cb_submit:
        if kwh_model is None:
            st.error("CatBoost models not loaded. Check that all .joblib files are in the repo.")
        else:
            raw = {
                'number_of_air_conditioners':     cb_ac,
                'ac_power_hp':                    cb_ac_hp,
                'number_of_refrigerators':        cb_fridge,
                'number_of_televisions':          cb_tv,
                'number_of_fans':                 cb_fans,
                'number_of_computers':            cb_pc,
                'average_daily_usage_hours':      cb_hours,
                'season':                         cb_season.lower(),
                'house_size_m2':                  float(cb_house),
                'insulation_quality':             cb_insul.lower(),
                'has_water_heater':               1 if cb_heater == "Yes" else 0,
                'washing_machine_usage_per_week': cb_washing,
            }
            with st.spinner("Running CatBoost…"):
                try:
                    scaled = preprocess_electricity(raw, cb_scaler)
                    kwh    = float(kwh_model.predict(scaled).flatten()[0])
                    bill   = float(bill_model.predict(scaled).flatten()[0])

                    st.success("Prediction complete!")
                    m1, m2, m3, m4 = st.columns(4)
                    m1.metric("🔋 Consumption",   f"{kwh:,.1f} kWh")
                    m2.metric("💰 Monthly Bill",   f"{bill:,.1f} EGP")
                    m3.metric("📅 Annual Bill",    f"{bill*12:,.0f} EGP")
                    m4.metric("📊 Daily Cost",     f"{bill/30:,.1f} EGP")

                    color, zone_label, zone_note = cost_zone(bill)
                    st.markdown(f"""
                    <div class="result-card" style="border-left-color:{color};">
                        <span style="font-family:'Space Mono',monospace; font-weight:700;
                                     color:{color}; font-size:0.95rem;">{zone_label}</span><br>
                        <span style="color:#8b949e; font-size:0.85rem;">{zone_note}</span>
                    </div>
                    """, unsafe_allow_html=True)
                except Exception as e:
                    st.error(f"Prediction failed: {e}")
                    st.exception(e)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — LightGBM (Smart City model)
# ══════════════════════════════════════════════════════════════════════════════
with tab2:
    st.markdown("""
    <span class="model-badge">LightGBM Regressor</span>
    <span class="model-badge">Output: USD/month</span>
    <p class="default-note" style="margin-top:0.6rem;">
        ℹ️ Smart-city site model. Inputs default to example values. Leave unchanged if unsure.
    </p>
    """, unsafe_allow_html=True)

    lgbm_model, lgbm_scaler = load_lgbm()

    LGBM_COLS = [
        'Site Area (square meters)', 'Water Consumption (liters/day)',
        'Recycling Rate (%)', 'Utilisation Rate (%)', 'Air Quality Index (AQI)',
        'Issue Resolution Time (hours)', 'Resident Count (number of people)',
        'Structure Type_Industrial', 'Structure Type_Mixed-use', 'Structure Type_Residential'
    ]

    with st.form("lgbm_form"):
        st.markdown("#### 📐 Site Information")
        lc1, lc2, lc3 = st.columns(3)
        with lc1:
            l_area     = st.number_input("Site Area (m²)",          min_value=100, value=1360, step=50)
            l_resident = st.number_input("Resident Count",           min_value=1, value=6, step=1)
            l_util     = st.slider("Utilisation Rate (%)",           0, 100, 59)
        with lc2:
            l_water    = st.number_input("Water Consumption (L/day)", min_value=0.0, value=2519.0, step=50.0)
            l_recycle  = st.slider("Recycling Rate (%)",              0, 100, 68)
            l_aqi      = st.number_input("Air Quality Index (AQI)",   min_value=0, max_value=500, value=51)
        with lc3:
            l_issue    = st.number_input("Issue Resolution Time (hrs)", min_value=0, value=34, step=1)
            l_struct   = st.selectbox("Structure Type",
                                      ["Commercial", "Industrial", "Mixed-use", "Residential"],
                                      index=2)

        lgbm_submit = st.form_submit_button("⚡ Predict with LightGBM", use_container_width=True)

    if lgbm_submit:
        if lgbm_model is None:
            st.error("LightGBM model not loaded. Check that lightgbm_regressor_model.joblib and scaler.joblib are in the repo.")
        else:
            lgbm_input = {
                'Site Area (square meters)':          l_area,
                'Water Consumption (liters/day)':     l_water,
                'Recycling Rate (%)':                 l_recycle,
                'Utilisation Rate (%)':               l_util,
                'Air Quality Index (AQI)':            l_aqi,
                'Issue Resolution Time (hours)':      l_issue,
                'Resident Count (number of people)':  l_resident,
                'Structure Type_Industrial':          int(l_struct == "Industrial"),
                'Structure Type_Mixed-use':           int(l_struct == "Mixed-use"),
                'Structure Type_Residential':         int(l_struct == "Residential"),
            }
            with st.spinner("Running LightGBM…"):
                try:
                    df_in  = pd.DataFrame([lgbm_input], columns=LGBM_COLS)
                    scaled = lgbm_scaler.transform(df_in)
                    cost   = float(lgbm_model.predict(scaled)[0])

                    st.success("Prediction complete!")
                    lm1, lm2, lm3 = st.columns(3)
                    lm1.metric("💡 Monthly Cost",    f"${cost:,.2f} USD")
                    lm2.metric("📅 Annual Estimate", f"${cost*12:,.2f} USD")
                    lm3.metric("📊 Daily Average",   f"${cost/30:,.2f} USD")

                    if cost < 500:
                        color, zone_label, zone_note = "#00e5ff", "Low Consumption Zone", "This site has a very efficient electricity footprint."
                    elif cost < 1500:
                        color, zone_label, zone_note = "#FFD700", "Moderate Consumption Zone", "Typical range for sites of this type and size."
                    else:
                        color, zone_label, zone_note = "#FF6B6B", "High Consumption Zone", "Consider energy audits or efficiency improvements."

                    st.markdown(f"""
                    <div class="result-card" style="border-left-color:{color};">
                        <span style="font-family:'Space Mono',monospace; font-weight:700;
                                     color:{color}; font-size:0.95rem;">{zone_label}</span><br>
                        <span style="color:#8b949e; font-size:0.85rem;">{zone_note}</span>
                    </div>
                    """, unsafe_allow_html=True)
                except Exception as e:
                    st.error(f"Prediction failed: {e}")
                    st.exception(e)

st.divider()
st.markdown(
    "<p style='text-align:center; color:#30363d; font-size:0.78rem; font-family:Space Mono,monospace;'>"
    "CatBoost · LightGBM · Streamlit · Smart Energy Analytics</p>",
    unsafe_allow_html=True,
)
