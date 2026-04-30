import os
import io
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from fpdf import FPDF
from datetime import datetime

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

[data-testid="stForm"] {
    background: #0d1117;
    border: 1px solid #21262d;
    border-radius: 14px;
    padding: 1.5rem;
}

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

[data-testid="stDownloadButton"] button {
    background: linear-gradient(135deg, #28a745, #20c997) !important;
    color: white !important;
    font-family: 'Space Mono', monospace !important;
    font-weight: 700 !important;
    border: none !important;
    border-radius: 10px !important;
    width: 100% !important;
}

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
.warning-banner {
    background: rgba(255,75,75,0.1);
    border: 2px solid #ff4b4b;
    border-radius: 12px;
    padding: 1rem 1.4rem;
    text-align: center;
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

# ── MEAN DEFAULTS (from feature_scaler.joblib) ─────────────────────────────────
MEANS = {
    'number_of_air_conditioners':     2.01,
    'ac_power_hp':                    2.25,
    'number_of_refrigerators':        1.51,
    'number_of_televisions':          1.49,
    'number_of_fans':                 1.99,
    'number_of_computers':            1.00,
    'average_daily_usage_hours':      6.49,
    'house_size_m2':                  132.15,
    'has_water_heater':               0.50,
    'washing_machine_usage_per_week': 3.01,
}

BASE_DIR         = os.path.dirname(os.path.abspath(__file__))
SCALER_PATH      = os.path.join(BASE_DIR, "feature_scaler.joblib")
CATBOOST_KWH     = os.path.join(BASE_DIR, "catboost_kwh_model.joblib")
CATBOOST_BILL    = os.path.join(BASE_DIR, "catboost_bill_model.joblib")
LGBM_MODEL_PATH  = os.path.join(BASE_DIR, "lightgbm_regressor_model.joblib")
LGBM_SCALER_PATH = os.path.join(BASE_DIR, "scaler.joblib")

MONTHS = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']

# ── LOADERS ───────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading CatBoost models...")
def load_catboost():
    missing = [p for p in (CATBOOST_KWH, CATBOOST_BILL, SCALER_PATH) if not os.path.exists(p)]
    if missing:
        st.warning(f"CatBoost files missing: {[os.path.basename(p) for p in missing]}")
        return None, None, None
    return joblib.load(CATBOOST_KWH), joblib.load(CATBOOST_BILL), joblib.load(SCALER_PATH)

@st.cache_resource(show_spinner="Loading LightGBM model...")
def load_lgbm():
    missing = [p for p in (LGBM_MODEL_PATH, LGBM_SCALER_PATH) if not os.path.exists(p)]
    if missing:
        st.warning(f"LightGBM files missing: {[os.path.basename(p) for p in missing]}")
        return None, None
    return joblib.load(LGBM_MODEL_PATH), joblib.load(LGBM_SCALER_PATH)

# ── PREPROCESSING ──────────────────────────────────────────────────────────────
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
        df['insulation_quality_low'] = df['insulation_quality_medium'] = 0
    for col in ELECTRICITY_COLS:
        if col not in df.columns:
            df[col] = 0
    return scaler.transform(df[ELECTRICITY_COLS])

# ── HELPERS ────────────────────────────────────────────────────────────────────
def cost_zone_egp(val):
    if val < 300:   return "#00e5ff", "Low Consumption Zone",      "Great efficiency -- below average for this configuration."
    elif val < 700: return "#FFD700", "Moderate Consumption Zone", "Typical range for a household of this size."
    else:           return "#FF6B6B", "High Consumption Zone",     "Consider energy-saving measures or appliance upgrades."

def cost_zone_usd(val):
    if val < 500:    return "#00e5ff", "Low Consumption Zone",      "This site has a very efficient electricity footprint."
    elif val < 1500: return "#FFD700", "Moderate Consumption Zone", "Typical range for sites of this type and size."
    else:            return "#FF6B6B", "High Consumption Zone",     "Consider energy audits or efficiency improvements."

def monthly_trend(base_value: float) -> list:
    """Simulates 12-month seasonality -- summer peaks, winter dips."""
    return [round(base_value * (1 + 0.25 * np.cos((i - 6) / 1.9)), 2) for i in range(12)]

def plot_trend(values: list, label: str, currency: str, color: str) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=MONTHS, y=values, mode='lines+markers',
        line=dict(color=color, width=3),
        marker=dict(size=8, color=color),
        fill='tozeroy',
        fillcolor=f'rgba({int(color[1:3],16)},{int(color[3:5],16)},{int(color[5:7],16)},0.12)',
        name=label,
    ))
    fig.update_layout(
        plot_bgcolor='#0d1117', paper_bgcolor='#0d1117',
        font=dict(color='#8b949e', family='Outfit'),
        xaxis=dict(showgrid=False, color='#8b949e'),
        yaxis=dict(showgrid=True, gridcolor='#21262d', color='#8b949e',
                   title=f"Estimated Cost ({currency})"),
        margin=dict(l=20, r=20, t=30, b=20),
        height=320,
    )
    return fig

def generate_pdf_catboost(inputs: dict, kwh: float, bill: float, ex_rate: float) -> bytes:
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", 'B', 18)
    pdf.set_text_color(30, 60, 114)
    pdf.cell(0, 12, "Electricity Prediction Report", ln=True, align='C')
    pdf.set_font("Arial", '', 10)
    pdf.set_text_color(100, 100, 100)
    pdf.cell(0, 8, f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}  |  Model: CatBoost", ln=True, align='C')
    pdf.ln(6)

    # Results box
    pdf.set_fill_color(230, 245, 255)
    pdf.set_draw_color(30, 60, 114)
    pdf.set_font("Arial", 'B', 13)
    pdf.set_text_color(30, 60, 114)
    pdf.cell(0, 10, "Prediction Results", ln=True, fill=True)
    pdf.set_font("Arial", '', 11)
    pdf.set_text_color(30, 30, 30)
    pdf.cell(95, 9, f"Monthly Consumption: {kwh:,.2f} kWh", border=1, ln=False)
    pdf.cell(95, 9, f"Monthly Bill: {bill:,.2f} EGP", border=1, ln=True)
    pdf.cell(95, 9, f"Annual Bill: {bill*12:,.2f} EGP", border=1, ln=False)
    pdf.cell(95, 9, f"USD Equivalent: ${bill/ex_rate:,.2f}", border=1, ln=True)
    pdf.ln(6)

    # Budget alert
    pdf.set_font("Arial", 'B', 11)
    if bill > 1000:
        pdf.set_text_color(200, 0, 0)
        pdf.cell(0, 9, "STATUS: OVER BUDGET (> 1,000 EGP/month)", ln=True)
    else:
        pdf.set_text_color(0, 150, 0)
        pdf.cell(0, 9, "STATUS: WITHIN BUDGET", ln=True)

    pdf.ln(4)
    # Inputs table
    pdf.set_font("Arial", 'B', 12)
    pdf.set_text_color(30, 60, 114)
    pdf.cell(0, 9, "Input Data", ln=True)
    pdf.set_font("Arial", '', 10)
    pdf.set_text_color(30, 30, 30)
    for k, v in inputs.items():
        label = k.replace('_', ' ').title()
        pdf.cell(95, 8, f"{label}", border='LTB')
        pdf.cell(95, 8, f"{v}", border='RTB', ln=True)

    pdf.ln(8)
    pdf.set_font("Arial", 'I', 9)
    pdf.set_text_color(150, 150, 150)
    pdf.cell(0, 8, "Smart Electricity Prediction Suite  |  CatBoost + LightGBM", ln=True, align='C')
    return pdf.output(dest='S').encode('latin-1')

def generate_pdf_lgbm(inputs: dict, cost_usd: float, ex_rate: float) -> bytes:
    cost_egp = cost_usd * ex_rate
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", 'B', 18)
    pdf.set_text_color(30, 60, 114)
    pdf.cell(0, 12, "Smart City Electricity Cost Report", ln=True, align='C')
    pdf.set_font("Arial", '', 10)
    pdf.set_text_color(100, 100, 100)
    pdf.cell(0, 8, f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}  |  Model: LightGBM", ln=True, align='C')
    pdf.ln(6)

    pdf.set_fill_color(230, 245, 255)
    pdf.set_draw_color(30, 60, 114)
    pdf.set_font("Arial", 'B', 13)
    pdf.set_text_color(30, 60, 114)
    pdf.cell(0, 10, "Prediction Results", ln=True, fill=True)
    pdf.set_font("Arial", '', 11)
    pdf.set_text_color(30, 30, 30)
    pdf.cell(95, 9, f"Monthly Cost: ${cost_usd:,.2f} USD", border=1, ln=False)
    pdf.cell(95, 9, f"EGP Equivalent: {cost_egp:,.2f} EGP", border=1, ln=True)
    pdf.cell(95, 9, f"Annual Estimate: ${cost_usd*12:,.2f} USD", border=1, ln=False)
    pdf.cell(95, 9, f"Daily Average: ${cost_usd/30:,.2f} USD", border=1, ln=True)
    pdf.ln(6)

    pdf.set_font("Arial", 'B', 11)
    if cost_usd > 1500:
        pdf.set_text_color(200, 0, 0)
        pdf.cell(0, 9, "STATUS: HIGH CONSUMPTION (> $1,500/month)", ln=True)
    elif cost_usd > 500:
        pdf.set_text_color(200, 140, 0)
        pdf.cell(0, 9, "STATUS: MODERATE CONSUMPTION", ln=True)
    else:
        pdf.set_text_color(0, 150, 0)
        pdf.cell(0, 9, "STATUS: LOW / EFFICIENT", ln=True)

    pdf.ln(4)
    pdf.set_font("Arial", 'B', 12)
    pdf.set_text_color(30, 60, 114)
    pdf.cell(0, 9, "Site Input Data", ln=True)
    pdf.set_font("Arial", '', 10)
    pdf.set_text_color(30, 30, 30)
    for k, v in inputs.items():
        pdf.cell(115, 8, f"{k}", border='LTB')
        pdf.cell(75, 8, f"{v}", border='RTB', ln=True)

    pdf.ln(8)
    pdf.set_font("Arial", 'I', 9)
    pdf.set_text_color(150, 150, 150)
    pdf.cell(0, 8, "Smart Electricity Prediction Suite  |  CatBoost + LightGBM", ln=True, align='C')
    return pdf.output(dest='S').encode('latin-1')

# ── SIDEBAR ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### Settings")
    ex_rate = st.number_input("USD -> EGP Exchange Rate", min_value=1.0, value=48.5, step=0.5,
                               help="Used to convert USD predictions to EGP and for PDF reports.")
    budget_egp = st.number_input("Monthly Budget Alert (EGP)", min_value=0, value=1000, step=50,
                                  help="App will warn you if predicted bill exceeds this.")

# ── HEADER ─────────────────────────────────────────────────────────────────────
st.markdown("""
<div style="padding: 1.5rem 0 0.5rem 0;">
    <span class="model-badge">ELECTRICITY SUITE</span>
    <span class="model-badge">CatBoost</span>
    <span class="model-badge">LightGBM</span>
    <h1 style="margin-top:0.8rem; font-size:2rem; color:#e6edf3; line-height:1.3;">
        Electricity Cost &<br><span style="color:#00e5ff;">Energy Predictor</span>
    </h1>
    <p style="color:#8b949e; font-size:0.9rem; margin-top:0.4rem;">
        Two ML models | PDF report | 12-month forecast | Budget alert<br>
        Inputs default to dataset averages -- only fill in what you know.
    </p>
</div>
""", unsafe_allow_html=True)

st.divider()

# ── TABS ───────────────────────────────────────────────────────────────────────
tab1, tab2 = st.tabs(["CatBoost  --  Home Electricity (kWh + EGP)", "LightGBM  --  Smart City Site (USD)"])

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 -- CatBoost
# ══════════════════════════════════════════════════════════════════════════════
with tab1:
    st.markdown("""
    <span class="model-badge">CatBoost Regressor</span>
    <span class="model-badge">Dual Output: kWh + EGP</span>
    <p class="default-note" style="margin-top:0.6rem;">
        All inputs default to dataset means. Leave unchanged if unsure.
    </p>
    """, unsafe_allow_html=True)

    kwh_model, bill_model, cb_scaler = load_catboost()

    with st.form("catboost_form"):
        st.markdown("#### Home & Appliances")
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
            cb_house   = st.number_input("House Size (m2)", min_value=10.0, value=round(MEANS['house_size_m2'], 0))
            cb_season  = st.selectbox("Season", ["Summer", "Winter"], index=0)
            cb_insul   = st.selectbox("Insulation Quality", ["High", "Medium", "Low"], index=0)

        cb_submit = st.form_submit_button("Predict with CatBoost", use_container_width=True)

    if cb_submit:
        if kwh_model is None:
            st.error("CatBoost models not loaded. Check all .joblib files are in the repo.")
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
            with st.spinner("Running CatBoost..."):
                try:
                    scaled = preprocess_electricity(raw, cb_scaler)
                    kwh    = float(kwh_model.predict(scaled).flatten()[0])
                    bill   = float(bill_model.predict(scaled).flatten()[0])

                    # ── Budget alert ───────────────────────────────────────────
                    if bill > budget_egp:
                        st.markdown(f"""
                        <div class="warning-banner">
                            <h3 style="color:#ff4b4b; margin:0;">Over Budget Alert</h3>
                            <p style="color:#ff4b4b; margin:0.3rem 0 0 0;">
                                Predicted bill <b>{bill:,.1f} EGP</b> exceeds your budget of <b>{budget_egp:,} EGP</b>
                            </p>
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.success(f"Within budget! Predicted bill: {bill:,.1f} EGP (limit: {budget_egp:,} EGP)")

                    # ── Metrics ────────────────────────────────────────────────
                    m1, m2, m3, m4 = st.columns(4)
                    m1.metric("Consumption",   f"{kwh:,.1f} kWh")
                    m2.metric("Monthly Bill",  f"{bill:,.1f} EGP")
                    m3.metric("Annual Bill",   f"{bill*12:,.0f} EGP")
                    m4.metric("Daily Cost",    f"{bill/30:,.1f} EGP")

                    # ── Zone card ──────────────────────────────────────────────
                    color, zone_label, zone_note = cost_zone_egp(bill)
                    st.markdown(f"""
                    <div class="result-card" style="border-left-color:{color};">
                        <span style="font-family:'Space Mono',monospace; font-weight:700;
                                     color:{color}; font-size:0.95rem;">{zone_label}</span><br>
                        <span style="color:#8b949e; font-size:0.85rem;">{zone_note}</span>
                    </div>
                    """, unsafe_allow_html=True)

                    st.markdown("<br>", unsafe_allow_html=True)

                    # ── Chart + PDF side by side ───────────────────────────────
                    chart_col, pdf_col = st.columns([3, 1])

                    with chart_col:
                        st.markdown("#### 12-Month Cost Forecast")
                        trend_vals = monthly_trend(bill)
                        fig = plot_trend(trend_vals, "Estimated Monthly Bill", "EGP", "#00e5ff")
                        st.plotly_chart(fig, use_container_width=True)

                    with pdf_col:
                        st.markdown("#### Download Report")
                        st.markdown("<br>", unsafe_allow_html=True)
                        human_inputs = {
                            "Air Conditioners":    cb_ac,
                            "AC Power (HP)":       cb_ac_hp,
                            "Refrigerators":       cb_fridge,
                            "Televisions":         cb_tv,
                            "Fans":                cb_fans,
                            "Computers":           cb_pc,
                            "Daily Hours":         cb_hours,
                            "House Size (m2)":     cb_house,
                            "Season":              cb_season,
                            "Insulation":          cb_insul,
                            "Water Heater":        cb_heater,
                            "Washing (times/wk)":  cb_washing,
                        }
                        pdf_bytes = generate_pdf_catboost(human_inputs, kwh, bill, ex_rate)
                        st.download_button(
                            label="Download PDF Report",
                            data=pdf_bytes,
                            file_name=f"electricity_report_{datetime.now().strftime('%Y%m%d_%H%M')}.pdf",
                            mime="application/pdf",
                            use_container_width=True,
                        )
                        st.markdown(f"""
                        <div style="margin-top:0.8rem; font-size:0.78rem; color:#8b949e; text-align:center;">
                            Exchange rate used:<br>
                            <b style="color:#e6edf3;">1 USD = {ex_rate} EGP</b>
                        </div>
                        """, unsafe_allow_html=True)

                except Exception as e:
                    st.error(f"Prediction failed: {e}")
                    st.exception(e)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 -- LightGBM
# ══════════════════════════════════════════════════════════════════════════════
with tab2:
    st.markdown("""
    <span class="model-badge">LightGBM Regressor</span>
    <span class="model-badge">Output: USD/month</span>
    <p class="default-note" style="margin-top:0.6rem;">
        Smart-city site model. Inputs default to example values. Leave unchanged if unsure.
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
        st.markdown("#### Site Information")
        lc1, lc2, lc3 = st.columns(3)
        with lc1:
            l_area     = st.number_input("Site Area (m2)",             min_value=100,   value=1360,   step=50)
            l_resident = st.number_input("Resident Count",              min_value=1,     value=6,      step=1)
            l_util     = st.slider("Utilisation Rate (%)",              0, 100, 59)
        with lc2:
            l_water    = st.number_input("Water Consumption (L/day)",   min_value=0.0,   value=2519.0, step=50.0)
            l_recycle  = st.slider("Recycling Rate (%)",                0, 100, 68)
            l_aqi      = st.number_input("Air Quality Index (AQI)",     min_value=0,     max_value=500, value=51)
        with lc3:
            l_issue    = st.number_input("Issue Resolution Time (hrs)", min_value=0,     value=34,     step=1)
            l_struct   = st.selectbox("Structure Type",
                                      ["Commercial", "Industrial", "Mixed-use", "Residential"],
                                      index=2)

        lgbm_submit = st.form_submit_button("Predict with LightGBM", use_container_width=True)

    if lgbm_submit:
        if lgbm_model is None:
            st.error("LightGBM model not loaded. Check lightgbm_regressor_model.joblib and scaler.joblib are in the repo.")
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
            with st.spinner("Running LightGBM..."):
                try:
                    df_in  = pd.DataFrame([lgbm_input], columns=LGBM_COLS)
                    scaled = lgbm_scaler.transform(df_in)
                    cost   = float(lgbm_model.predict(scaled)[0])
                    cost_egp = cost * ex_rate

                    # ── Budget alert ───────────────────────────────────────────
                    budget_usd = budget_egp / ex_rate
                    if cost > budget_usd:
                        st.markdown(f"""
                        <div class="warning-banner">
                            <h3 style="color:#ff4b4b; margin:0;">Over Budget Alert</h3>
                            <p style="color:#ff4b4b; margin:0.3rem 0 0 0;">
                                Predicted cost <b>${cost:,.2f}</b> exceeds your budget of <b>${budget_usd:,.2f}</b>
                            </p>
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.success(f"Within budget! Predicted cost: ${cost:,.2f} (limit: ${budget_usd:,.2f})")

                    # ── Metrics ────────────────────────────────────────────────
                    lm1, lm2, lm3, lm4 = st.columns(4)
                    lm1.metric("Monthly Cost",    f"${cost:,.2f}")
                    lm2.metric("In EGP",         f"{cost_egp:,.1f}")
                    lm3.metric("Annual Estimate", f"${cost*12:,.2f}")
                    lm4.metric("Daily Average",   f"${cost/30:,.2f}")

                    # ── Zone card ──────────────────────────────────────────────
                    color, zone_label, zone_note = cost_zone_usd(cost)
                    st.markdown(f"""
                    <div class="result-card" style="border-left-color:{color};">
                        <span style="font-family:'Space Mono',monospace; font-weight:700;
                                     color:{color}; font-size:0.95rem;">{zone_label}</span><br>
                        <span style="color:#8b949e; font-size:0.85rem;">{zone_note}</span>
                    </div>
                    """, unsafe_allow_html=True)

                    st.markdown("<br>", unsafe_allow_html=True)

                    # ── Chart + PDF side by side ───────────────────────────────
                    chart_col2, pdf_col2 = st.columns([3, 1])

                    with chart_col2:
                        st.markdown("#### 12-Month Cost Forecast")
                        trend_vals_usd = monthly_trend(cost)
                        fig2 = plot_trend(trend_vals_usd, "Estimated Monthly Cost", "USD", "#7c4dff")
                        st.plotly_chart(fig2, use_container_width=True)

                    with pdf_col2:
                        st.markdown("#### Download Report")
                        st.markdown("<br>", unsafe_allow_html=True)
                        pdf_bytes2 = generate_pdf_lgbm(lgbm_input, cost, ex_rate)
                        st.download_button(
                            label="Download PDF Report",
                            data=pdf_bytes2,
                            file_name=f"smartcity_report_{datetime.now().strftime('%Y%m%d_%H%M')}.pdf",
                            mime="application/pdf",
                            use_container_width=True,
                        )
                        st.markdown(f"""
                        <div style="margin-top:0.8rem; font-size:0.78rem; color:#8b949e; text-align:center;">
                            Exchange rate used:<br>
                            <b style="color:#e6edf3;">1 USD = {ex_rate} EGP</b>
                        </div>
                        """, unsafe_allow_html=True)

                except Exception as e:
                    st.error(f"Prediction failed: {e}")
                    st.exception(e)

st.divider()
st.markdown(
    "<p style='text-align:center; color:#30363d; font-size:0.78rem; font-family:Space Mono,monospace;'>"
    "CatBoost | LightGBM | Streamlit | Smart Energy Analytics</p>",
    unsafe_allow_html=True,
)
