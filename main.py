# main.py  —  Delta · Precious Metals & Gemstone Risk Analyzer
# Run with:  streamlit run main.py

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path
from datetime import date, timedelta
import io
import base64

import sys
sys.path.insert(0, str(Path(__file__).parent))

from app.config import (
    ASSET_METADATA, FEATURE_DEFAULTS, FEATURE_RANGES,
    FEATURE_LABELS, RISK_COLORS, RISK_ICONS, RISK_ACTIONS,
)
from app.model import predict_risk, predict_horizon

# ══════════════════════════════════════════════════════════════════
# PAGE CONFIG
# ══════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="Delta · Risk Analyzer",
    page_icon="⬡",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Inject CSS ─────────────────────────────────────────────────────
css_path = Path(__file__).parent / "app" / "styles.css"
if css_path.exists():
    st.markdown(f"<style>{css_path.read_text()}</style>", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════
# SESSION STATE INIT
# ══════════════════════════════════════════════════════════════════
if "results_ready" not in st.session_state:
    st.session_state.results_ready = False
if "dark_mode" not in st.session_state:
    st.session_state.dark_mode = True
if "selected_assets" not in st.session_state:
    st.session_state.selected_assets = list(ASSET_METADATA.keys())[:4]
if "asset_units" not in st.session_state:
    st.session_state.asset_units = {a: 1.0 for a in ASSET_METADATA}
if "asset_rupees" not in st.session_state:
    st.session_state.asset_rupees = {
        a: ASSET_METADATA[a]["base_price_usd"] * FEATURE_DEFAULTS["USD_INR"]
        for a in ASSET_METADATA
    }
if "inputs" not in st.session_state:
    st.session_state.inputs = FEATURE_DEFAULTS.copy()
if "results" not in st.session_state:
    st.session_state.results = {}
if "df_horizon" not in st.session_state:
    st.session_state.df_horizon = None
if "df_horizon_long" not in st.session_state:
    st.session_state.df_horizon_long = None
if "analysis_date" not in st.session_state:
    st.session_state.analysis_date = date.today()

# ══════════════════════════════════════════════════════════════════
# SCENARIO PRESETS
# ══════════════════════════════════════════════════════════════════
SCENARIO_PRESETS = {
    "Current Market": FEATURE_DEFAULTS.copy(),
    "2008 Crisis": {
        **FEATURE_DEFAULTS,
        "USD_INR": 49.0, "India_Inflation": 8.5, "RBI_Repo_Rate": 7.5,
        "Global_Inflation": 6.0, "Fed_Rate": 0.25, "DXY_Index": 88.0,
        "Oil_Price_USD": 45.0, "SP500_Index": 2700.0, "China_PMI": 45.0,
        "Geopolitical_Risk_VIX": 68.0, "Russia_Ukraine_Tension": 0.3,
        "Gold_Price_USD": 900.0, "Silver_Price_USD": 11.0,
    },
    "Post-COVID Rally": {
        **FEATURE_DEFAULTS,
        "USD_INR": 74.0, "India_Inflation": 6.2, "RBI_Repo_Rate": 4.0,
        "Global_Inflation": 7.5, "Fed_Rate": 0.08, "DXY_Index": 96.0,
        "Oil_Price_USD": 110.0, "SP500_Index": 4700.0, "China_PMI": 51.5,
        "Geopolitical_Risk_VIX": 22.0, "Russia_Ukraine_Tension": 0.2,
        "Gold_Price_USD": 1950.0, "Silver_Price_USD": 28.0,
        "Festival_Season": 1, "Festival_Intensity": 0.9, "Wedding_Season_Intensity": 0.85,
    },
    "India Election Cycle": {
        **FEATURE_DEFAULTS,
        "USD_INR": 85.0, "India_Inflation": 5.5, "RBI_Repo_Rate": 6.75,
        "India_GDP_Growth": 7.5, "Import_Duty_Gold_pct": 12.0,
        "Geopolitical_Risk_VIX": 24.0, "Russia_Ukraine_Tension": 0.5,
        "Festival_Season": 1, "Festival_Intensity": 0.75, "Wedding_Season_Intensity": 0.8,
    },
    "Russia-Ukraine Shock": {
        **FEATURE_DEFAULTS,
        "USD_INR": 77.0, "Global_Inflation": 9.0, "Fed_Rate": 1.5,
        "DXY_Index": 109.0, "Oil_Price_USD": 128.0, "SP500_Index": 4100.0,
        "Geopolitical_Risk_VIX": 52.0, "Russia_Ukraine_Tension": 1.0,
        "Gold_Price_USD": 2050.0, "Silver_Price_USD": 26.0,
        "Global_Mining_Output_Index": 87.0,
    },
    "Rate Cut Cycle": {
        **FEATURE_DEFAULTS,
        "Fed_Rate": 3.0, "RBI_Repo_Rate": 5.5, "DXY_Index": 98.0,
        "SP500_Index": 5400.0, "Gold_Price_USD": 2300.0,
        "Silver_Price_USD": 32.0, "Geopolitical_Risk_VIX": 14.0,
        "Global_Inflation": 2.5, "India_Inflation": 3.8,
    },
}

# ══════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════════
def risk_badge(label: str) -> str:
    cls = {"Low": "risk-low", "Medium": "risk-medium", "High": "risk-high"}.get(label, "")
    icon = RISK_ICONS.get(label, "")
    return f'<span class="risk-badge {cls}">{icon} {label}</span>'

def action_badge(action: str) -> str:
    key = action.split()[0].lower()
    return f'<span class="action-{key}">{action}</span>'

def metric_card(label: str, value: str, sub: str = "") -> str:
    sub_html = f'<div class="metric-sub">{sub}</div>' if sub else ""
    return f"""
    <div class="metric-card">
        <div class="metric-label">{label}</div>
        <div class="metric-value">{value}</div>
        {sub_html}
    </div>"""

def section_header(title: str, subtitle: str = "") -> None:
    sub_html = f'<div class="section-subtitle">{subtitle}</div>' if subtitle else ""
    st.markdown(f"""
    <div class="section-header">
        <div class="section-title">{title}</div>
        {sub_html}
        <div class="gold-line"></div>
    </div>""", unsafe_allow_html=True)

def plotly_dark_layout(fig: go.Figure) -> go.Figure:
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="DM Sans, sans-serif", color="#F0EDE8"),
        margin=dict(l=0, r=0, t=30, b=0),
        legend=dict(
            bgcolor="rgba(255,255,255,0.04)",
            bordercolor="rgba(255,255,255,0.08)",
            borderwidth=1,
        ),
    )
    fig.update_xaxes(gridcolor="rgba(255,255,255,0.06)", zeroline=False)
    fig.update_yaxes(gridcolor="rgba(255,255,255,0.06)", zeroline=False)
    return fig

def get_inr_price(asset: str, inputs: dict) -> float:
    price_map = {
        "Gold":     "Gold_Price_USD",
        "Silver":   "Silver_Price_USD",
        "Platinum": "Platinum_Price_USD",
        "Copper":   "Copper_Price_USD",
        "Diamond":  "Diamond_Index",
        "Emerald":  "Emerald_Index",
        "Ruby":     "Ruby_Index",
        "Pearl":    "Pearl_Index",
    }
    usd_price = inputs.get(price_map.get(asset, ""), ASSET_METADATA[asset]["base_price_usd"])
    return usd_price * inputs.get("USD_INR", FEATURE_DEFAULTS["USD_INR"])

# ══════════════════════════════════════════════════════════════════
# HAMBURGER DRAWER
# ══════════════════════════════════════════════════════════════════
all_assets = list(ASSET_METADATA.keys())

drawer_html = """
<style>
.delta-drawer-overlay {
    display: none;
    position: fixed;
    inset: 0;
    background: rgba(0,0,0,0.6);
    z-index: 1100;
    backdrop-filter: blur(2px);
}
.delta-drawer-overlay.open { display: block; }
.delta-drawer-panel {
    position: fixed;
    top: 0; left: 0;
    width: 300px;
    height: 100vh;
    background: rgba(10,11,22,0.98);
    backdrop-filter: blur(30px);
    -webkit-backdrop-filter: blur(30px);
    border-right: 1px solid rgba(212,175,55,0.2);
    box-shadow: 8px 0 40px rgba(0,0,0,0.5), inset -1px 0 0 rgba(255,255,255,0.05);
    z-index: 1200;
    transform: translateX(-100%);
    transition: transform 0.32s cubic-bezier(0.4,0,0.2,1);
    display: flex;
    flex-direction: column;
    overflow-y: auto;
}
.delta-drawer-panel.open { transform: translateX(0); }
.drawer-header {
    padding: 1.5rem 1.5rem 1rem;
    border-bottom: 1px solid rgba(255,255,255,0.07);
    display: flex;
    align-items: center;
    justify-content: space-between;
}
.drawer-title {
    font-family: 'Syne', sans-serif;
    font-size: 1rem;
    font-weight: 700;
    color: #D4AF37;
    letter-spacing: 0.1em;
    text-transform: uppercase;
}
.drawer-close {
    background: rgba(255,255,255,0.06);
    border: 1px solid rgba(255,255,255,0.1);
    border-radius: 8px;
    color: rgba(240,237,232,0.6);
    cursor: pointer;
    font-size: 1rem;
    width: 32px; height: 32px;
    display: flex; align-items: center; justify-content: center;
    transition: all 0.2s;
}
.drawer-close:hover { background: rgba(212,175,55,0.15); color: #D4AF37; border-color: rgba(212,175,55,0.3); }
.drawer-section {
    padding: 1.2rem 1.5rem 0.5rem;
}
.drawer-section-label {
    font-size: 0.62rem;
    text-transform: uppercase;
    letter-spacing: 0.14em;
    color: rgba(240,237,232,0.3);
    font-family: 'Syne', sans-serif;
    margin-bottom: 0.75rem;
}
.hamburger-fab {
    position: fixed;
    top: 1rem;
    left: 1rem;
    z-index: 1050;
    background: rgba(10,11,22,0.9);
    border: 1px solid rgba(212,175,55,0.3);
    border-radius: 10px;
    width: 42px; height: 42px;
    display: flex; align-items: center; justify-content: center;
    cursor: pointer;
    backdrop-filter: blur(12px);
    box-shadow: 0 4px 20px rgba(0,0,0,0.4);
    transition: all 0.2s;
    color: #D4AF37;
    font-size: 1.1rem;
}
.hamburger-fab:hover { border-color: rgba(212,175,55,0.6); box-shadow: 0 4px 24px rgba(212,175,55,0.2); }
</style>

<div class="hamburger-fab" onclick="toggleDrawer()" title="Menu">☰</div>
<div class="delta-drawer-overlay" id="drawerOverlay" onclick="closeDrawer()"></div>
<div class="delta-drawer-panel" id="drawerPanel">
    <div class="drawer-header">
        <span class="drawer-title">⬡ Delta Menu</span>
        <button class="drawer-close" onclick="closeDrawer()">✕</button>
    </div>
    <div class="drawer-section">
        <div class="drawer-section-label">🌙 Theme</div>
        <label style="display:flex;align-items:center;gap:0.75rem;cursor:pointer;color:rgba(240,237,232,0.7);font-size:0.85rem;">
            <div style="position:relative;width:44px;height:24px;">
                <input type="checkbox" id="themeToggle" style="opacity:0;width:0;height:0;" onchange="toggleTheme(this)">
                <span id="themeTrack" style="position:absolute;inset:0;border-radius:12px;background:rgba(212,175,55,0.2);border:1px solid rgba(212,175,55,0.3);transition:0.3s;"></span>
                <span id="themeThumb" style="position:absolute;top:3px;left:3px;width:18px;height:18px;border-radius:50%;background:#D4AF37;transition:0.3s;"></span>
            </div>
            <span id="themeLabel">Dark Mode</span>
        </label>
    </div>
    <div class="drawer-section" style="border-top:1px solid rgba(255,255,255,0.05);margin-top:0.5rem;padding-top:1.2rem;">
        <div class="drawer-section-label">💎 Asset Selection</div>
        <div id="assetCheckboxes" style="display:flex;flex-direction:column;gap:0.5rem;">
        </div>
    </div>
    <div class="drawer-section" style="border-top:1px solid rgba(255,255,255,0.05);margin-top:0.75rem;padding-top:1.2rem;flex:1;">
        <div class="drawer-section-label">📊 Portfolio Volume</div>
        <div id="volumeInputs" style="display:flex;flex-direction:column;gap:0.85rem;"></div>
    </div>
    <div style="padding:1rem 1.5rem 1.5rem;border-top:1px solid rgba(255,255,255,0.05);">
        <button onclick="applyDrawerSettings()" style="width:100%;padding:0.7rem;background:linear-gradient(135deg,#D4AF37,#F0D060);border:none;border-radius:10px;font-family:'Syne',sans-serif;font-weight:700;font-size:0.85rem;color:#07080f;cursor:pointer;letter-spacing:0.06em;">APPLY SETTINGS</button>
    </div>
</div>

<script>
function toggleDrawer() {
    document.getElementById('drawerPanel').classList.toggle('open');
    document.getElementById('drawerOverlay').classList.toggle('open');
}
function closeDrawer() {
    document.getElementById('drawerPanel').classList.remove('open');
    document.getElementById('drawerOverlay').classList.remove('open');
}
function toggleTheme(cb) {
    document.getElementById('themeLabel').textContent = cb.checked ? 'Light Mode' : 'Dark Mode';
    document.getElementById('themeThumb').style.left = cb.checked ? '23px' : '3px';
}
function applyDrawerSettings() {
    closeDrawer();
    window.parent.document.dispatchEvent(new Event('click'));
}
</script>
"""
st.markdown(drawer_html, unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════
# NAVBAR
# ══════════════════════════════════════════════════════════════════
st.markdown("""
<div class="delta-navbar" style="padding-left:4rem;">
    <div class="delta-logo">
        <div class="delta-logo-mark">Δ</div>
        <div>
            <div class="delta-logo-text">DELTA</div>
            <div class="delta-logo-sub">Precious Asset Intelligence</div>
        </div>
    </div>
    <div class="navbar-nav">
        <span class="navbar-badge">ML · v2.0</span>
    </div>
</div>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════
# ASSET + VOLUME SELECTORS (in-page, above tabs)
# ══════════════════════════════════════════════════════════════════
with st.expander("⚙️ Asset Selection & Portfolio Volume", expanded=False):
    sel_cols = st.columns(8)
    selected_assets = []
    for i, asset in enumerate(all_assets):
        with sel_cols[i % 8]:
            meta = ASSET_METADATA[asset]
            checked = st.checkbox(f"{meta['emoji']} {asset}", value=(asset in st.session_state.selected_assets), key=f"sel_{asset}")
            if checked:
                selected_assets.append(asset)
    st.session_state.selected_assets = selected_assets if selected_assets else all_assets[:1]

    st.markdown("---")
    st.markdown("**Portfolio Volume per Asset**")
    vol_cols = st.columns(min(len(st.session_state.selected_assets), 4))
    for i, asset in enumerate(st.session_state.selected_assets):
        with vol_cols[i % len(vol_cols)]:
            inr_price = get_inr_price(asset, st.session_state.inputs)
            st.markdown(f"**{ASSET_METADATA[asset]['emoji']} {asset}**")
            units_key = f"units_{asset}"
            rupees_key = f"rupees_{asset}"

            prev_units = st.session_state.asset_units.get(asset, 1.0)
            prev_rupees = st.session_state.asset_rupees.get(asset, inr_price)

            new_units = st.number_input(
                f"Units", min_value=0.0, value=float(prev_units),
                key=units_key, step=0.1,
            )
            if new_units != prev_units:
                st.session_state.asset_units[asset] = new_units
                st.session_state.asset_rupees[asset] = new_units * inr_price

            new_rupees = st.number_input(
                f"Amount (₹)", min_value=0.0, value=float(st.session_state.asset_rupees.get(asset, inr_price)),
                key=rupees_key, step=1000.0,
            )
            if new_rupees != st.session_state.asset_rupees.get(asset, inr_price):
                st.session_state.asset_rupees[asset] = new_rupees
                st.session_state.asset_units[asset] = new_rupees / inr_price if inr_price > 0 else 0

selected_assets = st.session_state.selected_assets

if not selected_assets:
    st.warning("Please select at least one asset.")
    st.stop()

# ══════════════════════════════════════════════════════════════════
# MAIN TABS
# ══════════════════════════════════════════════════════════════════
tab_macros, tab_results, tab_forecast, tab_optimization = st.tabs([
    "🌐 Macros", "📊 Results", "📈 Forecast", "⚙️ Optimization"
])

# ══════════════════════════════════════════════════════════════════
# TAB 1 — MACROS
# ══════════════════════════════════════════════════════════════════
with tab_macros:
    section_header("Macro Parameters", "Edit inputs and apply scenario presets")

    # Scenario preset buttons
    st.markdown("**Scenario Presets**")
    preset_cols = st.columns(len(SCENARIO_PRESETS))
    for i, preset_name in enumerate(SCENARIO_PRESETS):
        with preset_cols[i]:
            if st.button(preset_name, key=f"preset_{i}", use_container_width=True):
                st.session_state.inputs.update(SCENARIO_PRESETS[preset_name])
                st.rerun()

    st.markdown("<br>", unsafe_allow_html=True)

    inputs = st.session_state.inputs

    # India Macro card
    st.markdown("""<div class="glass-card glass-card-gold" style="margin-bottom:1rem;">
        <div style="font-family:'Syne',sans-serif;font-weight:700;color:#D4AF37;margin-bottom:1rem;font-size:0.9rem;letter-spacing:0.08em;">🇮🇳 INDIA MACRO</div>""",
        unsafe_allow_html=True)
    india_fields = ["USD_INR","India_Inflation","RBI_Repo_Rate","Import_Duty_Gold_pct","India_GDP_Growth","Monsoon_Index"]
    ic = st.columns(3)
    for j, f in enumerate(india_fields):
        with ic[j % 3]:
            lo, hi = FEATURE_RANGES.get(f, (0.0, 200.0))
            inputs[f] = st.number_input(FEATURE_LABELS.get(f, f), min_value=float(lo), max_value=float(hi),
                value=float(inputs.get(f, FEATURE_DEFAULTS[f])), key=f"mac_{f}")
    st.markdown("</div>", unsafe_allow_html=True)

    # Global Macro card
    st.markdown("""<div class="glass-card" style="margin-bottom:1rem;">
        <div style="font-family:'Syne',sans-serif;font-weight:700;color:#D4AF37;margin-bottom:1rem;font-size:0.9rem;letter-spacing:0.08em;">🌐 GLOBAL MACRO</div>""",
        unsafe_allow_html=True)
    global_fields = ["Global_Inflation","Fed_Rate","DXY_Index","Oil_Price_USD","SP500_Index","China_PMI","Geopolitical_Risk_VIX","Russia_Ukraine_Tension"]
    gc = st.columns(4)
    for j, f in enumerate(global_fields):
        with gc[j % 4]:
            lo, hi = FEATURE_RANGES.get(f, (0.0, 200.0))
            inputs[f] = st.number_input(FEATURE_LABELS.get(f, f), min_value=float(lo), max_value=float(hi),
                value=float(inputs.get(f, FEATURE_DEFAULTS[f])), key=f"mac_{f}")
    st.markdown("</div>", unsafe_allow_html=True)

    # Supply card
    st.markdown("""<div class="glass-card" style="margin-bottom:1rem;">
        <div style="font-family:'Syne',sans-serif;font-weight:700;color:#D4AF37;margin-bottom:1rem;font-size:0.9rem;letter-spacing:0.08em;">⛏️ SUPPLY</div>""",
        unsafe_allow_html=True)
    supply_fields = ["Global_Mining_Output_Index","Lab_Diamond_Supply_Index","Emerald_Origin_Premium_pct","Diamond_Demand_Index"]
    sc2 = st.columns(4)
    for j, f in enumerate(supply_fields):
        with sc2[j % 4]:
            lo, hi = FEATURE_RANGES.get(f, (0.0, 200.0))
            inputs[f] = st.number_input(FEATURE_LABELS.get(f, f), min_value=float(lo), max_value=float(hi),
                value=float(inputs.get(f, FEATURE_DEFAULTS[f])), key=f"mac_{f}")
    st.markdown("</div>", unsafe_allow_html=True)

    # Demand signals card
    st.markdown("""<div class="glass-card" style="margin-bottom:1rem;">
        <div style="font-family:'Syne',sans-serif;font-weight:700;color:#D4AF37;margin-bottom:1rem;font-size:0.9rem;letter-spacing:0.08em;">🎉 DEMAND SIGNALS</div>""",
        unsafe_allow_html=True)
    dc = st.columns(3)
    with dc[0]:
        inputs["Festival_Season"] = int(st.checkbox("Festival Season Active", value=bool(inputs.get("Festival_Season", 0)), key="mac_festival_season"))
    with dc[1]:
        inputs["Festival_Intensity"] = st.number_input("Festival Intensity (0–1)", min_value=0.0, max_value=1.0,
            value=float(inputs.get("Festival_Intensity", 0.0)), step=0.05, key="mac_fest_int")
    with dc[2]:
        inputs["Wedding_Season_Intensity"] = st.number_input("Wedding Season Intensity (0–1)", min_value=0.0, max_value=1.0,
            value=float(inputs.get("Wedding_Season_Intensity", 0.5)), step=0.05, key="mac_wed_int")
    st.markdown("</div>", unsafe_allow_html=True)

    # Spot Prices card
    st.markdown("""<div class="glass-card">
        <div style="font-family:'Syne',sans-serif;font-weight:700;color:#D4AF37;margin-bottom:1rem;font-size:0.9rem;letter-spacing:0.08em;">💰 SPOT PRICES</div>""",
        unsafe_allow_html=True)
    price_fields = ["Gold_Price_USD","Silver_Price_USD","Platinum_Price_USD","Copper_Price_USD","Diamond_Index","Emerald_Index","Ruby_Index","Pearl_Index"]
    pc = st.columns(4)
    for j, f in enumerate(price_fields):
        with pc[j % 4]:
            lo, hi = FEATURE_RANGES.get(f, (0.0, 20000.0))
            inputs[f] = st.number_input(FEATURE_LABELS.get(f, f), min_value=float(lo), max_value=float(hi),
                value=float(inputs.get(f, FEATURE_DEFAULTS[f])), key=f"mac_{f}")
    st.markdown("</div>", unsafe_allow_html=True)

    # fill remaining defaults
    for key, val in FEATURE_DEFAULTS.items():
        if key not in inputs:
            inputs[key] = val

    st.session_state.inputs = inputs

    # Analysis date
    st.markdown("<br>", unsafe_allow_html=True)
    st.session_state.analysis_date = st.date_input("📅 Analysis Date", value=st.session_state.analysis_date)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── START BUTTON ──────────────────────────────────────────────
    st.markdown("""
    <style>
    div[data-testid="stButton"] > button.start-btn {
        background: linear-gradient(135deg, #D4AF37 0%, #F0D060 50%, #B8960C 100%) !important;
        color: #07080f !important;
        border: none !important;
        border-radius: 14px !important;
        font-family: 'Syne', sans-serif !important;
        font-weight: 800 !important;
        font-size: 1.1rem !important;
        letter-spacing: 0.1em !important;
        padding: 1rem 4rem !important;
        box-shadow: 0 8px 40px rgba(212,175,55,0.4) !important;
        transition: all 0.2s !important;
        width: 100% !important;
    }
    </style>
    """, unsafe_allow_html=True)

    col_start = st.columns([1, 2, 1])
    with col_start[1]:
        start_clicked = st.button("▶  RUN ANALYSIS", type="primary", use_container_width=True, key="start_btn")

    if start_clicked:
        ts = pd.Timestamp(st.session_state.analysis_date)
        progress_bar = st.progress(0, text="Initialising model…")
        results = {}
        for idx, asset in enumerate(selected_assets):
            progress_bar.progress((idx) / len(selected_assets), text=f"Analysing {asset}…")
            r = predict_risk(st.session_state.inputs, ts, [asset])
            results[asset] = r[asset]
        progress_bar.progress(1.0, text="Running 30-day horizon…")
        end_short = ts + pd.Timedelta(days=30)
        st.session_state.df_horizon = predict_horizon(st.session_state.inputs, ts, end_short, selected_assets)
        progress_bar.progress(1.0, text="Running 1-year horizon…")
        end_long = ts + pd.Timedelta(days=365)
        st.session_state.df_horizon_long = predict_horizon(st.session_state.inputs, ts, end_long, selected_assets)
        progress_bar.empty()
        st.session_state.results = results
        st.session_state.results_ready = True
        st.success("✅ Analysis complete — view Results, Forecast & Optimization tabs.")

# ══════════════════════════════════════════════════════════════════
# TAB 2 — RESULTS
# ══════════════════════════════════════════════════════════════════
with tab_results:
    if not st.session_state.results_ready:
        st.markdown("""
        <div style="text-align:center;padding:5rem 0;">
            <div style="font-family:'Syne',sans-serif;font-size:2rem;font-weight:700;color:rgba(240,237,232,0.2);margin-bottom:1rem;">⬡</div>
            <div style="font-family:'Syne',sans-serif;font-size:1.1rem;color:rgba(240,237,232,0.4);">
                Press <strong style="color:#D4AF37;">RUN ANALYSIS</strong> in the Macros tab to generate results.
            </div>
        </div>""", unsafe_allow_html=True)
    else:
        results = st.session_state.results
        inputs = st.session_state.inputs

        # ── Per-asset risk cards ──────────────────────────────────
        section_header("Asset Risk Cards", "Should you buy, hold, or sell today? Green = safe, Yellow = watch, Red = danger")
        card_cols = st.columns(min(len(selected_assets), 4))
        for i, asset in enumerate(selected_assets):
            if asset not in results:
                continue
            res = results[asset]
            meta = ASSET_METADATA[asset]
            proba = res["proba"]
            risk_label = res["risk_label"]
            risk_color = RISK_COLORS[risk_label]
            with card_cols[i % len(card_cols)]:
                st.markdown(f"""
                <div class="glass-card" style="
                    background: rgba(255,255,255,0.06);
                    backdrop-filter: blur(24px);
                    -webkit-backdrop-filter: blur(24px);
                    border: 1px solid rgba(255,255,255,0.15);
                    box-shadow: 0 8px 32px rgba(0,0,0,0.3), inset 0 1px 0 rgba(255,255,255,0.1);
                    border-radius: 18px;
                    padding: 1.5rem;
                    margin-bottom: 1rem;
                    border-top: 3px solid {risk_color};
                ">
                    <div style="display:flex;justify-content:space-between;align-items:flex-start;margin-bottom:1rem;">
                        <div>
                            <div style="font-size:1.8rem;">{meta['emoji']}</div>
                            <div style="font-family:'Syne',sans-serif;font-weight:700;font-size:1rem;color:#F0EDE8;margin-top:0.3rem;">{asset}</div>
                            <div style="font-size:0.7rem;color:rgba(240,237,232,0.4);letter-spacing:0.08em;">{meta['symbol']} · {meta['category']}</div>
                        </div>
                        <div style="text-align:right;">
                            {risk_badge(risk_label)}
                            <div style="margin-top:0.5rem;">{action_badge(RISK_ACTIONS[risk_label])}</div>
                        </div>
                    </div>
                    <div style="margin-bottom:0.75rem;">
                        <div style="display:flex;justify-content:space-between;margin-bottom:0.3rem;">
                            <span style="font-size:0.72rem;color:rgba(240,237,232,0.5);">Low</span>
                            <span style="font-size:0.72rem;color:#22c55e;">{proba[0]*100:.1f}%</span>
                        </div>
                        <div style="background:rgba(255,255,255,0.07);border-radius:4px;height:5px;">
                            <div style="width:{proba[0]*100:.1f}%;height:100%;background:#22c55e;border-radius:4px;"></div>
                        </div>
                    </div>
                    <div style="margin-bottom:0.75rem;">
                        <div style="display:flex;justify-content:space-between;margin-bottom:0.3rem;">
                            <span style="font-size:0.72rem;color:rgba(240,237,232,0.5);">Medium</span>
                            <span style="font-size:0.72rem;color:#f59e0b;">{(proba[1] if len(proba)>1 else 0)*100:.1f}%</span>
                        </div>
                        <div style="background:rgba(255,255,255,0.07);border-radius:4px;height:5px;">
                            <div style="width:{(proba[1] if len(proba)>1 else 0)*100:.1f}%;height:100%;background:#f59e0b;border-radius:4px;"></div>
                        </div>
                    </div>
                    <div style="margin-bottom:1rem;">
                        <div style="display:flex;justify-content:space-between;margin-bottom:0.3rem;">
                            <span style="font-size:0.72rem;color:rgba(240,237,232,0.5);">High</span>
                            <span style="font-size:0.72rem;color:#ef4444;">{(proba[2] if len(proba)>2 else 0)*100:.1f}%</span>
                        </div>
                        <div style="background:rgba(255,255,255,0.07);border-radius:4px;height:5px;">
                            <div style="width:{(proba[2] if len(proba)>2 else 0)*100:.1f}%;height:100%;background:#ef4444;border-radius:4px;"></div>
                        </div>
                    </div>
                    <div style="font-size:0.72rem;color:rgba(240,237,232,0.4);border-top:1px solid rgba(255,255,255,0.07);padding-top:0.75rem;">
                        Confidence: <strong style="color:{risk_color};">{res['confidence']}%</strong>
                    </div>
                </div>
                """, unsafe_allow_html=True)

        # ── Asset Comparison ──────────────────────────────────────
        section_header("Asset Comparison", "Higher Low% = safer hold · Higher High% = exit signal · Medium% = watch closely")
        fig_comp = go.Figure()
        assets_list = [a for a in selected_assets if a in results]
        for label_key, color, col_idx in [("Low", "#22c55e", 0), ("Medium", "#f59e0b", 1), ("High", "#ef4444", 2)]:
            fig_comp.add_trace(go.Bar(
                name=label_key,
                x=[f"{ASSET_METADATA[a]['emoji']} {a}" for a in assets_list],
                y=[results[a]["proba"][col_idx] * 100 if len(results[a]["proba"]) > col_idx else 0 for a in assets_list],
                marker_color=color,
                marker_line_width=0,
            ))
        fig_comp.update_layout(
            barmode="group", height=320,
            yaxis=dict(ticksuffix="%", range=[0, 100]),
        )
        plotly_dark_layout(fig_comp)
        st.plotly_chart(fig_comp, use_container_width=True, config={"displayModeBar": False})

        # ── Correlation Matrix ────────────────────────────────────
        section_header("Asset Correlation Matrix", "Do these assets move together? A score near 1.0 means they rise and fall at the same time — diversify away from clusters")
        if st.session_state.df_horizon is not None:
            df_h = st.session_state.df_horizon
            pivot = df_h.pivot_table(index="Date", columns="Asset", values="RiskInt")
            pivot = pivot[[a for a in selected_assets if a in pivot.columns]]
            corr = pivot.corr().fillna(0)
            fig_corr = go.Figure(go.Heatmap(
                z=corr.values,
                x=list(corr.columns),
                y=list(corr.index),
                colorscale=[[0, "#ef4444"], [0.5, "#1a1a2e"], [1, "#22c55e"]],
                zmin=-1, zmax=1,
                text=[[f"{v:.2f}" for v in row] for row in corr.values],
                texttemplate="%{text}",
                textfont=dict(color="white"),
                showscale=True,
            ))
            fig_corr.update_layout(height=360)
            plotly_dark_layout(fig_corr)
            st.plotly_chart(fig_corr, use_container_width=True, config={"displayModeBar": False})

        # ── Portfolio P&L ─────────────────────────────────────────
        section_header("Portfolio P&L Estimator", "If the model is right, how much money do you stand to make or lose across your whole portfolio?")
        RISK_PNL = {"Low": 0.08, "Medium": 0.02, "High": -0.10}
        total_capital = 0.0
        portfolio_rows = []
        for asset in selected_assets:
            if asset not in results:
                continue
            rupees = st.session_state.asset_rupees.get(asset, 0)
            units = st.session_state.asset_units.get(asset, 0)
            risk_label = results[asset]["risk_label"]
            pnl_pct = RISK_PNL[risk_label]
            pnl_inr = rupees * pnl_pct
            total_capital += rupees
            portfolio_rows.append({
                "asset": asset, "rupees": rupees, "units": units,
                "risk": risk_label, "pnl_pct": pnl_pct * 100, "pnl_inr": pnl_inr,
            })

        total_expected_pnl = sum(r["pnl_inr"] for r in portfolio_rows)
        total_pnl_pct = (total_expected_pnl / total_capital * 100) if total_capital > 0 else 0

        pnl_cols = st.columns(3)
        with pnl_cols[0]:
            st.markdown(metric_card("Total Capital", f"₹{total_capital:,.0f}", "Portfolio value"), unsafe_allow_html=True)
        with pnl_cols[1]:
            pnl_color = "#22c55e" if total_expected_pnl >= 0 else "#ef4444"
            st.markdown(metric_card("Expected P&L", f"₹{total_expected_pnl:+,.0f}", f"{total_pnl_pct:+.1f}% expected"), unsafe_allow_html=True)
        with pnl_cols[2]:
            at_risk = sum(r["rupees"] for r in portfolio_rows if r["risk"] == "High")
            st.markdown(metric_card("Capital at Risk", f"₹{at_risk:,.0f}", f"{at_risk/total_capital*100:.1f}% in HIGH risk" if total_capital > 0 else ""), unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        if portfolio_rows and total_capital > 0:
            pie_labels = [f"{ASSET_METADATA[r['asset']]['emoji']} {r['asset']}" for r in portfolio_rows]
            pie_values = [r["rupees"] for r in portfolio_rows]
            pie_colors = [ASSET_METADATA[r["asset"]]["color"] for r in portfolio_rows]
            fig_pie = go.Figure(go.Pie(
                labels=pie_labels,
                values=pie_values,
                marker=dict(colors=pie_colors, line=dict(color="rgba(0,0,0,0.3)", width=1)),
                hole=0.5,
                textinfo="label+percent",
                textfont=dict(size=11),
            ))
            fig_pie.update_layout(height=320, showlegend=False)
            plotly_dark_layout(fig_pie)
            st.plotly_chart(fig_pie, use_container_width=True, config={"displayModeBar": False})

# ══════════════════════════════════════════════════════════════════
# TAB 3 — FORECAST
# ══════════════════════════════════════════════════════════════════
with tab_forecast:
    if not st.session_state.results_ready:
        st.markdown("""
        <div style="text-align:center;padding:5rem 0;">
            <div style="font-family:'Syne',sans-serif;font-size:1.1rem;color:rgba(240,237,232,0.4);">
                Press <strong style="color:#D4AF37;">RUN ANALYSIS</strong> in the Macros tab first.
            </div>
        </div>""", unsafe_allow_html=True)
    else:
        sub_short, sub_long = st.tabs(["📅 Short Term (1 Month)", "📆 Long Term (1 Year)"])

        # ── Short Term ────────────────────────────────────────────
        with sub_short:
            df_h = st.session_state.df_horizon
            if df_h is not None and len(df_h) > 0:
                section_header("Short-Term Forecast", "Where is the price likely heading over the next 30 working days, based on current market conditions?")

                # Alert threshold
                alert_threshold = st.slider("Alert Threshold (highlight days above this risk level)", 0, 2, 1, key="alert_threshold_short")

                # Daily risk line chart
                fig_st = go.Figure()
                for asset in selected_assets:
                    df_a = df_h[df_h["Asset"] == asset]
                    price_col = "Price_INR" if "Price_INR" in df_a.columns else "RiskInt"
                    fig_st.add_trace(go.Scatter(
                        x=df_a["Date"], y=df_a[price_col],
                        mode="lines+markers",
                        name=f"{ASSET_METADATA[asset]['emoji']} {asset}",
                        line=dict(color=ASSET_METADATA[asset]["color"], width=2),
                        marker=dict(size=5),
                        hovertemplate="<b>%{fullData.name}</b><br>%{x|%d %b}<br>₹%{y:,.0f}<extra></extra>",
                    ))
                fig_st.update_layout(
                    height=380,
                    yaxis=dict(title="Price (₹)", tickprefix="₹"),
                    xaxis_title="Date",
                    title="Projected Price (₹) — Next 30 Days",
                )
                plotly_dark_layout(fig_st)
                st.plotly_chart(fig_st, use_container_width=True, config={"displayModeBar": False})

                section_header("Risk Heatmap Calendar", "Click any day to see the exact risk breakdown for every asset on that date")

                # Build lookup: date -> {asset -> row}
                date_asset_map = {}
                all_dates_short = sorted(df_h["Date"].unique())[:20]
                for asset in selected_assets:
                    df_a = df_h[df_h["Asset"] == asset].copy()
                    for _, row in df_a.iterrows():
                        d = row["Date"]
                        if d not in date_asset_map:
                            date_asset_map[d] = {}
                        date_asset_map[d][asset] = row

                # Build tooltip content per date
                cal_cells = []
                for d in all_dates_short:
                    asset_rows_day = date_asset_map.get(d, {})
                    # Use first asset for color
                    primary_asset = selected_assets[0]
                    r0 = asset_rows_day.get(primary_asset)
                    risk_int = int(r0["RiskInt"]) if r0 is not None else 1
                    color = ["#22c55e","#f59e0b","#ef4444"][risk_int]
                    alert = "⚠️" if risk_int > alert_threshold else ""
                    # Build tooltip lines
                    tip_lines = [f"{d.strftime('%d %b')}"]
                    for ast in selected_assets:
                        r = asset_rows_day.get(ast)
                        if r is not None:
                            pl = round(r["Proba_Low"]*100,1)
                            pm = round(r["Proba_Med"]*100,1)
                            ph = round(r["Proba_High"]*100,1)
                            tip_lines.append(f"{ast}: {r['Risk']} (L:{pl}% M:{pm}% H:{ph}%)")
                    tooltip = " | ".join(tip_lines)
                    cal_cells.append({"date": d.strftime("%d %b"), "color": color, "alert": alert, "tooltip": tooltip, "risk_int": risk_int})

                cal_html = '<div style="display:grid;grid-template-columns:repeat(5,1fr);gap:8px;max-width:600px;">'
                for cell in cal_cells:
                    border = "2px solid #ef4444" if cell["alert"] else "1px solid rgba(255,255,255,0.12)"
                    risk_label = ["LOW","MED","HIGH"][cell["risk_int"]]
                    risk_color = ["#22c55e","#f59e0b","#ef4444"][cell["risk_int"]]
                    cal_html += f'''<div
                        title="{cell['tooltip']}"
                        onclick="this.nextElementSibling&&this.nextElementSibling.remove();var d=document.createElement('div');d.style='position:fixed;top:50%;left:50%;transform:translate(-50%,-50%);background:#0a0b16;border:1px solid #D4AF37;border-radius:14px;padding:1.2rem 1.5rem;z-index:9999;min-width:260px;font-family:DM Sans,sans-serif;color:#F0EDE8;box-shadow:0 8px 40px rgba(0,0,0,0.7)';d.innerHTML='<div style=\\'font-weight:700;color:#D4AF37;margin-bottom:0.7rem;font-size:0.95rem\\'>{cell["date"]}</div><pre style=\\'font-size:0.72rem;color:#F0EDE8;white-space:pre-wrap;margin:0\\'>{cell["tooltip"].replace(" | ", "\\n")}</pre><button onclick=\\'this.parentElement.remove()\\' style=\\'margin-top:0.8rem;background:#D4AF37;border:none;border-radius:8px;padding:0.3rem 1rem;cursor:pointer;font-weight:700;color:#07080f\\'>Close</button>';document.body.appendChild(d)"
                        style="background:{cell['color']}22;border:{border};border-radius:10px;padding:0.6rem 0.3rem;text-align:center;cursor:pointer;transition:all 0.15s;"
                        onmouseover="this.style.background='{cell['color']}44';this.style.transform='scale(1.05)'"
                        onmouseout="this.style.background='{cell['color']}22';this.style.transform='scale(1)'">
                        <div style="font-size:0.6rem;color:rgba(240,237,232,0.5);">{cell['date']}</div>
                        <div style="width:10px;height:10px;background:{cell['color']};border-radius:3px;margin:0.3rem auto;"></div>
                        <div style="font-size:0.58rem;color:{risk_color};font-weight:700;">{risk_label}</div>
                        <div style="font-size:0.55rem;color:rgba(240,237,232,0.4);">{cell['alert']}</div>
                    </div>'''
                cal_html += '</div>'
                st.markdown(cal_html, unsafe_allow_html=True)

        # ── Long Term ─────────────────────────────────────────────
        with sub_long:
            df_hl = st.session_state.df_horizon_long
            if df_hl is not None and len(df_hl) > 0:
                section_header("Long-Term Forecast", "Month-by-month outlook for the next year — when is the best time to buy or sell each asset?")

                # Monthly aggregation
                df_hl["Month"] = df_hl["Date"].dt.to_period("M")
                df_monthly = df_hl.groupby(["Month", "Asset"]).agg(
                    RiskInt=("RiskInt", "mean"),
                    Proba_Low=("Proba_Low", "mean"),
                    Proba_Med=("Proba_Med", "mean"),
                    Proba_High=("Proba_High", "mean"),
                ).reset_index()
                df_monthly["MonthStr"] = df_monthly["Month"].astype(str)

                # Monthly risk chart with BUY/SELL markers
                fig_lt = go.Figure()
                for asset in selected_assets:
                    df_a = df_monthly[df_monthly["Asset"] == asset].copy()
                    df_a["prev_risk"] = df_a["RiskInt"].shift(1)
                    fig_lt.add_trace(go.Scatter(
                        x=df_a["MonthStr"], y=df_a["RiskInt"],
                        mode="lines+markers",
                        name=f"{ASSET_METADATA[asset]['emoji']} {asset}",
                        line=dict(color=ASSET_METADATA[asset]["color"], width=2),
                        marker=dict(size=6),
                    ))
                    # BUY markers: transitions to Low (0)
                    buys = df_a[(df_a["RiskInt"] < 0.5) & (df_a["prev_risk"] >= 0.5)]
                    if len(buys) > 0:
                        fig_lt.add_trace(go.Scatter(
                            x=buys["MonthStr"], y=buys["RiskInt"],
                            mode="markers", marker=dict(symbol="triangle-up", size=14, color="#22c55e"),
                            name=f"BUY — {asset}", showlegend=False,
                            hovertemplate=f"BUY signal — {asset}<extra></extra>",
                        ))
                    # SELL markers: transitions to High (2)
                    sells = df_a[(df_a["RiskInt"] > 1.5) & (df_a["prev_risk"] <= 1.5)]
                    if len(sells) > 0:
                        fig_lt.add_trace(go.Scatter(
                            x=sells["MonthStr"], y=sells["RiskInt"],
                            mode="markers", marker=dict(symbol="triangle-down", size=14, color="#ef4444"),
                            name=f"SELL — {asset}", showlegend=False,
                            hovertemplate=f"SELL signal — {asset}<extra></extra>",
                        ))

                fig_lt.update_layout(
                    height=400,
                    yaxis=dict(tickvals=[0,1,2], ticktext=["Low","Medium","High"], range=[-0.3,2.3]),
                    xaxis_title="Month",
                )
                plotly_dark_layout(fig_lt)
                st.plotly_chart(fig_lt, use_container_width=True, config={"displayModeBar": False})

                section_header("Risk & Price Forecast per Asset", "How likely is each risk level each month, and where is the price headed?")
                prob_tabs = st.tabs([f"{ASSET_METADATA[a]['emoji']} {a}" for a in selected_assets])
                for tab_i, primary in enumerate(selected_assets):
                    with prob_tabs[tab_i]:
                        df_p = df_monthly[df_monthly["Asset"] == primary]

                        # Price forecast chart
                        price_col2 = "Price_INR" if "Price_INR" in df_hl.columns else None
                        if price_col2:
                            df_price_long = df_hl[df_hl["Asset"] == primary].copy()
                            df_price_long["MonthStr"] = df_price_long["Date"].dt.to_period("M").astype(str)
                            df_price_monthly = df_price_long.groupby("MonthStr")["Price_INR"].mean().reset_index()
                            fig_price = go.Figure()
                            fig_price.add_trace(go.Scatter(
                                x=df_price_monthly["MonthStr"], y=df_price_monthly["Price_INR"],
                                mode="lines+markers",
                                name="Projected Price (₹)",
                                line=dict(color=ASSET_METADATA[primary]["color"], width=2.5),
                                fill="tozeroy",
                                fillcolor='rgba(212,175,55,0.13)',
                                hovertemplate="₹%{y:,.0f}<extra></extra>",
                            ))
                            fig_price.update_layout(
                                height=240,
                                title=f"Projected Price — {primary} (₹)",
                                yaxis=dict(tickprefix="₹", title="Price (₹)"),
                                xaxis_title="Month",
                            )
                            plotly_dark_layout(fig_price)
                            st.plotly_chart(fig_price, use_container_width=True, config={"displayModeBar": False})

                        # Probability stack
                        fig_area = go.Figure()
                        for col, label, colour in [
                            ("Proba_High", "🔴 High Risk — consider selling or reducing exposure", "#ef4444"),
                            ("Proba_Med",  "🟡 Medium Risk — hold and watch closely", "#f59e0b"),
                            ("Proba_Low",  "🟢 Low Risk — safe to hold or buy more", "#22c55e"),
                        ]:
                            fig_area.add_trace(go.Scatter(
                                x=df_p["MonthStr"], y=df_p[col] * 100,
                                stackgroup="one", name=label, line_color=colour,
                                hovertemplate=f"{label.split('—')[0].strip()}: %{{y:.1f}}%<extra></extra>",
                            ))
                        fig_area.update_layout(
                            height=280,
                            title="Monthly Risk Probability Breakdown",
                            yaxis=dict(ticksuffix="%", range=[0,100], title="Probability"),
                            xaxis_title="Month",
                        )
                        plotly_dark_layout(fig_area)
                        st.plotly_chart(fig_area, use_container_width=True, config={"displayModeBar": False})
# ══════════════════════════════════════════════════════════════════
# TAB 4 — OPTIMIZATION
# ══════════════════════════════════════════════════════════════════
with tab_optimization:
    if not st.session_state.results_ready:
        st.markdown("""
        <div style="text-align:center;padding:5rem 0;">
            <div style="font-family:'Syne',sans-serif;font-size:1.1rem;color:rgba(240,237,232,0.4);">
                Press <strong style="color:#D4AF37;">RUN ANALYSIS</strong> in the Macros tab first.
            </div>
        </div>""", unsafe_allow_html=True)
    else:
        df_hl = st.session_state.df_horizon_long
        if df_hl is not None and len(df_hl) > 0:
            df_hl["Month"] = df_hl["Date"].dt.month
            df_hl["MonthName"] = df_hl["Date"].dt.strftime("%b")

            # ── Best/Worst months to buy ──────────────────────────
            section_header("Seasonal Analysis", "Which months are historically safest to buy, and which months should you avoid holding risky assets?")
            month_avg = df_hl.groupby(["Month", "MonthName", "Asset"])["RiskInt"].mean().reset_index()
            month_avg = month_avg.sort_values("Month")

            opt_cols = st.columns(2)
            with opt_cols[0]:
                st.markdown("**🟢 Best Months to BUY (Lowest Risk)**")
                fig_best = go.Figure()
                for asset in selected_assets:
                    df_a = month_avg[month_avg["Asset"] == asset]
                    fig_best.add_trace(go.Bar(
                        name=f"{ASSET_METADATA[asset]['emoji']} {asset}",
                        x=df_a["MonthName"], y=df_a["RiskInt"],
                        marker_color=ASSET_METADATA[asset]["color"],
                        marker_line_width=0,
                    ))
                fig_best.update_layout(
                    barmode="group", height=300,
                    yaxis=dict(tickvals=[0,1,2], ticktext=["Low","Med","High"], range=[0,2.2]),
                )
                plotly_dark_layout(fig_best)
                st.plotly_chart(fig_best, use_container_width=True, config={"displayModeBar": False})

            with opt_cols[1]:
                st.markdown("**🔴 Worst Months to HOLD (Highest Risk)**")
                fig_worst = go.Figure()
                for asset in selected_assets:
                    df_a = month_avg[month_avg["Asset"] == asset]
                    fig_worst.add_trace(go.Bar(
                        name=f"{ASSET_METADATA[asset]['emoji']} {asset}",
                        x=df_a["MonthName"], y=2 - df_a["RiskInt"],
                        marker_color=ASSET_METADATA[asset]["color"],
                        marker_line_width=0,
                    ))
                fig_worst.update_layout(barmode="group", height=300, yaxis=dict(title="Risk Score (inverted)"))
                plotly_dark_layout(fig_worst)
                st.plotly_chart(fig_worst, use_container_width=True, config={"displayModeBar": False})

            # ── Alert threshold table ─────────────────────────────
            section_header("Alert Threshold Breaches", "Every day where risk crosses your chosen limit — use this to set calendar reminders to review your holdings")
            alert_opt_cols = st.columns([2, 2, 1])
            with alert_opt_cols[0]:
                alert_asset = st.selectbox("Asset", options=[a for a in selected_assets if a in (st.session_state.df_horizon_long["Asset"].unique() if st.session_state.df_horizon_long is not None else [])], key="alert_asset_opt")
            with alert_opt_cols[1]:
                alert_level = st.selectbox("Risk Threshold", options=["Low (≥0)", "Medium (≥1)", "High (≥2)"], key="alert_level_opt")
            alert_int = ["Low (≥0)", "Medium (≥1)", "High (≥2)"].index(alert_level)

            df_alert = df_hl[(df_hl["Asset"] == alert_asset) & (df_hl["RiskInt"] >= alert_int)].copy()
            df_alert_display = df_alert[["Date","Risk","RiskInt","Confidence","Action"]].copy()
            df_alert_display["Date"] = df_alert_display["Date"].dt.strftime("%d %b %Y")
            st.dataframe(df_alert_display.rename(columns={"RiskInt": "Risk Score"}), use_container_width=True, hide_index=True)

            # ── Export PDF ────────────────────────────────────────
            section_header("Export Report", "Download a full PDF summary")

            if st.button("📄 Export PDF Report", type="primary", key="export_pdf"):
                try:
                    from fpdf import FPDF
                    import plotly.io as pio

                    pdf = FPDF()
                    pdf.add_page()
                    pdf.set_font("Helvetica", "B", 20)
                    pdf.set_text_color(212, 175, 55)
                    pdf.cell(0, 12, "DELTA — Precious Asset Risk Report", ln=True, align="C")
                    pdf.set_font("Helvetica", "", 10)
                    pdf.set_text_color(100, 100, 100)
                    pdf.cell(0, 6, f"Generated: {date.today().strftime('%d %B %Y')}  |  Analysis Date: {st.session_state.analysis_date}", ln=True, align="C")
                    pdf.ln(6)

                    # Macro snapshot
                    pdf.set_font("Helvetica", "B", 13)
                    pdf.set_text_color(30, 30, 30)
                    pdf.cell(0, 8, "Macro Snapshot", ln=True)
                    pdf.set_font("Helvetica", "", 9)
                    macro_snap = {
                        "USD/INR": f"{inputs.get('USD_INR', 0):.2f}",
                        "India Inflation": f"{inputs.get('India_Inflation', 0):.1f}%",
                        "RBI Repo Rate": f"{inputs.get('RBI_Repo_Rate', 0):.2f}%",
                        "Fed Rate": f"{inputs.get('Fed_Rate', 0):.2f}%",
                        "VIX": f"{inputs.get('Geopolitical_Risk_VIX', 0):.1f}",
                        "Oil (USD)": f"${inputs.get('Oil_Price_USD', 0):.1f}",
                        "S&P 500": f"{inputs.get('SP500_Index', 0):,.0f}",
                        "DXY": f"{inputs.get('DXY_Index', 0):.1f}",
                    }
                    for k, v in macro_snap.items():
                        pdf.set_text_color(80, 80, 80)
                        pdf.cell(60, 6, k + ":", border=0)
                        pdf.set_text_color(30, 30, 30)
                        pdf.cell(0, 6, v, ln=True)
                    pdf.ln(4)

                    # Asset results
                    pdf.set_font("Helvetica", "B", 13)
                    pdf.set_text_color(30, 30, 30)
                    pdf.cell(0, 8, "Asset Risk Summary", ln=True)
                    pdf.set_font("Helvetica", "B", 9)
                    pdf.set_fill_color(240, 235, 220)
                    pdf.cell(50, 7, "Asset", border=1, fill=True)
                    pdf.cell(30, 7, "Risk", border=1, fill=True)
                    pdf.cell(30, 7, "Action", border=1, fill=True)
                    pdf.cell(30, 7, "Confidence", border=1, fill=True)
                    pdf.cell(30, 7, "P(Low)", border=1, fill=True)
                    pdf.cell(30, 7, "P(High)", ln=True, border=1, fill=True)
                    pdf.set_font("Helvetica", "", 9)
                    for asset in selected_assets:
                        if asset not in results:
                            continue
                        res = results[asset]
                        proba = res["proba"]
                        pdf.cell(50, 6, f"{ASSET_METADATA[asset]['emoji']} {asset}", border=1)
                        pdf.cell(30, 6, res["risk_label"], border=1)
                        pdf.cell(30, 6, RISK_ACTIONS[res["risk_label"]], border=1)
                        pdf.cell(30, 6, f"{res['confidence']}%", border=1)
                        pdf.cell(30, 6, f"{proba[0]*100:.1f}%", border=1)
                        pdf.cell(30, 6, f"{(proba[2] if len(proba)>2 else 0)*100:.1f}%", border=1, ln=True)
                    pdf.ln(4)

                    # Portfolio summary
                    pdf.set_font("Helvetica", "B", 13)
                    pdf.set_text_color(30, 30, 30)
                    pdf.cell(0, 8, "Portfolio Summary", ln=True)
                    pdf.set_font("Helvetica", "", 9)
                    RISK_PNL = {"Low": 0.08, "Medium": 0.02, "High": -0.10}
                    total_cap = 0
                    total_pnl = 0
                    for asset in selected_assets:
                        if asset not in results:
                            continue
                        rupees = st.session_state.asset_rupees.get(asset, 0)
                        risk_label = results[asset]["risk_label"]
                        pnl = rupees * RISK_PNL[risk_label]
                        total_cap += rupees
                        total_pnl += pnl
                        pdf.cell(60, 6, f"{asset}:", border=0)
                        pdf.cell(0, 6, f"Rs {rupees:,.0f}  |  Expected: Rs {pnl:+,.0f}  ({RISK_PNL[risk_label]*100:+.0f}%)", ln=True)
                    pdf.set_font("Helvetica", "B", 10)
                    pdf.cell(60, 7, "Total Portfolio:")
                    pdf.cell(0, 7, f"Rs {total_cap:,.0f}  |  Expected P&L: Rs {total_pnl:+,.0f}  ({(total_pnl/total_cap*100 if total_cap>0 else 0):+.1f}%)", ln=True)
                    pdf.ln(4)

                    # Buy/sell signals from long term
                    pdf.set_font("Helvetica", "B", 13)
                    pdf.set_text_color(30, 30, 30)
                    pdf.cell(0, 8, "Key Buy/Sell Signals (1-Year Horizon)", ln=True)
                    pdf.set_font("Helvetica", "", 9)
                    if st.session_state.df_horizon_long is not None:
                        df_signals = st.session_state.df_horizon_long.copy()
                        df_signals["Month"] = df_signals["Date"].dt.to_period("M")
                        df_signals_m = df_signals.groupby(["Month","Asset"])["RiskInt"].mean().reset_index()
                        for asset in selected_assets:
                            df_a = df_signals_m[df_signals_m["Asset"]==asset].copy()
                            df_a["prev"] = df_a["RiskInt"].shift(1)
                            buys = df_a[df_a["RiskInt"] < 0.5]
                            sells = df_a[df_a["RiskInt"] > 1.5]
                            if len(buys) > 0:
                                pdf.set_text_color(34, 197, 94)
                                pdf.cell(0, 6, f"BUY  {asset}: {', '.join(str(m) for m in buys['Month'].head(3).astype(str))}", ln=True)
                            if len(sells) > 0:
                                pdf.set_text_color(239, 68, 68)
                                pdf.cell(0, 6, f"SELL {asset}: {', '.join(str(m) for m in sells['Month'].head(3).astype(str))}", ln=True)
                    pdf.set_text_color(100, 100, 100)
                    pdf.ln(6)
                    pdf.set_font("Helvetica", "I", 8)
                    pdf.cell(0, 6, "DELTA · Precious Asset Intelligence · Not financial advice.", ln=True, align="C")

                    # Inline chart images
                    try:
                        # Results comparison chart
                        fig_exp = go.Figure()
                        for label_key, color, col_idx in [("Low","#22c55e",0),("Medium","#f59e0b",1),("High","#ef4444",2)]:
                            fig_exp.add_trace(go.Bar(
                                name=label_key,
                                x=[a for a in selected_assets if a in results],
                                y=[results[a]["proba"][col_idx]*100 if len(results[a]["proba"])>col_idx else 0 for a in selected_assets if a in results],
                                marker_color=color,
                            ))
                        fig_exp.update_layout(
                            barmode="group", height=300,
                            paper_bgcolor="white", plot_bgcolor="white",
                            font=dict(color="black"),
                        )
                        img_bytes = pio.to_image(fig_exp, format="png", width=700, height=300)
                        img_path = "/tmp/delta_chart_compare.png"
                        with open(img_path, "wb") as f:
                            f.write(img_bytes)
                        pdf.add_page()
                        pdf.set_font("Helvetica", "B", 12)
                        pdf.set_text_color(30,30,30)
                        pdf.cell(0, 8, "Asset Risk Probability Comparison", ln=True)
                        pdf.image(img_path, x=10, w=190)
                    except Exception:
                        pass  # chart embed optional

                    pdf_bytes = pdf.output(dest="S").encode("latin-1")
                    b64 = base64.b64encode(pdf_bytes).decode()
                    href = f'<a href="data:application/pdf;base64,{b64}" download="delta_report_{date.today()}.pdf" style="display:inline-block;padding:0.7rem 2rem;background:linear-gradient(135deg,#D4AF37,#F0D060);color:#07080f;border-radius:10px;font-family:Syne,sans-serif;font-weight:700;text-decoration:none;letter-spacing:0.06em;">⬇️ Download PDF Report</a>'
                    st.markdown(href, unsafe_allow_html=True)
                    st.success("PDF generated successfully!")
                except ImportError:
                    st.error("fpdf2 not installed. Run: pip install fpdf2")
                except Exception as e:
                    st.error(f"PDF generation failed: {e}")

# ── Footer ─────────────────────────────────────────────────────────
st.markdown("""
<div style="text-align:center;padding:3rem 0 1rem;color:var(--text-dim);font-size:0.75rem;letter-spacing:0.08em;">
    DELTA · Precious Asset Intelligence &nbsp;·&nbsp; ML-powered risk classification
    &nbsp;·&nbsp; Not financial advice
</div>
""", unsafe_allow_html=True)