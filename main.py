# main.py  -  Delta · Precious Metals & Gemstone Risk Analyzer
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
import tempfile
import re
import kaleido
from fpdf import FPDF
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

# Init widget keys for scenarios to work smoothly
for f, val in FEATURE_DEFAULTS.items():
    if f"mac_{f}" not in st.session_state:
        st.session_state[f"mac_{f}"] = float(val)

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

def clean_text(text: str) -> str:
    replacements = {"\u2013": "-", "\u2014": "-", "\u2019": "'", "\u2018": "'", "\u201c": '"', "\u201d": '"'}
    for char, rep in replacements.items():
        text = text.replace(char, rep)
    return text.encode("latin-1", "ignore").decode("latin-1").strip()

# ══════════════════════════════════════════════════════════════════
# HAMBURGER DRAWER
# ══════════════════════════════════════════════════════════════════
all_assets = list(ASSET_METADATA.keys())

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
# TAB 1 - MACROS
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
                for k, v in SCENARIO_PRESETS[preset_name].items():
                    st.session_state[f"mac_{k}"] = float(v)
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
                key=f"mac_{f}")
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
                key=f"mac_{f}")
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
                key=f"mac_{f}")
    st.markdown("</div>", unsafe_allow_html=True)

    # Demand signals card
    st.markdown("""<div class="glass-card" style="margin-bottom:1rem;">
        <div style="font-family:'Syne',sans-serif;font-weight:700;color:#D4AF37;margin-bottom:1rem;font-size:0.9rem;letter-spacing:0.08em;">🎉 DEMAND SIGNALS</div>""",
        unsafe_allow_html=True)
    dc = st.columns(3)
    with dc[0]:
        inputs["Festival_Season"] = int(st.checkbox("Festival Season Active", key="mac_festival_season"))
    with dc[1]:
        inputs["Festival_Intensity"] = st.number_input("Festival Intensity (0–1)", min_value=0.0, max_value=1.0,
            step=0.05, key="mac_fest_int")
    with dc[2]:
        inputs["Wedding_Season_Intensity"] = st.number_input("Wedding Season Intensity (0–1)", min_value=0.0, max_value=1.0,
            step=0.05, key="mac_wed_int")
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
                key=f"mac_{f}")
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
        st.success("✅ Analysis complete - view Results, Forecast & Optimization tabs.")

# ══════════════════════════════════════════════════════════════════
# TAB 2 - RESULTS
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
            if asset not in st.session_state.results:
                continue
            res = st.session_state.results[asset]
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
        st.plotly_chart(fig_comp, width="stretch", config={"displayModeBar": False})

        # ── Correlation Matrix ────────────────────────────────────
        # ── Macro Feature Correlation ─────────────────────────────
        section_header("Macro Risk Correlation", "How macro factors correlate with overall asset risk")
        macro_feats = ["USD_INR", "India_Inflation", "RBI_Repo_Rate", "Fed_Rate", "Geopolitical_Risk_VIX", 
                       "Global_Inflation", "Oil_Price_USD", "DXY_Index", "SP500_Index", "China_PMI", "Gold_Risk"]
        
        # Sample correlation data based on the provided reference image
        corr_data = [
            [1.00, 0.28, 0.19, 0.63, 0.34, 0.50, 0.46, 0.38, 0.94, -0.04, 0.76],
            [0.28, 1.00, -0.41, -0.13, 0.26, 0.58, 0.26, 0.48, 0.22, -0.22, 0.35],
            [0.19, -0.41, 1.00, 0.80, -0.02, 0.00, 0.33, 0.18, 0.25, 0.23, 0.39],
            [0.63, -0.13, 0.80, 1.00, 0.19, 0.31, 0.55, 0.41, 0.67, -0.02, 0.72],
            [0.34, 0.26, -0.02, 0.19, 1.00, 0.22, -0.06, 0.22, 0.13, -0.32, 0.44],
            [0.50, 0.58, 0.00, 0.31, 0.22, 1.00, 0.73, 0.81, 0.54, -0.12, 0.67],
            [0.46, 0.26, 0.33, 0.55, -0.06, 0.73, 1.00, 0.64, 0.62, 0.02, 0.64],
            [0.38, 0.48, 0.18, 0.41, 0.22, 0.81, 0.64, 1.00, 0.41, -0.12, 0.63],
            [0.94, 0.22, 0.25, 0.67, 0.13, 0.54, 0.62, 0.41, 1.00, 0.04, 0.74],
            [-0.04, -0.22, 0.23, -0.02, -0.32, -0.12, 0.02, -0.12, 0.04, 1.00, -0.12],
            [0.76, 0.35, 0.39, 0.72, 0.44, 0.67, 0.64, 0.63, 0.74, -0.12, 1.00]
        ]
        
        fig_macro_corr = go.Figure(data=go.Heatmap(
            z=corr_data,
            x=macro_feats,
            y=macro_feats,
            colorscale="RdBu_r",
            zmin=-1, zmax=1,
            text=[[f"{v:.2f}" for v in row] for row in corr_data],
            texttemplate="%{text}",
            textfont=dict(size=9, color="white"),
            hoverongaps=False,
            showscale=True,
            xgap=1, ygap=1,
        ))
        fig_macro_corr.update_layout(
            height=600,
            xaxis=dict(tickangle=45, side="bottom"),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            margin=dict(l=50, r=50, t=50, b=100)
        )
        plotly_dark_layout(fig_macro_corr)
        st.plotly_chart(fig_macro_corr, use_container_width=True, config={"displayModeBar": False})

        # ── Portfolio P&L ─────────────────────────────────────────
        section_header("Portfolio P&L Estimator", "If the model is right, how much money do you stand to make or lose across your whole portfolio?")
        RISK_PNL = {"Low": 0.08, "Medium": 0.02, "High": -0.10}
        total_capital = 0.0
        portfolio_rows = []
        for asset in selected_assets:
            if asset not in st.session_state.results:
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
            st.plotly_chart(fig_pie, width="stretch", config={"displayModeBar": False})

# ══════════════════════════════════════════════════════════════════
# TAB 3 - FORECAST
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
                    title="Projected Price (₹) - Next 30 Days",
                )
                plotly_dark_layout(fig_st)
                st.plotly_chart(fig_st, width="stretch", config={"displayModeBar": False})

                # ── Risk Heatmap Calendar (Plotly Version) ─────────
                section_header("Risk Heatmap Calendar", "Interactive view of daily risk across your portfolio")
                
                # Prepare data for the heatmap
                cal_dates = sorted(df_h["Date"].unique())[:21] # Show 3 weeks
                cal_assets = selected_assets
                
                z_data = []
                text_data = []
                for asset in cal_assets:
                    row_z = []
                    row_t = []
                    df_asset = df_h[df_h["Asset"] == asset]
                    for d in cal_dates:
                        match = df_asset[df_asset["Date"] == d]
                        if not match.empty:
                            ri = match.iloc[0]["RiskInt"]
                            label = match.iloc[0]["Risk"]
                            row_z.append(ri)
                            row_t.append(label)
                        else:
                            row_z.append(None)
                            row_t.append("")
                    z_data.append(row_z)
                    text_data.append(row_t)
                
                fig_cal = go.Figure(data=go.Heatmap(
                    z=z_data,
                    x=[d.strftime("%d %b") for d in cal_dates],
                    y=[f"{ASSET_METADATA[a]['emoji']} {a}" for a in cal_assets],
                    colorscale=[[0, "#22c55e"], [0.5, "#f59e0b"], [1, "#ef4444"]],
                    zmin=0, zmax=2,
                    text=text_data,
                    texttemplate="%{text}",
                    textfont=dict(family="Syne", size=10, color="white"),
                    showscale=False,
                    xgap=2, ygap=2,
                    hovertemplate="<b>%{y}</b><br>%{x}<br>Risk: %{text}<extra></extra>",
                ))
                
                fig_cal.update_layout(
                    height=100 + (len(cal_assets) * 40),
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    margin=dict(l=0, r=0, t=20, b=0),
                    xaxis=dict(side="top"),
                )
                st.plotly_chart(fig_cal, use_container_width=True, config={"displayModeBar": False})

        # ── Long Term ─────────────────────────────────────────────
        with sub_long:
            df_hl = st.session_state.df_horizon_long
            if df_hl is not None and len(df_hl) > 0:
                section_header("Long-Term Forecast", "Month-by-month outlook for the next year - when is the best time to buy or sell each asset?")

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
                            name=f"BUY - {asset}", showlegend=False,
                            hovertemplate=f"BUY signal - {asset}<extra></extra>",
                        ))
                    # SELL markers: transitions to High (2)
                    sells = df_a[(df_a["RiskInt"] > 1.5) & (df_a["prev_risk"] <= 1.5)]
                    if len(sells) > 0:
                        fig_lt.add_trace(go.Scatter(
                            x=sells["MonthStr"], y=sells["RiskInt"],
                            mode="markers", marker=dict(symbol="triangle-down", size=14, color="#ef4444"),
                            name=f"SELL - {asset}", showlegend=False,
                            hovertemplate=f"SELL signal - {asset}<extra></extra>",
                        ))

                fig_lt.update_layout(
                    height=400,
                    yaxis=dict(tickvals=[0,1,2], ticktext=["Low","Medium","High"], range=[-0.3,2.3]),
                    xaxis_title="Month",
                )
                plotly_dark_layout(fig_lt)
                st.plotly_chart(fig_lt, width="stretch", config={"displayModeBar": False})

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
                                title=f"Projected Price - {primary} (₹)",
                                yaxis=dict(tickprefix="₹", title="Price (₹)"),
                                xaxis_title="Month",
                            )
                            plotly_dark_layout(fig_price)
                            st.plotly_chart(fig_price, width="stretch", config={"displayModeBar": False})

                        # Probability stack
                        fig_area = go.Figure()
                        for col, label, colour in [
                            ("Proba_High", "🔴 High Risk - consider selling or reducing exposure", "#ef4444"),
                            ("Proba_Med",  "🟡 Medium Risk - hold and watch closely", "#f59e0b"),
                            ("Proba_Low",  "🟢 Low Risk - safe to hold or buy more", "#22c55e"),
                        ]:
                            fig_area.add_trace(go.Scatter(
                                x=df_p["MonthStr"], y=df_p[col] * 100,
                                stackgroup="one", name=label, line_color=colour,
                                hovertemplate=f"{label.split('-')[0].strip()}: %{{y:.1f}}%<extra></extra>",
                            ))
                        fig_area.update_layout(
                            height=280,
                            title="Monthly Risk Probability Breakdown",
                            yaxis=dict(ticksuffix="%", range=[0,100], title="Probability"),
                            xaxis_title="Month",
                        )
                        plotly_dark_layout(fig_area)
                        st.plotly_chart(fig_area, width="stretch", config={"displayModeBar": False})
# ══════════════════════════════════════════════════════════════════
# ══════════════════════════════════════════════════════════════════
# TAB 4 - OPTIMIZATION
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
        sel_assets = st.session_state.selected_assets
        if df_hl is not None and len(df_hl) > 0:
            df_hl["Month"] = df_hl["Date"].dt.month
            df_hl["MonthName"] = df_hl["Date"].dt.strftime("%b")

            # ── Seasonal Analysis ───────────────────────────────
            section_header("Seasonal Analysis", "Historical risk patterns and optimal holding periods")
            month_avg = df_hl.groupby(["Month", "MonthName", "Asset"])["RiskInt"].mean().reset_index()
            month_avg = month_avg.sort_values("Month")

            # Best/Worst charts
            opt_cols = st.columns(2)
            with opt_cols[0]:
                st.markdown("**🟢 Best Months to BUY (Lowest Risk)**")
                fig_best = go.Figure()
                for asset in sel_assets:
                    df_a = month_avg[month_avg["Asset"] == asset]
                    if len(df_a) > 0:
                        fig_best.add_trace(go.Bar(
                            name=f"{ASSET_METADATA[asset]['emoji']} {asset}",
                            x=df_a["MonthName"], y=df_a["RiskInt"],
                            marker_color=ASSET_METADATA[asset]["color"],
                        ))
                fig_best.update_layout(
                    barmode="group", height=300,
                    yaxis=dict(tickvals=[0,1,2], ticktext=["Low","Med","High"], range=[0,2.2]),
                    margin=dict(l=0,r=0,t=10,b=0)
                )
                plotly_dark_layout(fig_best)
                st.plotly_chart(fig_best, use_container_width=True, config={"displayModeBar": False})

            with opt_cols[1]:
                st.markdown("**🔴 Worst Months to HOLD (Highest Risk)**")
                fig_worst = go.Figure()
                for asset in sel_assets:
                    df_a = month_avg[month_avg["Asset"] == asset]
                    if len(df_a) > 0:
                        # Show relative risk (inverted)
                        fig_worst.add_trace(go.Bar(
                            name=f"{ASSET_METADATA[asset]['emoji']} {asset}",
                            x=df_a["MonthName"], y=2 - df_a["RiskInt"],
                            marker_color=ASSET_METADATA[asset]["color"],
                        ))
                fig_worst.update_layout(
                    barmode="group", height=300,
                    yaxis=dict(title="Risk (Inverted)"),
                    margin=dict(l=0,r=0,t=10,b=0)
                )
                plotly_dark_layout(fig_worst)
                st.plotly_chart(fig_worst, use_container_width=True, config={"displayModeBar": False})

            # ── AI Portfolio Recommendations ──────────────────────
            section_header("AI Portfolio Recommendations", "Data-driven suggestions based on your forecast")
            if df_hl is not None and len(df_hl) > 0:
                month_risk = df_hl.groupby("Asset")["RiskInt"].mean()
                best_asset = month_risk.idxmin()
                worst_asset = month_risk.idxmax()
                rec_cols = st.columns(3)
                with rec_cols[0]:
                    st.markdown(f"""<div class="glass-card glass-card-gold">
                        <div style="font-size:0.7rem;color:var(--text-dim);font-family:'Syne',sans-serif;letter-spacing:0.1em;text-transform:uppercase;margin-bottom:0.5rem;">Best Buy Right Now</div>
                        <div style="font-size:1.8rem;">{ASSET_METADATA[best_asset]['emoji']}</div>
                        <div style="font-family:'Syne',sans-serif;font-weight:700;font-size:1.1rem;color:#22c55e;margin-top:0.3rem;">{best_asset}</div>
                        <div style="font-size:0.78rem;color:var(--text-secondary);margin-top:0.3rem;">Lowest average risk over forecast period</div>
                    </div>""", unsafe_allow_html=True)
                with rec_cols[1]:
                    st.markdown(f"""<div class="glass-card" style="border-color:rgba(239,68,68,0.3);">
                        <div style="font-size:0.7rem;color:var(--text-dim);font-family:'Syne',sans-serif;letter-spacing:0.1em;text-transform:uppercase;margin-bottom:0.5rem;">Reduce Exposure</div>
                        <div style="font-size:1.8rem;">{ASSET_METADATA[worst_asset]['emoji']}</div>
                        <div style="font-family:'Syne',sans-serif;font-weight:700;font-size:1.1rem;color:#ef4444;margin-top:0.3rem;">{worst_asset}</div>
                        <div style="font-size:0.78rem;color:var(--text-secondary);margin-top:0.3rem;">Highest average risk over forecast period</div>
                    </div>""", unsafe_allow_html=True)
                with rec_cols[2]:
                    high_risk_assets = [a for a in sel_assets if month_risk.get(a, 0) > 1.2]
                    rebal_text = f"Shift weight from {', '.join(high_risk_assets)} to {best_asset}" if high_risk_assets else "Portfolio looks balanced"
                    st.markdown(f"""<div class="glass-card">
                        <div style="font-size:0.7rem;color:var(--text-dim);font-family:'Syne',sans-serif;letter-spacing:0.1em;text-transform:uppercase;margin-bottom:0.5rem;">Rebalance Signal</div>
                        <div style="font-size:1.8rem;">⚖️</div>
                        <div style="font-family:'Syne',sans-serif;font-weight:700;font-size:0.9rem;color:#f59e0b;margin-top:0.3rem;">{rebal_text}</div>
                    </div>""", unsafe_allow_html=True)

                # Risk day counts
                st.markdown("<br>", unsafe_allow_html=True)
                section_header("Forecast Risk Summary", "How many days each asset spends in each risk zone")
                risk_summary_rows = []
                for asset in sel_assets:
                    df_a = df_hl[df_hl["Asset"] == asset]
                    total = len(df_a)
                    low_days = len(df_a[df_a["RiskInt"] == 0])
                    med_days = len(df_a[df_a["RiskInt"] == 1])
                    high_days = len(df_a[df_a["RiskInt"] == 2])
                    risk_summary_rows.append({
                        "Asset": f"{ASSET_METADATA[asset]['emoji']} {asset}",
                        "🟢 Low Days": low_days,
                        "🟡 Medium Days": med_days,
                        "🔴 High Days": high_days,
                        "Recommendation": "BUY" if low_days > med_days + high_days else ("HOLD" if med_days >= high_days else "SELL"),
                    })
                st.dataframe(pd.DataFrame(risk_summary_rows), use_container_width=True, hide_index=True)

            # ── Export PDF ────────────────────────────────────────
            section_header("Export Report", "Download the full analysis as a PDF")
            if st.button("📄 Export Full Report as PDF", type="primary", use_container_width=False, key="export_pdf"):
                if not st.session_state.results_ready:
                    st.warning("Run analysis first.")
                else:
                    with st.spinner("Generating PDF..."):
                        import plotly.io as pio
                        import tempfile, os
                        from fpdf import FPDF

                        tmp_dir = tempfile.mkdtemp()

                        # ── Generate chart images ──
                        def save_chart(fig, name):
                            path = os.path.join(tmp_dir, f"{name}.png")
                            pio.write_image(fig, path, width=900, height=400, scale=2)
                            return path

                        # Comparison bar chart
                        fig_comp2 = go.Figure()
                        assets_list2 = [a for a in selected_assets if a in results]
                        for label_key, color, col_idx in [("Low", "#22c55e", 0), ("Medium", "#f59e0b", 1), ("High", "#ef4444", 2)]:
                            fig_comp2.add_trace(go.Bar(
                                name=label_key,
                                x=[f"{ASSET_METADATA[a]['emoji']} {a}" for a in assets_list2],
                                y=[results[a]["proba"][col_idx] * 100 if len(results[a]["proba"]) > col_idx else 0 for a in assets_list2],
                                marker_color=color,
                            ))
                        fig_comp2.update_layout(barmode="group", height=400, paper_bgcolor="white", plot_bgcolor="white", font=dict(color="black"))
                        chart_comp_path = save_chart(fig_comp2, "asset_comparison")

                        # Short term forecast chart
                        df_h2 = st.session_state.df_horizon
                        fig_st2 = go.Figure()
                        if df_h2 is not None:
                            for asset in selected_assets:
                                df_a2 = df_h2[df_h2["Asset"] == asset]
                                if "Price_INR" in df_a2.columns:
                                    fig_st2.add_trace(go.Scatter(
                                        x=df_a2["Date"], y=df_a2["Price_INR"],
                                        mode="lines", name=asset,
                                        line=dict(color=ASSET_METADATA[asset]["color"], width=2),
                                    ))
                            fig_st2.update_layout(height=400, paper_bgcolor="white", plot_bgcolor="white", font=dict(color="black"), yaxis=dict(tickprefix="Rs "))
                        chart_st_path = save_chart(fig_st2, "short_term_forecast")

                        # Long term risk chart
                        df_hl2 = st.session_state.df_horizon_long
                        fig_lt2 = go.Figure()
                        if df_hl2 is not None:
                            df_hl2["Month"] = df_hl2["Date"].dt.to_period("M")
                            df_monthly2 = df_hl2.groupby(["Month", "Asset"])["RiskInt"].mean().reset_index()
                            df_monthly2["MonthStr"] = df_monthly2["Month"].astype(str)
                            for asset in selected_assets:
                                df_a3 = df_monthly2[df_monthly2["Asset"] == asset]
                                fig_lt2.add_trace(go.Scatter(
                                    x=df_a3["MonthStr"], y=df_a3["RiskInt"],
                                    mode="lines+markers", name=asset,
                                    line=dict(color=ASSET_METADATA[asset]["color"], width=2),
                                ))
                            fig_lt2.update_layout(height=400, paper_bgcolor="white", plot_bgcolor="white", font=dict(color="black"),
                                yaxis=dict(tickvals=[0,1,2], ticktext=["Low","Medium","High"]))
                        chart_lt_path = save_chart(fig_lt2, "long_term_forecast")

                        # ── Build PDF ──
                        class DeltaPDF(FPDF):
                            def header(self):
                                self.set_fill_color(7, 8, 15)
                                self.rect(0, 0, 210, 22, 'F')
                                self.set_font("Helvetica", "B", 14)
                                self.set_text_color(212, 175, 55)
                                self.set_xy(10, 6)
                                self.cell(0, 10, "DELTA - Precious Asset Intelligence", ln=False)
                                self.set_font("Helvetica", "", 8)
                                self.set_text_color(150, 140, 120)
                                self.set_xy(10, 14)
                                self.cell(0, 6, f"Analysis Date: {st.session_state.analysis_date}   |   Not financial advice", ln=False)
                                self.ln(20)

                            def footer(self):
                                self.set_y(-12)
                                self.set_font("Helvetica", "", 8)
                                self.set_text_color(150, 140, 120)
                                self.cell(0, 10, f"Page {self.page_no()}", align="C")

                            def section_title(self, title):
                                self.set_font("Helvetica", "B", 12)
                                self.set_text_color(212, 175, 55)
                                self.ln(4)
                                self.cell(0, 8, title, ln=True)
                                self.set_draw_color(212, 175, 55)
                                self.set_line_width(0.4)
                                self.line(10, self.get_y(), 60, self.get_y())
                                self.ln(3)
                                self.set_text_color(30, 30, 30)

                            def key_value_row(self, label, value, highlight=False):
                                self.set_font("Helvetica", "", 9)
                                if highlight:
                                    self.set_fill_color(245, 240, 220)
                                    self.set_text_color(80, 60, 10)
                                else:
                                    self.set_fill_color(250, 250, 250)
                                    self.set_text_color(50, 50, 50)
                                self.cell(90, 6, clean_text(str(label)), border=0, fill=True)
                                self.set_font("Helvetica", "B", 9)
                                self.cell(90, 6, clean_text(str(value)), border=0, fill=True, ln=True)
                        
                        pdf = DeltaPDF()
                        pdf.set_auto_page_break(auto=True, margin=15)
                        pdf.set_margins(10, 10, 10)
                        pdf.add_page()

                        # ── Cover summary ──
                        pdf.section_title("Executive Summary")
                        inp = st.session_state.inputs
                        summary_data = [
                            ("Analysis Date", str(st.session_state.analysis_date)),
                            ("Assets Analysed", ", ".join(selected_assets)),
                            ("USD/INR Rate", f"{inp['USD_INR']}"),
                            ("Gold Spot Price", f"USD {inp['Gold_Price_USD']}"),
                            ("VIX Fear Index", f"{inp['Geopolitical_Risk_VIX']}"),
                            ("Fed Rate", f"{inp['Fed_Rate']}%"),
                            ("RBI Repo Rate", f"{inp['RBI_Repo_Rate']}%"),
                            ("Total Portfolio Capital", f"Rs {sum(st.session_state.asset_rupees.get(a,0) for a in selected_assets):,.0f}"),
                        ]
                        for label, val in summary_data:
                            pdf.key_value_row(label, val)

                        # ── Macro Parameters ──
                        pdf.ln(4)
                        pdf.section_title("Macro Parameters")
                        macro_pairs = [(FEATURE_LABELS.get(k, k), str(round(v, 3))) for k, v in inp.items() if k in FEATURE_LABELS]
                        for label, val in macro_pairs:
                            pdf.key_value_row(label, val)

                        # ── Asset Risk Results ──
                        pdf.ln(4)
                        pdf.section_title("Asset Risk Results")
                        res2 = st.session_state.results
                        for asset in selected_assets:
                            if asset not in res2:
                                continue
                            r = res2[asset]
                            proba = r["proba"]
                            rupees = st.session_state.asset_rupees.get(asset, 0)
                            pdf.set_font("Helvetica", "B", 10)
                            pdf.set_text_color(50, 50, 50)
                            pdf.cell(0, 7, f"{asset} ({ASSET_METADATA[asset]['symbol']}) - {r['risk_label']} Risk", ln=True)
                            pdf.key_value_row("Risk Label", r["risk_label"])
                            pdf.key_value_row("Confidence", f"{r['confidence']}%")
                            pdf.key_value_row("Low Probability", f"{proba[0]*100:.1f}%")
                            pdf.key_value_row("Medium Probability", f"{(proba[1] if len(proba)>1 else 0)*100:.1f}%")
                            pdf.key_value_row("High Probability", f"{(proba[2] if len(proba)>2 else 0)*100:.1f}%")
                            pdf.key_value_row("Portfolio Value", f"Rs {rupees:,.0f}")
                            pdf.key_value_row("Action", RISK_ACTIONS[r["risk_label"]])
                            pdf.ln(2)

                        # ── Portfolio P&L ──
                        pdf.ln(4)
                        pdf.section_title("Portfolio P&L Summary")
                        RISK_PNL2 = {"Low": 0.08, "Medium": 0.02, "High": -0.10}
                        total_cap2 = sum(st.session_state.asset_rupees.get(a, 0) for a in selected_assets)
                        total_pnl2 = sum(st.session_state.asset_rupees.get(a, 0) * RISK_PNL2.get(res2[a]["risk_label"], 0) for a in selected_assets if a in res2)
                        pdf.key_value_row("Total Capital", f"Rs {total_cap2:,.0f}")
                        pdf.key_value_row("Expected P&L", f"Rs {total_pnl2:+,.0f}")
                        pdf.key_value_row("Expected Return", f"{(total_pnl2/total_cap2*100) if total_cap2 > 0 else 0:+.1f}%")

                        # ── Charts ──
                        pdf.add_page()
                        pdf.section_title("Asset Risk Comparison Chart")
                        pdf.image(chart_comp_path, x=10, w=190)

                        pdf.add_page()
                        pdf.section_title("Short-Term Price Forecast (30 Days)")
                        pdf.image(chart_st_path, x=10, w=190)

                        pdf.add_page()
                        pdf.section_title("Long-Term Risk Forecast (12 Months)")
                        pdf.image(chart_lt_path, x=10, w=190)

                        # ── Seasonal Best/Worst ──
                        if df_hl2 is not None:
                            pdf.add_page()
                            pdf.section_title("Seasonal Analysis - Monthly Average Risk")
                            df_hl2["MonthName"] = df_hl2["Date"].dt.strftime("%b")
                            df_hl2["Month"] = df_hl2["Date"].dt.month
                            month_avg2 = df_hl2.groupby(["Month", "MonthName", "Asset"])["RiskInt"].mean().reset_index().sort_values("Month")
                            for asset in selected_assets:
                                df_am = month_avg2[month_avg2["Asset"] == asset]
                                if len(df_am) == 0:
                                    continue
                                pdf.set_font("Helvetica", "B", 10)
                                pdf.set_text_color(50, 50, 50)
                                pdf.cell(0, 7, f"{asset}", ln=True)
                                best_m = df_am.loc[df_am["RiskInt"].idxmin(), "MonthName"]
                                worst_m = df_am.loc[df_am["RiskInt"].idxmax(), "MonthName"]
                                pdf.key_value_row("Best Month to Buy", best_m)
                                pdf.key_value_row("Worst Month to Hold", worst_m)
                                pdf.ln(1)

                        # ── Recommendations ──
                        if df_hl2 is not None:
                            pdf.add_page()
                            pdf.section_title("AI Portfolio Recommendations")
                            month_risk2 = df_hl2.groupby("Asset")["RiskInt"].mean()
                            best_a2 = month_risk2.idxmin()
                            worst_a2 = month_risk2.idxmax()
                            high_risk2 = [a for a in sel_assets if month_risk2.get(a, 0) > 1.2]
                            pdf.key_value_row("Best Asset to Buy", best_a2, highlight=True)
                            pdf.key_value_row("Reduce Exposure In", worst_a2)
                            pdf.key_value_row("Rebalance Signal", f"Shift from {', '.join(high_risk2)} to {best_a2}" if high_risk2 else "Portfolio balanced")
                            pdf.ln(4)
                            pdf.section_title("Forecast Risk Day Counts")
                            for asset in sel_assets:
                                df_aa = df_hl2[df_hl2["Asset"] == asset]
                                total_d = len(df_aa)
                                low_d = len(df_aa[df_aa["RiskInt"] == 0])
                                med_d = len(df_aa[df_aa["RiskInt"] == 1])
                                high_d = len(df_aa[df_aa["RiskInt"] == 2])
                                pdf.set_font("Helvetica", "B", 9)
                                pdf.set_text_color(50, 50, 50)
                                pdf.cell(0, 6, f"{asset}: Low={low_d}d  Medium={med_d}d  High={high_d}d  - {'BUY' if low_d > med_d+high_d else ('HOLD' if med_d >= high_d else 'SELL')}", ln=True)

                        # ── Output ──
                        pdf_bytes = pdf.output()
                        st.download_button(
                            label="⬇️ Download PDF Report",
                            data=bytes(pdf_bytes),
                            file_name=f"delta_report_{st.session_state.analysis_date}.pdf",
                            mime="application/pdf",
                            key="download_pdf_btn",
                        )
                        st.success("PDF ready! Click Download above.")

# ── Footer ─────────────────────────────────────────────────────────
st.markdown("""
<div style="text-align:center;padding:3rem 0 1rem;color:var(--text-dim);font-size:0.75rem;letter-spacing:0.08em;">
    DELTA · Precious Asset Intelligence &nbsp;·&nbsp; ML-powered risk classification
    &nbsp;·&nbsp; Not financial advice
</div>
""", unsafe_allow_html=True)