# utils/model.py
# Loads the pkl and runs predictions

import joblib
import numpy as np
import pandas as pd
import math
import streamlit as st
from pathlib import Path

MODEL_PATH = Path(__file__).parent.parent / "metal_risk_model.pkl"

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

def forecast_prices(user_inputs: dict, assets: list, n_days: int) -> dict:
    """
    Recursive multi-step price forecast using lag-based XGBoost.
    Returns {asset: [price_day1, price_day2, ..., price_dayN]} in USD.
    """
    artefact = load_model()
    pfm  = artefact["price_forecast_models"]
    pfsc = artefact["price_forecast_scalers"]
    lags = [1,2,3,5,7,14,21,30]
    
    results = {}
    for asset in assets:
        model  = pfm[asset]
        scaler = pfsc[asset]
        series = list(pfm[f'{asset}_last_series'])  # last 30 known prices
        
        forecasted = []
        for _ in range(n_days):
            s = pd.Series(series)
            row = {}
            for lag in lags:
                row[f'lag_{lag}'] = s.iloc[-lag] if len(s) >= lag else s.iloc[0]
            row['rolling_mean_7']  = s.iloc[-7:].mean()
            row['rolling_mean_30'] = s.iloc[-30:].mean() if len(s) >= 30 else s.mean()
            row['rolling_std_7']   = s.iloc[-7:].std()
            row['rolling_std_30']  = s.iloc[-30:].std() if len(s) >= 30 else s.std()
            row['pct_change_1']    = (s.iloc[-1] - s.iloc[-2]) / s.iloc[-2] if len(s) >= 2 else 0
            row['pct_change_7']    = (s.iloc[-1] - s.iloc[-7]) / s.iloc[-7] if len(s) >= 7 else 0
            
            X_row = scaler.transform([list(row.values())])
            next_price = float(model.predict(X_row)[0])
            forecasted.append(next_price)
            series.append(next_price)
        
        results[asset] = forecasted
    return results

def predict_price(user_inputs: dict, date: "pd.Timestamp", assets: list) -> dict:
    """Returns {asset: predicted_next_day_price_USD}"""
    artefact = load_model()
    scaler = artefact["scaler"]
    price_models = artefact["price_models"]
    X = build_feature_vector(user_inputs, date).reshape(1, -1)
    X_scaled = scaler.transform(X)
    return {asset: float(price_models[asset].predict(X_scaled)[0]) for asset in assets}

def _cyclic(val, period):
    return math.sin(2 * math.pi * val / period), math.cos(2 * math.pi * val / period)

def build_feature_vector(user_inputs: dict, date: "pd.Timestamp") -> np.ndarray:
    """
    Constructs the 89-feature vector expected by the model.
    user_inputs must contain all macro + asset price fields from FEATURE_DEFAULTS.
    """
    u = user_inputs
    month   = date.month
    dow     = date.dayofweek
    woy     = date.isocalendar().week
    quarter = (month - 1) // 3 + 1
    is_weekend = int(dow >= 5)

    sin_m, cos_m   = _cyclic(month, 12)
    sin_d, cos_d   = _cyclic(dow, 7)
    sin_w, cos_w   = _cyclic(woy, 52)

    # Derived macro features (mirroring notebook)
    macro_stress = (u["Geopolitical_Risk_VIX"] / 30) + (u["Global_Inflation"] / 10) + (u["Russia_Ukraine_Tension"])
    rate_spread  = u["RBI_Repo_Rate"] - u["Fed_Rate"]
    inr_pressure = (u["USD_INR"] - 65) / 20   # normalised drift from 2018 base

    festival_wedding = u["Festival_Intensity"] * u["Wedding_Season_Intensity"]

    # Per-asset rolling stubs (user can't supply these live; use sensible defaults)
    def asset_rows(name, price_col, vol_col):
        p = u.get(price_col, 0)
        v = u.get(vol_col, 0)
        vol5  = p * 0.008
        vol15 = p * 0.015
        vol30 = p * 0.022
        ret1  = 0.002
        ret7  = 0.008
        return [p, v, vol5, vol15, vol30, ret1, ret7]

    gold_rows     = asset_rows("Gold",     "Gold_Price_USD",   "Gold_Volume")
    silver_rows   = asset_rows("Silver",   "Silver_Price_USD", "Silver_Volume")
    plat_rows     = asset_rows("Platinum", "Platinum_Price_USD","Platinum_Volume")
    copper_rows   = asset_rows("Copper",   "Copper_Price_USD", "Copper_Volume")
    diamond_rows  = asset_rows("Diamond",  "Diamond_Index",    "Diamond_Volume")
    emerald_rows  = asset_rows("Emerald",  "Emerald_Index",    "Emerald_Volume")
    ruby_rows     = asset_rows("Ruby",     "Ruby_Index",       "Ruby_Volume")
    pearl_rows    = asset_rows("Pearl",    "Pearl_Index",      "Pearl_Volume")

    row = [
        u["USD_INR"], u["India_Inflation"], u["RBI_Repo_Rate"],
        u["Import_Duty_Gold_pct"], u["India_GDP_Growth"], u["Monsoon_Index"],
        u["Global_Inflation"], u["Fed_Rate"], u["DXY_Index"],
        u["Oil_Price_USD"], u["SP500_Index"], u["China_PMI"],
        u["Geopolitical_Risk_VIX"], u["Russia_Ukraine_Tension"],
        u["Global_Mining_Output_Index"], u["Lab_Diamond_Supply_Index"],
        u["Emerald_Origin_Premium_pct"], u["Diamond_Demand_Index"],
        macro_stress, rate_spread, inr_pressure,
        u["Festival_Season"], u["Festival_Intensity"], u["Wedding_Season_Intensity"],
        is_weekend, quarter, festival_wedding,
        sin_m, cos_m, sin_d, cos_d, sin_w, cos_w,
        *gold_rows, *silver_rows, *plat_rows, *copper_rows,
        *diamond_rows, *emerald_rows, *ruby_rows, *pearl_rows,
    ]
    return np.array(row, dtype=float)


def predict_risk(user_inputs: dict, date: "pd.Timestamp", assets: list) -> dict:
    """
    Returns per-asset dict:
      { asset: { "risk_label": str, "risk_int": int, "proba": [p0,p1,p2] } }
    """
    artefact = load_model()
    scaler   = artefact["scaler"]
    models   = artefact["tuned_models"]
    labels   = artefact["risk_labels"]   # ["Low","Medium","High"]

    X = build_feature_vector(user_inputs, date).reshape(1, -1)
    X_scaled = scaler.transform(X)

    results = {}
    for asset in assets:
        model  = models[asset]
        pred   = int(model.predict(X_scaled)[0])
        try:
            proba = model.predict_proba(X_scaled)[0].tolist()
        except Exception:
            proba = [0.0, 0.0, 0.0]
            proba[pred] = 1.0
        label = labels[pred]
        results[asset] = {
            "risk_label": label,
            "risk_int":   pred,
            "proba":      proba,
            "confidence": round(max(proba) * 100, 1),
        }
    return results


def predict_horizon(user_inputs: dict, start_date, end_date, assets: list) -> pd.DataFrame:
    """
    Runs prediction over every trading day in [start_date, end_date].
    Returns a DataFrame with columns: date, asset, risk_label, risk_int, confidence, action
    """
    from app.config import RISK_ACTIONS
    dates = pd.bdate_range(start=start_date, end=end_date)
    usd_inr = user_inputs.get("USD_INR", 84.0)
    price_usd_forecast = forecast_prices(user_inputs, assets, len(dates))
    rows  = []
    for day_idx, d in enumerate(dates):
        day_inputs = user_inputs.copy()
        # Drift macro slightly per day to simulate changing environment
        noise = np.random.default_rng(int(d.strftime("%Y%m%d"))).normal(0, 0.01, 10)
        day_inputs["Geopolitical_Risk_VIX"] = max(10, user_inputs["Geopolitical_Risk_VIX"] + noise[0] * 3)
        day_inputs["USD_INR"]               = max(70, user_inputs["USD_INR"] + noise[1] * 0.5)
        day_inputs["SP500_Index"]           = max(2000, user_inputs["SP500_Index"] + noise[2] * 40)
        day_inputs["Oil_Price_USD"]         = max(30, user_inputs["Oil_Price_USD"] + noise[3] * 2)
        day_inputs["Festival_Season"]       = int(d.month in [10, 11, 4, 5])
        day_inputs["Festival_Intensity"]    = 0.8 if d.month in [10, 11] else (0.7 if d.month in [4, 5] else 0.1)
        day_inputs["Wedding_Season_Intensity"] = 0.9 if d.month in [10, 11, 12] else (0.7 if d.month in [4, 5] else 0.2)
        day_inputs["Quarter"]               = (d.month - 1) // 3 + 1
        day_inputs["Is_Weekend"]            = int(d.dayofweek >= 5)

        risk_res = predict_risk(day_inputs, d, assets)
        price_res = predict_price(day_inputs, d, assets)
        for asset, res in risk_res.items():
            price_col_map = {
                "Gold": "Gold_Price_USD", "Silver": "Silver_Price_USD",
                "Platinum": "Platinum_Price_USD", "Copper": "Copper_Price_USD",
                "Diamond": "Diamond_Index", "Emerald": "Emerald_Index",
                "Ruby": "Ruby_Index", "Pearl": "Pearl_Index",
            }
            base_usd = day_inputs.get(price_col_map.get(asset, ""), 0)
            # Deterministic daily drift seeded by date+asset so it's reproducible
            rng = np.random.default_rng(int(d.strftime("%Y%m%d")) + hash(asset) % 10000)
            drift = {"Low": 0.0003, "Medium": 0.0001, "High": -0.0004}[res["risk_label"]]
            day_noise = rng.normal(drift, 0.006)
            days_from_start = (d - dates[0]).days
            price_usd = base_usd * ((1 + day_noise) ** days_from_start)
            price_inr = price_usd * day_inputs.get("USD_INR", 84.0)
            rows.append({
                "Date":       d,
                "Asset":      asset,
                "Risk":       res["risk_label"],
                "RiskInt":    res["risk_int"],
                "Confidence": res["confidence"],
                "Action":     RISK_ACTIONS[res["risk_label"]],
                "Proba_Low":  res["proba"][0],
                "Proba_Med":  res["proba"][1] if len(res["proba"]) > 1 else 0,
                "Proba_High": res["proba"][2] if len(res["proba"]) > 2 else 0,
                "Price_INR":  round(price_usd_forecast[asset][day_idx] * usd_inr, 2),
            })
    return pd.DataFrame(rows)