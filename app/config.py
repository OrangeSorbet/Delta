# utils/config.py
# Delta — Investment Portfolio Optimizer
# Central config, constants, and default values

ASSET_METADATA = {
    "Gold": {
        "symbol": "XAU",
        "emoji": "🥇",
        "base_price_usd": 2050,
        "color": "#D4AF37",
        "category": "Metal",
        "description": "The ultimate safe-haven asset. Surges during geopolitical stress and inflation.",
    },
    "Silver": {
        "symbol": "XAG",
        "emoji": "🥈",
        "base_price_usd": 24,
        "color": "#C0C0C0",
        "category": "Metal",
        "description": "Dual role as precious metal and industrial commodity. Highly sensitive to Fed policy.",
    },
    "Platinum": {
        "symbol": "XPT",
        "emoji": "⚪",
        "base_price_usd": 960,
        "color": "#E5E4E2",
        "category": "Metal",
        "description": "Rarer than gold, driven by auto industry demand and supply constraints from South Africa.",
    },
    "Copper": {
        "symbol": "XCU",
        "emoji": "🟤",
        "base_price_usd": 8500,
        "color": "#B87333",
        "category": "Metal",
        "description": "The economic bellwether. China PMI and global manufacturing drive copper prices.",
    },
    "Diamond": {
        "symbol": "DIA",
        "emoji": "💎",
        "base_price_usd": 5500,
        "color": "#B9F2FF",
        "category": "Gemstone",
        "description": "Luxury demand asset. Festival seasons and wedding intensity are key price drivers.",
    },
    "Emerald": {
        "symbol": "EMR",
        "emoji": "💚",
        "base_price_usd": 3200,
        "color": "#50C878",
        "category": "Gemstone",
        "description": "Colombian origin premium commands 30-60% price uplift. Monsoon index affects supply.",
    },
    "Ruby": {
        "symbol": "RUB",
        "emoji": "❤️",
        "base_price_usd": 4100,
        "color": "#E0115F",
        "category": "Gemstone",
        "description": "Burmese rubies are the rarest. Geopolitical tension and India wedding season are catalysts.",
    },
    "Pearl": {
        "symbol": "PRL",
        "emoji": "🫧",
        "base_price_usd": 1800,
        "color": "#F5F5F0",
        "category": "Gemstone",
        "description": "Supply-driven market. Monsoon index and Japan-India trade dynamics shape pricing.",
    },
}

FEATURE_DEFAULTS = {
    # India Macro
    "USD_INR": 95.62,                  # Live: May 13, 2026 (Morningstar)
    "India_Inflation": 3.48,           # CPI April 2026 (MoSPI, released May 12)
    "RBI_Repo_Rate": 5.25,             # Held unchanged Apr 8, 2026 MPC meeting
    "Import_Duty_Gold_pct": 15.0,      # Raised back to 15% from 6% (May 13, 2026)
    "India_GDP_Growth": 6.9,           # RBI FY27 Q1 projection (Apr 8 MPC)
    "Monsoon_Index": 98.0,             # IMD 2026 pre-season forecast: near-normal
    # Global Macro
    "Global_Inflation": 4.0,           # Elevated: US CPI 3.8%, PCE 3.5%; Iran war driving energy pass-through
    "Fed_Rate": 3.75,                  # Upper bound; held at 3.50–3.75% since Mar 2026 (Apr 29 FOMC)
    "DXY_Index": 98.3,                 # DXY May 12, 2026 (Trading Economics / Yahoo Finance)
    "Oil_Price_USD": 101.5,            # Crude ~$101/bbl; Strait of Hormuz disruption (Yahoo Finance, May 13)
    "SP500_Index": 7400.0,             # S&P 500 close May 12, 2026 (Yahoo Finance)
    "China_PMI": 51.5,                 # Manufacturing PMI at 5-yr high May 2026 (Trading Economics)
    "Geopolitical_Risk_VIX": 18.0,     # VIX ~18 (Yahoo Finance, May 13, 2026)
    "Russia_Ukraine_Tension": 0.65,    # War ongoing; no major escalation, frozen front lines
    # Supply
    "Global_Mining_Output_Index": 100.0,
    "Lab_Diamond_Supply_Index": 135.0, # Continued surge; price collapse ongoing in lab segment
    "Emerald_Origin_Premium_pct": 44.0,
    "Diamond_Demand_Index": 93.0,      # Natural diamond demand depressed by lab supply glut
    # Demand signals
    "Festival_Season": 0,
    "Festival_Intensity": 0.0,
    "Wedding_Season_Intensity": 0.9,   # Peak Indian wedding season (Akha Teej + May)
    "Is_Weekend": 0,
    "Quarter": 2,
    # Asset price defaults (USD or index)
    "Gold_Price_USD": 4700.0,          # Spot ~$4,700/oz (JM Bullion / Yahoo Finance, May 13)
    "Gold_Volume": 56000,
    "Silver_Price_USD": 87.3,          # Spot ~$87.29/oz (JM Bullion, May 13)
    "Silver_Volume": 230000,
    "Platinum_Price_USD": 2135.0,      # Spot ~$2,135/oz (JM Bullion, May 13)
    "Platinum_Volume": 8600,
    "Copper_Price_USD": 14550.0,       # ~$6.60/lb = ~$14,550/mt, all-time high (Trading Economics, May 13)
    "Copper_Volume": 530000,
    "Diamond_Index": 4700.0,
    "Diamond_Volume": 1900,
    "Emerald_Index": 3600.0,
    "Emerald_Volume": 840,
    "Ruby_Index": 4400.0,
    "Ruby_Volume": 635,
    "Pearl_Index": 1950.0,
    "Pearl_Volume": 1580,
}

FEATURE_RANGES = {
    "USD_INR":                  (58.0,  105.0),
    "India_Inflation":          (1.0,   16.0),
    "RBI_Repo_Rate":            (2.5,   10.0),
    "Import_Duty_Gold_pct":     (0.0,   25.0),
    "India_GDP_Growth":         (-12.0, 12.0),
    "Monsoon_Index":            (45.0,  135.0),
    "Global_Inflation":         (0.0,   18.0),
    "Fed_Rate":                 (0.0,   8.0),
    "DXY_Index":                (78.0,  125.0),
    "Oil_Price_USD":            (15.0,  160.0),
    "SP500_Index":              (1800.0, 7500.0),
    "China_PMI":                (38.0,  58.0),
    "Geopolitical_Risk_VIX":    (8.0,   90.0),
    "Russia_Ukraine_Tension":   (0.0,   1.0),
    "Global_Mining_Output_Index": (65.0, 130.0),
    "Lab_Diamond_Supply_Index": (60.0,  200.0),
    "Emerald_Origin_Premium_pct": (5.0, 110.0),
    "Diamond_Demand_Index":     (55.0,  155.0),
    "Gold_Price_USD":           (800.0, 5000.0),
    "Silver_Price_USD":         (8.0,   60.0),
    "Platinum_Price_USD":       (400.0, 2500.0),
    "Copper_Price_USD":         (3500.0, 16000.0),
    "Diamond_Index":            (1500.0, 13000.0),
    "Emerald_Index":            (800.0,  9000.0),
    "Ruby_Index":               (1000.0, 12000.0),
    "Pearl_Index":              (400.0,  6000.0),
}

FEATURE_LABELS = {
    "USD_INR":                    "USD / INR Exchange Rate",
    "India_Inflation":            "India Inflation (%)",
    "RBI_Repo_Rate":              "RBI Repo Rate (%)",
    "Import_Duty_Gold_pct":       "Gold Import Duty (%)",
    "India_GDP_Growth":           "India GDP Growth (%)",
    "Monsoon_Index":              "Monsoon Index",
    "Global_Inflation":           "Global Inflation (%)",
    "Fed_Rate":                   "US Fed Rate (%)",
    "DXY_Index":                  "US Dollar Index (DXY)",
    "Oil_Price_USD":              "Crude Oil Price (USD/bbl)",
    "SP500_Index":                "S&P 500 Index",
    "China_PMI":                  "China Manufacturing PMI",
    "Geopolitical_Risk_VIX":      "VIX (Fear Index)",
    "Russia_Ukraine_Tension":     "Geopolitical Tension (0-1)",
    "Global_Mining_Output_Index": "Global Mining Output Index",
    "Lab_Diamond_Supply_Index":   "Lab Diamond Supply Index",
    "Emerald_Origin_Premium_pct": "Emerald Origin Premium (%)",
    "Diamond_Demand_Index":       "Diamond Demand Index",
    "Gold_Price_USD":             "Gold Spot Price (USD/oz)",
    "Silver_Price_USD":           "Silver Spot Price (USD/oz)",
    "Platinum_Price_USD":         "Platinum Spot Price (USD/oz)",
    "Copper_Price_USD":           "Copper Price (USD/MT)",
    "Diamond_Index":              "Diamond Index (USD)",
    "Emerald_Index":              "Emerald Index (USD)",
    "Ruby_Index":                 "Ruby Index (USD)",
    "Pearl_Index":                "Pearl Index (USD)",
}

RISK_COLORS = {
    "Low":    "#22c55e",
    "Medium": "#f59e0b",
    "High":   "#ef4444",
}

RISK_ICONS = {
    "Low":    "🟢",
    "Medium": "🟡",
    "High":   "🔴",
}

RISK_ACTIONS = {
    "Low":    "BUY",
    "Medium": "HOLD",
    "High":   "SELL / AVOID",
}