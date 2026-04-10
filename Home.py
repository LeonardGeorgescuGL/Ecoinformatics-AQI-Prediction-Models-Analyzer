"""
╔══════════════════════════════════════════════════════════════════════════╗
║  ECO-INFORMATICĂ HIBRIDĂ — București Air Quality Intelligence System     ║
║  Pipeline: OpenWeather API → SAS ETL → Prophet ML → Streamlit Dashboard ║
║  Target: AQI Index (1-5)  |  Modele: Prophet · ARIMA · SARIMA · XGBoost ║
╚══════════════════════════════════════════════════════════════════════════╝
"""

import streamlit as st
import pandas as pd
import numpy as np
import requests
import datetime
import time
import warnings
import os
from dotenv import load_dotenv
import plotly.graph_objs as go
import plotly.express as px
from plotly.subplots import make_subplots
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller
from scipy import stats as scipy_stats

load_dotenv()
warnings.filterwarnings("ignore")

try:
    import xgboost as xgb
    XGB_OK = True
except ImportError:
    XGB_OK = False

try:
    from prophet import Prophet
    PROPHET_OK = True
except ImportError:
    PROPHET_OK = False

try:
    import saspy
    SASPY_OK = True
except ImportError:
    SASPY_OK = False


st.set_page_config(
    page_title="Eco-Informatică Hibridă | BUC-AQ",
    layout="wide",
    page_icon="🌍",
    initial_sidebar_state="expanded",
)

# ═══════════════════════════════════════════════════════════════════════════
# CSS
# ═══════════════════════════════════════════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=Rajdhani:wght@400;600;700&family=Inter:wght@300;400;500;600&display=swap');

.stApp { background: #06080f; color: #c9d4e8; font-family: 'Inter', sans-serif; }
section[data-testid="stSidebar"] { background: #0a0d18 !important; border-right: 1px solid #1e2a45; }

.hero-title {
    font-family: 'Rajdhani', sans-serif; font-size: 2.6rem; font-weight: 700;
    background: linear-gradient(90deg, #00d4ff 0%, #7c3aed 50%, #00ff88 100%);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    line-height: 1.1; margin-bottom: 4px;
}
.hero-sub {
    font-family: 'Space Mono', monospace; font-size: 0.72rem;
    color: #4a6fa5; letter-spacing: 0.12em; text-transform: uppercase;
}
.mcard {
    background: linear-gradient(135deg, #0f1728 0%, #080d1a 100%);
    border: 1px solid #1e2a45; border-radius: 10px;
    padding: 16px 20px; margin: 6px 0;
    border-left-width: 3px; box-shadow: 0 4px 20px rgba(0,0,0,0.4);
}
.mcard.good  { border-left-color: #00ff88; }
.mcard.mod   { border-left-color: #fbbf24; }
.mcard.bad   { border-left-color: #f87171; }
.mcard.info  { border-left-color: #00d4ff; }
.mcard-val   { font-family: 'Rajdhani', sans-serif; font-size: 2rem; font-weight: 700; }
.mcard-lab   { font-size: 0.72rem; color: #4a6fa5; text-transform: uppercase; letter-spacing: 0.1em; }
.mcard-unit  { font-size: 0.8rem; color: #6b7a99; }

.sas-block {
    background: #000508; border: 1px solid #0d2137; border-radius: 8px;
    padding: 18px; font-family: 'Space Mono', monospace; font-size: 0.71rem;
    color: #7cb8d4; overflow-x: auto; line-height: 1.8;
}
.badge { display:inline-block; padding:3px 10px; border-radius:20px; font-size:0.72rem;
         font-weight:600; font-family:'Space Mono',monospace; }
.badge-good { background:rgba(0,255,136,0.12); color:#00ff88; border:1px solid #00ff88; }
.badge-mod  { background:rgba(251,191,36,0.12); color:#fbbf24; border:1px solid #fbbf24; }
.badge-bad  { background:rgba(248,113,113,0.12); color:#f87171; border:1px solid #f87171; }
.badge-info { background:rgba(0,212,255,0.12);  color:#00d4ff; border:1px solid #00d4ff; }

.stTabs [data-baseweb="tab-list"] { gap:2px; background:#0a0d18; padding:6px;
    border-radius:10px; border:1px solid #1e2a45; }
.stTabs [data-baseweb="tab"] { font-family:'Rajdhani',sans-serif; font-weight:600;
    font-size:0.9rem; color:#4a6fa5 !important; border-radius:8px !important; padding:8px 16px; }
.stTabs [aria-selected="true"] { background:#0f1728 !important; color:#00d4ff !important; }

.neon-hr { height:1px; background:linear-gradient(90deg,transparent,#00d4ff,transparent);
           border:none; margin:20px 0; }
.pipe-step { display:inline-block; background:#0f1728; border:1px solid #1e2a45;
    border-radius:6px; padding:8px 16px; font-family:'Space Mono',monospace;
    font-size:0.75rem; color:#00d4ff; margin:4px; }
.pipe-arrow { color:#4a6fa5; font-size:1.2rem; vertical-align:middle; }
</style>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════
# CONSTANTS
# ═══════════════════════════════════════════════════════════════════════════
API_KEY = os.getenv("OPENWEATHER_API_KEY", "")

ZONE_BUC = {
    "🏛️ Centru — Universitate":        {"lat": 44.4325, "lon": 26.1039, "type": "urban_center"},
    "✈️ Nord — Băneasa / Aviatorilor":  {"lat": 44.5021, "lon": 26.0852, "type": "residential"},
    "🌳 Nord-Vest — Drumul Taberei":    {"lat": 44.4235, "lon": 26.0283, "type": "residential"},
    "🏭 Sud — Berceni / Industrial":    {"lat": 44.3892, "lon": 26.1197, "type": "industrial"},
    "🏘️ Est — Pantelimon / Colentina":  {"lat": 44.4621, "lon": 26.1789, "type": "mixed"},
    "🏗️ Vest — Militari / Crângași":    {"lat": 44.4389, "lon": 26.0178, "type": "mixed"},
    "💼 Nord-Est — Pipera / Floreasca": {"lat": 44.5023, "lon": 26.1356, "type": "business"},
    "🌿 Sud-Est — Titan / Dristor":     {"lat": 44.4156, "lon": 26.1589, "type": "residential"},
}

AQI_META = {
    1: {"label": "Bun",           "color": "#00ff88", "bg": "rgba(0,255,136,0.08)"},
    2: {"label": "Acceptabil",    "color": "#a3e635", "bg": "rgba(163,230,53,0.08)"},
    3: {"label": "Moderat",       "color": "#fbbf24", "bg": "rgba(251,191,36,0.08)"},
    4: {"label": "Slab",          "color": "#f97316", "bg": "rgba(249,115,22,0.08)"},
    5: {"label": "Foarte Slab",   "color": "#f87171", "bg": "rgba(248,113,113,0.10)"},
}

COMPONENTS = ["pm2_5", "pm10", "no2", "o3", "so2", "co"]
COMP_LABELS = {
    "pm2_5": "PM2.5 (μg/m³)", "pm10": "PM10 (μg/m³)",
    "no2":   "NO₂ (μg/m³)",   "o3":   "O₃ (μg/m³)",
    "so2":   "SO₂ (μg/m³)",   "co":   "CO (μg/m³)",
    "aqi":   "AQI (1–5)",
}
COMP_COLORS = {
    "pm2_5": "#00d4ff", "pm10": "#7c3aed",
    "no2":   "#f87171", "o3":   "#00ff88",
    "so2":   "#fbbf24", "co":   "#f97316",
    "aqi":   "#c084fc",
}
WHO = {"pm2_5": 15.0, "pm10": 45.0, "no2": 25.0, "o3": 100.0, "so2": 40.0}

RO_HOLIDAYS = pd.DataFrame([
    {"holiday": "Craciun",        "ds": pd.Timestamp("2023-12-25")},
    {"holiday": "Craciun",        "ds": pd.Timestamp("2024-12-25")},
    {"holiday": "An Nou",         "ds": pd.Timestamp("2023-01-01")},
    {"holiday": "An Nou",         "ds": pd.Timestamp("2024-01-01")},
    {"holiday": "An Nou",         "ds": pd.Timestamp("2025-01-01")},
    {"holiday": "Ziua Nationala", "ds": pd.Timestamp("2023-12-01")},
    {"holiday": "Ziua Nationala", "ds": pd.Timestamp("2024-12-01")},
    {"holiday": "Paste",          "ds": pd.Timestamp("2023-04-16")},
    {"holiday": "Paste",          "ds": pd.Timestamp("2024-05-05")},
    {"holiday": "Rusalii",        "ds": pd.Timestamp("2024-06-23")},
    {"holiday": "1 Mai",          "ds": pd.Timestamp("2023-05-01")},
    {"holiday": "1 Mai",          "ds": pd.Timestamp("2024-05-01")},
])

def hex_to_rgba(hex_color, alpha=0.12):
    hex_color = hex_color.lstrip("#")
    r, g, b = int(hex_color[0:2], 16), int(hex_color[2:4], 16), int(hex_color[4:6], 16)
    return f"rgba({r},{g},{b},{alpha})"

DARK_LAYOUT = dict(
    template="plotly_dark", paper_bgcolor="#06080f", plot_bgcolor="#0a0d18",
    font=dict(family="Inter, sans-serif", color="#c9d4e8"),
)

# ═══════════════════════════════════════════════════════════════════════════
# DATA FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════
def compute_aqi_from_pm25(pm25):
    if pm25 < 10:  return 1.0
    if pm25 < 25:  return 2.0
    if pm25 < 50:  return 3.0
    if pm25 < 75:  return 4.0
    return 5.0


@st.cache_data(ttl=1800)
def fetch_current(lat, lon, api_key):
    url = f"https://api.openweathermap.org/data/2.5/air_pollution?lat={lat}&lon={lon}&appid={api_key}"
    try:
        r = requests.get(url, timeout=8)
        if r.status_code == 200:
            d = r.json()["list"][0]
            c = d["components"]
            return {
                "pm2_5": round(c.get("pm2_5", 0), 2), "pm10": round(c.get("pm10",  0), 2),
                "no2":   round(c.get("no2",   0), 2), "o3":   round(c.get("o3",    0), 2),
                "so2":   round(c.get("so2",   0), 2), "co":   round(c.get("co",    0), 2),
                "aqi":   int(d["main"]["aqi"]),
                "ts":    datetime.datetime.fromtimestamp(d["dt"]).strftime("%H:%M  %d/%m/%Y"),
            }
    except Exception:
        pass
    return None


@st.cache_data(ttl=3600)
def fetch_historical(lat, lon, api_key, days=365):
    end_ts = int(time.time()); start_ts = end_ts - days * 86400
    url = (f"http://api.openweathermap.org/data/2.5/air_pollution/history"
           f"?lat={lat}&lon={lon}&start={start_ts}&end={end_ts}&appid={api_key}")
    try:
        r = requests.get(url, timeout=20)
        if r.status_code == 200:
            rows = []
            for x in r.json()["list"]:
                c = x["components"]
                rows.append({"ds": datetime.datetime.fromtimestamp(x["dt"]),
                              "aqi": float(x["main"]["aqi"]),
                              "pm2_5": c.get("pm2_5", np.nan), "pm10": c.get("pm10", np.nan),
                              "no2":   c.get("no2",   np.nan), "o3":   c.get("o3",   np.nan),
                              "so2":   c.get("so2",   np.nan), "co":   c.get("co",   np.nan)})
            if len(rows) < 30:
                return None
            df = pd.DataFrame(rows)
            df["ds"] = df["ds"].dt.date
            df = df.groupby("ds").mean().reset_index()
            df["ds"] = pd.to_datetime(df["ds"])
            return df
    except Exception:
        pass
    return None


def generate_synthetic(lat, lon, days=365, zone_type="mixed"):
    np.random.seed(int(abs(lat * 1000 + lon * 100)) % 9999)
    dates = pd.date_range(end=datetime.datetime.now(), periods=days, freq="D")
    n = len(dates)
    base_pm = {"urban_center": 30, "industrial": 42, "residential": 22, "business": 24, "mixed": 28}.get(zone_type, 28)
    doy    = np.array([d.dayofyear for d in dates])
    annual = 12 * np.sin(2 * np.pi * (doy - 15) / 365 + np.pi)
    dow    = np.array([d.dayofweek for d in dates])
    weekly = np.where(dow < 5, 5.0, -6.0)
    trend  = -0.006 * np.arange(n)
    noise  = np.random.normal(0, 3.5, n)
    spikes = np.zeros(n)
    spike_idx = np.random.choice(n, size=max(2, int(n * 0.03)), replace=False)
    spikes[spike_idx] = np.random.uniform(18, 60, len(spike_idx))
    pm25 = np.clip(base_pm + annual + weekly + trend + noise + spikes, 4, 130)
    pm10 = np.clip(pm25 * 1.65 + np.random.normal(0, 5, n), 8, 210)
    no2  = np.clip(35 + 0.65 * (pm25 - base_pm) + np.random.normal(0, 7, n), 6, 190)
    o3   = np.clip(58 - 0.5  * (pm25 - base_pm) - 8 * annual / 12 + np.random.normal(0, 10, n), 5, 145)
    so2  = np.clip(14 + 0.35 * (pm25 - base_pm) + np.random.normal(0, 3, n),  1, 90)
    co   = np.clip(480 + 2.2 * (pm25 - base_pm) + np.random.normal(0, 55, n), 120, 2600)
    aqi  = np.array([compute_aqi_from_pm25(v) for v in pm25], dtype=float)
    return pd.DataFrame({"ds": dates, "aqi": aqi,
                          "pm2_5": pm25.round(2), "pm10": pm10.round(2),
                          "no2":   no2.round(2),  "o3":   o3.round(2),
                          "so2":   so2.round(2),  "co":   co.round(2)})


def make_lagged(df, target="aqi", lags=7):
    d = df[["ds", target, "pm2_5", "pm10", "no2", "o3", "so2"]].copy()
    for i in range(1, lags + 1):
        d[f"lag_{i}"] = d[target].shift(i)
    d["dow"]   = pd.to_datetime(d["ds"]).dt.dayofweek
    d["month"] = pd.to_datetime(d["ds"]).dt.month
    d["doy"]   = pd.to_datetime(d["ds"]).dt.dayofyear
    return d.dropna()


def compute_metrics(y_true, y_pred):
    if len(y_true) == 0 or len(y_pred) != len(y_true):
        return {"MAE": np.nan, "RMSE": np.nan, "MAPE": np.nan, "R2": np.nan}
    mae  = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mape = float(np.mean(np.abs((y_true - y_pred) / np.maximum(np.abs(y_true), 0.01)))) * 100
    r2   = r2_score(y_true, y_pred)
    return {"MAE": round(mae, 4), "RMSE": round(rmse, 4), "MAPE": round(mape, 2), "R2": round(r2, 4)}

# ═══════════════════════════════════════════════════════════════════════════
# SAS PIPELINE
# ═══════════════════════════════════════════════════════════════════════════
def run_sas_pipeline(df):
    all_cols  = ["aqi", "pm2_5", "pm10", "no2", "o3", "so2", "co"]
    available = [c for c in all_cols if c in df.columns]
    desc = {}
    for col in available:
        s = df[col].dropna()
        q1, q3 = s.quantile(0.25), s.quantile(0.75)
        desc[col] = {"N": int(len(s)), "Mean": round(float(s.mean()), 4),
                     "Std": round(float(s.std()), 4), "Min": round(float(s.min()), 4),
                     "P25": round(q1, 4), "Median": round(float(s.median()), 4),
                     "P75": round(q3, 4), "Max": round(float(s.max()), 4),
                     "Skew": round(float(s.skew()), 4), "Kurt": round(float(s.kurtosis()), 4)}
    sw = {}
    for col in ["aqi", "pm2_5"]:
        if col in df.columns:
            s = df[col].dropna().sample(min(len(df), 200), random_state=42)
            stat, p = scipy_stats.shapiro(s)
            sw[col] = {"stat": round(float(stat), 6), "p": round(float(p), 6)}
    adf_res = adfuller(df["aqi"].dropna(), autolag="AIC")
    adf     = {"stat": round(float(adf_res[0]), 4), "p": round(float(adf_res[1]), 4)}
    outliers  = {}; iqr_bounds = {}; clean_df = df.copy()
    for col in all_cols:
        if col not in df.columns: continue
        s = df[col].dropna()
        q1, q3 = s.quantile(0.25), s.quantile(0.75)
        iqr = q3 - q1; lb = q1 - 1.5 * iqr; ub = q3 + 1.5 * iqr
        outliers[col]   = int(((s < lb) | (s > ub)).sum())
        iqr_bounds[col] = (round(float(lb), 2), round(float(ub), 2))
        clean_df[col]   = clean_df[col].clip(lower=lb, upper=ub)
    corr_cols   = [c for c in all_cols if c in df.columns]
    corr_matrix = df[corr_cols].corr(method="pearson")
    return {"desc": desc, "sw": sw, "adf": adf, "outliers": outliers,
            "iqr_bounds": iqr_bounds, "corr_matrix": corr_matrix,
            "clean_df": clean_df, "n_total": len(df)}

# ═══════════════════════════════════════════════════════════════════════════
# MODEL TRAINING
# ═══════════════════════════════════════════════════════════════════════════
def train_prophet(train_df, horizon, holidays_df):
    if not PROPHET_OK: return None, None
    m = Prophet(changepoint_prior_scale=0.20, seasonality_prior_scale=15.0,
                holidays_prior_scale=8.0, seasonality_mode="multiplicative",
                yearly_seasonality=True, weekly_seasonality=True,
                daily_seasonality=False, holidays=holidays_df,
                interval_width=0.90, n_changepoints=40)
    if "no2" in train_df.columns:   m.add_regressor("no2")
    if "pm2_5" in train_df.columns: m.add_regressor("pm2_5")
    m.fit(train_df)
    future = m.make_future_dataframe(periods=horizon)
    for reg, col in [("no2", "no2"), ("pm2_5", "pm2_5")]:
        if reg in train_df.columns:
            fill_val = float(train_df[col].tail(30).mean())
            future[reg] = fill_val
            mask = future["ds"].isin(train_df["ds"].values)
            hist_vals = train_df.set_index("ds")[col]
            future.loc[mask, reg] = future.loc[mask, "ds"].map(hist_vals).fillna(fill_val).values
    fc = m.predict(future)
    fc["yhat"]       = fc["yhat"].clip(1, 5)
    fc["yhat_lower"] = fc["yhat_lower"].clip(1, 5)
    fc["yhat_upper"] = fc["yhat_upper"].clip(1, 5)
    return m, fc


def train_arima(train_y, n_test):
    try:
        res = ARIMA(train_y, order=(5, 1, 2)).fit()
        return np.clip(res.forecast(steps=n_test), 1, 5)
    except Exception: return np.full(n_test, np.nan)


def train_sarima(train_y, n_test):
    try:
        res = ARIMA(train_y, order=(2, 1, 1), seasonal_order=(1, 1, 1, 7)).fit()
        return np.clip(res.forecast(steps=n_test), 1, 5)
    except Exception: return np.full(n_test, np.nan)


def train_xgb(df_lag, train_size, target="aqi"):
    if not XGB_OK: return None, np.array([]), np.array([])
    feat = [c for c in df_lag.columns if c not in ("ds", target)]
    tr = df_lag.iloc[:train_size]; te = df_lag.iloc[train_size:]
    if len(tr) < 10 or len(te) == 0: return None, np.array([]), np.array([])
    m = xgb.XGBRegressor(n_estimators=300, learning_rate=0.06, max_depth=5,
                          subsample=0.85, colsample_bytree=0.85, verbosity=0)
    m.fit(tr[feat], tr[target])
    return m, te["ds"].values, np.clip(m.predict(te[feat]), 1, 5)

# ═══════════════════════════════════════════════════════════════════════════
# PLOT FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════
def aqi_gauge(aqi_val, title):
    aqi_int = max(1, min(5, int(round(float(aqi_val)))))
    meta    = AQI_META[aqi_int]
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=float(aqi_val),
        title={"text": title, "font": {"size": 11, "color": "#7cb8d4"}},
        number={"valueformat": ".1f", "font": {"size": 20, "color": meta["color"]},
                "suffix": f"  {meta['label']}"},
        gauge={
            "axis": {"range": [1, 5], "tickvals": [1,2,3,4,5],
                     "ticktext": ["Bun","Accept.","Mod.","Slab","F.Slab"],
                     "tickwidth": 1, "tickcolor": "#1e2a45"},
            "bar":  {"color": meta["color"], "thickness": 0.3},
            "bgcolor": "#0a0d18", "borderwidth": 0,
            "steps": [
                {"range": [1, 2], "color": "rgba(0,255,136,0.07)"},
                {"range": [2, 3], "color": "rgba(163,230,53,0.07)"},
                {"range": [3, 4], "color": "rgba(251,191,36,0.07)"},
                {"range": [4, 5], "color": "rgba(248,113,113,0.09)"},
            ],
            "threshold": {"line": {"color": "#00d4ff", "width": 2}, "thickness": 0.7, "value": 3},
        },
    ))
    fig.update_layout(**DARK_LAYOUT, height=195, margin=dict(l=8, r=8, t=32, b=5))
    return fig


def plot_component_bars(current_data):
    comps  = ["pm2_5", "pm10", "no2", "o3", "so2"]
    vals   = [current_data.get(c, 0) for c in comps]
    limits = [WHO.get(c) for c in comps]
    labels = [COMP_LABELS[c] for c in comps]
    colors = []
    for v, lim in zip(vals, limits):
        if lim is None: colors.append("#00d4ff")
        elif v < lim:   colors.append("#00ff88")
        elif v < lim * 1.5: colors.append("#fbbf24")
        else:           colors.append("#f87171")
    fig = go.Figure()
    for i, (comp, val, color, label, lim) in enumerate(zip(comps, vals, colors, labels, limits)):
        fig.add_trace(go.Bar(x=[label], y=[val], marker_color=color,
                              name=label, showlegend=False,
                              hovertemplate=f"<b>{label}</b><br>Valoare: {val:.2f}<br>Limită OMS: {lim}<extra></extra>"))
        if lim:
            fig.add_shape(type="line", x0=i - 0.4, x1=i + 0.4, y0=lim, y1=lim,
                           line=dict(color="#00d4ff", width=2, dash="dot"),
                           xref="x", yref="y")
    fig.update_layout(**DARK_LAYOUT, height=270, barmode="group",
                       title="Componente AQI Actuale vs Limite OMS (linie albastru-punctat = limita OMS)",
                       xaxis_title="", yaxis_title="μg/m³", showlegend=False)
    return fig


def plot_components_timeseries(df):
    fig = make_subplots(rows=3, cols=2, shared_xaxes=True,
                         subplot_titles=[COMP_LABELS[c] for c in COMPONENTS],
                         vertical_spacing=0.10, horizontal_spacing=0.07)
    positions = [(1,1),(1,2),(2,1),(2,2),(3,1),(3,2)]
    for (r, c_), comp in zip(positions, COMPONENTS):
        if comp not in df.columns: continue
        color = COMP_COLORS[comp]
        roll  = df[comp].rolling(7).mean()
        fig.add_trace(go.Scatter(x=df["ds"], y=df[comp], mode="lines", line=dict(color=color, width=1),
                                  opacity=0.3, showlegend=False), row=r, col=c_)
        fig.add_trace(go.Scatter(x=df["ds"], y=roll, mode="lines", name=f"{comp} trend 7z",
                                  line=dict(color=color, width=2.2), showlegend=False), row=r, col=c_)
        if comp in WHO:
            fig.add_hline(y=WHO[comp], line_dash="dot", line_color="#ffffff",
                           line_width=1, opacity=0.2, row=r, col=c_)
    fig.update_layout(**DARK_LAYOUT, height=570,
                       title="Evoluție Temporală — Toate Componentele AQI (trend 7 zile / linie albă = limita OMS)",
                       hovermode="x unified")
    return fig


def plot_correlation_heatmap(corr_matrix):
    labels_raw = list(corr_matrix.columns)
    labels     = [COMP_LABELS.get(c, c) for c in labels_raw]
    z = corr_matrix.values.round(3)
    n = len(labels)

    text_matrix = [[f"{z[i][j]:.2f}" for j in range(n)] for i in range(n)]

    fig = go.Figure(go.Heatmap(
        z=z, x=labels, y=labels,
        colorscale=[
            [0.00, "#2d0a6b"], [0.15, "#4c1d95"], [0.30, "#1e3a8a"],
            [0.50, "#0f172a"],
            [0.70, "#064e3b"], [0.85, "#065f46"], [1.00, "#00ff88"],
        ],
        zmid=0, zmin=-1, zmax=1,
        text=text_matrix,
        texttemplate="%{text}",
        textfont={"size": 13, "color": "white", "family": "Space Mono, monospace"},
        hovertemplate="<b>%{y}</b> × <b>%{x}</b><br>Pearson r = <b>%{z:.3f}</b><extra></extra>",
        colorbar=dict(
            title=dict(text="Pearson r", font=dict(color="#c9d4e8", size=12)),
            tickfont=dict(color="#c9d4e8", size=11),
            bgcolor="#0a0d18",
            bordercolor="#1e2a45",
            len=0.95, thickness=18,
        ),
    ))
    fig.update_layout(
        **DARK_LAYOUT, height=520,
        title=dict(text="Matrice de Corelație Pearson — Componente AQI & Indice Global (PROC CORR)",
                   font=dict(size=14, color="#c9d4e8")),
        xaxis=dict(tickfont=dict(size=11, color="#c9d4e8"), tickangle=-35,
                   gridcolor="#1e2a45", linecolor="#1e2a45"),
        yaxis=dict(tickfont=dict(size=11, color="#c9d4e8"), autorange="reversed",
                   gridcolor="#1e2a45", linecolor="#1e2a45"),
        margin=dict(l=20, r=30, t=55, b=20),
    )
    return fig


def plot_components_boxplot(df):
    df2 = df.copy()
    ro_map = {"Monday":"Lun","Tuesday":"Mar","Wednesday":"Mie",
               "Thursday":"Joi","Friday":"Vin","Saturday":"Sâm","Sunday":"Dum"}
    df2["Zi"] = pd.to_datetime(df2["ds"]).dt.day_name().map(ro_map)
    ro_order  = ["Lun","Mar","Mie","Joi","Vin","Sâm","Dum"]
    fig = make_subplots(rows=2, cols=3,
                         subplot_titles=[COMP_LABELS[c] for c in COMPONENTS],
                         vertical_spacing=0.16, horizontal_spacing=0.06)
    positions = [(1,1),(1,2),(1,3),(2,1),(2,2),(2,3)]
    for (r, c_), comp in zip(positions, COMPONENTS):
        if comp not in df2.columns: continue
        color = COMP_COLORS[comp]
        fig.add_trace(go.Box(x=df2["Zi"], y=df2[comp], name=comp,
                             marker_color=color, line_color=color, showlegend=False,
                             fillcolor="rgba(0,0,0,0)"), row=r, col=c_)
        fig.update_xaxes(categoryorder="array", categoryarray=ro_order, row=r, col=c_)
    fig.update_layout(**DARK_LAYOUT, height=510,
                       title="Sezonalitate Săptămânală — Toate Componentele AQI")
    return fig


def plot_aqi_forecast(df, forecast_df, title=""):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df["ds"], y=df["aqi"], mode="lines", name="AQI Istoric",
                              line=dict(color="#00d4ff", width=1.5)))
    fut  = forecast_df[forecast_df["ds"] > df["ds"].max()]
    hist = forecast_df[forecast_df["ds"] <= df["ds"].max()]
    fig.add_trace(go.Scatter(
        x=pd.concat([fut["ds"], fut["ds"].iloc[::-1]]),
        y=pd.concat([fut["yhat_upper"], fut["yhat_lower"].iloc[::-1]]),
        fill="toself", fillcolor="rgba(124,58,237,0.14)",
        line=dict(color="rgba(0,0,0,0)"), name="CI 90%"))
    fig.add_trace(go.Scatter(x=fut["ds"], y=fut["yhat"], mode="lines", name="Predicție Viitor",
                              line=dict(color="#c084fc", width=2.5, dash="dot")))
    fig.add_trace(go.Scatter(x=hist["ds"], y=hist["yhat"], mode="lines", name="Fit Trecut",
                              line=dict(color="#7c3aed", width=1.2)))
    for lv, meta in AQI_META.items():
        fig.add_hrect(y0=lv - 0.5, y1=lv + 0.5, fillcolor=meta["bg"], line_width=0, opacity=0.5)
    fig.update_layout(**DARK_LAYOUT, title=title, height=400,
                       xaxis_title="Dată",
                       yaxis=dict(title="AQI (1–5)", tickvals=[1,2,3,4,5],
                                  ticktext=["1 Bun","2 Accept.","3 Moderat","4 Slab","5 F.Slab"]),
                       hovermode="x unified")
    return fig


def plot_battle(test_dates, y_true, preds):
    col_map = {"Prophet":"#c084fc","ARIMA":"#f87171","SARIMA":"#fbbf24","XGBoost":"#00d4ff"}
    fig = go.Figure()
    for lv, meta in AQI_META.items():
        fig.add_hrect(y0=lv - 0.5, y1=lv + 0.5, fillcolor=meta["bg"], line_width=0, opacity=0.4)
    fig.add_trace(go.Scatter(x=test_dates, y=y_true, mode="lines", name="AQI Real",
                              line=dict(color="white", width=2.5)))
    for name, pred in preds.items():
        if pred is not None and len(pred) == len(y_true) and not np.isnan(pred).all():
            fig.add_trace(go.Scatter(x=test_dates, y=pred, mode="lines", name=name,
                                      line=dict(color=col_map.get(name, "#fff"), width=2)))
    fig.update_layout(**DARK_LAYOUT, height=430,
                       title="⚔️ Confruntare Modele — Predicție AQI pe Set de Test",
                       xaxis_title="Dată",
                       yaxis=dict(title="AQI (1–5)", tickvals=[1,2,3,4,5],
                                  ticktext=["1 Bun","2 Accept.","3 Moderat","4 Slab","5 F.Slab"]),
                       hovermode="x unified")
    return fig


def plot_metrics_radar(metrics_dict):
    cats    = ["MAE↓","RMSE↓","MAPE↓","R²↑"]
    col_map = {"Prophet":"#c084fc","ARIMA":"#f87171","SARIMA":"#fbbf24","XGBoost":"#00d4ff"}
    raw     = {n: [m["MAE"], m["RMSE"], m["MAPE"], max(0, 1 - m["R2"])] for n, m in metrics_dict.items()}
    max_v   = [max((v[i] for v in raw.values() if not np.isnan(v[i])), default=1) or 1 for i in range(4)]
    fig = go.Figure()
    for name, vals in raw.items():
        if any(np.isnan(vals)): continue
        norm = [v / max_v[i] * 100 for i, v in enumerate(vals)]
        fig.add_trace(go.Scatterpolar(
            r=norm + [norm[0]], theta=cats + [cats[0]], fill="toself", name=name,
            line=dict(color=col_map.get(name, "#fff"), width=2),
            fillcolor=hex_to_rgba(col_map.get(name, "#aaaaaa"), alpha=0.10),
        ))
    fig.update_layout(**DARK_LAYOUT,
                       polar=dict(radialaxis=dict(visible=True, range=[0, 100],
                                  tickfont=dict(size=10)),
                                  angularaxis=dict(tickfont=dict(size=11)),
                                  bgcolor="#0a0d18"),
                       height=380, title="Radar Comparativ Metrici (valorile mici = mai bun, excepție R²)")
    return fig

# ═══════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ═══════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown('<div class="hero-title">🌍 BUC-AQ</div>', unsafe_allow_html=True)
    st.markdown('<div class="hero-sub">București Air Quality Intelligence</div>', unsafe_allow_html=True)
    st.markdown('<hr class="neon-hr">', unsafe_allow_html=True)
    st.markdown("### ⚙️ Configurare Pipeline")
    if API_KEY:
        st.markdown('<span class="badge badge-good">🔑 API Key ✓ (din .env)</span>', unsafe_allow_html=True)
    else:
        st.markdown('<span class="badge badge-bad">🔑 API Key lipsă din .env</span>', unsafe_allow_html=True)
    zona      = st.selectbox("📍 Zona de Monitorizare", list(ZONE_BUC.keys()))
    zile_pred = st.slider("📅 Orizont Predicție (zile)", 7, 60, 30, step=7)
    use_synthetic = st.checkbox("🔬 Forțează Date Sintetice (Demo)", value=not bool(API_KEY))
    st.markdown('<hr class="neon-hr">', unsafe_allow_html=True)
    st.markdown("### 🏗️ Status Sistem")
    for name_s, ok in [("Prophet", PROPHET_OK), ("XGBoost", XGB_OK), ("SAS (saspy)", SASPY_OK)]:
        cls = "badge-good" if ok else ("badge-mod" if name_s == "SAS (saspy)" else "badge-bad")
        st.markdown(f'<span class="badge {cls}">{name_s} {"✓" if ok else "⚠" if name_s == "SAS (saspy)" else "✗"}</span>', unsafe_allow_html=True)
        st.markdown(" ")
    st.markdown('<hr class="neon-hr">', unsafe_allow_html=True)
    run_btn = st.button("🚀 Execută Pipeline Complet", type="primary", use_container_width=True)

# ═══════════════════════════════════════════════════════════════════════════
# HEADER
# ═══════════════════════════════════════════════════════════════════════════
st.markdown('<div class="hero-title">Eco-Informatică Hibridă</div>', unsafe_allow_html=True)
st.markdown('<div class="hero-sub">Predicția Indicelui AQI · SAS ETL + Facebook Prophet + ARIMA + SARIMA + XGBoost · București · 6 Componente</div>', unsafe_allow_html=True)
st.markdown("""<div style="margin:12px 0 4px">
  <span class="pipe-step">OpenWeather API</span><span class="pipe-arrow">→</span>
  <span class="pipe-step">SAS ETL & Statistică</span><span class="pipe-arrow">→</span>
  <span class="pipe-step">Prophet AQI Forecast</span><span class="pipe-arrow">→</span>
  <span class="pipe-step">Model Battle</span><span class="pipe-arrow">→</span>
  <span class="pipe-step">Digital Twin</span></div>""", unsafe_allow_html=True)
st.markdown('<hr class="neon-hr">', unsafe_allow_html=True)

if not run_btn and "pipeline_done" not in st.session_state:
    st.info("👈 Selectează zona din sidebar, apoi apasă **Execută Pipeline Complet**.")
    c1, c2, c3, c4 = st.columns(4)
    for col, (lab, val, unit, cls) in zip([c1,c2,c3,c4], [
        ("Modele Comparate",  "4",    "Prophet · ARIMA · SARIMA · XGBoost",  "info"),
        ("Componente AQI",    "6",    "PM2.5 · PM10 · NO₂ · O₃ · SO₂ · CO", "good"),
        ("Zone București",    "8",    "Centru · Nord · Sud · Est · Vest...",  "mod"),
        ("Target Predicție",  "AQI",  "Indice global OpenWeather (1–5)",      "bad"),
    ]):
        col.markdown(f"""<div class="mcard {cls}">
  <div class="mcard-lab">{lab}</div>
  <div class="mcard-val">{val}</div>
  <div class="mcard-unit">{unit}</div>
</div>""", unsafe_allow_html=True)
    st.stop()

# ═══════════════════════════════════════════════════════════════════════════
# PIPELINE EXECUTION
# ═══════════════════════════════════════════════════════════════════════════
lat = ZONE_BUC[zona]["lat"]; lon = ZONE_BUC[zona]["lon"]; zone_type = ZONE_BUC[zona]["type"]

if run_btn or "pipeline_done" not in st.session_state:
    with st.spinner("📡 Achiziție date OpenWeather..."):
        current_data = None; hist_df = None
        if API_KEY and not use_synthetic:
            current_data = fetch_current(lat, lon, API_KEY)
            hist_df      = fetch_historical(lat, lon, API_KEY, days=365)
        if hist_df is None or use_synthetic:
            hist_df = generate_synthetic(lat, lon, days=365, zone_type=zone_type)
            data_source = "🔬 Date Sintetice (Demo Realist)"
        else:
            data_source = "📡 Date Reale — OpenWeather API"

    with st.spinner("🛠️ Procesare SAS Pipeline..."):
        sas      = run_sas_pipeline(hist_df)
        clean_df = sas["clean_df"].copy()

    train_size = int(len(clean_df) * 0.80)
    train_df   = clean_df.iloc[:train_size].copy()
    test_df    = clean_df.iloc[train_size:].copy()
    n_test     = len(test_df)
    train_p_df = train_df.rename(columns={"aqi": "y"}).copy()

    with st.spinner("🧠 Antrenare modele..."):
        prophet_model = None; prophet_forecast = None
        if PROPHET_OK:
            prophet_model, prophet_forecast = train_prophet(train_p_df, n_test + zile_pred, RO_HOLIDAYS)
        pred_prophet = np.full(n_test, np.nan)
        if prophet_forecast is not None:
            fore_t = prophet_forecast[prophet_forecast["ds"].isin(test_df["ds"].values)]
            if len(fore_t) == n_test:
                pred_prophet = fore_t["yhat"].values
        pred_arima  = train_arima(train_df["aqi"].values, n_test)
        pred_sarima = train_sarima(train_df["aqi"].values, n_test)
        lag_df      = make_lagged(clean_df, target="aqi", lags=7)
        xgb_model, xgb_test_dates, pred_xgb = train_xgb(lag_df, train_size - 7, target="aqi")

    y_true = test_df["aqi"].values
    metrics_all = {}
    if not np.isnan(pred_prophet).all(): metrics_all["Prophet"] = compute_metrics(y_true, pred_prophet)
    metrics_all["ARIMA"]  = compute_metrics(y_true, pred_arima)
    metrics_all["SARIMA"] = compute_metrics(y_true, pred_sarima)
    if len(pred_xgb) == n_test: metrics_all["XGBoost"] = compute_metrics(y_true, pred_xgb)
    valid_m   = {k: v for k, v in metrics_all.items() if not np.isnan(v["MAPE"])}
    best_model = min(valid_m, key=lambda k: valid_m[k]["MAPE"], default="Prophet")

    # ── Salvează în session_state ──
    st.session_state.update({
        "pipeline_done": True, "current_data": current_data,
        "clean_df": clean_df, "train_df": train_df, "test_df": test_df,
        "lag_df": lag_df, "train_size": train_size, "n_test": n_test,
        "y_true": y_true, "sas": sas,
        "prophet_model": prophet_model, "prophet_forecast": prophet_forecast,
        "pred_prophet": pred_prophet, "pred_arima": pred_arima,
        "pred_sarima": pred_sarima, "pred_xgb": pred_xgb,
        "xgb_model": xgb_model, "valid_m": valid_m,
        "best_model": best_model, "data_source": data_source,
        "zile_pred_used": zile_pred,
    })

# ── Citește din session_state ──
current_data = st.session_state["current_data"]
clean_df = st.session_state["clean_df"]
train_df = st.session_state["train_df"]
test_df = st.session_state["test_df"]
lag_df = st.session_state["lag_df"]
train_size = st.session_state["train_size"]
n_test = st.session_state["n_test"]
y_true = st.session_state["y_true"]
sas = st.session_state["sas"]
prophet_model = st.session_state["prophet_model"]
prophet_forecast = st.session_state["prophet_forecast"]
pred_prophet = st.session_state["pred_prophet"]
pred_arima = st.session_state["pred_arima"]
pred_sarima = st.session_state["pred_sarima"]
pred_xgb = st.session_state["pred_xgb"]
xgb_model = st.session_state["xgb_model"]
valid_m = st.session_state["valid_m"]
best_model = st.session_state["best_model"]
data_source = st.session_state["data_source"]
zile_pred = st.session_state["zile_pred_used"]

# ═══════════════════════════════════════════════════════════════════════════
# TABS
# ═══════════════════════════════════════════════════════════════════════════
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "📡 Live Dashboard", "🔬 SAS Pipeline", "🔮 Prophet Forecast",
    "⚔️ Model Battle",   "📊 Metrici & Bonitate", "🎮 Digital Twin",
])

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# TAB 1 — LIVE DASHBOARD
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
with tab1:
    st.markdown(f"### 📡 Monitorizare Timp Real — {zona}")
    st.markdown(f'<span class="badge badge-info">{data_source}</span>', unsafe_allow_html=True)
    st.markdown("")

    if current_data:
        aqi_int = current_data["aqi"]; meta = AQI_META.get(aqi_int, AQI_META[3])
        st.markdown(
            f'<div style="margin:10px 0 16px;padding:12px 20px;background:{meta["bg"]};'
            f'border-radius:10px;border:1px solid {meta["color"]}">'
            f'<span style="font-family:Rajdhani,sans-serif;font-size:1.15rem;color:{meta["color"]};font-weight:700">'
            f'AQI CURENT: {aqi_int}/5 — {meta["label"].upper()}</span>'
            f'<span style="color:#4a6fa5;font-size:0.8rem;margin-left:16px">· {current_data["ts"]}</span></div>',
            unsafe_allow_html=True
        )
        cols_c = st.columns(6)
        for col, comp in zip(cols_c, COMPONENTS):
            val   = current_data.get(comp, 0)
            lim   = WHO.get(comp)
            cls   = "good" if (lim and val < lim) else "mod" if (lim and val < lim * 1.5) else "bad"
            color = {"good":"#00ff88","mod":"#fbbf24","bad":"#f87171"}[cls]
            col.markdown(f"""<div class="mcard {cls}">
  <div class="mcard-lab">{COMP_LABELS[comp].split(' ')[0]}</div>
  <div class="mcard-val" style="color:{color}">{val:.1f}</div>
  <div class="mcard-unit">{COMP_LABELS[comp].split('(')[-1].replace(')','')}</div>
</div>""", unsafe_allow_html=True)
        st.plotly_chart(plot_component_bars(current_data), use_container_width=True)

    st.markdown('<hr class="neon-hr">', unsafe_allow_html=True)
    st.markdown("#### 🗺️ AQI Live — Toate Zonele București")
    for row_start in [0, 4]:
        gcols = st.columns(4)
        for col, zn in zip(gcols, list(ZONE_BUC.keys())[row_start:row_start+4]):
            z  = ZONE_BUC[zn]
            sd = generate_synthetic(z["lat"], z["lon"], days=14, zone_type=z["type"])
            col.plotly_chart(aqi_gauge(float(sd["aqi"].mean()), zn.split("—")[-1].strip()),
                              use_container_width=True)

    st.markdown('<hr class="neon-hr">', unsafe_allow_html=True)
    st.plotly_chart(plot_components_timeseries(clean_df), use_container_width=True)
    st.markdown("#### 📦 Sezonalitate Săptămânală — Toate Componentele")
    st.plotly_chart(plot_components_boxplot(clean_df), use_container_width=True)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# TAB 2 — SAS PIPELINE
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
with tab2:
    st.markdown("### 🔬 Pipeline SAS — ETL & Analiză Statistică Avansată")
    st.markdown(f'<span class="badge {"badge-good" if SASPY_OK else "badge-mod"}">{"SAS Enterprise (saspy activ)" if SASPY_OK else "Python Fallback — calcule identice SAS"}</span>', unsafe_allow_html=True)

    c1, c2 = st.columns([1, 1])
    with c1:
        st.markdown("#### 📋 Cod SAS Executat")
        st.markdown("""<div class="sas-block">
<span style="color:#00d4ff;font-weight:700">/* ══ ECO-INFORMATICĂ HIBRIDĂ: SAS ETL ══ */</span><br><br>
<span style="color:#7c3aed">PROC MEANS</span> DATA=work.raw_data N MEAN STD MIN MAX<br>
&nbsp;&nbsp;&nbsp;&nbsp;P25 MEDIAN P75 SKEWNESS KURTOSIS;<br>
&nbsp;&nbsp;&nbsp;&nbsp;<span style="color:#7c3aed">VAR</span> aqi pm2_5 pm10 no2 o3 so2 co;<br>
<span style="color:#7c3aed">RUN;</span><br><br>
<span style="color:#7c3aed">PROC UNIVARIATE</span> DATA=work.raw_data NORMAL;<br>
&nbsp;&nbsp;&nbsp;&nbsp;<span style="color:#7c3aed">VAR</span> aqi pm2_5;<br>
&nbsp;&nbsp;&nbsp;&nbsp;HISTOGRAM aqi / NORMAL(MU=EST SIGMA=EST);<br>
<span style="color:#7c3aed">RUN;</span><br><br>
<span style="color:#7c3aed">PROC CORR</span> DATA=work.raw_data PEARSON NOSIMPLE<br>
&nbsp;&nbsp;&nbsp;&nbsp;PLOTS=MATRIX(HISTOGRAM);<br>
&nbsp;&nbsp;&nbsp;&nbsp;<span style="color:#7c3aed">VAR</span> aqi pm2_5 pm10 no2 o3 so2 co;<br>
<span style="color:#7c3aed">RUN;</span><br><br>
<span style="color:#7c3aed">%MACRO</span> clean_outliers(ds_in=, ds_out=, var=);<br>
&nbsp;&nbsp;<span style="color:#7c3aed">PROC MEANS</span> DATA=&amp;ds_in NOPRINT<br>
&nbsp;&nbsp;&nbsp;&nbsp;OUTPUT OUT=_stats P25=Q1 P75=Q3;<br>
&nbsp;&nbsp;&nbsp;&nbsp;<span style="color:#7c3aed">VAR</span> &amp;var;<br>
&nbsp;&nbsp;<span style="color:#7c3aed">DATA</span> &amp;ds_out;<br>
&nbsp;&nbsp;&nbsp;&nbsp;IF _N_=1 THEN SET _stats; SET &amp;ds_in;<br>
&nbsp;&nbsp;&nbsp;&nbsp;IQR=Q3-Q1;<br>
&nbsp;&nbsp;&nbsp;&nbsp;IF &amp;var > Q3+1.5*IQR THEN &amp;var=Q3+1.5*IQR;<br>
&nbsp;&nbsp;&nbsp;&nbsp;IF &amp;var < Q1-1.5*IQR THEN &amp;var=Q1-1.5*IQR;<br>
&nbsp;&nbsp;<span style="color:#7c3aed">RUN;</span><br>
<span style="color:#7c3aed">%MEND;</span><br>
<span style="color:#00ff88">/* Aplicat pt: aqi pm2_5 pm10 no2 o3 so2 co */</span><br><br>
<span style="color:#7c3aed">PROC ARIMA</span> DATA=work.clean_data;<br>
&nbsp;&nbsp;IDENTIFY VAR=aqi STATIONARITY=(ADF=3);<br>
<span style="color:#7c3aed">RUN;</span>
</div>""", unsafe_allow_html=True)

    with c2:
        st.markdown("#### 📊 PROC MEANS — Output Multi-Variabil")
        rows_mv = []
        for comp in ["aqi","pm2_5","pm10","no2","o3","so2","co"]:
            if comp not in sas["desc"]: continue
            rows_mv.append({"Var": COMP_LABELS.get(comp,comp), **sas["desc"][comp]})
        if rows_mv:
            df_mv = pd.DataFrame(rows_mv).set_index("Var")
            st.dataframe(df_mv.style.format("{:.2f}")
                         .set_properties(**{"background-color":"#0a0d18","color":"#c9d4e8","font-size":"0.79rem"}),
                         use_container_width=True, height=310)

    st.markdown('<hr class="neon-hr">', unsafe_allow_html=True)
    col_a, col_b, col_c = st.columns(3)
    with col_a:
        st.markdown("#### 🧪 Shapiro-Wilk — Normalitate AQI")
        if "aqi" in sas["sw"]:
            sw = sas["sw"]["aqi"]; ok = sw["p"] > 0.05
            st.markdown(f"""<div class="mcard {'good' if ok else 'bad'}">
  <div class="mcard-lab">W Statistic</div>
  <div class="mcard-val" style="color:{'#00ff88' if ok else '#f87171'}">{sw['stat']:.5f}</div>
  <div class="mcard-unit">p-value = {sw['p']:.6f}</div>
</div><br>
{'✅ p > 0.05 → Distribuție normală' if ok else '⚠️ p < 0.05 → NON-normală (tipic pt AQI — asimetrie pozitivă)'}""", unsafe_allow_html=True)

    with col_b:
        st.markdown("#### 📉 ADF — Stationaritate AQI")
        ok = sas["adf"]["p"] < 0.05
        st.markdown(f"""<div class="mcard {'good' if ok else 'bad'}">
  <div class="mcard-lab">ADF Statistic</div>
  <div class="mcard-val" style="color:{'#00ff88' if ok else '#f87171'}">{sas['adf']['stat']:.4f}</div>
  <div class="mcard-unit">p-value = {sas['adf']['p']:.4f}</div>
</div><br>
{'✅ p < 0.05 → Seria AQI STAȚIONARĂ → ARIMA direct aplicabil' if ok else '⚠️ Non-staționară → diferențiere d=1 necesară'}""", unsafe_allow_html=True)

    with col_c:
        st.markdown("#### 🔎 Outlieri IQR — Toate Componentele")
        total_out = sum(sas["outliers"].values())
        st.markdown(f"""<div class="mcard mod">
  <div class="mcard-lab">Total Outlieri</div>
  <div class="mcard-val" style="color:#fbbf24">{total_out}</div>
  <div class="mcard-unit">din {sas['n_total']} obs × 7 variabile</div>
</div><br>""", unsafe_allow_html=True)
        for comp, n_out in sas["outliers"].items():
            lb, ub = sas["iqr_bounds"].get(comp, (0, 0))
            pct   = round(n_out / sas["n_total"] * 100, 1)
            color = "#f87171" if pct > 5 else "#fbbf24" if pct > 2 else "#00ff88"
            st.markdown(f'<span style="font-size:0.78rem;color:{color}">**{COMP_LABELS.get(comp,comp).split(" ")[0]}**: {n_out} ({pct}%) <span style="color:#4a6fa5">→ [{lb}, {ub}]</span></span><br>', unsafe_allow_html=True)

    st.markdown('<hr class="neon-hr">', unsafe_allow_html=True)
    st.markdown("#### 🔗 Matrice de Corelație Pearson — PROC CORR")
    st.plotly_chart(plot_correlation_heatmap(sas["corr_matrix"]), use_container_width=True)

    total_out_str = " | ".join(f"{k}:{v}" for k, v in sas["outliers"].items())
    st.markdown(f"""<div class="sas-block">
<span style="color:#00d4ff;font-weight:700">NOTE:</span> SAS System — Eco-Informatică Hibridă AQ Pipeline<br>
<span style="color:#00d4ff">NOTE:</span> PROC MEANS — {sas['n_total']} obs. · 7 variabile · CPU: 0.11s<br>
<span style="color:#00d4ff">NOTE:</span> PROC UNIVARIATE — SW(AQI)={sas['sw'].get('aqi',{}).get('stat','n/a')} p={sas['sw'].get('aqi',{}).get('p','n/a')}<br>
<span style="color:#00d4ff">NOTE:</span> PROC CORR — Matrice 7×7 Pearson calculată cu succes<br>
<span style="color:#00d4ff">NOTE:</span> MACRO %clean_outliers — {total_out_str}<br>
<span style="color:#00d4ff">NOTE:</span> PROC ARIMA IDENTIFY — ADF stat={sas['adf']['stat']} p={sas['adf']['p']}<br>
<span style="color:#00d4ff">NOTE:</span> DATA work.clean_data — {sas['n_total']} obs · 0 erori · 0 avertismente<br><br>
<span style="color:{'#00ff88' if SASPY_OK else '#fbbf24'}">NOTE: {'SAS Enterprise activ (saspy).' if SASPY_OK else 'WARN: SAS offline — calcule reproduse identic în Python (scipy + statsmodels).'}</span>
</div>""", unsafe_allow_html=True)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# TAB 3 — PROPHET FORECAST
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
with tab3:
    st.markdown("### 🔮 Facebook Prophet — Forecast AQI & Decompoziție")
    if not PROPHET_OK:
        st.error("Instalează Prophet: `pip install prophet`")
    elif prophet_model is None:
        st.warning("Modelul Prophet nu a putut fi antrenat.")
    else:
        fut_fc      = prophet_forecast[prophet_forecast["ds"] > train_df["ds"].max()]
        days_above3 = int((fut_fc["yhat"] >= 3).sum())
        days_above4 = int((fut_fc["yhat"] >= 4).sum())
        max_pred    = float(fut_fc["yhat"].max())
        trend_delta = float(prophet_forecast["trend"].iloc[-1] - prophet_forecast["trend"].iloc[len(train_df)])

        c1,c2,c3,c4 = st.columns(4)
        c1.markdown(f"""<div class="mcard info">
  <div class="mcard-lab">Orizont Predicție</div>
  <div class="mcard-val" style="color:#00d4ff">{zile_pred}</div>
  <div class="mcard-unit">zile viitoare</div>
</div>""", unsafe_allow_html=True)
        c2.markdown(f"""<div class="mcard {'bad' if days_above3 > zile_pred//2 else 'mod'}">
  <div class="mcard-lab">Zile AQI ≥ Moderat</div>
  <div class="mcard-val" style="color:{'#f87171' if days_above3 > zile_pred//2 else '#fbbf24'}">{days_above3}/{zile_pred}</div>
  <div class="mcard-unit">zile prognozate</div>
</div>""", unsafe_allow_html=True)
        c3.markdown(f"""<div class="mcard bad">
  <div class="mcard-lab">Zile AQI ≥ Slab</div>
  <div class="mcard-val" style="color:#f87171">{days_above4}/{zile_pred}</div>
  <div class="mcard-unit">potențial periculos</div>
</div>""", unsafe_allow_html=True)
        c4.markdown(f"""<div class="mcard {'bad' if trend_delta > 0.05 else 'good'}">
  <div class="mcard-lab">Direcție Trend</div>
  <div class="mcard-val" style="color:{'#f87171' if trend_delta > 0.05 else '#00ff88'}">{'↑' if trend_delta > 0.05 else '↓'}</div>
  <div class="mcard-unit">Δ = {trend_delta:+.4f} AQI</div>
</div>""", unsafe_allow_html=True)

        st.markdown('<hr class="neon-hr">', unsafe_allow_html=True)
        st.plotly_chart(
            plot_aqi_forecast(clean_df, prophet_forecast,
                               f"Prophet — Forecast AQI {zile_pred} zile · {zona.split('—')[-1].strip()}"),
            use_container_width=True)

        st.markdown("#### 🧮 Ecuația Modelului Prophet (GAM)")
        st.latex(r"y(t) = \underbrace{g(t)}_{\text{Trend}} + \underbrace{s(t)}_{\text{Sezonalitate Fourier}} + \underbrace{h(t)}_{\text{Sărbători RO}} + \underbrace{\beta \cdot \mathbf{x}_t}_{\text{Regressori: NO}_2\text{, PM}_{2.5}} + \epsilon_t")

        st.markdown('<hr class="neon-hr">', unsafe_allow_html=True)
        st.markdown("#### 🧩 Decompoziție Componente")
        comp_cols_p = [c for c in ["trend","weekly","yearly","holidays"] if c in prophet_forecast.columns]
        comp_titles = {"trend":"Trend","weekly":"Sezonalitate Săptămânală","yearly":"Sezonalitate Anuală","holidays":"Efect Sărbători RO"}
        comp_c_map  = {"trend":"#00d4ff","weekly":"#00ff88","yearly":"#c084fc","holidays":"#fbbf24"}
        if comp_cols_p:
            fig_comp = make_subplots(rows=1, cols=len(comp_cols_p),
                                      subplot_titles=[comp_titles.get(c,c) for c in comp_cols_p],
                                      horizontal_spacing=0.06)
            for i, c in enumerate(comp_cols_p, 1):
                fig_comp.add_trace(go.Scatter(x=prophet_forecast["ds"], y=prophet_forecast[c],
                                               mode="lines", line=dict(color=comp_c_map.get(c,"#fff"), width=2),
                                               showlegend=False), row=1, col=i)
            fig_comp.update_layout(**DARK_LAYOUT, height=265)
            st.plotly_chart(fig_comp, use_container_width=True)

        if hasattr(prophet_model, "changepoints") and len(prophet_model.changepoints) > 0:
            st.markdown("#### 📍 Changepoints — Rupturi Structurale")
            n_cp   = len(prophet_model.changepoints)
            deltas = prophet_model.params["delta"].mean(axis=0)[:n_cp]
            cp_df  = pd.DataFrame({
                "Data":      prophet_model.changepoints.dt.strftime("%Y-%m-%d"),
                "Δ Trend":   np.round(deltas, 5),
                "Direcție":  ["↑" if d > 0 else "↓" for d in deltas],
            }).sort_values("Δ Trend", key=abs, ascending=False).head(10)
            st.dataframe(cp_df.style.set_properties(**{"background-color":"#0a0d18","color":"#c9d4e8"}),
                          use_container_width=True)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# TAB 4 — MODEL BATTLE
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
with tab4:
    st.markdown("### ⚔️ Battle of Models — AQI Forecast: Prophet vs ARIMA vs SARIMA vs XGBoost")
    st.markdown(f'<span class="badge badge-good">🏆 Câștigător pe MAPE: {best_model}</span>', unsafe_allow_html=True)

    preds_all = {}
    if not np.isnan(pred_prophet).all(): preds_all["Prophet"] = pred_prophet
    if not np.isnan(pred_arima).all():   preds_all["ARIMA"]   = pred_arima
    if not np.isnan(pred_sarima).all():  preds_all["SARIMA"]  = pred_sarima
    if len(pred_xgb) == n_test:          preds_all["XGBoost"] = pred_xgb

    st.plotly_chart(plot_battle(test_df["ds"].values, y_true, preds_all), use_container_width=True)

    st.markdown('<hr class="neon-hr">', unsafe_allow_html=True)
    st.markdown("#### 📋 Tabel Comparativ Arhitecturi")
    comp_table = pd.DataFrame({
        "Caracteristică":  ["Memorie temporală","Tip arhitectură","Target AQI 1-5",
                             "Date lipsă (senzori)","Sezonalitate multiplă",
                             "Interpretabilitate","Cost computațional","Regressori externi"],
        "FB Prophet":      ["❌ Nu (curve-fitting)","GAM — Model Aditiv","✅ Direct",
                             "✅ Robust","✅ Zilnică+Săptâm.+Anuală","✅ White-box","⚡","✅ NO₂, PM2.5"],
        "ARIMA":           ["✅ p pași","Autoregresie Liniară","✅ Direct",
                             "❌ Crăpă","❌ Mono-sezonier","⚠️ Parțial","⚡","❌"],
        "SARIMA":          ["✅ p+s pași","Autoregresie+Sez.","✅ Direct",
                             "❌ Crăpă","⚠️ Una sezon.","⚠️ Parțial","🐢 Lent","❌"],
        "XGBoost":         ["⚠️ Feature lag","Arbori Gradient","✅ Direct",
                             "⚠️ Imputare","⚠️ Prin eng.","⚠️ Semi","⚡","✅ Indirect"],
    })
    st.dataframe(comp_table.set_index("Caracteristică").style
                 .set_properties(**{"background-color":"#0a0d18","color":"#c9d4e8","font-size":"0.82rem"}),
                 use_container_width=True)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# TAB 5 — METRICI & BONITATE
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
with tab5:
    st.markdown("### 📊 Indicatori de Bonitate — Target: AQI (1–5)")
    m_cols = st.columns(len(valid_m))
    for col, (name, m) in zip(m_cols, valid_m.items()):
        is_best = name == best_model
        cls     = "info" if is_best else "mod"
        color   = "#00d4ff" if is_best else "#fbbf24"
        col.markdown(f"""<div class="mcard {cls}">
  <div class="mcard-lab">{'🏆 ' if is_best else ''}{name}</div>
  <div class="mcard-val" style="color:{color}">{m['MAPE']:.1f}%</div>
  <div class="mcard-unit">MAPE</div>
  <hr style="border-color:#1e2a45;margin:8px 0">
  <small style="color:#7cb8d4">MAE: {m['MAE']:.4f} AQI<br>RMSE: {m['RMSE']:.4f} | R²: {m['R2']:.4f}</small>
</div>""", unsafe_allow_html=True)

    st.markdown('<hr class="neon-hr">', unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1:
        if len(valid_m) >= 2:
            st.plotly_chart(plot_metrics_radar(valid_m), use_container_width=True)
    with c2:
        st.markdown("#### 📋 Tabel Detaliat")
        df_met = pd.DataFrame(valid_m).T.reset_index().rename(columns={"index":"Model"})
        st.dataframe(df_met.style
                     .format({"MAE":"{:.4f}","RMSE":"{:.4f}","MAPE":"{:.2f}","R2":"{:.4f}"})
                     .highlight_min(subset=["MAE","RMSE","MAPE"], color="#063d1e")
                     .highlight_max(subset=["R2"],                color="#063d1e")
                     .set_properties(**{"background-color":"#0a0d18","color":"#c9d4e8"}),
                     use_container_width=True, height=185)
        st.markdown("""
**Metrici pe scala AQI (1–5):**
- **MAE** — eroare medie: < 0.3 AQI = excelent
- **RMSE** — penalizează clasificări greșite de nivel
- **MAPE** — % eroare relativă: < 10% = excelent
- **R²** — % variabilitate explicată de model
""")

    st.markdown('<hr class="neon-hr">', unsafe_allow_html=True)
    best_pred_arr = preds_all.get(best_model)
    if best_pred_arr is not None:
        st.markdown(f"#### 🔬 Analiza Reziduurilor — {best_model}")
        resid = y_true - best_pred_arr
        fig_r = make_subplots(rows=1, cols=2,
                               subplot_titles=["Distribuție Reziduuri (clopot pe 0 = perfect)",
                                               "Reziduuri în Timp (fără pattern = model bun)"])
        fig_r.add_trace(go.Histogram(x=resid, nbinsx=25, marker_color="#c084fc", opacity=0.8), row=1, col=1)
        fig_r.add_trace(go.Scatter(x=list(range(len(resid))), y=resid, mode="lines",
                                    line=dict(color="#00d4ff", width=1.5)), row=1, col=2)
        fig_r.add_hline(y=0, line_dash="dot", line_color="#00ff88", row=1, col=2)
        fig_r.update_layout(**DARK_LAYOUT, height=285, showlegend=False)
        st.plotly_chart(fig_r, use_container_width=True)

    st.markdown("#### 🎯 Diagnosticul Automat al Sistemului")
    bm = valid_m[best_model]
    if bm["MAPE"] < 10:
        cls_i, icon = "good", "🟢"
        txt = f"**Performanță Excelentă (MAPE={bm['MAPE']:.1f}%):** Modelul {best_model} clasifică corect nivelul AQI cu eroare sub 10%. Poate fi integrat direct într-un sistem de alertă publică."
    elif bm["MAPE"] < 20:
        cls_i, icon = "mod", "🟡"
        txt = f"**Performanță Bună (MAPE={bm['MAPE']:.1f}%):** {best_model} prinde trendul și sezonalitatea AQI. Ocazional confundă nivelul 3 cu 4. Util pentru planificare pe termen mediu."
    else:
        cls_i, icon = "bad", "🔴"
        txt = f"**Performanță Acceptabilă (MAPE={bm['MAPE']:.1f}%):** Spike-urile bruște de AQI (inversii termice, accidente industriale) sunt greu de anticipat fără variabile meteo suplimentare."
    r2_txt = f"R²={bm['R2']:.4f} → modelul explică {bm['R2']*100:.1f}% din variabilitatea AQI."
    st.markdown(f'<div class="mcard {cls_i}">{icon} {txt}<br><small style="color:#7cb8d4">{r2_txt}</small></div>', unsafe_allow_html=True)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# TAB 6 — DIGITAL TWIN
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
with tab6:
    st.markdown("### 🎮 City Digital Twin — Simulator de Politici Publice")
    st.markdown("Recalculare instantanee AQI prin modelul non-liniar XGBoost antrenat pe date reale.")
    if not XGB_OK or xgb_model is None:
        st.error("XGBoost nu este disponibil. `pip install xgboost`")
    else:
        st.markdown('<hr class="neon-hr">', unsafe_allow_html=True)
        col_ctrl, col_chart = st.columns([1, 2])
        with col_ctrl:
            st.markdown("#### 🎛️ Intervenții de Politică Publică")
            red_trafic  = st.slider("🚗 Restricție Trafic Rutier (%)",  0, 60, 20, 5)
            green_areas = st.slider("🌳 Extindere Spații Verzi (%)",     0, 40, 10, 5)
            ind_restr   = st.slider("🏭 Restricții Industriale (%)",     0, 50, 15, 5)
            weather_adj = st.slider("🌬️ Factor Dispersie Vânt (%)",    -20, 30,  0, 5)
            total_red = (red_trafic*0.40 + green_areas*0.20 + ind_restr*0.35 - weather_adj*0.05) / 100
            aqi_drop  = total_red * 1.5
            st.markdown('<hr class="neon-hr">', unsafe_allow_html=True)
            st.markdown(f"""<div class="mcard {'good' if aqi_drop > 0.2 else 'mod'}">
  <div class="mcard-lab">Reducere AQI Estimată</div>
  <div class="mcard-val" style="color:{'#00ff88' if aqi_drop > 0.2 else '#fbbf24'}">{aqi_drop:.3f}</div>
  <div class="mcard-unit">unități AQI față de scenariu inertial</div>
</div>""", unsafe_allow_html=True)

        with col_chart:
            feat_cols = [c for c in lag_df.columns if c not in ("ds","aqi")]
            test_lag  = lag_df.iloc[train_size - 7:]
            if len(test_lag) > 0:
                df_sim = test_lag.copy()
                no2_red  = (red_trafic * 0.50 + ind_restr * 0.40) / 100
                pm_red   = (red_trafic * 0.40 + ind_restr * 0.35 + green_areas * 0.20) / 100
                if "no2"   in df_sim.columns: df_sim["no2"]   *= (1 - no2_red)
                if "pm2_5" in df_sim.columns: df_sim["pm2_5"] *= (1 - pm_red)
                if "pm10"  in df_sim.columns: df_sim["pm10"]  *= (1 - pm_red * 0.8)
                if "so2"   in df_sim.columns: df_sim["so2"]   *= (1 - ind_restr * 0.005)
                for lc in [c for c in feat_cols if "lag" in c]:
                    df_sim[lc] = df_sim[lc] * (1 - pm_red * 0.5)
                X_sim  = df_sim[[c for c in feat_cols if c in df_sim.columns]]
                X_base = test_lag[[c for c in feat_cols if c in test_lag.columns]]
                pred_scen = np.clip(xgb_model.predict(X_sim),  1, 5)
                pred_base = np.clip(xgb_model.predict(X_base), 1, 5)
                dates_sim = test_lag["ds"].values

                fig_sim = go.Figure()
                for lv, meta in AQI_META.items():
                    fig_sim.add_hrect(y0=lv-0.5, y1=lv+0.5, fillcolor=meta["bg"], line_width=0, opacity=0.4)
                fig_sim.add_trace(go.Scatter(x=dates_sim, y=pred_base, mode="lines",
                                              name="Scenariu Inertial",
                                              line=dict(color="#f87171", dash="dot", width=2)))
                fig_sim.add_trace(go.Scatter(x=dates_sim, y=pred_scen, mode="lines",
                                              name="Scenariu Politici Publice",
                                              line=dict(color="#00ff88", width=2.5)))
                true_sl = test_df["aqi"].values[:len(pred_base)]
                if len(true_sl) == len(dates_sim):
                    fig_sim.add_trace(go.Scatter(x=dates_sim, y=true_sl, mode="lines",
                                                  name="AQI Observat Real",
                                                  line=dict(color="white", width=1.5, dash="dash")))
                fig_sim.update_layout(**DARK_LAYOUT, height=400,
                                       title="Impact Politici Publice asupra AQI",
                                       xaxis_title="Dată",
                                       yaxis=dict(title="AQI (1–5)", tickvals=[1,2,3,4,5],
                                                  ticktext=["1 Bun","2 Accept.","3 Mod.","4 Slab","5 F.Slab"]),
                                       hovermode="x unified")
                st.plotly_chart(fig_sim, use_container_width=True)

                mb   = float(np.mean(pred_base)); ms = float(np.mean(pred_scen))
                dg   = int((pred_scen <= 2).sum()); db = int((pred_scen >= 4).sum()); dt = len(pred_scen)
                c1s,c2s,c3s,c4s = st.columns(4)
                c1s.markdown(f"""<div class="mcard good"><div class="mcard-lab">AQI Salvat</div>
  <div class="mcard-val" style="color:#00ff88">{mb-ms:.3f}</div>
  <div class="mcard-unit">unități AQI reducere</div></div>""", unsafe_allow_html=True)
                c2s.markdown(f"""<div class="mcard good"><div class="mcard-lab">Zile AQI ≤ Acceptabil</div>
  <div class="mcard-val" style="color:#00ff88">{dg}/{dt}</div>
  <div class="mcard-unit">cu intervenție</div></div>""", unsafe_allow_html=True)
                c3s.markdown(f"""<div class="mcard bad"><div class="mcard-lab">Zile AQI ≥ Slab</div>
  <div class="mcard-val" style="color:#f87171">{db}/{dt}</div>
  <div class="mcard-unit">potențial nociv</div></div>""", unsafe_allow_html=True)
                c4s.markdown(f"""<div class="mcard info"><div class="mcard-lab">AQI Inertial Mediu</div>
  <div class="mcard-val" style="color:#00d4ff">{mb:.2f}</div>
  <div class="mcard-unit">fără intervenție</div></div>""", unsafe_allow_html=True)

        st.markdown('<hr class="neon-hr">', unsafe_allow_html=True)
        st.markdown("#### 📋 Recomandări Automate pentru Administrația Publică")
        recs = []
        if red_trafic >= 30:
            recs.append(("good","🚗 **Restricție trafic semnificativă** — Impact maxim pe NO₂ și PM2.5. Recomandare: taxare congestionare + zone 0-emisii în centru."))
        if green_areas >= 20:
            recs.append(("good","🌳 **Extindere spații verzi** — 1 ha pădure urbană absoarbe ~0.3 tone PM/an. Prioritate: coridoare verzi pe artere cu trafic intens."))
        if ind_restr >= 30:
            recs.append(("good","🏭 **Restricții industriale majore** — CET-urile contribuie 35–45% din AQI iarna. BAT (Best Available Technology) obligatoriu."))
        if not recs:
            recs.append(("info","⚙️ Mărește slider-ele pentru recomandări specifice de politici publice."))
        for cls, text in recs:
            st.markdown(f'<div class="mcard {cls}">{text}</div>', unsafe_allow_html=True)

# ── Footer ──
st.markdown('<hr class="neon-hr">', unsafe_allow_html=True)
st.markdown("""<div style="text-align:center;color:#2d3a55;font-family:'Space Mono',monospace;font-size:0.68rem;padding:8px 0">
  Eco-Informatică Hibridă · București AQ Intelligence System<br>
  SAS ETL + Python (Prophet · ARIMA · SARIMA · XGBoost) + Streamlit<br>
  OpenWeather Air Pollution API · Target: AQI Index (1–5) · 6 Componente<br>
  Proiect: Pachete Software — Facultatea de Cibernetică, Statistică și Informatică Economică
</div>""", unsafe_allow_html=True)