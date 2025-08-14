#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# --- Make both project root and ./src discoverable for imports ---
import os, sys
APP_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(APP_DIR)
for p in (APP_DIR, ROOT_DIR):
    if p not in sys.path:
        sys.path.insert(0, p)

import importlib
import inspect
import platform
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="BTC Long/Short Playbook (Not Finincal Advice only for education)", layout="wide", page_icon="📈")
st.title("📈 BTC Long/Short Playbook  (Not Finincal Advice only for education)")

# ---------- Helper: import with fallback between flat and src.* ----------
def import_with_fallback(name_plain: str, name_pkg: str):
    try:
        return importlib.import_module(name_plain)
    except ModuleNotFoundError:
        return importlib.import_module(name_pkg)

# Try to import modules (show errors on-page instead of blank screen)
try:
    settings_mod = import_with_fallback("settings", "src.settings")
    data_mod = import_with_fallback("data", "src.data")
    indicators_mod = import_with_fallback("indicators", "src.indicators")
    strategy_mod = import_with_fallback("strategy", "src.strategy")
    plotter_mod = import_with_fallback("plotter", "src.plotter")
except Exception as e:
    st.error("Import error while loading modules:")
    st.exception(e)
    st.stop()

# Pull the symbols we need from modules (no matter their package path)
try:
    Settings = getattr(settings_mod, "Settings")
    RiskSettings = getattr(settings_mod, "RiskSettings")
    load_data = getattr(data_mod, "load_data")
    prepare_anchor = getattr(data_mod, "prepare_anchor")
    normalize_ohlcv = getattr(data_mod, "normalize_ohlcv")
    compute_indicators = getattr(strategy_mod, "compute_indicators")
    compute_anchor_bias = getattr(strategy_mod, "compute_anchor_bias")
    generate_signals = getattr(strategy_mod, "generate_signals")
    compute_stops_targets = getattr(strategy_mod, "compute_stops_targets")
    latest_signal = getattr(strategy_mod, "latest_signal")
    build_figure = getattr(plotter_mod, "build_figure")
except Exception as e:
    st.error("Your modules are missing expected functions/classes:")
    st.exception(e)
    with st.expander("Loaded module files", expanded=True):
        def fpath(m):
            try:
                return inspect.getfile(m)
            except Exception:
                return "<unknown>"
        st.json({
            "settings": fpath(settings_mod),
            "data": fpath(data_mod),
            "indicators": fpath(indicators_mod),
            "strategy": fpath(strategy_mod),
            "plotter": fpath(plotter_mod),
        })
    st.stop()

# ---------- Preset symbols ----------
YAHOO_PRESETS = {
    "Crypto": ["BTC-USD", "ETH-USD", "SOL-USD", "BNB-USD", "XRP-USD", "DOGE-USD", "ADA-USD", "LINK-USD"],
    "Stocks": ["AAPL", "MSFT", "NVDA", "AMZN", "META", "GOOGL", "TSLA", "AMD"],
    "Commodities/ETFs": ["GLD", "SLV", "USO", "UNG", "UUP", "DXY", "SPY", "QQQ"],
}
BINANCE_PRESETS = {
    "Crypto": ["BTC/USDT", "ETH/USDT", "SOL/USDT", "BNB/USDT", "XRP/USDT", "DOGE/USDT", "ADA/USDT", "LINK/USDT"],
}

# ---------- Sidebar controls ----------
with st.sidebar:
    st.header("Data")
    source_choice = st.selectbox(
        "Source",
        ["Yahoo Finance", "Binance Perps (USDT-M)", "Binance Spot"],
        index=1,
    )

    if source_choice == "Yahoo Finance":
        asset_class = st.selectbox("Asset class", list(YAHOO_PRESETS.keys()), index=0)
        default_list = YAHOO_PRESETS[asset_class]
        preset_symbol = st.selectbox("Preset symbol", default_list, index=0)
        custom_symbol = st.text_input("Or enter custom Yahoo symbol (e.g., TSLA, BTC-USD)", "")
        selected_symbol = custom_symbol.strip() or preset_symbol
        data_source = f"Yahoo Finance ({selected_symbol})"

    else:
        # Binance sources: only Crypto symbols make sense
        asset_class = "Crypto"
        default_list = BINANCE_PRESETS[asset_class]
        preset_symbol = st.selectbox("Preset symbol", default_list, index=0)
        custom_symbol = st.text_input("Or enter custom Binance symbol (e.g., BTC/USDT, ETH/USDT)", "")
        selected_symbol = custom_symbol.strip().upper() or preset_symbol
        if source_choice.startswith("Binance Perps"):
            data_source = "Binance Perps (BTC/USDT)"
        else:
            data_source = "Binance Spot (BTC/USDT)"

    entry_tf = st.selectbox("Entry timeframe", ["1m", "5m", "15m", "1h", "4h", "1d"], index=3)
    anchor_tf = st.selectbox("Anchor (higher) timeframe", ["15m", "1h", "4h", "1d", "1wk"], index=3)
    lookback_days = st.slider("Lookback days", 3, 365, 120, 1)
    limit_ccxt = st.slider("Max bars (Binance only)", 200, 5000, 1500, 100)

    st.header("Strategy")
    strategy_choice = st.selectbox(
        "Strategy",
        ["Baseline", "Scalping (EMA9/21 + VWAP + RSI filter)"],
        index=0,
        help="Scalping works best on 1m–5m charts. Uses EMA9/21 cross with VWAP & RSI filters.",
    )

    st.header("Indicators & Thresholds")
    setts = Settings(
        adx_thresh=st.slider("ADX trend threshold", 10.0, 40.0, 20.0, 1.0),
        rsi_bull_floor=st.slider("RSI bull-floor", 35.0, 60.0, 45.0, 1.0),
        rsi_bear_ceiling=st.slider("RSI bear-ceiling", 40.0, 70.0, 55.0, 1.0),
        bb_length=st.number_input("Bollinger length", 10, 60, 20, 1),
        bb_mult=st.slider("Bollinger mult", 1.0, 3.5, 2.0, 0.1),
        donchian_len=st.number_input("Donchian length", 10, 60, 20, 1),
        ema_fast=st.number_input("EMA fast", 5, 50, 20, 1),
        ema_slow=st.number_input("EMA slow", 20, 200, 50, 1),
        sma_trend=st.number_input("SMA trend (HTF)", 50, 400, 200, 5),
        atr_len=st.number_input("ATR length", 5, 50, 14, 1),
        swing_lookback=st.number_input("Swing lookback (structure)", 5, 30, 10, 1),
        vol_ma_len=st.number_input("Volume MA length", 5, 50, 20, 1),
        breakout_vol_mult=st.slider("Breakout volume multiple", 1.0, 3.0, 1.2, 0.1),
    )

    st.header("Risk & Targets")
    # Suggest tighter defaults for scalping but keep fully editable
    default_risk_pct = 0.35 if strategy_choice.startswith("Scalping") else 0.75
    default_atr_mult = 0.8 if strategy_choice.startswith("Scalping") else 1.0
    risk = RiskSettings(
        account_equity=st.number_input("Account equity ($)", 100.0, 10_000_000.0, 10_000.0, 100.0),
        risk_pct=st.slider("Risk per trade (%)", 0.05, 2.0, float(default_risk_pct), 0.05),
        stop_method=st.selectbox("Stop method", ["Structure", "ATR x"], index=1 if strategy_choice.startswith("Scalping") else 0),
        atr_mult=st.slider("ATR multiple (if ATR stop)", 0.3, 3.0, float(default_atr_mult), 0.1),
        tp1_r=st.slider("TP1 (R multiple)", 0.5, 3.0, 1.2 if strategy_choice.startswith("Scalping") else 1.5, 0.1),
        tp2_r=st.slider("TP2 (R multiple)", 0.5, 5.0, 1.8 if strategy_choice.startswith("Scalping") else 2.0, 0.1),
    )

    st.header("Chart Options")
    use_heikin = st.checkbox("Use Heikin Ashi candles (visual only)", False)
    log_scale = st.checkbox("Log scale", False)
    max_bars_display = st.slider("Bars to display", 200, 3000, 800, 50)

    st.header("Overlays & Panels")
    show_ma = st.checkbox("Show SMA200", True)
    show_ema = st.checkbox("Show EMA20/50", True)
    show_bb = st.checkbox("Show Bollinger Bands", False)
    show_dc = st.checkbox("Show Donchian Channels", False)
    show_volume = st.checkbox("Show Volume", True)
    show_rsi = st.checkbox("Show RSI", True)
    show_adx = st.checkbox("Show ADX", True)
    show_atr = st.checkbox("Show ATR", False)
    show_signals_long = st.checkbox("Show Long signals", True)
    show_signals_short = st.checkbox("Show Short signals", True)
    show_stops_tps = st.checkbox("Show Stop/TP lines for latest signal", True)

    st.header("Debug & Cache")
    debug_mode = st.checkbox("Debug mode", True)
    c1, c2 = st.columns(2)
    with c1:
        if st.button("Reload"):
            st.rerun()
    with c2:
        if st.button("Clear cache & reload"):
            try:
                st.cache_data.clear()
            except Exception:
                pass
            st.rerun()

# ---------- Optional: environment + module info for quick troubleshooting ----------
# with st.expander("Environment", expanded=False):
#     st.write({
#         "python": sys.version,
#         "platform": platform.platform(),
#         "cwd": os.getcwd(),
#         "__file__": os.path.abspath(__file__),
#     })
# with st.expander("Loaded module files", expanded=False):
#     def fpath(m):
#         try:
#             return inspect.getfile(m)
#         except Exception:
#             return "<unknown>"
#     st.json({
#         "settings": fpath(settings_mod),
#         "data": fpath(data_mod),
#         "indicators": fpath(indicators_mod),
#         "strategy": fpath(strategy_mod),
#         "plotter": fpath(plotter_mod),
#     })

# ---------- Helpers for Scalping (local fallback) ----------
def _ensure_scalping_indicators_local(df: pd.DataFrame) -> pd.DataFrame:
    """Add EMA9/21/200, RSI14, VWAP (daily reset). Safe if columns already exist."""
    out = df.copy()
    c = out["close"].astype(float)
    v = out["volume"] if "volume" in out.columns else pd.Series(0, index=out.index, dtype=float)

    # EMAs
    if "EMA9" not in out.columns:
        out["EMA9"] = c.ewm(span=9, adjust=False).mean()
    if "EMA21" not in out.columns:
        out["EMA21"] = c.ewm(span=21, adjust=False).mean()
    if "EMA200" not in out.columns:
        out["EMA200"] = c.ewm(span=200, adjust=False).mean()

    # RSI(14)
    if "RSI" not in out.columns:
        delta = c.diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)
        roll_up = gain.ewm(alpha=1/14, adjust=False).mean()
        roll_down = loss.ewm(alpha=1/14, adjust=False).mean()
        rs = roll_up / (roll_down.replace(0, np.nan))
        out["RSI"] = 100 - (100 / (1 + rs))

    # VWAP (session reset daily); if no volume, will return NaN
    if "VWAP" not in out.columns:
        try:
            tp = (out["high"] + out["low"] + out["close"]) / 3.0
            day = out.index.normalize()
            cum_vol = v.groupby(day).cumsum()
            cum_pv = (tp * v).groupby(day).cumsum()
            out["VWAP"] = (cum_pv / cum_vol).replace([np.inf, -np.inf], np.nan).ffill()
        except Exception:
            out["VWAP"] = np.nan
    return out

def _generate_signals_scalping_local(df: pd.DataFrame, bias_value, setts) -> pd.DataFrame:
    """EMA9/21 cross with VWAP + trend filter. Returns a copy with 'Signal' column."""
    d = _ensure_scalping_indicators_local(df)
    ema9, ema21, ema200 = d["EMA9"], d["EMA21"], d["EMA200"]
    close = d["close"]
    vwap = d.get("VWAP", pd.Series(np.nan, index=d.index))
    rsi = d.get("RSI", pd.Series(np.nan, index=d.index))

    # Cross conditions
    cross_up = (ema9 > ema21) & (ema9.shift(1) <= ema21.shift(1))
    cross_dn = (ema9 < ema21) & (ema9.shift(1) >= ema21.shift(1))

    # Trend filters; if VWAP missing, ignore it
    vwap_ok_long = (close > vwap) | vwap.isna()
    vwap_ok_short = (close < vwap) | vwap.isna()
    trend_long = (close > ema200) & vwap_ok_long
    trend_short = (close < ema200) & vwap_ok_short

    # RSI sanity (avoid extremes)
    rsi_ok_long = (rsi > 45) & (rsi < 70) | rsi.isna()
    rsi_ok_short = (rsi < 55) & (rsi > 30) | rsi.isna()

    # Anchor bias filter if provided as string
    def bias_allows(side: str) -> pd.Series:
        b = str(bias_value).lower()
        if "bull" in b:
            return pd.Series(True if side == "long" else False, index=d.index)
        if "bear" in b:
            return pd.Series(True if side == "short" else False, index=d.index)
        return pd.Series(True, index=d.index)

    allow_long = bias_allows("long")
    allow_short = bias_allows("short")

    long_cond = cross_up & trend_long & rsi_ok_long & allow_long
    short_cond = cross_dn & trend_short & rsi_ok_short & allow_short

    out = d.copy()
    out["Signal"] = ""
    out.loc[long_cond, "Signal"] = "LONG"
    out.loc[short_cond, "Signal"] = "SHORT"
    return out

# ---------- Main pipeline with on-page error reporting ----------
try:
    with st.status("Loading data...", expanded=debug_mode) as status:
        # Try to pass selected_symbol if your load_data supports it; fallback otherwise.
        try:
            df_entry = load_data(data_source, entry_tf, lookback_days, limit_ccxt, symbol=selected_symbol)
        except TypeError:
            df_entry = load_data(data_source, entry_tf, lookback_days, limit_ccxt)
        df_entry = normalize_ohlcv(df_entry)
        status.update(label=f"Loaded {len(df_entry)} rows for {selected_symbol}", state="complete")

    if df_entry is None or df_entry.empty:
        st.error("Loader returned 0 rows. Try different source/timeframe or clear cache.")
        with st.expander("Diagnostics", expanded=True):
            st.write({
                "source": source_choice, "data_source": data_source,
                "symbol": selected_symbol, "entry_tf": entry_tf,
                "lookback_days": lookback_days, "limit_ccxt": limit_ccxt
            })
        st.stop()

    # Prepare higher timeframe anchor
    df_anchor = prepare_anchor(df_entry, anchor_tf, entry_tf=entry_tf)

    # Compute indicators on entry and anchor
    df_e = compute_indicators(df_entry, setts)
    df_a = compute_indicators(df_anchor, setts)
    bias = compute_anchor_bias(df_a, setts)

    # Trim for display
    df_e = df_e.iloc[-max_bars_display:].copy()

    # Signals and risk levels (choose strategy)
    if strategy_choice.startswith("Scalping"):
        # Use strategy module if available; else local fallback
        if hasattr(strategy_mod, "generate_signals_scalping"):
            df_sig = strategy_mod.generate_signals_scalping(df_e, bias, setts)
        else:
            df_sig = _generate_signals_scalping_local(df_e, bias, setts)
    else:
        df_sig = generate_signals(df_e, bias, setts)

    df_st = compute_stops_targets(df_sig, risk, setts)
    latest_idx, latest_side = latest_signal(df_st)

    # Header context (bias, regime, ADX)
    last_row = df_sig.iloc[-1] if not df_sig.empty else pd.Series(dtype=float)
    try:
        adx_val = float(last_row.get("ADX", float("nan")))
        regime = "Trending" if bool(last_row.get("Trending", False)) else "Ranging"
    except Exception:
        adx_val, regime = float("nan"), "Unknown"
    bias_text = f"Symbol: {selected_symbol} • Higher TF bias: {str(bias).upper()} • Regime: {regime} • ADX: {adx_val:.1f}"
    st.subheader(bias_text)

    # Build and show chart
    fig, latest_info_md = build_figure(
        df_e=df_e, df_st=df_st, risk=risk, setts=setts,
        latest_idx=latest_idx, latest_side=latest_side,
        options=dict(
            use_heikin=use_heikin, log_scale=log_scale,
            show_ma=show_ma, show_ema=show_ema, show_bb=show_bb, show_dc=show_dc,
            show_volume=show_volume, show_rsi=show_rsi, show_adx=show_adx, show_atr=show_atr,
            show_signals_long=show_signals_long, show_signals_short=show_signals_short,
            show_stops_tps=show_stops_tps,
        ),
    )

    if latest_info_md:
        st.markdown(latest_info_md)
    st.plotly_chart(fig, use_container_width=True, theme="streamlit")

    # with st.expander("Diagnostics", expanded=debug_mode):
    #     st.write("Entry rows:", len(df_entry), " • Anchor rows:", len(df_anchor))
    #     try:
    #         st.write("Entry date range:", df_entry.index.min(), "→", df_entry.index.max())
    #     except Exception:
    #         pass
    #     st.write("Columns:", list(df_entry.columns))
    #     st.write({"Selected source": source_choice, "Selected symbol": selected_symbol})

except Exception as e:
    st.error("Unhandled exception during app execution:")
    st.exception(e)