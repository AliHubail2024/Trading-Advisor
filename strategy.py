import numpy as np
import pandas as pd
from typing import Optional, Tuple, List

from settings import Settings, RiskSettings
from indicators import sma, ema, rsi, atr, bollinger_bands, donchian, slope

def heikin_ashi(df: pd.DataFrame) -> pd.DataFrame:
    ha = pd.DataFrame(index=df.index, columns=["open", "high", "low", "close"], dtype=float)
    ha["close"] = (df["open"] + df["high"] + df["low"] + df["close"]) / 4.0
    ha["open"] = 0.0
    if len(df) > 0:
        ha.iloc[0, ha.columns.get_loc("open")] = (df["open"].iloc[0] + df["close"].iloc[0]) / 2.0
        for i in range(1, len(df)):
            ha.iloc[i, ha.columns.get_loc("open")] = (ha["open"].iloc[i - 1] + ha["close"].iloc[i - 1]) / 2.0
    ha["high"] = np.vstack([df["high"], ha["open"], ha["close"]]).max(axis=0)
    ha["low"] = np.vstack([df["low"], ha["open"], ha["close"]]).min(axis=0)
    return ha.rename(columns={"open": "ha_open", "high": "ha_high", "low": "ha_low", "close": "ha_close"})

def swing_low(df: pd.DataFrame, lookback: int = 10) -> pd.Series:
    return df["low"].rolling(lookback, min_periods=lookback).min()

def swing_high(df: pd.DataFrame, lookback: int = 10) -> pd.Series:
    return df["high"].rolling(lookback, min_periods=lookback).max()

def compute_indicators(df: pd.DataFrame, setts: Settings) -> pd.DataFrame:
    df = df.copy()
    for col in ["open", "high", "low", "close"]:
        if col not in df.columns:
            return pd.DataFrame()

    df["SMA200"] = sma(df["close"], setts.sma_trend)
    df["EMA20"] = ema(df["close"], setts.ema_fast)
    df["EMA50"] = ema(df["close"], setts.ema_slow)
    df["RSI"] = rsi(df["close"], 14)
    df["ATR"] = atr(df["high"], df["low"], df["close"], setts.atr_len)

    bb_u, bb_m, bb_l = bollinger_bands(df["close"], setts.bb_length, setts.bb_mult)
    df["BB_Upper"], df["BB_Middle"], df["BB_Lower"] = bb_u, bb_m, bb_l
    df["BB_Width"] = (bb_u - bb_l) / bb_m.replace(0, np.nan)

    dcu, dcm, dcl = donchian(df["high"], df["low"], setts.donchian_len)
    df["DC_Upper"], df["DC_Mid"], df["DC_Lower"] = dcu, dcm, dcl

    df["ADX"] = adx(df["high"], df["low"], df["close"], 14)
    df["Vol_MA"] = sma(df["volume"], setts.vol_ma_len) if "volume" in df.columns else np.nan

    df["SMA200_Slope"] = slope(df["SMA200"], lookback=5)
    return df

def adx(high: pd.Series, low: pd.Series, close: pd.Series, length: int = 14) -> pd.Series:
    up_move = high.diff()
    down_move = -low.diff()
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    plus_dm = pd.Series(plus_dm, index=high.index)
    minus_dm = pd.Series(minus_dm, index=high.index)

    tr = (pd.concat([
        (high - low).abs(),
        (high - close.shift(1)).abs(),
        (low - close.shift(1)).abs()
    ], axis=1).max(axis=1))

    tr_n = tr.ewm(alpha=1 / length, adjust=False, min_periods=length).mean()
    plus_di = 100 * (plus_dm.ewm(alpha=1 / length, adjust=False, min_periods=length).mean() / tr_n)
    minus_di = 100 * (minus_dm.ewm(alpha=1 / length, adjust=False, min_periods=length).mean() / tr_n)
    dx = 100 * ((plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan))
    adx_val = dx.ewm(alpha=1 / length, adjust=False, min_periods=length).mean()
    return adx_val

def compute_anchor_bias(anchor_df: pd.DataFrame, setts: Settings) -> str:
    valid = anchor_df.dropna()
    if valid.empty:
        return "neutral"
    last = valid.iloc[-1]
    if pd.isna(last.get("SMA200", np.nan)):
        return "neutral"
    slope_val = float(last.get("SMA200_Slope", 0.0))
    if last["close"] > last["SMA200"] and slope_val > 0:
        return "long"
    if last["close"] < last["SMA200"] and slope_val < 0:
        return "short"
    return "neutral"

def generate_signals(df: pd.DataFrame, bias: str, setts: Settings) -> pd.DataFrame:
    df = df.copy()
    if df.empty:
        return df

    df["Trending"] = df["ADX"] > setts.adx_thresh
    df["Ranging"] = df["ADX"] < setts.adx_thresh

    df["CrossUp_EMA20"] = (df["close"] > df["EMA20"]) & (df["close"].shift(1) <= df["EMA20"].shift(1))
    df["CrossDown_EMA20"] = (df["close"] < df["EMA20"]) & (df["close"].shift(1) >= df["EMA20"].shift(1))

    df["Pulled_to_EMA50"] = (df["low"] <= df["EMA50"]) | (df["close"] <= df["EMA20"])

    df["RSI_Bull_Range"] = df["RSI"] > setts.rsi_bull_floor
    df["RSI_Bear_Range"] = df["RSI"] < setts.rsi_bear_ceiling

    df["Vol_Exp"] = False
    if "volume" in df.columns and "Vol_MA" in df.columns:
        df["Vol_Exp"] = df["volume"] > (setts.breakout_vol_mult * df["Vol_MA"].fillna(0))

    df["Breakout_Up"] = (df["close"] > df["DC_Upper"]) & (df["close"].shift(1) <= df["DC_Upper"].shift(1))
    df["Breakout_Down"] = (df["close"] < df["DC_Lower"]) & (df["close"].shift(1) >= df["DC_Lower"].shift(1))

    df["At_BB_Lower"] = df["close"] <= df["BB_Lower"]
    df["At_BB_Upper"] = df["close"] >= df["BB_Upper"]

    cond_trend_long = (
        (bias in ["long", "neutral"]) &
        df["Trending"] &
        (df["close"] > df["SMA200"]) &
        (df["SMA200_Slope"] > 0) &
        df["Pulled_to_EMA50"] &
        df["CrossUp_EMA20"] &
        df["RSI_Bull_Range"]
    )

    cond_trend_short = (
        (bias in ["short", "neutral"]) &
        df["Trending"] &
        (df["close"] < df["SMA200"]) &
        (df["SMA200_Slope"] < 0) &
        df["Pulled_to_EMA50"] &
        df["CrossDown_EMA20"] &
        df["RSI_Bear_Range"]
    )

    cond_breakout_long = (
        (bias in ["long", "neutral"]) &
        df["Breakout_Up"] &
        (df["ADX"] > (setts.adx_thresh - 2)) &
        (df["ADX"] > df["ADX"].shift(1)) &
        (df["Vol_Exp"] | df["Trending"])
    )

    cond_breakout_short = (
        (bias in ["short", "neutral"]) &
        df["Breakout_Down"] &
        (df["ADX"] > (setts.adx_thresh - 2)) &
        (df["ADX"] > df["ADX"].shift(1)) &
        (df["Vol_Exp"] | df["Trending"])
    )

    cond_range_long = (
        df["Ranging"] &
        df["At_BB_Lower"] &
        (df["RSI"] < 35) &
        (df["close"] > df["close"].shift(1))
    )

    cond_range_short = (
        df["Ranging"] &
        df["At_BB_Upper"] &
        (df["RSI"] > 65) &
        (df["close"] < df["close"].shift(1))
    )

    df["Long_Signal"] = cond_trend_long | cond_breakout_long | cond_range_long
    df["Short_Signal"] = cond_trend_short | cond_breakout_short | cond_range_short

    reasons: List[str] = []
    for i in range(len(df)):
        r = []
        if df["Long_Signal"].iloc[i]:
            if cond_trend_long.iloc[i]: r.append("Trend pullback long")
            if cond_breakout_long.iloc[i]: r.append("Breakout long")
            if cond_range_long.iloc[i]: r.append("Range bounce long")
        if df["Short_Signal"].iloc[i]:
            if cond_trend_short.iloc[i]: r.append("Trend rally short")
            if cond_breakout_short.iloc[i]: r.append("Breakdown short")
            if cond_range_short.iloc[i]: r.append("Range sell short")
        reasons.append(", ".join(r))
    df["Signal_Reason"] = reasons

    return df

def compute_stops_targets(df: pd.DataFrame, risk: RiskSettings, setts: Settings) -> pd.DataFrame:
    df = df.copy()
    if df.empty:
        return df

    recent_sw_low = swing_low(df, setts.swing_lookback)
    recent_sw_high = swing_high(df, setts.swing_lookback)

    entry = df["close"]
    long_stop_structure = recent_sw_low
    short_stop_structure = recent_sw_high

    long_stop_atr = entry - risk.atr_mult * df["ATR"]
    short_stop_atr = entry + risk.atr_mult * df["ATR"]

    use_structure = (risk.stop_method == "Structure")
    df["Long_Stop"] = np.where(use_structure, long_stop_structure, long_stop_atr)
    df["Short_Stop"] = np.where(use_structure, short_stop_structure, short_stop_atr)

    long_risk = (entry - df["Long_Stop"]).clip(lower=1e-12)
    short_risk = (df["Short_Stop"] - entry).clip(lower=1e-12)

    df["Long_TP1"] = entry + risk.tp1_r * long_risk
    df["Long_TP2"] = entry + risk.tp2_r * long_risk
    df["Short_TP1"] = entry - risk.tp1_r * short_risk
    df["Short_TP2"] = entry - risk.tp2_r * short_risk

    return df

def position_size(entry: float, stop: float, equity: float, risk_pct: float) -> Tuple[float, float]:
    if entry is None or stop is None or np.isnan(entry) or np.isnan(stop):
        return 0.0, 0.0
    risk_amount = equity * (risk_pct / 100.0)
    stop_dist = abs(entry - stop)
    if stop_dist <= 0:
        return 0.0, 0.0
    qty = risk_amount / stop_dist
    return float(qty), float(risk_amount)

def latest_signal(df_st: pd.DataFrame) -> Tuple[Optional[pd.Timestamp], Optional[str]]:
    if df_st is None or df_st.empty:
        return None, None
    last_has_long = bool(df_st["Long_Signal"].iloc[-1]) if "Long_Signal" in df_st else False
    last_has_short = bool(df_st["Short_Signal"].iloc[-1]) if "Short_Signal" in df_st else False
    if last_has_long:
        return df_st.index[-1], "long"
    if last_has_short:
        return df_st.index[-1], "short"
    longs = df_st.index[df_st.get("Long_Signal", pd.Series(False, index=df_st.index))].tolist()
    shorts = df_st.index[df_st.get("Short_Signal", pd.Series(False, index=df_st.index))].tolist()
    last_long = longs[-1] if len(longs) else None
    last_short = shorts[-1] if len(shorts) else None
    if last_long and last_short:
        idx = max(last_long, last_short)
        side = "long" if idx == last_long else "short"
        return idx, side
    if last_long:
        return last_long, "long"
    if last_short:
        return last_short, "short"
    return None, None