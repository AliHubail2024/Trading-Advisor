import numpy as np
import pandas as pd
from typing import Dict, Optional
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from settings import RiskSettings, Settings
from strategy import heikin_ashi, position_size

def build_figure(
    df_e: pd.DataFrame,
    df_st: pd.DataFrame,
    risk: RiskSettings,
    setts: Settings,
    latest_idx: Optional[pd.Timestamp],
    latest_side: Optional[str],
    options: Dict,
):
    # Defaults for options
    use_heikin = bool(options.get("use_heikin", False))
    log_scale = bool(options.get("log_scale", False))
    show_ma = bool(options.get("show_ma", True))
    show_ema = bool(options.get("show_ema", True))
    show_bb = bool(options.get("show_bb", False))
    show_dc = bool(options.get("show_dc", False))
    show_volume = bool(options.get("show_volume", True))
    show_rsi = bool(options.get("show_rsi", True))
    show_adx = bool(options.get("show_adx", True))
    show_atr = bool(options.get("show_atr", False))
    show_signals_long = bool(options.get("show_signals_long", True))
    show_signals_short = bool(options.get("show_signals_short", True))
    show_stops_tps = bool(options.get("show_stops_tps", True))

    rows = 3
    fig = make_subplots(
        rows=rows,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.02,
        specs=[[{"secondary_y": False}], [{"secondary_y": False}], [{"secondary_y": True}]],
        row_heights=[0.62, 0.18, 0.20],
    )

    # Prepare plot dataframe
    plot_df = df_e[["open", "high", "low", "close", "volume"]].copy()
    if use_heikin:
        ha = heikin_ashi(df_e)
        plot_df["open"], plot_df["high"], plot_df["low"], plot_df["close"] = ha["ha_open"], ha["ha_high"], ha["ha_low"], ha["ha_close"]

    plot_df = plot_df.dropna(subset=["open", "high", "low", "close"])

    # Guard log scale with non-positive values
    if log_scale and not plot_df.empty:
        if (plot_df[["open","high","low","close"]] <= 0).any().any():
            log_scale = False  # auto-disable to show data

    if not plot_df.empty:
        fig.add_trace(
            go.Candlestick(
                x=plot_df.index,
                open=plot_df["open"],
                high=plot_df["high"],
                low=plot_df["low"],
                close=plot_df["close"],
                name="Price",
                increasing_line_color="#2ECC71",
                decreasing_line_color="#E74C3C",
                showlegend=True,
                opacity=0.9,
            ),
            row=1, col=1
        )
    else:
        fig.add_annotation(text="No plottable OHLC rows", showarrow=False, xref="paper", yref="paper", x=0.02, y=0.95)

    # Overlays
    if show_ma and "SMA200" in df_e:
        fig.add_trace(go.Scatter(x=df_e.index, y=df_e["SMA200"], mode="lines", name="SMA200",
                                 line=dict(color="#1F77B4", width=1.5)), row=1, col=1)

    if show_ema and "EMA20" in df_e and "EMA50" in df_e:
        fig.add_trace(go.Scatter(x=df_e.index, y=df_e["EMA20"], mode="lines", name="EMA20",
                                 line=dict(color="#9B59B6", width=1.2)), row=1, col=1)
        fig.add_trace(go.Scatter(x=df_e.index, y=df_e["EMA50"], mode="lines", name="EMA50",
                                 line=dict(color="#F1C40F", width=1.2)), row=1, col=1)

    if show_bb and all(c in df_e for c in ["BB_Upper", "BB_Middle", "BB_Lower"]):
        fig.add_trace(go.Scatter(x=df_e.index, y=df_e["BB_Upper"], mode="lines", name="BB Upper",
                                 line=dict(color="gray", width=1, dash="dot")), row=1, col=1)
        fig.add_trace(go.Scatter(x=df_e.index, y=df_e["BB_Middle"], mode="lines", name="BB Middle",
                                 line=dict(color="gray", width=1, dash="dash")), row=1, col=1)
        fig.add_trace(go.Scatter(x=df_e.index, y=df_e["BB_Lower"], mode="lines", name="BB Lower",
                                 line=dict(color="gray", width=1, dash="dot")), row=1, col=1)

    if show_dc and all(c in df_e for c in ["DC_Upper", "DC_Lower"]):
        fig.add_trace(go.Scatter(x=df_e.index, y=df_e["DC_Upper"], mode="lines", name="Donchian Upper",
                                 line=dict(color="#E67E22", width=1, dash="dot")), row=1, col=1)
        fig.add_trace(go.Scatter(x=df_e.index, y=df_e["DC_Lower"], mode="lines", name="Donchian Lower",
                                 line=dict(color="#E67E22", width=1, dash="dot")), row=1, col=1)

    # Volume
    if show_volume and "volume" in df_e:
        vol_colors = np.where(df_e["close"] >= df_e["close"].shift(1), "#4CAF50", "#E74C3C")
        fig.add_trace(go.Bar(x=df_e.index, y=df_e["volume"], name="Volume", marker_color=vol_colors, opacity=0.5),
                      row=2, col=1)

    # Indicators panel
    if show_rsi and "RSI" in df_e:
        fig.add_trace(go.Scatter(x=df_e.index, y=df_e["RSI"], name="RSI", line=dict(color="#16A085", width=1.5)),
                      row=3, col=1, secondary_y=False)
        fig.add_hline(y=70, line=dict(color="#95A5A6", dash="dot", width=1), row=3, col=1)
        fig.add_hline(y=30, line=dict(color="#95A5A6", dash="dot", width=1), row=3, col=1)

    if show_adx and "ADX" in df_e:
        fig.add_trace(go.Scatter(x=df_e.index, y=df_e["ADX"], name="ADX", line=dict(color="#34495E", width=1.5)),
                      row=3, col=1, secondary_y=True)

    if show_atr and "ATR" in df_e:
        fig.add_trace(go.Scatter(x=df_e.index, y=df_e["ATR"], name="ATR", line=dict(color="#C0392B", width=1.2)),
                      row=3, col=1, secondary_y=True)

    # Signal markers
    long_idx = df_st.index[df_st["Long_Signal"]].tolist() if (df_st is not None and not df_st.empty and "Long_Signal" in df_st) else []
    short_idx = df_st.index[df_st["Short_Signal"]].tolist() if (df_st is not None and not df_st.empty and "Short_Signal" in df_st) else []

    if long_idx:
        fig.add_trace(go.Scatter(
            x=long_idx, y=df_st.reindex(long_idx)["close"], mode="markers", name="Long Signal",
            marker=dict(symbol="triangle-up", size=12, color="#2ECC71", line=dict(color="black", width=0.5))
        ), row=1, col=1)

    if short_idx:
        fig.add_trace(go.Scatter(
            x=short_idx, y=df_st.reindex(short_idx)["close"], mode="markers", name="Short Signal",
            marker=dict(symbol="triangle-down", size=12, color="#E74C3C", line=dict(color="black", width=0.5))
        ), row=1, col=1)

    latest_info_md = ""
    if show_stops_tps and (latest_idx is not None) and (df_st is not None) and (not df_st.empty) and (latest_idx in df_st.index):
        row = df_st.loc[latest_idx]
        side = latest_side
        entry_price = float(row["close"])

        if side == "long":
            stop_price = float(row["Long_Stop"])
            tp1 = float(row["Long_TP1"])
            tp2 = float(row["Long_TP2"])
            color = "#2ECC71"
        else:
            stop_price = float(row["Short_Stop"])
            tp1 = float(row["Short_TP1"])
            tp2 = float(row["Short_TP2"])
            color = "#E74C3C"

        qty, risk_amt = position_size(entry_price, stop_price, risk.account_equity, risk.risk_pct)

        fig.add_hline(y=entry_price, line=dict(color=color, width=1.5),
                      annotation_text=f"Entry {entry_price:,.2f}", annotation_position="right", row=1, col=1)
        fig.add_hline(y=stop_price, line=dict(color="#C0392B", width=1.5, dash="dash"),
                      annotation_text=f"Stop {stop_price:,.2f}", annotation_position="right", row=1, col=1)
        fig.add_hline(y=tp1, line=dict(color="#27AE60", width=1, dash="dot"),
                      annotation_text=f"TP1 {tp1:,.2f}", annotation_position="right", row=1, col=1)
        fig.add_hline(y=tp2, line=dict(color="#1ABC9C", width=1, dash="dot"),
                      annotation_text=f"TP2 {tp2:,.2f}", annotation_position="right", row=1, col=1)

        r1 = (abs(tp1 - entry_price) / abs(entry_price - stop_price)) if abs(entry_price - stop_price) > 0 else float("nan")
        r2 = (abs(tp2 - entry_price) / abs(entry_price - stop_price)) if abs(entry_price - stop_price) > 0 else float("nan")

        latest_info_md = f"""
- Side: {side.upper()}
- Time: {latest_idx}
- Reason: {row.get('Signal_Reason', '')}
- Entry: ${entry_price:,.2f} • Stop: ${stop_price:,.2f} • TP1: ${tp1:,.2f} • TP2: ${tp2:,.2f}
- Risk: ${risk_amt:,.2f} ({risk.risk_pct:.2f}%) • Size: {qty:.6f} BTC (notional ~ ${qty*entry_price:,.2f})
- R to TP1: {r1:.2f}R • R to TP2: {r2:.2f}R
        """.strip()

    fig.update_layout(
        height=850,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        margin=dict(l=10, r=10, t=30, b=10),
        xaxis_rangeslider_visible=False,
        template="plotly_dark",
    )
    fig.update_yaxes(type="log" if log_scale else "linear", row=1, col=1)

    return fig, latest_info_md