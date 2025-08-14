from dataclasses import dataclass

@dataclass
class Settings:
    adx_thresh: float = 20.0
    rsi_bull_floor: float = 45.0
    rsi_bear_ceiling: float = 55.0
    bb_length: int = 20
    bb_mult: float = 2.0
    donchian_len: int = 20
    ema_fast: int = 20
    ema_slow: int = 50
    sma_trend: int = 200
    atr_len: int = 14
    swing_lookback: int = 10
    vol_ma_len: int = 20
    breakout_vol_mult: float = 1.2

@dataclass
class RiskSettings:
    account_equity: float = 10_000.0
    risk_pct: float = 0.75
    stop_method: str = "Structure"  # or "ATR x"
    atr_mult: float = 1.0
    tp1_r: float = 1.5
    tp2_r: float = 2.0