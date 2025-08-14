import numpy as np
import pandas as pd

def sma(series: pd.Series, length: int) -> pd.Series:
    return series.rolling(length, min_periods=length).mean()

def ema(series: pd.Series, length: int) -> pd.Series:
    return series.ewm(span=length, adjust=False, min_periods=length).mean()

def rsi(series: pd.Series, length: int = 14) -> pd.Series:
    delta = series.diff()
    gain = (delta.clip(lower=0)).ewm(alpha=1 / length, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / length, adjust=False).mean()
    rs = gain / (loss.replace(0, np.nan))
    return 100 - (100 / (1 + rs))

def true_range(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
    prev_close = close.shift(1)
    tr = pd.concat([
        (high - low).abs(),
        (high - prev_close).abs(),
        (low - prev_close).abs()
    ], axis=1).max(axis=1)
    return tr

def atr(high: pd.Series, low: pd.Series, close: pd.Series, length: int = 14) -> pd.Series:
    tr = true_range(high, low, close)
    return tr.ewm(alpha=1 / length, adjust=False, min_periods=length).mean()

def bollinger_bands(series: pd.Series, length: int = 20, mult: float = 2.0):
    basis = sma(series, length)
    dev = series.rolling(length, min_periods=length).std()
    upper = basis + mult * dev
    lower = basis - mult * dev
    return upper, basis, lower

def donchian(high: pd.Series, low: pd.Series, length: int = 20):
    upper = high.rolling(length, min_periods=length).max()
    lower = low.rolling(length, min_periods=length).min()
    mid = (upper + lower) / 2.0
    return upper, mid, lower

def slope(series: pd.Series, lookback: int = 5) -> pd.Series:
    return series - series.shift(lookback)