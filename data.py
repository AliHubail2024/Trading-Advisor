import pandas as pd
import numpy as np

# Optional Streamlit caching without hard dependency
try:
    import streamlit as st
    cache_data = st.cache_data
except Exception:  # bare mode or not installed
    def cache_data(**kwargs):
        def decorator(func):
            return func
        return decorator

def normalize_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    required = ["open", "high", "low", "close"]
    out_cols = ["open", "high", "low", "close", "volume"]

    if df is None or (hasattr(df, "empty") and df.empty):
        return pd.DataFrame(columns=out_cols)

    df = df.copy()

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [
            "_".join([str(c) for c in tup if c is not None]).strip().lower()
            for tup in df.columns.values
        ]
    else:
        df.columns = [str(c).strip().lower() for c in df.columns]

    rename_map = {
        "open": "open", "high": "high", "low": "low",
        "close": "close", "adj close": "close", "adj_close": "close", "adjclose": "close",
        "volume": "volume", "vol": "volume", "quote_volume": "volume",
    }
    for k, v in list(rename_map.items()):
        if k in df.columns and k != v:
            df.rename(columns={k: v}, inplace=True)

    for key in ["open", "high", "low", "close"]:
        if key not in df.columns:
            candidates = [c for c in df.columns if c.endswith(f"_{key}") or c == key]
            if candidates:
                df.rename(columns={candidates[0]: key}, inplace=True)

    if "volume" not in df.columns:
        df["volume"] = 0.0

    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index, errors="coerce")

    df = df.sort_index()
    df = df[~df.index.duplicated(keep="last")]

    # Keep only relevant columns
    df = df[[c for c in out_cols if c in df.columns]]

    if not set(required).issubset(df.columns):
        return pd.DataFrame(columns=out_cols)

    for c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=required)

    return df

def fetch_yf(symbol: str, interval: str, lookback_days: int) -> pd.DataFrame:
    try:
        import yfinance as yf
    except Exception as e:
        raise ImportError("yfinance not installed. pip install yfinance") from e

    interval = interval.lower()
    # yfinance realistic caps
    period_caps = {
        "1m": 7, "2m": 60, "5m": 60, "15m": 60, "30m": 60,
        "60m": 730, "90m": 730, "1h": 730,
        "1d": 3650, "5d": 3650, "1wk": 3650, "1mo": 3650, "3mo": 3650
    }

    if interval not in period_caps:
        if interval == "4h":
            interval = "1h"
        else:
            raise ValueError(f"Unsupported yfinance interval: {interval}")

    period_days = min(int(lookback_days), period_caps[interval])
    period_str = f"{period_days}d"

    # Note: yfinance accepts both "60m" and "1h". We'll normalize "1h" to "60m" to be safe.
    yf_interval = "60m" if interval == "1h" else interval

    df = yf.download(
        tickers=symbol,
        interval=yf_interval,
        period=period_str,
        auto_adjust=False,
        progress=False,
        threads=True,
        prepost=False,
    )
    return normalize_ohlcv(df)

def fetch_ccxt_binance(
    timeframe: str,
    limit: int = 1500,
    lookback_days: int | None = None,
    market: str = "usdm",
    symbol: str | None = None,
) -> pd.DataFrame:
    try:
        import ccxt
    except Exception as e:
        raise ImportError("ccxt not installed. pip install ccxt") from e

    m = market.lower()
    if m == "usdm":
        # Use 'future' to be compatible with older CCXT; subtype linear for clarity
        exchange = ccxt.binanceusdm({
            "enableRateLimit": True,
            "options": {"defaultType": "future", "defaultSubType": "linear"},
        })
        default_symbol = None
    elif m == "coinm":
        exchange = ccxt.binancecoinm({"enableRateLimit": True, "options": {"defaultType": "delivery"}})
        default_symbol = "BTC/USD:USD"
    elif m == "spot":
        exchange = ccxt.binance({"enableRateLimit": True})
        default_symbol = "BTC/USDT"
    else:
        raise ValueError("market must be 'spot','usdm','coinm'")

    exchange.load_markets()
    tf = timeframe.lower()

    # Validate timeframe if the exchange exposes a whitelist
    if getattr(exchange, "timeframes", None):
        if tf not in exchange.timeframes:
            raise ValueError(f"Timeframe '{tf}' not supported by {exchange.id}. Supported: {list(exchange.timeframes.keys())}")

    # Choose a valid symbol
    if m == "usdm":
        use_symbol = symbol or _pick_binance_usdm_symbol(exchange, preferred=symbol)
    else:
        use_symbol = symbol or default_symbol
        if use_symbol not in exchange.symbols:
            raise ValueError(f"{exchange.id} does not have market symbol {use_symbol}")

    # Compute 'since' from lookback
    since = None
    if lookback_days is not None:
        since = exchange.milliseconds() - int(lookback_days * 24 * 60 * 60 * 1000)

    # Helpful debug print shows in the Streamlit terminal
    try:
        import ccxt as _ccxt
        btc_examples = [s for s in exchange.symbols if "BTC" in s][:12]
        print(f"[ccxt] id={exchange.id} v{_ccxt.__version__} tf={tf} use_symbol={use_symbol} markets={len(exchange.symbols)}")
        print(f"[ccxt] BTC-like examples: {btc_examples}")
    except Exception:
        pass

    # Fetch data
    ohlcv = exchange.fetch_ohlcv(use_symbol, timeframe=tf, since=since, limit=limit)
    if not ohlcv:
        return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])

    df = pd.DataFrame(ohlcv, columns=["timestamp", "open", "high", "low", "close", "volume"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True).dt.tz_convert(None)
    df = df.set_index("timestamp").astype(float)
    return normalize_ohlcv(df)

def resample_ohlc(df: pd.DataFrame, rule: str) -> pd.DataFrame:
    df = normalize_ohlcv(df)
    if df.empty:
        return df
    agg = {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}
    out = df.resample(rule).agg(agg)
    return normalize_ohlcv(out).dropna()

def prepare_anchor(df: pd.DataFrame, anchor_tf: str, entry_tf: str | None = None) -> pd.DataFrame:
    df = normalize_ohlcv(df)
    if df.empty:
        return df
    if entry_tf is not None and anchor_tf == entry_tf:
        return df.copy()
    rule_map = {"15m": "15T", "1h": "1H", "4h": "4H", "1d": "1D", "1wk": "1W"}
    if anchor_tf not in rule_map:
        raise ValueError(f"Unsupported anchor_tf: {anchor_tf}")
    return resample_ohlc(df, rule_map[anchor_tf])

@cache_data(show_spinner=False, ttl=180)
def load_data(source: str, entry_tf: str, lookback_days: int, limit_ccxt: int) -> pd.DataFrame:
    """
    Unified loader (cached). Returns a normalized OHLCV DataFrame.
    """
    src = (source or "").lower()
    tf = (entry_tf or "").lower()

    if "yahoo" in src:
        if tf == "4h":
            base = fetch_yf("BTC-USD", "1h", lookback_days)
            return resample_ohlc(base, "4H")
        return fetch_yf("BTC-USD", tf, lookback_days)

    # Default to Binance USDT‑M perpetuals
    return fetch_ccxt_binance(
        timeframe=tf,
        limit=limit_ccxt,
        lookback_days=lookback_days,
        market="usdm",
        symbol="BTC/USDT",
    )




def _pick_binance_usdm_symbol(exchange, preferred: str | None = None) -> str:
    """
    Robustly detect the BTC USDT‑margined perpetual symbol for the installed CCXT version.
    Works across CCXT versions where market['type'] may be 'future' or 'swap'.
    """
    exchange.load_markets()

    # If user explicitly passed a symbol and it exists, use it.
    if preferred:
        if preferred in exchange.symbols:
            return preferred
        raise ValueError(f"Preferred symbol '{preferred}' not found on {exchange.id}")

    # Quick wins
    for s in ["BTC/USDT:USDT", "BTC/USDT"]:
        if s in exchange.symbols:
            return s

    # Programmatic search: linear, BTC base, USDT/USDC/BUSD quote, perpetual
    candidates = []
    for sym in exchange.symbols:
        try:
            m = exchange.market(sym)
            base_ok = m.get("base") == "BTC"
            quote_ok = m.get("quote") in ("USDT", "USDC", "BUSD")
            linear_ok = bool(m.get("linear"))
            type_ok = m.get("type") in ("swap", "future")  # older CCXT used 'future'
            # Perpetual detection across CCXT versions
            contract_type = str(m.get("contractType", "")).lower()
            is_perp = ("perpetual" in contract_type) or bool(m.get("swap")) or (m.get("contract") and m.get("expiry") in (None, 0))
            if base_ok and quote_ok and linear_ok and type_ok and is_perp:
                candidates.append(sym)
        except Exception:
            pass

    if candidates:
        # Prefer USDT, then shorter/cleaner symbol strings
        def rank(s: str):
            s_up = s.upper()
            return (
                0 if (":USDT" in s_up or "/USDT" in s_up) else 1,
                len(s),
                s,
            )
        candidates.sort(key=rank)
        return candidates[0]

    # Fallback: any BTC/USDT-like symbol
    btc_like = [s for s in exchange.symbols if "BTC" in s]
    raise ValueError(
        "Could not find a Binance USD‑M BTC perpetual symbol. "
        f"BTC-like examples: {btc_like[:20]}{' ...' if len(btc_like) > 20 else ''}"
    )