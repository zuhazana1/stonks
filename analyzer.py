import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
import ta


@dataclass
class StockMetrics:
    ticker: str
    current_price: float
    change_1d: float       # % return today
    change_5d: float       # % return past 5 days
    change_1mo: float      # % return past month
    change_3mo: float      # % return past 3 months
    rsi: float
    macd: float
    macd_signal: float
    bb_pct: float          # position within Bollinger Bands (0=lower, 1=upper)
    volume_ratio: float    # today's volume vs 20-day avg
    predicted_return_5d: float  # ML predicted % return over next 5 days
    composite_score: float
    raw_data: pd.DataFrame = field(repr=False)


def _safe_pct(series: pd.Series, periods: int) -> float:
    if len(series) <= periods:
        return 0.0
    val = series.iloc[-1]
    past = series.iloc[-(periods + 1)]
    if past == 0 or np.isnan(past) or np.isnan(val):
        return 0.0
    return (val - past) / past * 100.0


def _compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    close = df["Close"].squeeze()
    high = df["High"].squeeze()
    low = df["Low"].squeeze()
    volume = df["Volume"].squeeze()

    df = df.copy()
    df["rsi"] = ta.momentum.RSIIndicator(close=close, window=14).rsi()

    macd_ind = ta.trend.MACD(close=close)
    df["macd"] = macd_ind.macd()
    df["macd_signal"] = macd_ind.macd_signal()

    bb = ta.volatility.BollingerBands(close=close, window=20, window_dev=2)
    df["bb_upper"] = bb.bollinger_hband()
    df["bb_lower"] = bb.bollinger_lband()

    df["vol_ma20"] = volume.rolling(20).mean()

    # momentum features for ML
    df["ret_1d"] = close.pct_change(1)
    df["ret_5d"] = close.pct_change(5)
    df["ret_10d"] = close.pct_change(10)
    df["ret_20d"] = close.pct_change(20)

    return df


def _build_ml_features(df: pd.DataFrame) -> np.ndarray:
    cols = ["rsi", "macd", "macd_signal", "ret_1d", "ret_5d", "ret_10d", "ret_20d", "volume_ratio"]
    row = {}
    latest = df.iloc[-1]
    row["rsi"] = latest.get("rsi", 50)
    row["macd"] = latest.get("macd", 0)
    row["macd_signal"] = latest.get("macd_signal", 0)
    row["ret_1d"] = latest.get("ret_1d", 0)
    row["ret_5d"] = latest.get("ret_5d", 0)
    row["ret_10d"] = latest.get("ret_10d", 0)
    row["ret_20d"] = latest.get("ret_20d", 0)
    close = df["Close"].squeeze()
    vol = df["Volume"].squeeze()
    vol_ma = vol.rolling(20).mean()
    row["volume_ratio"] = (vol.iloc[-1] / vol_ma.iloc[-1]) if vol_ma.iloc[-1] > 0 else 1.0
    return np.array([list(row.values())])


def _train_and_predict(df: pd.DataFrame) -> float:
    """
    Train a simple GBR on rolling windows to predict 5-day forward return.
    Uses only the stock's own history (no cross-sectional data).
    """
    df = _compute_indicators(df)
    close = df["Close"].squeeze()
    vol = df["Volume"].squeeze()
    vol_ma = vol.rolling(20).mean()
    df["volume_ratio"] = vol / vol_ma

    feature_cols = ["rsi", "macd", "macd_signal", "ret_1d", "ret_5d", "ret_10d", "ret_20d", "volume_ratio"]
    df["target"] = close.pct_change(5).shift(-5)  # forward 5d return

    model_df = df[feature_cols + ["target"]].dropna()
    if len(model_df) < 30:
        return 0.0

    X = model_df[feature_cols].values
    y = model_df["target"].values * 100.0  # as %

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = GradientBoostingRegressor(n_estimators=100, max_depth=3, random_state=42)
    model.fit(X_scaled[:-5], y[:-5])  # don't train on last 5 (no target yet)

    last_features = scaler.transform(X[-1:])
    return float(model.predict(last_features)[0])


def analyze_stock(ticker: str, df: pd.DataFrame) -> StockMetrics | None:
    try:
        df = _compute_indicators(df)
        close = df["Close"].squeeze()
        vol = df["Volume"].squeeze()
        vol_ma = vol.rolling(20).mean()

        current_price = float(close.iloc[-1])
        change_1d = _safe_pct(close, 1)
        change_5d = _safe_pct(close, 5)
        change_1mo = _safe_pct(close, 21)
        change_3mo = _safe_pct(close, 63)

        rsi = float(df["rsi"].iloc[-1]) if not np.isnan(df["rsi"].iloc[-1]) else 50.0
        macd = float(df["macd"].iloc[-1]) if not np.isnan(df["macd"].iloc[-1]) else 0.0
        macd_signal = float(df["macd_signal"].iloc[-1]) if not np.isnan(df["macd_signal"].iloc[-1]) else 0.0

        bb_upper = float(df["bb_upper"].iloc[-1])
        bb_lower = float(df["bb_lower"].iloc[-1])
        bb_pct = (current_price - bb_lower) / (bb_upper - bb_lower) if (bb_upper - bb_lower) > 0 else 0.5

        vol_ratio = float(vol.iloc[-1] / vol_ma.iloc[-1]) if vol_ma.iloc[-1] > 0 else 1.0

        predicted_return_5d = _train_and_predict(df.copy())

        # Composite score weights: recent momentum + ML prediction + volume confirmation
        score = (
            0.30 * change_1d
            + 0.25 * change_5d
            + 0.15 * change_1mo
            + 0.15 * predicted_return_5d
            + 0.10 * (vol_ratio - 1.0) * 10  # bonus for high volume
            + 0.05 * (macd - macd_signal) * 100  # MACD crossover
        )

        return StockMetrics(
            ticker=ticker,
            current_price=current_price,
            change_1d=change_1d,
            change_5d=change_5d,
            change_1mo=change_1mo,
            change_3mo=change_3mo,
            rsi=rsi,
            macd=macd,
            macd_signal=macd_signal,
            bb_pct=bb_pct,
            volume_ratio=vol_ratio,
            predicted_return_5d=predicted_return_5d,
            composite_score=score,
            raw_data=df,
        )
    except Exception:
        return None


def rank_stocks(stock_data: dict[str, pd.DataFrame], top_n: int = 10) -> list[StockMetrics]:
    """Analyze and rank all stocks, returning the top N performers."""
    metrics_list = []
    total = len(stock_data)
    for i, (ticker, df) in enumerate(stock_data.items(), 1):
        if i % 50 == 0:
            print(f"  Analyzing {i}/{total}...")
        m = analyze_stock(ticker, df)
        if m is not None:
            metrics_list.append(m)

    metrics_list.sort(key=lambda x: x.composite_score, reverse=True)
    return metrics_list[:top_n]
