import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta


def get_sp500_tickers() -> list[str]:
    """Scrape current S&P 500 constituents from Wikipedia."""
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    tables = pd.read_html(url)
    df = tables[0]
    tickers = df["Symbol"].tolist()
    # Fix dot notation (BRK.B -> BRK-B for yfinance)
    tickers = [t.replace(".", "-") for t in tickers]
    return tickers


def fetch_price_data(tickers: list[str], period_days: int = 90) -> dict[str, pd.DataFrame]:
    """
    Download OHLCV data for a list of tickers.
    Returns a dict mapping ticker -> DataFrame with columns: Open, High, Low, Close, Volume.
    """
    end = datetime.today()
    start = end - timedelta(days=period_days)

    result = {}
    # yfinance batch download is faster than individual calls
    raw = yf.download(
        tickers,
        start=start.strftime("%Y-%m-%d"),
        end=end.strftime("%Y-%m-%d"),
        group_by="ticker",
        auto_adjust=True,
        progress=False,
        threads=True,
    )

    for ticker in tickers:
        try:
            if len(tickers) == 1:
                df = raw.copy()
            else:
                df = raw[ticker].copy()
            df = df.dropna(how="all")
            if len(df) >= 20:  # need enough history for indicators
                result[ticker] = df
        except (KeyError, TypeError):
            continue

    return result


def get_top_performers_data(n_top: int = 20, period_days: int = 90) -> dict[str, pd.DataFrame]:
    """Convenience wrapper: fetch all S&P 500 data ready for analysis."""
    tickers = get_sp500_tickers()
    print(f"Fetching data for {len(tickers)} S&P 500 tickers...")
    data = fetch_price_data(tickers, period_days=period_days)
    print(f"Successfully loaded data for {len(data)} tickers.")
    return data
