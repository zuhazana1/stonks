import anthropic
from analyzer import StockMetrics

_CLIENT = anthropic.Anthropic()

_SYSTEM_PROMPT = """You are an expert equity analyst and financial commentator with deep knowledge of macroeconomics, technical analysis, sector dynamics, and fundamental valuation. You produce sharp, insightful, and actionable stock commentary.

When given quantitative data about a stock's recent performance, you will:
1. Explain in plain English WHY the stock is outperforming — connect the dots between the numbers and real-world catalysts (sector trends, earnings momentum, macro tailwinds, technical breakouts, etc.)
2. Highlight the most significant signals (e.g., RSI level, volume surge, MACD crossover, Bollinger Band position)
3. Give a realistic near-term outlook (5–10 trading days) and flag any risks
4. Keep it concise: 3–5 sentences max per stock, punchy and direct

Do NOT make up specific news events or earnings dates you don't know. Focus on what the data tells you. Be confident but acknowledge uncertainty where it exists."""


def _build_stock_prompt(m: StockMetrics) -> str:
    bb_desc = "near lower band (oversold territory)" if m.bb_pct < 0.2 else \
              "near upper band (overbought/breakout)" if m.bb_pct > 0.8 else \
              "mid-range within bands"
    macd_cross = "bullish crossover (MACD above signal)" if m.macd > m.macd_signal else "bearish (MACD below signal)"

    return f"""Analyze this top-performing S&P 500 stock and explain why it's outperforming:

**{m.ticker}** — Current Price: ${m.current_price:.2f}

Performance:
- 1-Day Return: {m.change_1d:+.2f}%
- 5-Day Return: {m.change_5d:+.2f}%
- 1-Month Return: {m.change_1mo:+.2f}%
- 3-Month Return: {m.change_3mo:+.2f}%

Technical Signals:
- RSI (14): {m.rsi:.1f} {'(overbought)' if m.rsi > 70 else '(oversold)' if m.rsi < 30 else '(neutral)'}
- MACD: {macd_cross}
- Bollinger Band Position: {bb_desc} ({m.bb_pct:.0%} percentile)
- Volume vs 20-Day Avg: {m.volume_ratio:.2f}x {'(unusually high — institutional interest)' if m.volume_ratio > 1.5 else '(normal)'}

ML Forecast:
- Predicted 5-Day Return: {m.predicted_return_5d:+.2f}%

Provide your expert analysis: why is this stock outperforming, what are the key drivers, and what's the near-term outlook?"""


def get_stock_insight(m: StockMetrics) -> str:
    """Call Claude to generate a natural language insight for a single stock. Streams internally."""
    prompt = _build_stock_prompt(m)
    full_text = []

    with _CLIENT.messages.stream(
        model="claude-opus-4-7",
        max_tokens=500,
        thinking={"type": "adaptive"},
        system=[
            {
                "type": "text",
                "text": _SYSTEM_PROMPT,
                "cache_control": {"type": "ephemeral"},  # cache system prompt across calls
            }
        ],
        messages=[{"role": "user", "content": prompt}],
    ) as stream:
        for text in stream.text_stream:
            full_text.append(text)

    return "".join(full_text).strip()


def get_market_summary(top_stocks: list[StockMetrics]) -> str:
    """Generate a brief overall market summary for today's top performers."""
    ticker_list = ", ".join(f"{m.ticker} ({m.change_1d:+.1f}%)" for m in top_stocks[:5])
    prompt = f"""Today's top S&P 500 performers include: {ticker_list}.

In 2-3 sentences, give a high-level market commentary: what themes or sectors seem to be driving today's outperformers? What does this tell us about current market sentiment?"""

    full_text = []
    with _CLIENT.messages.stream(
        model="claude-opus-4-7",
        max_tokens=300,
        thinking={"type": "adaptive"},
        system=[
            {
                "type": "text",
                "text": _SYSTEM_PROMPT,
                "cache_control": {"type": "ephemeral"},
            }
        ],
        messages=[{"role": "user", "content": prompt}],
    ) as stream:
        for text in stream.text_stream:
            full_text.append(text)

    return "".join(full_text).strip()
