#!/usr/bin/env python3
"""
FastAPI backend for the Stonks dashboard.
Run: python3.14 -m uvicorn api:app --reload --port 8000
"""
import asyncio
import json
import threading
import time
from pathlib import Path

import anthropic as anthropic_module
import numpy as np
import pandas as pd
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from analyzer import StockMetrics, _compute_indicators, rank_stocks
from fetcher import get_top_performers_data
from insights import _SYSTEM_PROMPT, _build_stock_prompt

app = FastAPI(title="Stonks API")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

_cache: dict = {
    "status": "idle",       # idle | loading | ready | error
    "progress_msg": "Starting...",
    "top_stocks": [],       # list[StockMetrics]
    "stock_data": {},       # dict[ticker, DataFrame]  — raw OHLCV
    "timestamp": 0.0,
}


# ── Background data loader ────────────────────────────────────────────────────

def _load_data() -> None:
    _cache["status"] = "loading"
    _cache["progress_msg"] = "Downloading S&P 500 price data…"
    try:
        stock_data = get_top_performers_data()
        _cache["stock_data"] = stock_data
        _cache["progress_msg"] = f"Running ML analysis on {len(stock_data)} stocks…"
        top_stocks = rank_stocks(stock_data, top_n=15)
        _cache["top_stocks"] = top_stocks
        _cache["timestamp"] = time.time()
        _cache["status"] = "ready"
        _cache["progress_msg"] = "Ready"
    except Exception as exc:
        _cache["status"] = "error"
        _cache["progress_msg"] = str(exc)


@app.on_event("startup")
async def startup() -> None:
    threading.Thread(target=_load_data, daemon=True).start()


# ── Helpers ───────────────────────────────────────────────────────────────────

def _m2d(m: StockMetrics) -> dict:
    return {
        "ticker": m.ticker,
        "current_price": round(m.current_price, 2),
        "change_1d": round(m.change_1d, 2),
        "change_5d": round(m.change_5d, 2),
        "change_1mo": round(m.change_1mo, 2),
        "change_3mo": round(m.change_3mo, 2),
        "rsi": round(m.rsi, 1),
        "macd": round(m.macd, 4),
        "macd_signal": round(m.macd_signal, 4),
        "bb_pct": round(m.bb_pct, 2),
        "volume_ratio": round(m.volume_ratio, 2),
        "predicted_return_5d": round(m.predicted_return_5d, 2),
        "composite_score": round(m.composite_score, 2),
    }


def _safe(val) -> float | None:
    try:
        f = float(val)
        return None if (np.isnan(f) or np.isinf(f)) else round(f, 4)
    except (TypeError, ValueError):
        return None


# ── API Routes ────────────────────────────────────────────────────────────────

@app.get("/api/status")
async def get_status():
    return {
        "status": _cache["status"],
        "progress_msg": _cache["progress_msg"],
        "ticker_count": len(_cache["top_stocks"]),
        "timestamp": _cache["timestamp"],
    }


@app.get("/api/top-performers")
async def get_top_performers():
    if _cache["status"] != "ready":
        return JSONResponse({"error": "Data not ready yet"}, status_code=503)
    return [_m2d(m) for m in _cache["top_stocks"]]


@app.get("/api/stock/{ticker}/ohlcv")
async def get_ohlcv(ticker: str):
    df = _cache["stock_data"].get(ticker)
    if df is None:
        return JSONResponse({"error": "Ticker not found"}, status_code=404)

    result = []
    for ts, row in df.iterrows():
        try:
            o, h, l, c = float(row["Open"]), float(row["High"]), float(row["Low"]), float(row["Close"])
            v = int(row["Volume"])
            if any(np.isnan(x) for x in [o, h, l, c]):
                continue
            result.append({
                "time": ts.strftime("%Y-%m-%d"),
                "open": round(o, 2), "high": round(h, 2),
                "low": round(l, 2),  "close": round(c, 2),
                "volume": v,
            })
        except (ValueError, TypeError):
            continue
    return result


@app.get("/api/stock/{ticker}/indicators")
async def get_indicators(ticker: str):
    raw = _cache["stock_data"].get(ticker)
    if raw is None:
        return JSONResponse({"error": "Ticker not found"}, status_code=404)

    df = _compute_indicators(raw.copy())
    vol = df["Volume"].squeeze()
    vol_ma = vol.rolling(20).mean()
    df["volume_ratio"] = vol / vol_ma.replace(0, np.nan)

    rsi, macd, signal, bb_up, bb_lo = [], [], [], [], []

    for ts, row in df.iterrows():
        date = ts.strftime("%Y-%m-%d")
        if (v := _safe(row.get("rsi"))) is not None:
            rsi.append({"time": date, "value": v})
        if (v := _safe(row.get("macd"))) is not None:
            macd.append({"time": date, "value": v})
        if (v := _safe(row.get("macd_signal"))) is not None:
            signal.append({"time": date, "value": v})
        if (v := _safe(row.get("bb_upper"))) is not None:
            bb_up.append({"time": date, "value": v})
        if (v := _safe(row.get("bb_lower"))) is not None:
            bb_lo.append({"time": date, "value": v})

    return {"rsi": rsi, "macd": macd, "macd_signal": signal,
            "bb_upper": bb_up, "bb_lower": bb_lo}


@app.get("/api/insight/{ticker}")
async def stream_insight(ticker: str):
    metrics = next((m for m in _cache["top_stocks"] if m.ticker == ticker), None)
    if metrics is None:
        return JSONResponse({"error": "Ticker not in top performers"}, status_code=404)

    async def generate():
        client = anthropic_module.AsyncAnthropic()
        try:
            async with client.messages.stream(
                model="claude-opus-4-7",
                max_tokens=500,
                thinking={"type": "adaptive"},
                system=[{"type": "text", "text": _SYSTEM_PROMPT,
                         "cache_control": {"type": "ephemeral"}}],
                messages=[{"role": "user", "content": _build_stock_prompt(metrics)}],
            ) as stream:
                async for chunk in stream.text_stream:
                    yield f"data: {json.dumps({'text': chunk})}\n\n"
        except Exception as exc:
            yield f"data: {json.dumps({'text': f'[Error: {exc}]'})}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(generate(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


@app.get("/api/market-summary/stream")
async def stream_market_summary():
    if _cache["status"] != "ready":
        return JSONResponse({"error": "Not ready"}, status_code=503)

    top = _cache["top_stocks"]
    ticker_list = ", ".join(f"{m.ticker} ({m.change_1d:+.1f}%)" for m in top[:5])
    prompt = (
        f"Today's top S&P 500 performers include: {ticker_list}.\n\n"
        "In 2–3 sentences, give a high-level market commentary: what themes or sectors are driving "
        "these outperformers? What does this signal about current market sentiment?"
    )

    async def generate():
        client = anthropic_module.AsyncAnthropic()
        try:
            async with client.messages.stream(
                model="claude-opus-4-7",
                max_tokens=300,
                thinking={"type": "adaptive"},
                system=[{"type": "text", "text": _SYSTEM_PROMPT,
                         "cache_control": {"type": "ephemeral"}}],
                messages=[{"role": "user", "content": prompt}],
            ) as stream:
                async for chunk in stream.text_stream:
                    yield f"data: {json.dumps({'text': chunk})}\n\n"
        except Exception as exc:
            yield f"data: {json.dumps({'text': f'[Error: {exc}]'})}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(generate(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


# ── Serve frontend (must be last) ─────────────────────────────────────────────
app.mount("/", StaticFiles(directory=Path(__file__).parent / "frontend", html=True), name="frontend")
