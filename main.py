#!/usr/bin/env python3
"""
stonks — Daily S&P 500 Top Performers with AI Insights
Usage: python main.py [--top N] [--no-ai]
"""
import argparse
import os
from datetime import datetime

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich import box
from rich.text import Text
from rich.rule import Rule
from rich.progress import Progress, SpinnerColumn, TextColumn

from fetcher import get_top_performers_data
from analyzer import rank_stocks, StockMetrics
from insights import get_stock_insight, get_market_summary

console = Console()


def _color_pct(val: float) -> str:
    color = "green" if val >= 0 else "red"
    sign = "+" if val >= 0 else ""
    return f"[{color}]{sign}{val:.2f}%[/{color}]"


def _rsi_color(rsi: float) -> str:
    if rsi > 70:
        return f"[red]{rsi:.1f}[/red]"
    elif rsi < 30:
        return f"[green]{rsi:.1f}[/green]"
    return f"[yellow]{rsi:.1f}[/yellow]"


def build_summary_table(stocks: list[StockMetrics]) -> Table:
    table = Table(
        title="Top Performers — Summary",
        box=box.ROUNDED,
        show_header=True,
        header_style="bold cyan",
        border_style="cyan",
        expand=True,
    )
    table.add_column("#", style="bold", width=3, justify="right")
    table.add_column("Ticker", style="bold white", width=7)
    table.add_column("Price", justify="right", width=10)
    table.add_column("1D", justify="right", width=9)
    table.add_column("5D", justify="right", width=9)
    table.add_column("1M", justify="right", width=9)
    table.add_column("3M", justify="right", width=9)
    table.add_column("RSI", justify="right", width=7)
    table.add_column("Vol×", justify="right", width=7)
    table.add_column("ML 5D", justify="right", width=9)
    table.add_column("Score", justify="right", width=8)

    for i, m in enumerate(stocks, 1):
        vol_str = f"[{'green' if m.volume_ratio > 1.3 else 'white'}]{m.volume_ratio:.2f}x[/{'green' if m.volume_ratio > 1.3 else 'white'}]"
        ml_color = "green" if m.predicted_return_5d >= 0 else "red"
        ml_sign = "+" if m.predicted_return_5d >= 0 else ""

        table.add_row(
            str(i),
            m.ticker,
            f"${m.current_price:.2f}",
            Text.from_markup(_color_pct(m.change_1d)),
            Text.from_markup(_color_pct(m.change_5d)),
            Text.from_markup(_color_pct(m.change_1mo)),
            Text.from_markup(_color_pct(m.change_3mo)),
            Text.from_markup(_rsi_color(m.rsi)),
            Text.from_markup(vol_str),
            Text.from_markup(f"[{ml_color}]{ml_sign}{m.predicted_return_5d:.2f}%[/{ml_color}]"),
            f"{m.composite_score:.2f}",
        )

    return table


def print_stock_detail(rank: int, m: StockMetrics, insight: str) -> None:
    macd_cross = "↑ Bullish" if m.macd > m.macd_signal else "↓ Bearish"
    bb_desc = "Lower band" if m.bb_pct < 0.2 else "Upper band" if m.bb_pct > 0.8 else "Mid-range"
    color = "green" if m.change_1d >= 0 else "red"

    header = f"[bold white]#{rank}  {m.ticker}[/bold white]  [dim]${m.current_price:.2f}[/dim]  " \
             f"[{color}]{'▲' if m.change_1d >= 0 else '▼'} {abs(m.change_1d):.2f}% today[/{color}]"

    technicals = (
        f"  RSI: {_rsi_color(m.rsi)}  |  "
        f"MACD: {macd_cross}  |  "
        f"BB: {bb_desc} ({m.bb_pct:.0%})  |  "
        f"Volume: [{'green' if m.volume_ratio > 1.3 else 'white'}]{m.volume_ratio:.2f}x avg[/{'green' if m.volume_ratio > 1.3 else 'white'}]  |  "
        f"ML Forecast: [{'green' if m.predicted_return_5d >= 0 else 'red'}]{'+'  if m.predicted_return_5d >= 0 else ''}{m.predicted_return_5d:.2f}% (5d)[/{'green' if m.predicted_return_5d >= 0 else 'red'}]"
    )

    content = Text.from_markup(
        f"{technicals}\n\n[italic dim]AI Insight:[/italic dim]\n{insight}"
    )

    console.print(Panel(content, title=Text.from_markup(header), border_style=color, expand=True))


def main():
    parser = argparse.ArgumentParser(description="S&P 500 daily top performer analysis")
    parser.add_argument("--top", type=int, default=10, help="Number of top stocks to show (default: 10)")
    parser.add_argument("--no-ai", action="store_true", help="Skip AI insights (faster, no API cost)")
    args = parser.parse_args()

    today = datetime.today().strftime("%A, %B %d, %Y")

    console.print()
    console.print(Rule(f"[bold cyan]STONKS[/bold cyan]  [dim]S&P 500 Daily Intelligence Report[/dim]"))
    console.print(f"  [dim]{today}[/dim]")
    console.print()

    # Step 1: Fetch data
    with Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}"), console=console) as p:
        task = p.add_task("Downloading S&P 500 price data...", total=None)
        stock_data = get_top_performers_data()
        p.update(task, completed=True)

    console.print()

    # Step 2: Analyze & rank
    with Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}"), console=console) as p:
        task = p.add_task(f"Analyzing stocks and running ML models...", total=None)
        top_stocks = rank_stocks(stock_data, top_n=args.top)
        p.update(task, completed=True)

    console.print()

    # Step 3: Display summary table
    console.print(build_summary_table(top_stocks))
    console.print()

    if args.no_ai:
        console.print("[dim]AI insights skipped (--no-ai). Run without flag for full analysis.[/dim]")
        return

    if not os.environ.get("ANTHROPIC_API_KEY"):
        console.print("[yellow]Warning: ANTHROPIC_API_KEY not set. Skipping AI insights.[/yellow]")
        console.print("[dim]Set ANTHROPIC_API_KEY environment variable to enable AI-generated analysis.[/dim]")
        return

    # Step 4: Market summary
    console.print(Rule("[bold cyan]Market Summary[/bold cyan]"))
    console.print()
    with Progress(SpinnerColumn(), TextColumn("Generating market overview..."), console=console) as p:
        task = p.add_task("", total=None)
        summary = get_market_summary(top_stocks)
        p.update(task, completed=True)

    console.print(Panel(summary, border_style="cyan", expand=True))
    console.print()

    # Step 5: Per-stock AI insights
    console.print(Rule("[bold cyan]Individual Stock Analysis[/bold cyan]"))
    console.print()

    for i, m in enumerate(top_stocks, 1):
        with Progress(SpinnerColumn(), TextColumn(f"[dim]Analyzing {m.ticker}...[/dim]"), console=console, transient=True) as p:
            p.add_task("", total=None)
            insight = get_stock_insight(m)
        print_stock_detail(i, m, insight)
        console.print()

    console.print(Rule("[dim]End of Report[/dim]"))
    console.print()


if __name__ == "__main__":
    main()
