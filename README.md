# TradingPal-AI

Multi-agent AI investment advisor for academic thesis validation.
Combines **FinBERT** sentiment analysis, **technical indicators** (RSI/SMA),
the **FinSen dataset** (2007-2023), and **Google Gemini** to produce personalised
BUY / HOLD / SELL recommendations and full-year backtesting.

---

## Architecture

```
User Input (Streamlit UI)
        │
        ▼
┌───────────────────────────────────────────┐
│              Manager Agent                │  ← Google Gemini (gemini-2.5-flash-lite)
│   synthesises all signals → final decision│
└───────────┬───────────┬───────────────────┘
            │           │           │
    ┌───────▼─┐  ┌──────▼──┐  ┌────▼──────┐
    │Technical│  │Sentiment│  │  Risk     │
    │  Agent  │  │  Agent  │  │  Agent    │
    └────┬────┘  └────┬────┘  └────┬──────┘
         │            │            │
    yfinance     FinBERT +    RiskManager
    RSI/SMA      FinSen CSV   (stop-loss /
    Volatility   NewsAPI      take-profit)
```

---

## Installation

### 1. Clone & enter the project

```bash
cd TradingPal
```

### 2. Create a virtual environment

```bash
python -m venv venv
# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

> **Note:** `torch` and `transformers` (for FinBERT) can be large (~2 GB).
> If you only want the TextBlob fallback, you can skip them — the app degrades gracefully.

### 4. Configure environment

```bash
cp .env.example .env
```

Edit `.env`:

| Variable           | Required | Description                                                 |
|--------------------|----------|-------------------------------------------------------------|
| `GEMINI_API_KEY`   | Yes      | Google AI Studio key — get it at aistudio.google.com/apikey |
| `NEWS_API_KEY`     | No       | NewsAPI key — falls back to simulated news if absent         |
| `FINSEN_DATA_PATH` | No       | Path to FinSen CSV — falls back to simulated sentiment       |

### 5. (Optional) Download FinSen Dataset

The FinSen dataset provides historical financial news sentiment (2007-2023).

1. Go to Kaggle: [massive-stock-news-analysis-db-for-nlpbacktests](https://www.kaggle.com/datasets/miguelaenlle/massive-stock-news-analysis-db-for-nlpbacktests)
2. Download `US_Financial_News.csv`
3. Place it at: `data/finsen_data/US_Financial_News.csv`

The app works without it (uses simulated sentiment data as fallback).

### 6. Run the application

```bash
streamlit run app.py
```

Open **http://localhost:8501** in your browser.

---

## Project Structure

```
TradingPal/
├── app.py                          # Streamlit entry point + sidebar
├── config.py                       # Global constants & API keys
├── requirements.txt
├── .env.example
│
├── data/
│   ├── finsen_loader.py            # FinSen historical sentiment dataset
│   ├── market_data.py              # yfinance — live & historical prices
│   └── news_fetcher.py             # NewsAPI — current news (with fallback)
│
├── analysis/
│   ├── technical_indicators.py     # RSI, SMA, trading signals
│   ├── sentiment_analyzer.py       # FinBERT (TextBlob fallback)
│   └── risk_manager.py             # Position sizing, stop-loss, take-profit
│
├── agents/
│   ├── manager_agent.py            # Gemini orchestrator — final decision
│   ├── technical_agent.py          # Technical signal agent
│   ├── sentiment_agent.py          # Sentiment signal agent
│   └── risk_agent.py               # Risk calculation agent
│
├── backtesting/
│   ├── backtest_engine.py          # Full-year historical simulation
│   └── accuracy_reporter.py        # Metrics & CSV export
│
├── ui/
│   ├── components.py               # Reusable Plotly/Streamlit components
│   └── pages/
│       ├── trade_advisor.py        # Trade recommendation page
│       └── backtest_viewer.py      # Backtesting chart page
│
├── utils/
│   ├── helpers.py                  # Formatting utilities
│   └── validators.py               # Input validation
│
├── tests/
│   ├── test_technical.py
│   ├── test_sentiment.py
│   └── test_backtest.py
│
└── data_cache/                     # Local cache (git-ignored)
```

---

## How Backtesting Works

1. **Data**: downloads daily OHLCV prices for the full selected year via yfinance (~252 trading days).
2. **Signals**: for each day, calculates RSI(14) and SMA(20/50) on prior history only (no look-ahead bias), blended with FinSen historical sentiment.
3. **Simulation**: BUY signal → open position (risk-based sizing); SELL → close and record P&L. Starts with configurable initial capital (default $10,000).
4. **Chart**: 3-panel Plotly chart — candlestick + signals (Panel 1), equity curve (Panel 2), drawdown (Panel 3) — all synchronized and interactive.
5. **Metrics**: Total Return, Annualised Return, Sharpe Ratio, Max Drawdown, Win Rate.
6. **Accuracy**: compares directional calls against actual N-day forward returns.

---

## Telegram Daily Brief (bot/)

A headless bot, run by GitHub Actions every 30 minutes, that:

- sends a **morning recap** (default 08:00 Europe/Bucharest): benchmark moves, your watchlist
  with price / 1-day / 5-day change / RSI, top crypto & stock headlines (real RSS + Yahoo
  Finance news only — never simulated), and 1-2 **buy** and **sell/avoid** ideas from
  RSI/SMA + headline sentiment, written up by Gemini (Groq fallback, rule-based if no AI);
- lets you **edit the watchlist by chatting** with it — "add AMD and Solana",
  "stop tracking Tesla", "note on BTC: long-term hold", "send the recap at 7:30",
  "send me the recap now". The AI turns your message into actions; tickers are checked
  against Yahoo Finance before they're added. Shortcuts: `/list`, `/add X`, `/remove X`,
  `/brief`, `/help`.

The watchlist lives in [`bot/data/watchlist.json`](bot/data/watchlist.json) (you can also edit
it on GitHub directly) and the workflow commits changes back to the repo. Assets with
`"focus": "watch"` only show up on strong signals or big (≥5%) moves.
Messages are picked up on the next run, so replies take up to ~30 minutes.

### Setup

1. In Telegram, talk to **@BotFather** → `/newbot` → copy the **bot token**.
2. Send any message to your new bot, then open
   `https://api.telegram.org/bot<TOKEN>/getUpdates` in a browser and copy `"chat":{"id": ...}`.
3. In GitHub: **Settings → Secrets and variables → Actions → New repository secret** and add
   `TELEGRAM_BOT_TOKEN`, `TELEGRAM_CHAT_ID`, `GEMINI_API_KEY` (optional: `GROQ_API_KEY`).
4. Merge to the default branch (scheduled workflows only run from there), then test it via
   **Actions → Telegram bot → Run workflow** with *Send the daily brief now* ticked.

Only messages from `TELEGRAM_CHAT_ID` are obeyed. Local test run (prints instead of sending):

```bash
pip install -r bot/requirements.txt
python -m bot.main --dry-run --force-brief
```

---

## Running Tests

```bash
pytest tests/ -v
```

---

## Disclaimer

TradingPal-AI is an academic research application for thesis validation purposes only.
It does not constitute financial advice. Always consult a qualified financial advisor
before making investment decisions.
