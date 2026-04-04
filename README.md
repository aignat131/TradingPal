# TradingPal

AI-powered web app that advises users on stock and crypto trades based on their risk profile and preferences.

## Features

- Live price data via yfinance (falls back to sample CSV when unavailable)
- Rule-based risk assessment before any AI call
- Claude-powered trade advice with a clear BUY / SELL / HOLD / AVOID verdict
- Clean single-page UI — no frontend build step required

## Quick Start

```bash
# 1. Clone & enter the project
cd TradingPal

# 2. Create a virtual environment
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Configure environment
cp .env.example .env
# Edit .env and set your ANTHROPIC_API_KEY

# 5. Run
python app.py
```

Open http://localhost:5000 in your browser.

## Project Structure

```
TradingPal/
├── app.py                  # Flask app + single-page UI
├── requirements.txt
├── .env.example
├── models/
│   └── trading_advisor.py  # Claude API integration
├── utils/
│   ├── market_data.py      # yfinance / CSV price fetcher
│   └── risk_checks.py      # Rule-based risk scoring
├── prompts/
│   └── advisor_prompt.txt  # Prompt template for Claude
├── data/
│   └── sample_prices.csv   # Offline fallback prices
└── tests/
    └── test_risk.py        # Unit tests for risk_checks
```

## Running Tests

```bash
python tests/test_risk.py
# or with pytest
pytest tests/
```

## Configuration

| Variable            | Default                     | Description                          |
|---------------------|-----------------------------|--------------------------------------|
| `ANTHROPIC_API_KEY` | —                           | **Required.** Your Anthropic API key |
| `CLAUDE_MODEL`      | `claude-haiku-4-5-20251001` | Claude model to use for advice       |
| `PORT`              | `5000`                      | Port to listen on                    |
| `FLASK_DEBUG`       | `false`                     | Enable Flask debug mode              |

## Disclaimer

TradingPal is for informational purposes only and does not constitute financial advice. Always do your own research before making investment decisions.
