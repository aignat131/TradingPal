"""
Global configuration for TradingPal-AI.
All constants, paths, and environment variables are defined here.
"""
import os
from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------------------------
# API Keys — supports GEMINI_API_KEY and legacy ANTHROPIC_API_KEY fallback
# ---------------------------------------------------------------------------
GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY") or os.getenv("ANTHROPIC_API_KEY", "")
NEWS_API_KEY: str = os.getenv("NEWS_API_KEY", "")

# ---------------------------------------------------------------------------
# Data paths
# ---------------------------------------------------------------------------
FINSEN_DATA_PATH: str = os.getenv(
    "FINSEN_DATA_PATH", "data/finsen_data"
)
CACHE_DIR: str = "data_cache"

# ---------------------------------------------------------------------------
# Technical indicators
# ---------------------------------------------------------------------------
RSI_PERIOD: int = 14
SMA_SHORT: int = 20
SMA_LONG: int = 50
RSI_OVERSOLD: float = 30.0
RSI_OVERBOUGHT: float = 70.0

# ---------------------------------------------------------------------------
# Risk management
# ---------------------------------------------------------------------------
DEFAULT_RISK_PERCENT: float = 2.0
MAX_RISK_PERCENT: float = 5.0
MAX_POSITION_PCT: float = 0.25

# ---------------------------------------------------------------------------
# Backtesting defaults
# ---------------------------------------------------------------------------
BACKTEST_YEAR: int = 2024
INITIAL_CAPITAL: float = 10_000.0
BACKTEST_START: str = "2024-01-01"
BACKTEST_END: str = "2024-12-31"

# ---------------------------------------------------------------------------
# Model identifiers
# ---------------------------------------------------------------------------
GEMINI_MODEL: str = "gemini-2.5-flash-lite"
FINBERT_MODEL: str = "ProsusAI/finbert"
