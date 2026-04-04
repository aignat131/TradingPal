"""
Backtest Engine — simulates TradingPal-AI signals over a historical period.

How it works
------------
1. Downloads daily OHLCV data for the full period via yfinance (~252 days for 1 year).
2. For each trading day (rolling window):
   - Calculates RSI(14) and SMA(20/50) on the preceding price history.
   - Looks up pre-processed FinSen sentiment from SentimentCache (parquet).
   - Produces a composite signal: BUY / SELL / NEUTRAL.
3. Simulates trades:
   - BUY signal  → open a long position (position sizing by risk%).
   - SELL signal → close any open position and record P&L.
4. Computes performance metrics:
   Total Return, Annualised Return, Sharpe Ratio, Max Drawdown, Win Rate.

Requires the sentiment cache to be built first:
    python utils/preprocess_finsen.py
"""
import logging
from datetime import datetime
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from analysis.technical_indicators import TechnicalAnalyzer
from backtesting.sentiment_cache import SentimentCache
from data.market_data import MarketData
import config

logger = logging.getLogger(__name__)


class BacktestEngine:
    """Runs historical signal simulations and computes performance statistics."""

    def __init__(self, sentiment_cache: Optional[SentimentCache] = None) -> None:
        self._market = MarketData()
        self._tech = TechnicalAnalyzer()
        self._cache = sentiment_cache or SentimentCache()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run_backtest(
        self,
        ticker: str,
        start_date: str = config.BACKTEST_START,
        end_date: str = config.BACKTEST_END,
        initial_capital: float = config.INITIAL_CAPITAL,
        risk_percent: float = config.DEFAULT_RISK_PERCENT,
        recurring_amount: float = 0.0,
        recurring_period: str = "monthly",
    ) -> Dict:
        """
        Run a full backtest for *ticker* between *start_date* and *end_date*.

        Returns a result dict with:
          signals      — DataFrame of daily signals and prices
          trades       — list of completed trade dicts
          equity_curve — list of (date, portfolio_value) tuples
          metrics      — performance metrics dict
        """
        # Fetch prices: fetch extra history for indicator warm-up
        df = self._market.get_historical_prices_range(
            ticker,
            start=self._offset_date(start_date, -120),
            end=end_date,
        )
        if df.empty:
            logger.warning("No price data for %s in range %s–%s.", ticker, start_date, end_date)
            return self._empty_result(ticker)

        # Flatten MultiIndex columns
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [c[0] for c in df.columns]

        signals_df = self.generate_signals_for_period(df, ticker, start_date, end_date)
        trades, equity_curve = self._simulate_trades(
            signals_df, initial_capital, risk_percent,
            recurring_amount=recurring_amount, recurring_period=recurring_period,
        )
        metrics = self.calculate_performance(trades, equity_curve, initial_capital)

        return {
            "ticker": ticker,
            "start_date": start_date,
            "end_date": end_date,
            "initial_capital": initial_capital,
            "signals": signals_df,
            "trades": trades,
            "buy_log": [],
            "equity_curve": equity_curve,
            "metrics": metrics,
            "strategy": "RSI+News",
        }

    def generate_signals_for_period(
        self,
        df: pd.DataFrame,
        ticker: str,
        start_date: str,
        end_date: str,
    ) -> pd.DataFrame:
        """
        Generate daily signals for each date in [start_date, end_date].
        Uses a rolling window of prior prices so each signal is realistic
        (no look-ahead bias).

        Returns a DataFrame with columns:
          date, open, high, low, close, volume,
          rsi, sma_short, sma_long, tech_signal, tech_score,
          hist_sentiment, composite_signal
        """
        close = df["Close"].squeeze()
        rows = []
        start_ts = pd.Timestamp(start_date)
        end_ts = pd.Timestamp(end_date)

        for i, date in enumerate(df.index):
            if date < start_ts or date > end_ts:
                continue
            # Use all prices UP TO (but not including) current day — no look-ahead
            hist_prices = close.iloc[:i].tolist()
            if len(hist_prices) < config.SMA_LONG + 5:
                continue

            rsi = self._tech.calculate_rsi(hist_prices)
            sma_s = self._tech.calculate_sma(hist_prices, config.SMA_SHORT)
            sma_l = self._tech.calculate_sma(hist_prices, config.SMA_LONG)
            current_price = float(close.iloc[i])
            tech_signal = self._tech.generate_signal(rsi, sma_s, sma_l, current_price)
            tech_score = self._tech.get_technical_score(rsi, current_price, sma_s, sma_l)

            # Historical sentiment (FinSen or simulated)
            hist_sent = self._get_hist_sentiment(ticker, date)

            # Composite: 60% technical, 40% sentiment
            composite_score = tech_score * 0.6 + hist_sent * 0.4
            threshold = config.COMPOSITE_SIGNAL_THRESHOLD
            if composite_score > threshold:
                composite_signal = "BUY"
            elif composite_score < -threshold:
                composite_signal = "SELL"
            else:
                composite_signal = "NEUTRAL"

            rows.append(
                {
                    "date": date,
                    "open": float(df["Open"].squeeze().iloc[i]),
                    "high": float(df["High"].squeeze().iloc[i]),
                    "low": float(df["Low"].squeeze().iloc[i]),
                    "close": current_price,
                    "volume": float(df["Volume"].squeeze().iloc[i]),
                    "rsi": round(rsi, 2),
                    "sma_short": round(sma_s, 4),
                    "sma_long": round(sma_l, 4),
                    "tech_signal": tech_signal,
                    "tech_score": round(tech_score, 4),
                    "hist_sentiment": round(hist_sent, 4),
                    "composite_signal": composite_signal,
                    "composite_score": round(composite_score, 4),
                }
            )

        return pd.DataFrame(rows).set_index("date") if rows else pd.DataFrame()

    def calculate_performance(
        self,
        trades: List[Dict],
        equity_curve: List[Dict],
        initial_capital: float,
    ) -> Dict:
        """
        Compute performance metrics from completed trades and equity curve.

        Metrics:
          total_return_pct, annualised_return_pct, sharpe_ratio,
          max_drawdown_pct, win_rate_pct, total_trades,
          profitable_trades, avg_profit, avg_loss
        """
        if not equity_curve:
            return self._zero_metrics()

        eq_values = [e["value"] for e in equity_curve]
        final_value = eq_values[-1]
        total_return = (final_value - initial_capital) / initial_capital * 100

        # Annualised return
        n_days = len(equity_curve)
        ann_return = ((final_value / initial_capital) ** (252 / max(n_days, 1)) - 1) * 100

        # Daily returns for Sharpe
        eq_series = pd.Series(eq_values)
        daily_ret = eq_series.pct_change().dropna()
        sharpe = (
            float(daily_ret.mean() / daily_ret.std() * np.sqrt(252))
            if daily_ret.std() > 0
            else 0.0
        )

        # Max drawdown
        roll_max = eq_series.cummax()
        drawdown = (eq_series - roll_max) / roll_max * 100
        max_drawdown = float(drawdown.min())

        # Trade statistics
        total_trades = len(trades)
        if total_trades == 0:
            win_rate = 0.0
            avg_profit = 0.0
            avg_loss = 0.0
        else:
            profits = [t["pnl"] for t in trades if t.get("pnl", 0) > 0]
            losses = [t["pnl"] for t in trades if t.get("pnl", 0) <= 0]
            win_rate = len(profits) / total_trades * 100
            avg_profit = sum(profits) / len(profits) if profits else 0.0
            avg_loss = sum(losses) / len(losses) if losses else 0.0

        return {
            "total_return_pct": round(total_return, 2),
            "annualised_return_pct": round(ann_return, 2),
            "sharpe_ratio": round(sharpe, 3),
            "max_drawdown_pct": round(max_drawdown, 2),
            "win_rate_pct": round(win_rate, 1),
            "total_trades": total_trades,
            "profitable_trades": len(profits) if total_trades else 0,
            "avg_profit": round(avg_profit, 2),
            "avg_loss": round(avg_loss, 2),
            "final_value": round(final_value, 2),
        }

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def run_dca_backtest(
        self,
        ticker: str,
        start_date: str = config.BACKTEST_START,
        end_date: str = config.BACKTEST_END,
        initial_capital: float = config.INITIAL_CAPITAL,
        frequency: str = "weekly",
        recurring_amount: float = 0.0,
        recurring_period: str = "monthly",
    ) -> Dict:
        """
        Run a Dollar-Cost Averaging backtest: buy a fixed amount on a regular schedule
        regardless of price or indicators. No signals — purely time-based.
        """
        df = self._market.get_historical_prices_range(
            ticker,
            start=self._offset_date(start_date, -120),
            end=end_date,
        )
        if df.empty:
            return self._empty_result(ticker)

        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [c[0] for c in df.columns]

        start_ts = pd.Timestamp(start_date)
        end_ts = pd.Timestamp(end_date)
        price_df = df[(df.index >= start_ts) & (df.index <= end_ts)].copy()
        if price_df.empty:
            return self._empty_result(ticker)

        # Build a minimal signals-like DataFrame with price columns only
        signals_df = pd.DataFrame({
            "open": price_df["Open"].squeeze(),
            "high": price_df["High"].squeeze(),
            "low": price_df["Low"].squeeze(),
            "close": price_df["Close"].squeeze(),
            "volume": price_df["Volume"].squeeze(),
        })

        trades, equity_curve, buy_log = self._simulate_dca_trades(
            signals_df, initial_capital, frequency,
            recurring_amount=recurring_amount, recurring_period=recurring_period,
        )
        metrics = self.calculate_performance(trades, equity_curve, initial_capital)

        return {
            "ticker": ticker,
            "start_date": start_date,
            "end_date": end_date,
            "initial_capital": initial_capital,
            "signals": signals_df,
            "trades": trades,
            "buy_log": buy_log,
            "equity_curve": equity_curve,
            "metrics": metrics,
            "strategy": "DCA",
        }

    def run_hybrid_backtest(
        self,
        ticker: str,
        start_date: str = config.BACKTEST_START,
        end_date: str = config.BACKTEST_END,
        initial_capital: float = config.INITIAL_CAPITAL,
        risk_percent: float = config.DEFAULT_RISK_PERCENT,
        frequency: str = "weekly",
        recurring_amount: float = 0.0,
        recurring_period: str = "monthly",
    ) -> Dict:
        """
        Run a Hybrid backtest: DCA base buys on a regular schedule, doubled when
        RSI+News signal is BUY, skipped when signal is SELL.
        """
        df = self._market.get_historical_prices_range(
            ticker,
            start=self._offset_date(start_date, -120),
            end=end_date,
        )
        if df.empty:
            return self._empty_result(ticker)

        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [c[0] for c in df.columns]

        signals_df = self.generate_signals_for_period(df, ticker, start_date, end_date)
        if signals_df.empty:
            return self._empty_result(ticker)

        trades, equity_curve, buy_log = self._simulate_hybrid_trades(
            signals_df, initial_capital, frequency,
            recurring_amount=recurring_amount, recurring_period=recurring_period,
        )
        metrics = self.calculate_performance(trades, equity_curve, initial_capital)

        return {
            "ticker": ticker,
            "start_date": start_date,
            "end_date": end_date,
            "initial_capital": initial_capital,
            "signals": signals_df,
            "trades": trades,
            "buy_log": buy_log,
            "equity_curve": equity_curve,
            "metrics": metrics,
            "strategy": "Hybrid",
        }

    def _simulate_trades(
        self,
        signals_df: pd.DataFrame,
        initial_capital: float,
        risk_percent: float,
        recurring_amount: float = 0.0,
        recurring_period: str = "monthly",
    ):
        """
        Iterate through signals and simulate a long-only portfolio.
        Supports optional recurring contributions added at regular intervals.
        Returns (trades_list, equity_curve_list).
        Each trade is a closed (entry + exit) pair.
        """
        if signals_df.empty:
            return [], []

        capital = initial_capital
        position_shares = 0.0
        entry_price = 0.0
        entry_date = None
        trades = []
        equity_curve = []

        contrib_dates: set = set()
        if recurring_amount > 0:
            contrib_dates = set(self._get_schedule_dates(signals_df.index, recurring_period))

        total_invested = initial_capital

        for date, row in signals_df.iterrows():
            close = row["close"]
            signal = row["composite_signal"]

            # Add recurring contribution before any trading decision
            if date in contrib_dates:
                capital += recurring_amount
                total_invested += recurring_amount

            # Portfolio value
            portfolio_value = capital + position_shares * close
            equity_curve.append({"date": date, "value": round(portfolio_value, 2), "total_invested": round(total_invested, 2)})

            if signal == "BUY" and position_shares == 0 and capital > 0:
                risk_amount = capital * (risk_percent / 100)
                stop_dist = close * 0.02
                if stop_dist > 0:
                    shares = min(risk_amount / stop_dist, capital / close)
                    position_shares = round(shares, 6)
                    cost = position_shares * close
                    capital -= cost
                    entry_price = close
                    entry_date = date
                    logger.debug("BUY  %s @ %.2f  shares=%.4f", date, close, position_shares)

            elif signal == "SELL" and position_shares > 0:
                proceeds = position_shares * close
                pnl = proceeds - position_shares * entry_price
                capital += proceeds
                trades.append({
                    "type": "Closed",
                    "entry_date": entry_date,
                    "exit_date": date,
                    "entry_price": round(entry_price, 4),
                    "exit_price": round(close, 4),
                    "shares": round(position_shares, 6),
                    "pnl": round(pnl, 2),
                    "return_pct": round(pnl / (position_shares * entry_price) * 100, 2),
                })
                logger.debug("SELL %s @ %.2f  P&L=%.2f", date, close, pnl)
                position_shares = 0.0
                entry_price = 0.0
                entry_date = None

        # Close any open position at end of period
        if position_shares > 0 and not signals_df.empty:
            last_close = signals_df["close"].iloc[-1]
            proceeds = position_shares * last_close
            pnl = proceeds - position_shares * entry_price
            capital += proceeds
            trades.append({
                "type": "Open at end",
                "entry_date": entry_date,
                "exit_date": signals_df.index[-1],
                "entry_price": round(entry_price, 4),
                "exit_price": round(last_close, 4),
                "shares": round(position_shares, 6),
                "pnl": round(pnl, 2),
                "return_pct": round(pnl / (position_shares * entry_price) * 100, 2),
            })

        return trades, equity_curve

    def _get_hist_sentiment(self, ticker: str, date) -> float:
        """Return average sentiment score for 30 days before *date* from cache."""
        try:
            end = pd.Timestamp(date)
            start = end - pd.Timedelta(days=30)
            return self._cache.get_window_score(ticker, start, end)
        except Exception:
            return 0.0

    @staticmethod
    def _offset_date(date_str: str, days: int) -> str:
        dt = datetime.strptime(date_str, "%Y-%m-%d") + pd.Timedelta(days=days)
        return dt.strftime("%Y-%m-%d")

    @staticmethod
    def _get_schedule_dates(
        trading_dates: pd.DatetimeIndex,
        frequency: str,
    ) -> List:
        """Return a list of scheduled buy dates from the actual trading day index."""
        step = 5 if frequency == "weekly" else 21
        return [trading_dates[i] for i in range(0, len(trading_dates), step)]

    def _simulate_dca_trades(
        self,
        signals_df: pd.DataFrame,
        initial_capital: float,
        frequency: str,
        recurring_amount: float = 0.0,
        recurring_period: str = "monthly",
    ):
        """
        Simulate a pure DCA strategy: buy fixed amount on a regular schedule.
        Returns (summary_trades, equity_curve, buy_log).
        - summary_trades: single closing record used for metrics
        - buy_log: list of individual buys for display in UI
        """
        if signals_df.empty:
            return [], [], []

        schedule = self._get_schedule_dates(signals_df.index, frequency)
        total_periods = max(len(schedule), 1)
        buy_amount = initial_capital / total_periods

        contrib_dates: set = set()
        if recurring_amount > 0:
            contrib_dates = set(self._get_schedule_dates(signals_df.index, recurring_period))

        cash = initial_capital
        total_shares = 0.0
        buy_log = []
        equity_curve = []
        schedule_set = set(schedule)
        total_invested = initial_capital

        for date, row in signals_df.iterrows():
            close = row["close"]

            if date in contrib_dates:
                cash += recurring_amount
                total_invested += recurring_amount

            if date in schedule_set and cash >= buy_amount * 0.9:
                spend = min(buy_amount, cash)
                shares = spend / close
                total_shares += shares
                cash -= spend
                buy_log.append({
                    "date": date,
                    "action": "BUY",
                    "shares_bought": round(shares, 6),
                    "price_paid": round(close, 4),
                    "amount_spent": round(spend, 2),
                    "total_shares": round(total_shares, 6),
                    "portfolio_value": round(cash + total_shares * close, 2),
                })

            portfolio_value = cash + total_shares * close
            equity_curve.append({"date": date, "value": round(portfolio_value, 2), "total_invested": round(total_invested, 2)})

        # Single summary trade for metrics computation
        summary_trades = []
        if buy_log and total_shares > 0:
            last_close = signals_df["close"].iloc[-1]
            total_invested = sum(b["amount_spent"] for b in buy_log)
            final_value = cash + total_shares * last_close
            pnl = final_value - initial_capital
            summary_trades = [{
                "type": "DCA Summary",
                "entry_date": buy_log[0]["date"],
                "exit_date": signals_df.index[-1],
                "entry_price": buy_log[0]["price_paid"],
                "exit_price": round(last_close, 4),
                "shares": round(total_shares, 6),
                "num_buys": len(buy_log),
                "total_invested": round(total_invested, 2),
                "pnl": round(pnl, 2),
                "return_pct": round(pnl / initial_capital * 100, 2),
            }]

        return summary_trades, equity_curve, buy_log

    def _simulate_hybrid_trades(
        self,
        signals_df: pd.DataFrame,
        initial_capital: float,
        frequency: str,
        recurring_amount: float = 0.0,
        recurring_period: str = "monthly",
    ):
        """
        Simulate Hybrid: DCA base buys, doubled on BUY signal, skipped on SELL.
        Returns (summary_trades, equity_curve, buy_log).
        """
        if signals_df.empty:
            return [], [], []

        schedule = self._get_schedule_dates(signals_df.index, frequency)
        total_periods = max(len(schedule), 1)
        base_amount = initial_capital / total_periods / 2

        contrib_dates: set = set()
        if recurring_amount > 0:
            contrib_dates = set(self._get_schedule_dates(signals_df.index, recurring_period))

        cash = initial_capital
        total_shares = 0.0
        buy_log = []
        equity_curve = []
        schedule_set = set(schedule)
        total_invested = initial_capital

        for date, row in signals_df.iterrows():
            close = row["close"]
            signal = row.get("composite_signal", "NEUTRAL")

            if date in contrib_dates:
                cash += recurring_amount
                total_invested += recurring_amount

            if date in schedule_set and cash > 0:
                if signal == "BUY":
                    spend = min(base_amount * 2, cash)
                    action_label = "BUY ×2 (signal boost)"
                elif signal == "SELL":
                    spend = 0.0
                    action_label = "SKIPPED (sell signal)"
                else:
                    spend = min(base_amount, cash)
                    action_label = "BUY (base)"

                if spend > 0:
                    shares = spend / close
                    total_shares += shares
                    cash -= spend
                    buy_log.append({
                        "date": date,
                        "action": action_label,
                        "signal": signal,
                        "shares_bought": round(shares, 6),
                        "price_paid": round(close, 4),
                        "amount_spent": round(spend, 2),
                        "total_shares": round(total_shares, 6),
                        "portfolio_value": round(cash + total_shares * close, 2),
                    })
                elif signal == "SELL":
                    buy_log.append({
                        "date": date,
                        "action": action_label,
                        "signal": signal,
                        "shares_bought": 0.0,
                        "price_paid": round(close, 4),
                        "amount_spent": 0.0,
                        "total_shares": round(total_shares, 6),
                        "portfolio_value": round(cash + total_shares * close, 2),
                    })

            portfolio_value = cash + total_shares * close
            equity_curve.append({"date": date, "value": round(portfolio_value, 2), "total_invested": round(total_invested, 2)})

        # Single summary trade for metrics
        summary_trades = []
        if buy_log and total_shares > 0:
            last_close = signals_df["close"].iloc[-1]
            cost_basis = initial_capital - cash
            pnl = total_shares * last_close - cost_basis
            executed = [b for b in buy_log if b["shares_bought"] > 0]
            summary_trades = [{
                "type": "Hybrid Summary",
                "entry_date": executed[0]["date"] if executed else signals_df.index[0],
                "exit_date": signals_df.index[-1],
                "entry_price": executed[0]["price_paid"] if executed else 0.0,
                "exit_price": round(last_close, 4),
                "shares": round(total_shares, 6),
                "num_buys": len(executed),
                "pnl": round(pnl, 2),
                "return_pct": round(pnl / max(initial_capital, 1) * 100, 2),
            }]

        return summary_trades, equity_curve, buy_log

    def _empty_result(self, ticker: str) -> Dict:
        return {
            "ticker": ticker,
            "start_date": "",
            "end_date": "",
            "initial_capital": config.INITIAL_CAPITAL,
            "signals": pd.DataFrame(),
            "trades": [],
            "equity_curve": [],
            "metrics": self._zero_metrics(),
        }

    @staticmethod
    def _zero_metrics() -> Dict:
        return {
            "total_return_pct": 0.0,
            "annualised_return_pct": 0.0,
            "sharpe_ratio": 0.0,
            "max_drawdown_pct": 0.0,
            "win_rate_pct": 0.0,
            "total_trades": 0,
            "profitable_trades": 0,
            "avg_profit": 0.0,
            "avg_loss": 0.0,
            "final_value": config.INITIAL_CAPITAL,
        }
