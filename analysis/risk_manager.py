"""
Risk Manager — position sizing, stop-loss, take-profit, and risk validation.
"""
import logging

import config

logger = logging.getLogger(__name__)


class RiskManager:
    """Calculates risk parameters for a proposed trade."""

    # ------------------------------------------------------------------
    # Position sizing
    # ------------------------------------------------------------------

    def calculate_position_size(
        self,
        portfolio_value: float,
        risk_percent: float,
        stop_loss_percent: float,
        entry_price: float,
    ) -> float:
        """
        Return the maximum number of shares to buy given the risk budget.

        Formula: shares = (portfolio * risk%) / (entry * stop_loss%)
        """
        if entry_price <= 0 or stop_loss_percent <= 0:
            return 0.0
        risk_amount = portfolio_value * (risk_percent / 100)
        loss_per_share = entry_price * (stop_loss_percent / 100)
        shares = risk_amount / loss_per_share
        return round(shares, 4)

    # ------------------------------------------------------------------
    # Stop-loss / take-profit
    # ------------------------------------------------------------------

    def calculate_stop_loss(
        self,
        entry_price: float,
        atr: float,
        risk_multiplier: float = 1.5,
    ) -> float:
        """Return stop-loss price = entry - ATR * multiplier."""
        return round(max(0.0, entry_price - atr * risk_multiplier), 4)

    def calculate_take_profit(
        self,
        entry_price: float,
        atr: float,
        reward_risk_ratio: float = 2.0,
    ) -> float:
        """Return take-profit price = entry + ATR * ratio."""
        return round(entry_price + atr * reward_risk_ratio, 4)

    def calculate_risk_reward_ratio(
        self, entry: float, stop_loss: float, take_profit: float
    ) -> float:
        """Return reward/risk ratio (e.g. 2.0 means 2:1)."""
        risk = entry - stop_loss
        reward = take_profit - entry
        if risk <= 0:
            return 0.0
        return round(reward / risk, 2)

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def validate_position(
        self,
        amount: float,
        portfolio_value: float,
        max_position_pct: float = config.MAX_POSITION_PCT,
    ) -> bool:
        """Return True if *amount* does not exceed *max_position_pct* of portfolio."""
        if portfolio_value <= 0:
            return False
        return (amount / portfolio_value) <= max_position_pct

    # ------------------------------------------------------------------
    # ATR estimation
    # ------------------------------------------------------------------

    def estimate_atr(self, entry_price: float, volatility_pct: float) -> float:
        """
        Estimate Average True Range from annualised volatility.
        Daily move ≈ entry * vol% / sqrt(252)
        """
        if volatility_pct <= 0:
            return entry_price * 0.02  # default 2%
        daily_move = entry_price * (volatility_pct / 100) / (252 ** 0.5)
        return round(daily_move, 4)
