"""
Risk Agent — wraps RiskManager to compute position sizing and risk parameters
for a proposed trade.
"""
import logging
from typing import Dict

from analysis.risk_manager import RiskManager
from data.market_data import MarketData
import config

logger = logging.getLogger(__name__)

_RISK_PERCENT_MAP = {
    "low": 1.0,
    "medium": config.DEFAULT_RISK_PERCENT,
    "high": config.MAX_RISK_PERCENT,
}


class RiskAgent:
    """Calculates the risk plan for a trade given investment amount and tolerance."""

    def __init__(self) -> None:
        self._risk = RiskManager()
        self._market = MarketData()

    def analyze(
        self,
        ticker: str,
        investment_amount: float,
        risk_tolerance: str,
    ) -> Dict:
        """
        Return risk plan dict:
        {position_size, stop_loss, take_profit_1, take_profit_2,
         risk_reward_ratio, max_loss, valid_position}
        """
        snapshot = self._market.get_current_price(ticker)
        entry_price = snapshot.get("price", 0.0)
        volatility_pct = self._market.calculate_volatility(ticker)

        risk_pct = _RISK_PERCENT_MAP.get(risk_tolerance.lower(), config.DEFAULT_RISK_PERCENT)
        atr = self._risk.estimate_atr(entry_price, volatility_pct)

        stop_loss = self._risk.calculate_stop_loss(entry_price, atr, risk_multiplier=1.5)
        take_profit_1 = self._risk.calculate_take_profit(entry_price, atr, reward_risk_ratio=2.0)
        take_profit_2 = self._risk.calculate_take_profit(entry_price, atr, reward_risk_ratio=3.0)

        stop_loss_pct = ((entry_price - stop_loss) / entry_price * 100) if entry_price else 2.0
        position_size = self._risk.calculate_position_size(
            investment_amount, risk_pct, stop_loss_pct, entry_price
        )
        rrr = self._risk.calculate_risk_reward_ratio(entry_price, stop_loss, take_profit_1)
        max_loss = round(position_size * (entry_price - stop_loss), 2) if entry_price > stop_loss else 0.0
        valid = self._risk.validate_position(investment_amount, investment_amount * 4)

        return {
            "entry_price": round(entry_price, 4),
            "position_size": position_size,
            "stop_loss": stop_loss,
            "take_profit_1": take_profit_1,
            "take_profit_2": take_profit_2,
            "risk_reward_ratio": rrr,
            "max_loss": max_loss,
            "valid_position": valid,
            "risk_percent": risk_pct,
            "atr": round(atr, 4),
        }
