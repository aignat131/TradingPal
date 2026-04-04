"""
Accuracy Reporter — compares backtest predictions with actual price movements
and generates accuracy metrics and exportable reports.
"""
import csv
import io
import logging
from typing import Dict, List, Optional, Tuple

import pandas as pd

logger = logging.getLogger(__name__)


class AccuracyReporter:
    """Evaluates how well the model's directional calls matched reality."""

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def compare_with_reality(
        self,
        signals_df: pd.DataFrame,
        lookahead_days: int = 5,
    ) -> pd.DataFrame:
        """
        For each signal in *signals_df*, check whether the price moved in the
        predicted direction over the next *lookahead_days* trading days.

        Adds columns: actual_direction, correct, forward_return_pct
        Returns an augmented DataFrame.
        """
        if signals_df.empty:
            return signals_df

        df = signals_df.copy()
        df["forward_return_pct"] = (
            df["close"].shift(-lookahead_days) / df["close"] - 1
        ) * 100
        df["actual_direction"] = df["forward_return_pct"].apply(
            lambda r: "UP" if r > 0.5 else ("DOWN" if r < -0.5 else "FLAT")
        )
        df["predicted_direction"] = df["composite_signal"].map(
            {"BUY": "UP", "SELL": "DOWN", "NEUTRAL": "FLAT"}
        )
        df["correct"] = df["actual_direction"] == df["predicted_direction"]
        return df

    def calculate_accuracy_metrics(
        self,
        augmented_df: pd.DataFrame,
    ) -> Dict:
        """
        Return precision, recall, F1, and overall accuracy for BUY/SELL calls.
        Only evaluated on non-NEUTRAL signals.
        """
        if augmented_df.empty or "correct" not in augmented_df.columns:
            return self._zero_metrics()

        active = augmented_df[
            augmented_df["composite_signal"].isin(["BUY", "SELL"])
        ].dropna(subset=["correct"])

        if active.empty:
            return self._zero_metrics()

        total = len(active)
        correct = active["correct"].sum()
        accuracy = correct / total * 100

        # Per-class metrics
        buy_df = active[active["composite_signal"] == "BUY"]
        sell_df = active[active["composite_signal"] == "SELL"]

        buy_precision = (
            buy_df["correct"].sum() / len(buy_df) * 100 if len(buy_df) > 0 else 0.0
        )
        sell_precision = (
            sell_df["correct"].sum() / len(sell_df) * 100 if len(sell_df) > 0 else 0.0
        )

        return {
            "overall_accuracy_pct": round(accuracy, 1),
            "buy_signals": len(buy_df),
            "buy_precision_pct": round(buy_precision, 1),
            "sell_signals": len(sell_df),
            "sell_precision_pct": round(sell_precision, 1),
            "total_evaluated": total,
        }

    def generate_report(
        self,
        backtest_result: Dict,
        lookahead_days: int = 5,
    ) -> Tuple[str, Dict]:
        """
        Generate a CSV report string and accuracy metrics dict.

        Returns: (csv_string, accuracy_metrics_dict)
        """
        signals_df = backtest_result.get("signals", pd.DataFrame())
        trades = backtest_result.get("trades", [])
        metrics = backtest_result.get("metrics", {})

        augmented = self.compare_with_reality(signals_df, lookahead_days)
        accuracy = self.calculate_accuracy_metrics(augmented)

        # Build CSV
        output = io.StringIO()
        writer = csv.writer(output)

        # Header section
        writer.writerow(["TradingPal-AI Backtest Report"])
        writer.writerow(["Ticker", backtest_result.get("ticker", "")])
        writer.writerow(
            ["Period", f"{backtest_result.get('start_date','')} to {backtest_result.get('end_date','')}"]
        )
        writer.writerow(["Initial Capital", f"${backtest_result.get('initial_capital', 0):,.2f}"])
        writer.writerow([])

        # Performance metrics
        writer.writerow(["=== PERFORMANCE METRICS ==="])
        for k, v in metrics.items():
            writer.writerow([k.replace("_", " ").title(), v])
        writer.writerow([])

        # Accuracy metrics
        writer.writerow(["=== ACCURACY METRICS ==="])
        for k, v in accuracy.items():
            writer.writerow([k.replace("_", " ").title(), v])
        writer.writerow([])

        # Trades
        if trades:
            writer.writerow(["=== TRADES ==="])
            writer.writerow(
                ["Entry Date", "Exit Date", "Entry Price", "Exit Price",
                 "Shares", "P&L ($)", "Return (%)"]
            )
            for t in trades:
                writer.writerow(
                    [
                        t.get("entry_date", ""),
                        t.get("exit_date", ""),
                        t.get("entry_price", ""),
                        t.get("exit_price", ""),
                        t.get("shares", ""),
                        t.get("pnl", ""),
                        t.get("return_pct", ""),
                    ]
                )
            writer.writerow([])

        # Daily signals
        if not augmented.empty:
            writer.writerow(["=== DAILY SIGNALS ==="])
            cols = ["close", "rsi", "tech_signal", "composite_signal",
                    "forward_return_pct", "correct"]
            available = [c for c in cols if c in augmented.columns]
            writer.writerow(["Date"] + available)
            for date, row in augmented[available].iterrows():
                writer.writerow([date.strftime("%Y-%m-%d") if hasattr(date, "strftime") else date] + list(row))

        return output.getvalue(), accuracy

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _zero_metrics() -> Dict:
        return {
            "overall_accuracy_pct": 0.0,
            "buy_signals": 0,
            "buy_precision_pct": 0.0,
            "sell_signals": 0,
            "sell_precision_pct": 0.0,
            "total_evaluated": 0,
        }
