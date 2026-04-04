"""
Manager Agent — Gemini-powered orchestrator that synthesises all signals
and produces the final investment recommendation.

Gemini API pattern (preserved from working codebase):
    from google import genai
    client = genai.Client(api_key=...)
    message = client.models.generate_content(model="gemini-2.5-flash-lite", contents=prompt)
    response_text = message.text
"""
import json
import logging
import re
from typing import Any, Dict, Optional

import config

logger = logging.getLogger(__name__)

_FALLBACK_DECISION = {
    "decision": "HOLD",
    "confidence": 0.5,
    "explanation": (
        "Gemini API is not configured. "
        "This recommendation is based on rule-based signal aggregation only."
    ),
    "key_risks": ["API key missing — AI analysis unavailable."],
    "next_steps": ["Configure GEMINI_API_KEY and retry for full AI analysis."],
}

_PROMPT_TEMPLATE = """\
You are TradingPal-AI, an expert multi-agent investment advisor.
Analyse the following signals and produce a final investment recommendation.

=== ASSET ===
Ticker:            {ticker}
Investment Amount: ${investment_amount:,.2f}
Risk Tolerance:    {risk_tolerance}

=== TECHNICAL SIGNALS ===
Signal:            {technical_signal}
Technical Score:   {technical_score:.2f}  (range -1 to +1)
RSI(14):           {rsi:.1f}
SMA Trend:         {sma_trend}
Volatility:        {volatility:.1f}%

=== SENTIMENT SIGNALS ===
Aggregate Score:   {sentiment_score:.2f}  (range -1 to +1)
Trend:             {sentiment_trend}
News Items:        {news_count}
Historical:        {historical_context}

=== RISK PLAN ===
Position Size:     {position_size:.4f} shares
Stop-Loss:         ${stop_loss:.2f}
Take-Profit 1:     ${take_profit_1:.2f}
Take-Profit 2:     ${take_profit_2:.2f}
Risk/Reward:       {risk_reward_ratio:.2f}
Max Loss:          ${max_loss:.2f}

=== TASK ===
Based on ALL signals above, return ONLY valid JSON (no markdown fences) in this exact format:
{{
  "decision": "BUY" | "HOLD" | "SELL",
  "confidence": <float 0.0-1.0>,
  "explanation": "<2-4 sentence rationale>",
  "key_risks": ["<risk 1>", "<risk 2>", "<risk 3>"],
  "next_steps": ["<action 1>", "<action 2>", "<action 3>"]
}}
"""


class ManagerAgent:
    """
    Orchestrates technical, sentiment, and risk agents.
    Uses Gemini to produce the final recommendation.
    """

    def __init__(self) -> None:
        self._client = None
        self._gemini_available = False
        self._init_gemini()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate_recommendation(
        self,
        ticker: str,
        investment_amount: float,
        risk_tolerance: str,
        technical_signal: Dict,
        sentiment_signal: Dict,
        risk_plan: Dict,
    ) -> Dict:
        """
        Return the final recommendation dict:
        {decision, confidence, explanation, key_risks, next_steps}
        """
        prompt = _PROMPT_TEMPLATE.format(
            ticker=ticker,
            investment_amount=investment_amount,
            risk_tolerance=risk_tolerance,
            technical_signal=technical_signal.get("signal", "NEUTRAL"),
            technical_score=technical_signal.get("score", 0.0),
            rsi=technical_signal.get("rsi", 50.0),
            sma_trend=technical_signal.get("sma_trend", "neutral"),
            volatility=technical_signal.get("volatility", 0.0),
            sentiment_score=sentiment_signal.get("score", 0.0),
            sentiment_trend=sentiment_signal.get("trend", "stable"),
            news_count=sentiment_signal.get("news_count", 0),
            historical_context=sentiment_signal.get("historical_context", "N/A"),
            position_size=risk_plan.get("position_size", 0.0),
            stop_loss=risk_plan.get("stop_loss", 0.0),
            take_profit_1=risk_plan.get("take_profit_1", 0.0),
            take_profit_2=risk_plan.get("take_profit_2", 0.0),
            risk_reward_ratio=risk_plan.get("risk_reward_ratio", 0.0),
            max_loss=risk_plan.get("max_loss", 0.0),
        )

        if self._gemini_available:
            result = self._call_gemini(prompt)
            if result:
                return result

        # Deterministic fallback
        return self._deterministic_recommendation(
            technical_signal, sentiment_signal, ticker
        )

    def synthesize_signals(
        self,
        technical: Dict,
        sentiment: Dict,
        risk: Dict,
    ) -> float:
        """
        Return a composite score in [-1, 1] aggregating all signals.
        Weights: technical 50%, sentiment 30%, risk 20%.
        """
        t_score = technical.get("score", 0.0)
        s_score = sentiment.get("score", 0.0)
        r_rr = risk.get("risk_reward_ratio", 1.0)
        r_score = min(1.0, (r_rr - 1) / 2)  # map RR ratio to [-1,1]
        return round(t_score * 0.5 + s_score * 0.3 + r_score * 0.2, 4)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _init_gemini(self) -> None:
        if not config.GEMINI_API_KEY:
            logger.warning("GEMINI_API_KEY not set — AI mode disabled.")
            return
        try:
            from google import genai

            self._client = genai.Client(api_key=config.GEMINI_API_KEY)
            self._gemini_available = True
            logger.info("Gemini client initialised (model: %s).", config.GEMINI_MODEL)
        except Exception as exc:
            logger.error("Failed to init Gemini client: %s", exc)

    def _call_gemini(self, prompt: str) -> Optional[Dict]:
        try:
            message = self._client.models.generate_content(
                model=config.GEMINI_MODEL,
                contents=prompt,
            )
            raw = message.text.strip()
            # Strip markdown code fences if present
            raw = re.sub(r"^```(?:json)?\s*", "", raw, flags=re.MULTILINE)
            raw = re.sub(r"\s*```$", "", raw, flags=re.MULTILINE)
            return json.loads(raw)
        except json.JSONDecodeError as exc:
            logger.warning("Gemini returned invalid JSON: %s", exc)
            return self._extract_partial(message.text if "message" in dir() else "")
        except Exception as exc:
            logger.error("Gemini API call failed: %s", exc)
            return None

    def _extract_partial(self, text: str) -> Optional[Dict]:
        """Best-effort extraction if JSON parsing fails."""
        match = re.search(r"\b(BUY|SELL|HOLD)\b", text, re.IGNORECASE)
        if not match:
            return None
        decision = match.group(1).upper()
        return {
            "decision": decision,
            "confidence": 0.6,
            "explanation": text[:500] if text else "No explanation available.",
            "key_risks": ["Full JSON parse failed — partial extraction used."],
            "next_steps": ["Review raw Gemini output for details."],
        }

    def _deterministic_recommendation(
        self,
        technical: Dict,
        sentiment: Dict,
        ticker: str,
    ) -> Dict:
        """Rule-based fallback when Gemini is unavailable."""
        composite = self.synthesize_signals(technical, sentiment, {})
        if composite > 0.2:
            decision, confidence = "BUY", round(0.5 + composite * 0.5, 2)
        elif composite < -0.2:
            decision, confidence = "SELL", round(0.5 + abs(composite) * 0.5, 2)
        else:
            decision, confidence = "HOLD", 0.5

        return {
            "decision": decision,
            "confidence": min(confidence, 0.95),
            "explanation": (
                f"Rule-based analysis for {ticker}: "
                f"Technical score {technical.get('score', 0):.2f}, "
                f"Sentiment score {sentiment.get('score', 0):.2f}. "
                f"Composite signal: {composite:.2f}."
            ),
            "key_risks": [
                "AI analysis unavailable — results are rule-based only.",
                f"Volatility: {technical.get('volatility', 0):.1f}%",
                f"Sentiment trend: {sentiment.get('trend', 'stable')}",
            ],
            "next_steps": [
                f"Consider {decision.lower()}ing based on technical + sentiment alignment.",
                "Set stop-loss per the risk plan below.",
                "Monitor news flow for sentiment shifts.",
            ],
        }
