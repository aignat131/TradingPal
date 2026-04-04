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

_TRADE_OPPORTUNITIES_PROMPT = """\
You are TradingPal-AI, a beginner-friendly stock advisor.

=== USER CONTEXT ===
Ticker:            {ticker}
Investment Amount: ${investment_amount:,.2f}
Trading Frequency: {trade_frequency}
Time Horizon:      {horizon}
Recurring Income:  {recurring_info}

=== CURRENT SIGNALS ===
Technical Signal:  {technical_signal}
RSI(14):           {rsi:.1f}
Sentiment:         {sentiment_label}

=== PERSONAL CONTEXT ===
{extra_context}

=== TASK ===
Generate 3-5 specific trade actions the user could take to grow their portfolio.
Use plain English — no jargon. For long-term horizons (>3 months) suggest
weekly or monthly moves, not intraday trades.
If the user has recurring income, factor it into the strategy (e.g. suggest DCA entry points).
If the personal context mentions existing holdings, account for them.

Return ONLY valid JSON array (no markdown fences), exactly:
[
  {{"stock": "TICKER (Company Name)", "action": "BUY", "reason": "plain English reason under 20 words", "price_range": "e.g. $180-$185 or current market price"}},
  ...
]
"""

_STOCK_DISCOVERY_PROMPT = """\
You are TradingPal-AI. The user is currently looking at {ticker}.
They have approximately {remaining_pct:.0f}% of their portfolio uninvested.

Based on current market trends and growth potential, suggest 2-3 OTHER stocks or ETFs
(not {ticker}) that a beginner investor could consider adding to their portfolio.
Choose assets with strong recent momentum, positive news, or solid fundamentals.
Keep it simple — one reason per stock in plain English.

Return ONLY valid JSON array (no markdown fences):
[
  {{"ticker": "SYMBOL", "reason": "one plain-English sentence about why this looks promising"}},
  ...
]
"""

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
Recurring Income:  {recurring_info}

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

=== PERSONAL CONTEXT ===
{extra_context}

=== TASK ===
Based on ALL signals above and the personal context, return ONLY valid JSON (no markdown fences) in this exact format:
{{
  "decision": "BUY" | "HOLD" | "SELL",
  "confidence": <float 0.0-1.0>,
  "explanation": "<2-4 sentence rationale>",
  "key_risks": ["<risk 1>", "<risk 2>", "<risk 3>"],
  "next_steps": ["<action 1>", "<action 2>", "<action 3>"]
}}
"""

_PORTFOLIO_PROMPT = """\
You are TradingPal-AI, a portfolio advisor helping a beginner investor.

=== USER PROFILE ===
Budget:          ${budget:,.2f}
Risk Tolerance:  {risk}
Time Horizon:    {horizon}
Recurring:       {recurring_info}

=== AVAILABLE ASSETS & TECHNICAL SIGNALS ===
{signals_table}

=== TASK ===
Based on the user's profile and the technical signals above, rank the top 10 assets from
highest profit potential to lowest FOR THIS SPECIFIC USER.

Consider:
- Risk tolerance: Low → prefer stable large-caps & ETFs; High → OK with volatile crypto/growth stocks
- Time horizon: Intraday → prefer strong short-term momentum; Long-term → prefer fundamentals + trend
- RSI <30 = oversold (buy opportunity), RSI >70 = overbought (exercise caution)
- SMA trend: bullish = upward trend, bearish = downward trend
- Technical score: -1 to +1, higher = stronger buy signal

Return ONLY valid JSON array (no markdown fences), exactly 10 items sorted highest to lowest:
[
  {{
    "ticker": "SYMBOL",
    "signal": "BUY" | "SELL" | "HOLD",
    "score": <float 0.0-1.0, profit potential for this user>,
    "reasoning": "<2 concise plain-English sentences explaining why>",
    "suggested_allocation": <float, % of budget to allocate if BUY, else 0>
  }},
  ...
]
"""

_CONTEXT_ADVICE_PROMPT = """\
You are TradingPal-AI, a personalized financial advisor.

=== USER SITUATION ===
Ticker they are analysing: {ticker}
Investment amount:         ${investment_amount:,.2f}
Risk tolerance:            {risk_tolerance}
Time horizon:              {horizon}
Recurring income/savings:  {recurring_info}

=== CURRENT MARKET SIGNALS ===
Technical signal: {technical_signal}
RSI(14):          {rsi:.1f}
Sentiment:        {sentiment_label}

=== WHAT THE USER TOLD US ===
{extra_context}

=== TASK ===
Give specific, personalized advice that directly addresses what the user mentioned.
- If they say they already own shares (e.g. "I have $500 of AMZN"), factor that into your advice.
- If they mention a savings goal, relate the strategy to that goal.
- If they mention news or events, include that in your analysis.
- Be concrete, practical, and use plain English. No jargon.

Return ONLY valid JSON (no markdown fences):
{{
  "personal_advice": "<2-4 sentence advice tailored to their specific situation>",
  "context_risks": ["<risk specific to their situation>", "<risk 2>", "<risk 3>"],
  "context_actions": ["<concrete action 1 for their situation>", "<action 2>", "<action 3>"]
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
        extra_context: str = "",
        recurring_amount: float = 0.0,
        recurring_period: str = "monthly",
    ) -> Dict:
        """
        Return the final recommendation dict:
        {decision, confidence, explanation, key_risks, next_steps}
        """
        recurring_info = (
            f"${recurring_amount:,.0f} added {recurring_period}"
            if recurring_amount > 0
            else "None"
        )
        prompt = _PROMPT_TEMPLATE.format(
            ticker=ticker,
            investment_amount=investment_amount,
            risk_tolerance=risk_tolerance,
            recurring_info=recurring_info,
            extra_context=extra_context.strip() if extra_context else "None provided.",
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

    def generate_trade_opportunities(
        self,
        ticker: str,
        investment_amount: float,
        trade_frequency: str,
        horizon: str,
        technical_signal: Dict,
        sentiment_signal: Dict,
        extra_context: str = "",
        recurring_amount: float = 0.0,
        recurring_period: str = "monthly",
    ) -> list:
        """
        Return a list of actionable trade opportunities in plain English.
        Each item: {stock, action, reason, price_range}
        """
        score = sentiment_signal.get("score", 0.0)
        if score > 0.1:
            sentiment_label = "positive"
        elif score < -0.1:
            sentiment_label = "negative"
        else:
            sentiment_label = "mixed"

        recurring_info = (
            f"${recurring_amount:,.0f} added {recurring_period}"
            if recurring_amount > 0
            else "None"
        )
        prompt = _TRADE_OPPORTUNITIES_PROMPT.format(
            ticker=ticker,
            investment_amount=investment_amount,
            trade_frequency=trade_frequency,
            horizon=horizon,
            recurring_info=recurring_info,
            extra_context=extra_context.strip() if extra_context else "None provided.",
            technical_signal=technical_signal.get("signal", "NEUTRAL"),
            rsi=technical_signal.get("rsi", 50.0),
            sentiment_label=sentiment_label,
        )

        if self._gemini_available:
            result = self._call_gemini_list(prompt)
            if result:
                validated = []
                for item in result:
                    if isinstance(item, dict) and "stock" in item and "action" in item:
                        validated.append({
                            "stock": item.get("stock", ticker),
                            "action": item.get("action", "HOLD").upper(),
                            "reason": item.get("reason", ""),
                            "price_range": item.get("price_range", "current market price"),
                        })
                if validated:
                    return validated

        # Deterministic fallback
        sig = technical_signal.get("signal", "NEUTRAL")
        action_map = {"BUY": "BUY", "SELL": "SELL"}
        action = action_map.get(sig, "HOLD")
        reason_map = {
            "BUY": "Technical indicators suggest a buying opportunity right now",
            "SELL": "Technical indicators suggest caution — consider reducing exposure",
            "HOLD": "No strong signal — hold your current position and wait",
        }
        return [{"stock": ticker, "action": action, "reason": reason_map[action], "price_range": "current market price"}]

    def generate_context_advice(
        self,
        ticker: str,
        investment_amount: float,
        risk_tolerance: str,
        horizon: str,
        technical_signal: Dict,
        sentiment_signal: Dict,
        extra_context: str,
        recurring_amount: float = 0.0,
        recurring_period: str = "monthly",
    ) -> Optional[Dict]:
        """
        Generate personalized advice based on the user's extra context.
        Returns {personal_advice, context_risks, context_actions} or None.
        """
        if not extra_context or not extra_context.strip():
            return None

        score = sentiment_signal.get("score", 0.0)
        sentiment_label = "positive" if score > 0.1 else ("negative" if score < -0.1 else "mixed")
        recurring_info = (
            f"${recurring_amount:,.0f} added {recurring_period}"
            if recurring_amount > 0
            else "None"
        )

        prompt = _CONTEXT_ADVICE_PROMPT.format(
            ticker=ticker,
            investment_amount=investment_amount,
            risk_tolerance=risk_tolerance,
            horizon=horizon,
            recurring_info=recurring_info,
            technical_signal=technical_signal.get("signal", "NEUTRAL"),
            rsi=technical_signal.get("rsi", 50.0),
            sentiment_label=sentiment_label,
            extra_context=extra_context.strip(),
        )

        if self._gemini_available:
            result = self._call_gemini(prompt)
            if result and "personal_advice" in result:
                return result

        return None

    def generate_portfolio_recommendations(
        self,
        budget: float,
        risk: str,
        horizon: str,
        signals: Dict[str, Dict],
        recurring_amount: float = 0.0,
        recurring_period: str = "monthly",
    ) -> list:
        """
        Rank all provided tickers by profit potential for the user.
        Returns a list of up to 10 dicts: {ticker, signal, score, reasoning, suggested_allocation}
        """
        recurring_info = (
            f"${recurring_amount:,.0f} added {recurring_period}"
            if recurring_amount > 0
            else "None"
        )

        # Build signals table string
        rows = ["Ticker | Signal | Tech Score | RSI  | SMA Trend | Volatility"]
        rows.append("-" * 65)
        for ticker, sig in signals.items():
            rows.append(
                f"{ticker:<8} | {sig.get('signal', 'NEUTRAL'):<6} | "
                f"{sig.get('score', 0.0):+.3f}      | "
                f"{sig.get('rsi', 50.0):<5.1f} | "
                f"{sig.get('sma_trend', 'neutral'):<9} | "
                f"{sig.get('volatility', 0.0):.1f}%"
            )
        signals_table = "\n".join(rows)

        prompt = _PORTFOLIO_PROMPT.format(
            budget=budget,
            risk=risk,
            horizon=horizon,
            recurring_info=recurring_info,
            signals_table=signals_table,
        )

        if self._gemini_available:
            result = self._call_gemini_list(prompt)
            if result:
                validated = []
                for item in result:
                    if isinstance(item, dict) and "ticker" in item and "signal" in item:
                        validated.append({
                            "ticker": item.get("ticker", "").upper(),
                            "signal": item.get("signal", "HOLD").upper(),
                            "score": float(item.get("score", 0.5)),
                            "reasoning": item.get("reasoning", ""),
                            "suggested_allocation": float(item.get("suggested_allocation", 0)),
                        })
                if validated:
                    return validated[:10]

        # Deterministic fallback: sort by technical score
        items = []
        for ticker, sig in signals.items():
            tech_score = sig.get("score", 0.0)
            rsi = sig.get("rsi", 50.0)
            signal = sig.get("signal", "HOLD")
            profit_potential = round((tech_score + 1) / 2, 3)  # map [-1,1] → [0,1]
            items.append({
                "ticker": ticker,
                "signal": signal,
                "score": profit_potential,
                "reasoning": (
                    f"Technical score {tech_score:+.2f} with RSI at {rsi:.0f}. "
                    f"SMA trend is {sig.get('sma_trend', 'neutral')}."
                ),
                "suggested_allocation": round(100 / len(signals), 1) if signal == "BUY" else 0,
            })
        items.sort(key=lambda x: x["score"], reverse=True)
        return items[:10]

    def suggest_new_stocks(
        self,
        current_ticker: str,
        investment_amount: float,
        technical_signal: Dict,
        sentiment_signal: Dict,
        remaining_pct: float = 60.0,
    ) -> list:
        """
        Suggest 2-3 other stocks/ETFs the user could consider with remaining capital.
        Returns list of {ticker, reason}.
        """
        prompt = _STOCK_DISCOVERY_PROMPT.format(
            ticker=current_ticker,
            remaining_pct=remaining_pct,
        )

        if self._gemini_available:
            result = self._call_gemini_list(prompt)
            if result:
                validated = []
                for item in result:
                    if isinstance(item, dict) and "ticker" in item:
                        if item["ticker"].upper() != current_ticker.upper():
                            validated.append({
                                "ticker": item.get("ticker", "").upper(),
                                "reason": item.get("reason", ""),
                            })
                if validated:
                    return validated[:3]

        return [
            {"ticker": "SPY", "reason": "Broad US market ETF — steady long-term growth with low risk"},
            {"ticker": "QQQ", "reason": "Top tech companies ETF — strong growth potential"},
        ]

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

    def _call_gemini_list(self, prompt: str) -> Optional[list]:
        """Like _call_gemini but expects a JSON array at the top level."""
        try:
            message = self._client.models.generate_content(
                model=config.GEMINI_MODEL,
                contents=prompt,
            )
            raw = message.text.strip()
            raw = re.sub(r"^```(?:json)?\s*", "", raw, flags=re.MULTILINE)
            raw = re.sub(r"\s*```$", "", raw, flags=re.MULTILINE)
            result = json.loads(raw)
            if isinstance(result, list):
                return result
            # Gemini may wrap the list in a dict key
            if isinstance(result, dict):
                for v in result.values():
                    if isinstance(v, list):
                        return v
            return None
        except Exception as exc:
            logger.error("Gemini list call failed: %s", exc)
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
