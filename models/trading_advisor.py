import os
import re
import anthropic

_client = None


def _get_client():
    global _client
    if _client is None:
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError("ANTHROPIC_API_KEY is not set in environment variables.")
        _client = anthropic.Anthropic(api_key=api_key)
    return _client


def _load_prompt_template() -> str:
    prompt_path = os.path.join(os.path.dirname(__file__), "..", "prompts", "advisor_prompt.txt")
    with open(prompt_path, "r", encoding="utf-8") as f:
        return f.read()


def get_trade_advice(
    ticker: str,
    asset_type: str,
    direction: str,
    amount: float,
    risk_tolerance: str,
    horizon: str,
    market: dict,
    risk_summary: dict,
    extra_context: str = "",
) -> tuple[str, str]:
    """
    Returns (advice_text, verdict) where verdict is one of: BUY, SELL, HOLD, AVOID
    """
    template = _load_prompt_template()

    warnings_text = (
        "\n".join(f"- {w}" for w in risk_summary["warnings"])
        if risk_summary["warnings"]
        else "None"
    )

    prompt = template.format(
        ticker=ticker,
        asset_type=asset_type.capitalize(),
        direction=direction.capitalize(),
        amount=f"{amount:,.2f}",
        risk_tolerance=risk_tolerance.capitalize(),
        horizon=horizon.capitalize(),
        price=f"{market.get('price', 0):,.2f}",
        change_pct=f"{market.get('change_pct', 0):+.2f}",
        high_52w=f"{market.get('high_52w', 0):,.2f}",
        low_52w=f"{market.get('low_52w', 0):,.2f}",
        volume=market.get("volume", "N/A"),
        risk_level=risk_summary["level"],
        risk_warnings=warnings_text,
        extra_context=extra_context if extra_context else "None provided.",
    )

    client = _get_client()
    message = client.messages.create(
        model=os.getenv("CLAUDE_MODEL", "claude-haiku-4-5-20251001"),
        max_tokens=1024,
        messages=[{"role": "user", "content": prompt}],
    )

    response_text = message.content[0].text.strip()
    verdict = _extract_verdict(response_text)
    return response_text, verdict


def _extract_verdict(text: str) -> str:
    match = re.search(r"\b(BUY|SELL|HOLD|AVOID)\b", text, re.IGNORECASE)
    if match:
        return match.group(1).upper()
    return "HOLD"
