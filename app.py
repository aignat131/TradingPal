import os
from flask import Flask, render_template_string, request, jsonify
from dotenv import load_dotenv
from models.trading_advisor import get_trade_advice
from utils.market_data import get_price_data
from utils.risk_checks import assess_risk

load_dotenv()

app = Flask(__name__)

HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>TradingPal - AI Trade Advisor</title>
    <style>
        * { box-sizing: border-box; margin: 0; padding: 0; }
        body { font-family: 'Segoe UI', sans-serif; background: #0f0f1a; color: #e0e0e0; min-height: 100vh; }
        .header { background: linear-gradient(135deg, #1a1a2e, #16213e); padding: 24px 40px; border-bottom: 1px solid #2a2a4a; }
        .header h1 { font-size: 28px; color: #00d4aa; letter-spacing: 1px; }
        .header p { color: #888; font-size: 14px; margin-top: 4px; }
        .container { max-width: 860px; margin: 40px auto; padding: 0 20px; }
        .card { background: #1a1a2e; border: 1px solid #2a2a4a; border-radius: 12px; padding: 32px; margin-bottom: 24px; }
        .card h2 { font-size: 18px; color: #00d4aa; margin-bottom: 20px; }
        .form-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 16px; }
        .form-group { display: flex; flex-direction: column; gap: 6px; }
        .form-group.full { grid-column: 1 / -1; }
        label { font-size: 13px; color: #aaa; text-transform: uppercase; letter-spacing: 0.5px; }
        input, select { background: #0f0f1a; border: 1px solid #2a2a4a; border-radius: 8px; padding: 10px 14px;
                        color: #e0e0e0; font-size: 15px; outline: none; transition: border-color 0.2s; }
        input:focus, select:focus { border-color: #00d4aa; }
        select option { background: #1a1a2e; }
        button { margin-top: 20px; width: 100%; padding: 14px; background: #00d4aa; color: #0f0f1a;
                 border: none; border-radius: 8px; font-size: 16px; font-weight: 700; cursor: pointer;
                 transition: background 0.2s; }
        button:hover { background: #00b894; }
        button:disabled { background: #444; color: #888; cursor: not-allowed; }
        #result { display: none; }
        .result-header { display: flex; align-items: center; gap: 12px; margin-bottom: 16px; }
        .verdict { font-size: 22px; font-weight: 700; padding: 6px 18px; border-radius: 6px; }
        .verdict.BUY   { background: #00d4aa22; color: #00d4aa; border: 1px solid #00d4aa44; }
        .verdict.SELL  { background: #ff6b6b22; color: #ff6b6b; border: 1px solid #ff6b6b44; }
        .verdict.HOLD  { background: #f9ca2422; color: #f9ca24; border: 1px solid #f9ca2444; }
        .verdict.AVOID { background: #ff6b6b22; color: #ff9f43; border: 1px solid #ff9f4344; }
        .risk-badge { font-size: 12px; padding: 4px 10px; border-radius: 4px; }
        .risk-LOW    { background: #00d4aa22; color: #00d4aa; }
        .risk-MEDIUM { background: #f9ca2422; color: #f9ca24; }
        .risk-HIGH   { background: #ff6b6b22; color: #ff6b6b; }
        .ai-advice { line-height: 1.7; font-size: 15px; color: #ccc; white-space: pre-wrap; }
        .market-row { display: flex; gap: 24px; flex-wrap: wrap; margin-bottom: 16px; }
        .stat { background: #0f0f1a; border-radius: 8px; padding: 12px 20px; flex: 1; min-width: 120px; }
        .stat .label { font-size: 11px; color: #666; text-transform: uppercase; }
        .stat .value { font-size: 20px; font-weight: 700; color: #e0e0e0; margin-top: 4px; }
        .stat .change.pos { color: #00d4aa; }
        .stat .change.neg { color: #ff6b6b; }
        .warning { background: #ff9f4322; border: 1px solid #ff9f4344; border-radius: 8px;
                   padding: 12px 16px; color: #ff9f43; font-size: 13px; margin-bottom: 16px; }
        .spinner { display: inline-block; width: 18px; height: 18px; border: 2px solid #444;
                   border-top-color: #00d4aa; border-radius: 50%; animation: spin 0.7s linear infinite; }
        @keyframes spin { to { transform: rotate(360deg); } }
        .error-msg { color: #ff6b6b; font-size: 14px; margin-top: 12px; display: none; }
    </style>
</head>
<body>
    <div class="header">
        <h1>TradingPal</h1>
        <p>AI-powered trade advisor for stocks & crypto</p>
    </div>
    <div class="container">
        <div class="card">
            <h2>Get Trade Advice</h2>
            <div class="form-grid">
                <div class="form-group">
                    <label>Ticker / Symbol</label>
                    <input type="text" id="ticker" placeholder="e.g. AAPL, BTC-USD" value="AAPL">
                </div>
                <div class="form-group">
                    <label>Asset Type</label>
                    <select id="asset_type">
                        <option value="stock">Stock</option>
                        <option value="crypto">Crypto</option>
                    </select>
                </div>
                <div class="form-group">
                    <label>Trade Direction</label>
                    <select id="direction">
                        <option value="buy">Considering Buy</option>
                        <option value="sell">Considering Sell</option>
                    </select>
                </div>
                <div class="form-group">
                    <label>Investment Amount ($)</label>
                    <input type="number" id="amount" placeholder="e.g. 1000" value="1000" min="1">
                </div>
                <div class="form-group">
                    <label>Risk Tolerance</label>
                    <select id="risk_tolerance">
                        <option value="low">Low — Preserve capital</option>
                        <option value="medium" selected>Medium — Balanced</option>
                        <option value="high">High — Aggressive growth</option>
                    </select>
                </div>
                <div class="form-group">
                    <label>Investment Horizon</label>
                    <select id="horizon">
                        <option value="short">Short-term (&lt; 1 month)</option>
                        <option value="medium" selected>Medium (1–6 months)</option>
                        <option value="long">Long-term (6+ months)</option>
                    </select>
                </div>
                <div class="form-group full">
                    <label>Additional Context (optional)</label>
                    <input type="text" id="context" placeholder="e.g. I already hold 5 shares, expecting earnings next week">
                </div>
            </div>
            <button id="submit-btn" onclick="getAdvice()">Analyze Trade</button>
            <div class="error-msg" id="error-msg"></div>
        </div>

        <div class="card" id="result">
            <div class="result-header">
                <span class="verdict" id="verdict-badge">—</span>
                <span class="risk-badge" id="risk-badge">Risk: —</span>
                <span id="ticker-label" style="color:#666; font-size:13px;"></span>
            </div>
            <div class="market-row" id="market-row"></div>
            <div class="warning" id="risk-warnings" style="display:none;"></div>
            <div class="ai-advice" id="ai-advice"></div>
        </div>
    </div>

    <script>
        async function getAdvice() {
            const btn = document.getElementById('submit-btn');
            const errorEl = document.getElementById('error-msg');
            errorEl.style.display = 'none';
            btn.disabled = true;
            btn.innerHTML = '<span class="spinner"></span>  Analyzing...';

            const payload = {
                ticker:        document.getElementById('ticker').value.trim().toUpperCase(),
                asset_type:    document.getElementById('asset_type').value,
                direction:     document.getElementById('direction').value,
                amount:        parseFloat(document.getElementById('amount').value),
                risk_tolerance:document.getElementById('risk_tolerance').value,
                horizon:       document.getElementById('horizon').value,
                context:       document.getElementById('context').value.trim(),
            };

            try {
                const res = await fetch('/advise', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify(payload)
                });
                const data = await res.json();
                if (!res.ok) throw new Error(data.error || 'Unknown error');
                renderResult(data, payload.ticker);
            } catch (e) {
                errorEl.textContent = 'Error: ' + e.message;
                errorEl.style.display = 'block';
            } finally {
                btn.disabled = false;
                btn.textContent = 'Analyze Trade';
            }
        }

        function renderResult(data, ticker) {
            const result = document.getElementById('result');
            result.style.display = 'block';
            result.scrollIntoView({behavior: 'smooth', block: 'start'});

            const verdict = data.verdict || 'HOLD';
            const vBadge = document.getElementById('verdict-badge');
            vBadge.textContent = verdict;
            vBadge.className = 'verdict ' + verdict;

            const rBadge = document.getElementById('risk-badge');
            const rLevel = (data.risk_level || 'MEDIUM').toUpperCase();
            rBadge.textContent = 'Risk: ' + rLevel;
            rBadge.className = 'risk-badge risk-' + rLevel;

            document.getElementById('ticker-label').textContent = ticker;

            const m = data.market || {};
            const chg = m.change_pct || 0;
            const chgClass = chg >= 0 ? 'pos' : 'neg';
            const chgSign = chg >= 0 ? '+' : '';
            document.getElementById('market-row').innerHTML = `
                <div class="stat"><div class="label">Current Price</div><div class="value">$${(m.price||0).toLocaleString(undefined,{minimumFractionDigits:2,maximumFractionDigits:2})}</div></div>
                <div class="stat"><div class="label">24h Change</div><div class="value"><span class="change ${chgClass}">${chgSign}${chg.toFixed(2)}%</span></div></div>
                <div class="stat"><div class="label">52w High</div><div class="value">$${(m.high_52w||0).toLocaleString(undefined,{minimumFractionDigits:2,maximumFractionDigits:2})}</div></div>
                <div class="stat"><div class="label">52w Low</div><div class="value">$${(m.low_52w||0).toLocaleString(undefined,{minimumFractionDigits:2,maximumFractionDigits:2})}</div></div>
            `;

            const warnings = data.risk_warnings || [];
            const warnEl = document.getElementById('risk-warnings');
            if (warnings.length) {
                warnEl.innerHTML = '&#9888; ' + warnings.join('<br>&#9888; ');
                warnEl.style.display = 'block';
            } else {
                warnEl.style.display = 'none';
            }

            document.getElementById('ai-advice').textContent = data.advice || '';
        }
    </script>
</body>
</html>
"""


@app.route("/")
def index():
    return render_template_string(HTML)


@app.route("/advise", methods=["POST"])
def advise():
    data = request.get_json()
    ticker        = data.get("ticker", "").strip().upper()
    asset_type    = data.get("asset_type", "stock")
    direction     = data.get("direction", "buy")
    amount        = float(data.get("amount", 1000))
    risk_tolerance= data.get("risk_tolerance", "medium")
    horizon       = data.get("horizon", "medium")
    extra_context = data.get("context", "")

    if not ticker:
        return jsonify({"error": "Ticker symbol is required."}), 400

    market = get_price_data(ticker, asset_type)
    if "error" in market:
        return jsonify({"error": market["error"]}), 400

    risk_result = assess_risk(
        price=market["price"],
        amount=amount,
        risk_tolerance=risk_tolerance,
        change_pct=market.get("change_pct", 0),
        high_52w=market.get("high_52w"),
        low_52w=market.get("low_52w"),
    )

    advice, verdict = get_trade_advice(
        ticker=ticker,
        asset_type=asset_type,
        direction=direction,
        amount=amount,
        risk_tolerance=risk_tolerance,
        horizon=horizon,
        market=market,
        risk_summary=risk_result,
        extra_context=extra_context,
    )

    return jsonify({
        "verdict": verdict,
        "advice": advice,
        "market": market,
        "risk_level": risk_result["level"],
        "risk_warnings": risk_result["warnings"],
    })


if __name__ == "__main__":
    port = int(os.getenv("PORT", 5000))
    debug = os.getenv("FLASK_DEBUG", "false").lower() == "true"
    app.run(host="0.0.0.0", port=port, debug=debug)
