"""
CLIPS Expert System Engine — TradingPal
========================================
Wraps the clipspy library to run the rule-based investment advisor
defined in clips/rules/Baza_de_reguli.clp.

Public API
----------
run_clips_period(...)  -> dict
    Execute one CLIPS inference cycle and return the updated portfolio state.

run_simulation(...)  -> list[dict]
    Run N consecutive CLIPS cycles, propagating state between periods.

Both functions are pure Python — no Streamlit dependencies — so they can be
used from tests, notebooks, or any UI layer.

Requires: clipspy==1.0.6  (declared in requirements.txt)
"""
import logging
import os
import re

logger = logging.getLogger(__name__)

# Path to the CLIPS rule base, relative to the project root where streamlit runs.
# Updated to reflect the new clips/rules/ location.
_CLP_PATH = os.path.join("clips", "rules", "Baza_de_reguli.clp")


# ──────────────────────────────────────────────────────────────────────────────
# CaptureRouter
# ──────────────────────────────────────────────────────────────────────────────

def _make_capture_router():
    """
    Dynamically build a clips.Router subclass that captures CLIPS printout.

    Defined at call-time so that an ImportError (clipspy not installed)
    surfaces as a clean error message rather than crashing the module import.

    CLIPS writes to its own internal streams (stdout, t, wdisplay, …).
    This router intercepts all of them and accumulates the text in a per-instance
    buffer stored at the class level (keyed by id(self)) to avoid conflicts
    with __slots__ defined in the base clips.Router class.
    """
    import clips  # clipspy 1.0.6

    class CaptureRouter(clips.Router):
        # clips.Router uses __slots__; we cannot add instance attributes directly.
        # Use a class-level dict keyed by id(self) as the per-instance buffer.
        _buffers: dict = {}

        def __init__(self):
            super().__init__("capture", 100)  # priority 100 → intercepts before default routers
            CaptureRouter._buffers[id(self)] = []

        def query(self, name: str) -> bool:
            """Return True for every CLIPS stream we want to capture."""
            return name in ("stdout", "stderr", "stdin", "t",
                            "wdisplay", "wdialog", "wwarning", "werror")

        def write(self, name: str, message: str) -> None:
            CaptureRouter._buffers[id(self)].append(message)

        def read(self, name: str) -> int:
            return 0

        def unread(self, name: str, char: int) -> int:
            return 0

        def exit(self, code: int) -> None:
            pass

        def get_output(self) -> str:
            return "".join(CaptureRouter._buffers.get(id(self), []))

        def __del__(self):
            CaptureRouter._buffers.pop(id(self), None)

    return CaptureRouter()


# ──────────────────────────────────────────────────────────────────────────────
# Core inference cycle
# ──────────────────────────────────────────────────────────────────────────────

def run_clips_period(
    objective: str,
    strategy_clips: str,
    target_sum: float,
    periods: int,
    budget: float,
    ticker: str,
    rsi: float,
    price: float,
    ma200: float,
    qty: float,
    avg_price: float,
) -> dict:
    """
    Run one CLIPS inference cycle.

    Steps:
        1. Create an isolated clips.Environment()
        2. Attach CaptureRouter to intercept all CLIPS printout
        3. Load Baza_de_reguli.clp  (without env.reset() → deffacts are NOT loaded)
        4. Assert four facts from Python into the working memory
        5. env.run() → inference engine fires rules until saturation
        6. Read final working memory → extract updated slot values
        7. Infer BUY / SELL / HOLD by comparing qty/budget before vs. after

    Parameters
    ----------
    objective       : "Invest" or "Cash out"
    strategy_clips  : exact CLIPS string — "DCA fix", "hibrid", or "RSI"
    target_sum      : total budget / target to extract ($)
    periods         : number of periods remaining
    budget          : available cash ($)
    ticker          : asset ticker as a CLIPS SYMBOL (no quotes, no hyphens)
    rsi             : RSI(14) value (0–100)
    price           : current asset price ($)
    ma200           : 200-day moving average ($)
    qty             : units currently held
    avg_price       : average purchase price per unit ($)

    Returns
    -------
    dict with keys:
        clips_log       str   — raw CLIPS printout
        action          str   — "BUY", "SELL", or "HOLD"
        amount          float — dollar amount transacted
        new_budget      float — cash remaining after the action
        new_qty         float — units held after the action
        new_avg_price   float — updated average purchase price
        new_target_sum  float — updated target remaining
        new_periods     int   — periods remaining (decremented by 1)
        rule_fired      str   — name of the first triggered rule
        rules_triggered list  — all triggered rule names (in order)
        error           str|None — error message if the engine failed
    """
    result = {
        "clips_log": "",
        "action": "HOLD",
        "amount": 0.0,
        "new_budget": budget,
        "new_qty": qty,
        "new_avg_price": avg_price,
        "new_target_sum": target_sum,
        "new_periods": max(0, periods - 1),
        "rule_fired": "fallback-nicio-actiune",
        "rules_triggered": [],
        "error": None,
    }

    try:
        import clips  # clipspy 1.0.6

        env = clips.Environment()

        # Attach router BEFORE load so we also capture load-time printout
        capture = _make_capture_router()
        env.add_router(capture)

        env.load(_CLP_PATH)
        # Do NOT call env.reset() — avoids loading the test deffacts block

        # ── Assert facts ──────────────────────────────────────────────────────
        # stare-sistem: global system state
        env.assert_string(
            f'(stare-sistem '
            f'(faza initializare) '
            f'(obiectiv "{objective}") '
            f'(strategie "{strategy_clips}") '
            f'(suma-tinta {target_sum:.4f}) '
            f'(perioade-ramase {periods}) '
            f'(exista-fisier da) '
            f'(eof nu))'
        )
        # portofoliu: available cash
        env.assert_string(
            f'(portofoliu (buget-disponibil {budget:.4f}))'
        )
        # activ-piata: live market snapshot — (nume) is a CLIPS SYMBOL, no quotes
        env.assert_string(
            f'(activ-piata (nume {ticker}) (rsi {rsi:.4f}) '
            f'(pret {price:.4f}) (ma200 {ma200:.4f}))'
        )
        # detinere-activ: current holdings
        env.assert_string(
            f'(detinere-activ (nume-activ {ticker}) '
            f'(cantitate {qty:.6f}) (pret-mediu {avg_price:.4f}))'
        )

        # ── Run inference ─────────────────────────────────────────────────────
        env.run()
        result["clips_log"] = capture.get_output()

        # ── Read back working memory ──────────────────────────────────────────
        stare = {}
        portof = {}
        detinere = {}

        for fact in env.facts():
            tmpl_name = fact.template.name
            slots = dict(fact)

            if tmpl_name == "stare-sistem":
                stare = slots
            elif tmpl_name == "portofoliu":
                portof = slots
            elif tmpl_name == "detinere-activ":
                detinere = slots

        new_budget = float(portof.get("buget-disponibil", budget))
        new_qty = float(detinere.get("cantitate", qty))
        new_avg = float(detinere.get("pret-mediu", avg_price))
        new_target = float(stare.get("suma-tinta", target_sum))
        new_periods = int(stare.get("perioade-ramase", max(0, periods - 1)))

        # ── Infer action from delta ───────────────────────────────────────────
        # Python does not read "BUY/SELL" directly from CLIPS — it compares
        # the working memory state before and after env.run().
        budget_delta = new_budget - budget
        qty_delta = new_qty - qty

        if qty_delta > 1e-9:
            action = "BUY"
            amount = abs(budget_delta)
        elif qty_delta < -1e-9:
            action = "SELL"
            amount = abs(budget_delta)
        else:
            action = "HOLD"
            amount = 0.0

        # ── Extract triggered rule names from CLIPS printout ──────────────────
        log = result["clips_log"]
        rules_triggered = []
        for line in log.splitlines():
            m = re.search(r'\[([\w\-]+)\]', line)
            if m and any(prefix in line for prefix in ("EXEC ", "ALERTĂ ", "HOLD ")):
                rules_triggered.append(m.group(1))

        rule_fired = rules_triggered[0] if rules_triggered else "fallback-nicio-actiune"

        result.update({
            "action": action,
            "amount": amount,
            "new_budget": new_budget,
            "new_qty": new_qty,
            "new_avg_price": new_avg,
            "new_target_sum": new_target,
            "new_periods": new_periods,
            "rule_fired": rule_fired,
            "rules_triggered": rules_triggered,
        })

    except ImportError:
        result["error"] = (
            "clipspy is not installed. Run: pip install clipspy==1.0.6"
        )
    except Exception as exc:
        logger.exception("CLIPS engine error: %s", exc)
        result["error"] = str(exc)

    return result


# ──────────────────────────────────────────────────────────────────────────────
# Multi-period simulation
# ──────────────────────────────────────────────────────────────────────────────

def run_simulation(
    objective: str,
    strategy_clips: str,
    target_sum: float,
    n_periods: int,
    budget: float,
    ticker: str,
    rsi: float,
    price: float,
    ma200: float,
    qty: float,
    avg_price: float,
    frequency: str = "Monthly",
    predicted_prices: list | None = None,
    on_error=None,
) -> list[dict]:
    """
    Simulate N consecutive periods using the CLIPS inference engine.

    State (budget, qty, avg_price, target_sum, periods) propagates from one
    period to the next.  RSI, price, and MA200 remain fixed at the live
    snapshot value for the duration of the simulation.

    Parameters
    ----------
    on_error : callable(period, error_str) | None
        Optional callback invoked when a CLIPS error occurs on a given period.
        If None, the error is silently included in the returned row and the
        loop stops.

    Returns
    -------
    list of row dicts suitable for pd.DataFrame.  Each row contains:
        Period, Action, Amount ($), Cash Left ($), Portfolio Value ($),
        and optionally Predicted Price ($) / Predicted Portfolio ($).
    """
    period_label = {"Daily": "Day", "Weekly": "Week", "Monthly": "Month"}[frequency]

    rows = []
    cur_budget = budget
    cur_qty = qty
    cur_avg = avg_price
    cur_target = target_sum
    cur_periods = n_periods

    for period in range(1, n_periods + 1):
        if cur_periods <= 0:
            break

        r = run_clips_period(
            objective=objective,
            strategy_clips=strategy_clips,
            target_sum=cur_target,
            periods=cur_periods,
            budget=cur_budget,
            ticker=ticker,
            rsi=rsi,
            price=price,
            ma200=ma200,
            qty=cur_qty,
            avg_price=cur_avg,
        )

        if r["error"]:
            if on_error:
                on_error(period, r["error"])
            break

        portfolio_value = r["new_budget"] + r["new_qty"] * price

        pred_price = (
            predicted_prices[period - 1]
            if predicted_prices and period - 1 < len(predicted_prices)
            else None
        )
        pred_portfolio = (
            round(r["new_budget"] + r["new_qty"] * pred_price, 2)
            if pred_price is not None
            else None
        )

        row = {
            "Period": f"{period_label} {period}",
            "Action": r["action"],
            "Amount ($)": round(r["amount"], 2),
            "Cash Left ($)": round(r["new_budget"], 2),
            "Portfolio Value ($)": round(portfolio_value, 2),
        }
        if pred_price is not None:
            row["Predicted Price ($)"] = pred_price
            row["Predicted Portfolio ($)"] = pred_portfolio

        rows.append(row)

        cur_budget = r["new_budget"]
        cur_qty = r["new_qty"]
        cur_avg = r["new_avg_price"]
        cur_target = r["new_target_sum"]
        cur_periods = r["new_periods"]

        if r["action"] == "HOLD" and cur_target <= 0:
            break

    return rows
