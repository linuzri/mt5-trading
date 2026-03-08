"""
telegram_gate.py — Telegram approval gate for AutoResearch deployments.

When a KEEP experiment is found, autotrader sends a Telegram message asking
for approval. This module polls for your reply and returns the decision.

Commands you send from your phone:
  /deploy   → apply the param change to live bot
  /skip     → discard this keep, continue research loop
  /status   → show current baseline metrics
  /stop     → stop autotrader loop cleanly
"""

import os
import json
import time
import urllib.request
import urllib.parse
from datetime import datetime, timezone

# ── Config ─────────────────────────────────────────────────────────────────────
POLL_INTERVAL_SEC = 10     # check Telegram every 10s
APPROVAL_TIMEOUT  = 1800   # 30 minutes to approve/skip before auto-skip


def _api(token: str, method: str, params: dict = None) -> dict:
    """Call Telegram Bot API."""
    url = f"https://api.telegram.org/bot{token}/{method}"
    if params:
        url += "?" + urllib.parse.urlencode(params)
    try:
        with urllib.request.urlopen(url, timeout=10) as r:
            return json.loads(r.read())
    except Exception as e:
        return {"ok": False, "error": str(e)}


def _get_updates(token: str, offset: int = 0) -> list:
    """Fetch new messages from Telegram."""
    result = _api(token, "getUpdates", {
        "offset":  offset,
        "timeout": 5,
        "allowed_updates": '["message"]',
    })
    if result.get("ok"):
        return result.get("result", [])
    return []


def send_message(msg: str) -> bool:
    """Send a Telegram message. Returns True if sent."""
    token   = os.environ.get("TELEGRAM_BOT_TOKEN_AUTORESEARCH")
    chat_id = os.environ.get("TELEGRAM_CHAT_ID_AUTORESEARCH")
    if not token or not chat_id:
        return False
    try:
        data = json.dumps({
            "chat_id":    chat_id,
            "text":       msg,
            "parse_mode": "Markdown",
        }).encode()
        req = urllib.request.Request(
            f"https://api.telegram.org/bot{token}/sendMessage",
            data=data,
            headers={"Content-Type": "application/json"},
        )
        urllib.request.urlopen(req, timeout=5)
        return True
    except Exception as e:
        print(f"  ⚠ Telegram send failed: {e}")
        return False


def wait_for_approval(experiment: dict, timeout: int = APPROVAL_TIMEOUT) -> str:
    """
    Send approval request to Telegram and poll for response.

    Returns:
      'deploy'  — user sent /deploy
      'skip'    — user sent /skip
      'stop'    — user sent /stop
      'timeout' — no response within timeout seconds
      'no_telegram' — Telegram not configured
    """
    token   = os.environ.get("TELEGRAM_BOT_TOKEN_AUTORESEARCH")
    chat_id = os.environ.get("TELEGRAM_CHAT_ID_AUTORESEARCH")

    if not token or not chat_id:
        print("  ℹ Telegram not configured — auto-deploying (no approval gate)")
        return "no_telegram"

    param   = experiment["param"]
    old_val = experiment["old_value"]
    new_val = experiment["new_value"]
    wr      = experiment["win_rate"]
    pnl     = experiment["pnl"]
    dd      = experiment["drawdown"]
    ratio   = round(new_val / experiment.get("sl_atr_multiplier", 1.5), 2) if "tp" in param else "—"

    msg = (
        f"🔬 *AutoResearch — KEEP Found*\n\n"
        f"Param: `{param}`\n"
        f"Change: `{old_val}` → `{new_val}`\n"
        f"Rationale: _{experiment.get('rationale', '—')}_\n\n"
        f"📊 *Backtest Result (168h)*\n"
        f"  Win Rate : `{wr}%`\n"
        f"  PnL      : `${pnl}`\n"
        f"  Drawdown : `{dd}%`\n\n"
        f"Reply within {timeout//60} min:\n"
        f"  /deploy — apply to live bot\n"
        f"  /skip   — discard, keep researching\n"
        f"  /stop   — stop autotrader loop"
    )

    sent = send_message(msg)
    if not sent:
        print("  ⚠ Could not send Telegram approval request — auto-skipping")
        return "skip"

    print(f"  📱 Approval request sent to Telegram. Waiting up to {timeout//60} min...")
    print(f"     Reply /deploy or /skip on your phone.")

    # Get current update offset to ignore old messages
    updates = _get_updates(token, offset=0)
    offset  = 0
    if updates:
        offset = updates[-1]["update_id"] + 1

    deadline = time.time() + timeout

    while time.time() < deadline:
        remaining = int(deadline - time.time())
        time.sleep(POLL_INTERVAL_SEC)

        updates = _get_updates(token, offset=offset)
        for upd in updates:
            offset = upd["update_id"] + 1
            msg_obj = upd.get("message", {})
            text    = msg_obj.get("text", "").strip().lower()
            from_id = str(msg_obj.get("chat", {}).get("id", ""))

            # Only accept from the configured chat
            if from_id != str(chat_id):
                continue

            if text.startswith("/deploy"):
                send_message(
                    f"✅ *Deploying* `{param}` {old_val}→{new_val}\n"
                    f"Bot will restart shortly. Verification in 2h."
                )
                return "deploy"

            elif text.startswith("/skip"):
                send_message(
                    f"⏭ *Skipped* `{param}` {old_val}→{new_val}\n"
                    f"Continuing research loop..."
                )
                return "skip"

            elif text.startswith("/stop"):
                send_message("🛑 *AutoTrader stopping* after current experiment.")
                return "stop"

            elif text.startswith("/status"):
                _send_status()

        # Countdown reminder at 10 min remaining
        if 590 < remaining < 610:
            send_message(
                f"⏰ *10 min left* to approve `{param}` {old_val}→{new_val}\n"
                f"Reply /deploy or /skip"
            )

    send_message(
        f"⏱ *Timeout* — no response for `{param}` {old_val}→{new_val}\n"
        f"Auto-skipped. Research loop continues."
    )
    return "timeout"


def _send_status():
    """Send current MEMORY.md baseline to Telegram."""
    try:
        from pathlib import Path
        memory = Path("MEMORY.md").read_text()
        # Extract baseline metrics section
        lines  = memory.splitlines()
        in_baseline = False
        metrics = []
        for line in lines:
            if "BASELINE METRICS" in line:
                in_baseline = True
            elif in_baseline and line.startswith("##"):
                break
            elif in_baseline and line.strip():
                metrics.append(line.strip())
        block = "\n".join(metrics[:8])
        send_message(f"📊 *Current Baseline*\n```\n{block}\n```")
    except Exception:
        send_message("⚠ Could not read MEMORY.md")


if __name__ == "__main__":
    # Quick test — sends a test message and waits for reply
    print("Testing Telegram gate...")
    result = wait_for_approval({
        "param":     "tp_atr_multiplier",
        "old_value": 1.75,
        "new_value": 2.0,
        "rationale": "Test message — reply /skip",
        "win_rate":  52.0,
        "pnl":       55.0,
        "drawdown":  0.22,
    }, timeout=120)
    print(f"Result: {result}")
