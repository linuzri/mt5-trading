"""
autotrader.py — AutoTrader Research Loop (MT5 Edition)
Adapted from karpathy/autoresearch for BTCUSD trend-following strategy

HOW IT RUNS:
  python autotrader.py                    # loops forever, all params
  python autotrader.py --once             # single experiment then exit
  python autotrader.py --dry-run          # simulate without MT5 connection
  python autotrader.py --hours 48         # wider backtest window

REQUIREMENTS:
  pip install anthropic
  ANTHROPIC_API_KEY env var set
  MT5 terminal running + mt5_auth.json present (for live mode)

Run from inside btcusd/ directory.
"""

import sys
import io
#sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
#sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

import os
import sys
import json
import time
import argparse
import datetime
from pathlib import Path

import anthropic

# ── Config ────────────────────────────────────────────────────────────────────

MEMORY_FILE        = Path("MEMORY.md")
CALIB_LOG          = Path("calibration.jsonl")
CLAUDE_MODEL       = "claude-sonnet-4-20250514"
BACKTEST_HOURS     = 168
LOOP_DELAY_SECONDS = 60     # gap between experiments (seconds)
MAX_TOKENS         = 2048

# Deploy gate: require Telegram approval before pushing to live bot
# Set to False to auto-deploy every keep without asking (not recommended)
REQUIRE_APPROVAL   = True

# ── Anthropic client ──────────────────────────────────────────────────────────

client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))

# ── MEMORY.md helpers ─────────────────────────────────────────────────────────

def read_memory() -> str:
    if not MEMORY_FILE.exists():
        raise FileNotFoundError(f"{MEMORY_FILE} not found. Create it first.")
    return MEMORY_FILE.read_text(encoding="utf-8")


def write_memory(content: str):
    MEMORY_FILE.write_text(content, encoding="utf-8")


# ── Backtest: real MT5 integration ────────────────────────────────────────────

def run_backtest(strategy_name: str, params: dict,
                 hours: int = BACKTEST_HOURS, dry_run: bool = False) -> dict:
    """
    Calls backtest_mt5.run_backtest() with the agent-proposed params.

    params keys the agent may set (all optional):
      sl_atr_multiplier  : float   SL = ATR * this
      tp_atr_multiplier  : float   TP = ATR * this
      atr_period         : int     ATR lookback
      min_atr            : float   volatility floor filter
      h4_ema_fast        : int     H4 trend EMA fast period
      h4_ema_slow        : int     H4 trend EMA slow period
      h1_ema_period      : int     H1 entry EMA period
      lot_size           : float   position size
    """
    # Import here so MT5 is only loaded when needed
    from backtest_mt5 import run_backtest as _bt
    return _bt(strategy_name, params, hours=hours, dry_run=dry_run)


# ── Agent: propose mutation ────────────────────────────────────────────────────

def load_recent_experiments(n: int = 20) -> str:
    """Load last N experiments from calibration.jsonl as a summary string."""
    if not CALIB_LOG.exists():
        return "No experiments yet."
    lines = [l for l in CALIB_LOG.read_text().splitlines() if l.strip()]
    if not lines:
        return "No experiments yet."
    recent = lines[-n:]
    rows = []
    for l in recent:
        try:
            e = json.loads(l)
            rows.append(
                f"  {e['param']:<22} {str(e['old_value'])}->{str(e['new_value']):<8} "
                f"WR={e['win_rate']}% PnL=${e['pnl']} DD={e['drawdown']}% "
                f"-> {e['decision'].upper()}"
            )
        except Exception:
            continue
    return "\n".join(rows)


def propose_mutation(memory: str) -> dict:
    """
    Ask Claude to read MEMORY.md and recent experiment history,
    then propose ONE parameter mutation not recently tried.
    Returns structured dict.
    """
    recent_history = load_recent_experiments(20)

    system = """You are an autonomous trading strategy optimizer for a BTCUSD trend-following bot.

Read MEMORY.md and the RECENT EXPERIMENT HISTORY carefully.
Propose exactly ONE parameter mutation.

Rules:
- Only change params listed under MUTABLE PARAMS
- Change by exactly ONE step size
- Stay within [min, max] bounds
- NEVER propose a param+value combination already shown as DISCARD in recent history
- NEVER repeat the same experiment twice
- If a direction was discarded (e.g. tp 1.75->1.5 failed), try a DIFFERENT param entirely
- Target the param most likely to improve the WORST baseline metric
- Never exceed hard limits in CONSTRAINTS

Respond with ONLY valid JSON, no explanation, no markdown:
{
  "param": "sl_atr_multiplier",
  "old_value": 1.5,
  "new_value": 1.75,
  "rationale": "one sentence max"
}"""

    user_content = (
        f"MEMORY.md:\n\n{memory}\n\n"
        f"RECENT EXPERIMENT HISTORY (last 20):\n{recent_history}\n\n"
        f"Propose your next mutation. Avoid repeating discarded experiments. JSON only."
    )

    response = client.messages.create(
        model=CLAUDE_MODEL,
        max_tokens=MAX_TOKENS,
        system=system,
        messages=[{"role": "user", "content": user_content}]
    )

    raw = response.content[0].text.strip()
    if "```" in raw:
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
    return json.loads(raw.strip())


# ── Agent: keep/discard decision ──────────────────────────────────────────────

def evaluate_decision(memory: str, mutation: dict, result: dict) -> dict:
    """
    Two-step decision to avoid embedding MEMORY.md inside JSON (causes parse errors).

    Step 1: small JSON-only call → get decision + reason
    Step 2: if KEEP, plain-text call → get updated MEMORY.md content
    """
    payload = {
        "mutation": mutation,
        "backtest_result": result,
    }

    # ── Step 1: decision only (small, safe JSON) ──────────────────────────────
    step1_system = """You are evaluating a trading parameter mutation against DECISION RULES.
Apply the rules strictly.

Respond with ONLY this JSON (no markdown, no extra text):
{"decision": "keep", "reason": "one sentence explanation"}

decision must be exactly "keep" or "discard"."""

    step1_response = client.messages.create(
        model=CLAUDE_MODEL,
        max_tokens=200,
        system=step1_system,
        messages=[{
            "role": "user",
            "content": (
                f"MEMORY.md:\n\n{memory}\n\n"
                f"Experiment result:\n{json.dumps(payload, indent=2)}\n\n"
                f"Apply DECISION RULES. JSON only."
            )
        }]
    )

    raw1 = step1_response.content[0].text.strip()
    if "```" in raw1:
        raw1 = raw1.split("```")[1]
        if raw1.startswith("json"):
            raw1 = raw1[4:]
    verdict = json.loads(raw1.strip())
    decision = verdict["decision"]
    reason   = verdict.get("reason", "")

    # ── Step 2: if KEEP, get updated MEMORY.md as plain text ─────────────────
    updated_memory = None
    if decision == "keep":
        step2_system = """You are updating a MEMORY.md file after a successful trading experiment.
Output ONLY the complete updated MEMORY.md text — no explanation, no markdown fences, no preamble.
Start directly with the first line of the file (# MEMORY.md ...)."""

        step2_response = client.messages.create(
            model=CLAUDE_MODEL,
            max_tokens=MAX_TOKENS * 3,
            system=step2_system,
            messages=[{
                "role": "user",
                "content": (
                    f"Current MEMORY.md:\n\n{memory}\n\n"
                    f"Experiment that was KEPT:\n{json.dumps(payload, indent=2)}\n\n"
                    f"Update these sections:\n"
                    f"1. Current Parameters — set {mutation['param']} to {mutation['new_value']}\n"
                    f"2. BASELINE METRICS — update win_rate, pnl_per_session, max_drawdown, last_updated\n"
                    f"3. EXPERIMENT LOG SUMMARY — increment total_experiments_run and total_kept, set last_experiment\n\n"
                    f"Output the full updated MEMORY.md now."
                )
            }]
        )
        updated_memory = step2_response.content[0].text.strip()

    return {
        "decision":       decision,
        "reason":         reason,
        "updated_memory": updated_memory,
    }


# ── Calibration log ───────────────────────────────────────────────────────────

def log_experiment(record: dict):
    record["timestamp"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
    with open(CALIB_LOG, "a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")


# ── Telegram notify ───────────────────────────────────────────────────────────

def telegram_notify(msg: str):
    token   = os.environ.get("TELEGRAM_BOT_TOKEN_AUTORESEARCH")
    chat_id = os.environ.get("TELEGRAM_CHAT_ID_AUTORESEARCH")
    if not token or not chat_id:
        return
    try:
        import urllib.request
        data = json.dumps({"chat_id": chat_id, "text": msg, "parse_mode": "Markdown"}).encode()
        req  = urllib.request.Request(
            f"https://api.telegram.org/bot{token}/sendMessage",
            data=data, headers={"Content-Type": "application/json"}
        )
        urllib.request.urlopen(req, timeout=5)
    except Exception as e:
        print(f"  ⚠ Telegram failed: {e}")


# ── Single experiment ─────────────────────────────────────────────────────────

def run_one_experiment(hours: int = BACKTEST_HOURS, dry_run: bool = False) -> dict:
    sep = "─" * 58
    print(f"\n{sep}")
    print(f"  🔁  {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # 1. Read
    memory = read_memory()
    print(f"  📖  MEMORY.md read ({len(memory):,} chars)")

    # 2. Propose
    print(f"  🧬  Requesting mutation from Claude...")
    mutation = propose_mutation(memory)
    print(f"  🧬  {mutation['param']}: {mutation['old_value']} → {mutation['new_value']}")
    print(f"       {mutation['rationale']}")

    # 3. Backtest
    print(f"  ⏱   Running {hours}h backtest {'(dry-run)' if dry_run else '(live MT5)'}...")
    t0     = time.time()
    result = run_backtest("btcusd", {mutation["param"]: mutation["new_value"]},
                          hours=hours, dry_run=dry_run)
    elapsed = time.time() - t0
    print(f"  📊  win_rate={result['win_rate']}% | pnl=${result['pnl']} | "
          f"dd={result['drawdown']}% | trades={result['total_trades']} ({elapsed:.1f}s)")

    # 4. Decide
    print(f"  ⚖   Evaluating decision...")
    outcome  = evaluate_decision(memory, mutation, result)
    decision = outcome["decision"]
    reason   = outcome.get("reason", "")
    print(f"  {'✅ KEEP' if decision == 'keep' else '❌ DISCARD'}  {reason}")

    # 5. Update memory if kept
    if decision == "keep" and outcome.get("updated_memory"):
        write_memory(outcome["updated_memory"])
        print(f"  💾  MEMORY.md updated")

    # 6. Build record
    record = {
        "param":     mutation["param"],
        "old_value": mutation["old_value"],
        "new_value": mutation["new_value"],
        "rationale": mutation["rationale"],
        **result,
        "decision":  decision,
        "reason":    reason,
        "deployed":  False,
    }

    # 7. Approval gate + deploy (only for KEEP)
    if decision == "keep":
        if REQUIRE_APPROVAL:
            from telegram_gate import wait_for_approval
            approval = wait_for_approval({**mutation, **result}, timeout=1800)
        else:
            approval = "deploy"   # auto-deploy without asking

        if approval == "deploy":
            print(f"  🚀  Deploying to live bot...")
            from deploy import deploy as _deploy
            deployed = _deploy(
                param        = mutation["param"],
                new_value    = mutation["new_value"],
                old_value    = mutation["old_value"],
                backtest_wr  = result["win_rate"],
                backtest_pnl = result["pnl"],
                dry_run      = dry_run,
            )
            record["deployed"] = deployed
            print(f"  {'✅  Deploy complete' if deployed else '❌  Deploy failed'}")

        elif approval == "skip":
            print(f"  ⏭   Skipped by user — not deploying")
            telegram_notify(f"⏭ *Skipped* `{mutation['param']}` {mutation['old_value']}→{mutation['new_value']}")

        elif approval == "timeout":
            print(f"  ⏱   No response — auto-skipped")

        elif approval == "stop":
            print(f"  🛑  Stop requested via Telegram")
            record["stop_requested"] = True

        elif approval == "no_telegram":
            # Telegram not configured — just notify without gating
            telegram_notify(
                f"✅ *KEEP* `{mutation['param']}` {mutation['old_value']}→{mutation['new_value']}\n"
                f"WR: `{result['win_rate']}%` PnL: `${result['pnl']}`\n"
                f"⚠ Telegram gate not configured — not auto-deploying.\n"
                f"Apply manually via deploy.py"
            )

    else:
        # Discard — just notify
        telegram_notify(
            f"❌ *DISCARD* `{mutation['param']}` {mutation['old_value']}→{mutation['new_value']}\n"
            f"WR: `{result['win_rate']}%` PnL: `${result['pnl']}`\n"
            f"_{reason}_"
        )

    # 8. Log
    log_experiment(record)
    print(f"  📝  Logged → {CALIB_LOG}")

    return record


# ── Main ──────────────────────────────────────────────────────────────────────

def _generate_summary() -> str:
    """Generate a text summary from calibration.jsonl for Telegram."""
    if not CALIB_LOG.exists():
        return "No experiments logged."
    lines = [l for l in CALIB_LOG.read_text().splitlines() if l.strip()]
    if not lines:
        return "No experiments logged."

    exps = [json.loads(l) for l in lines]
    kept = [e for e in exps if e["decision"] == "keep"]
    total = len(exps)
    keep_rate = len(kept) / total * 100 if total > 0 else 0

    summary = f"*AutoResearch Summary*\n"
    summary += f"Total: {total} | Kept: {len(kept)} ({keep_rate:.0f}%)\n\n"

    if kept:
        best = max(kept, key=lambda x: x["pnl"])
        summary += f"*Best:* `{best['param']}` {best['old_value']}→{best['new_value']}\n"
        summary += f"WR: `{best['win_rate']}%` PnL: `${best['pnl']}` DD: `{best['drawdown']}%`\n\n"
        summary += "*All keeps:*\n"
        for e in kept:
            summary += f"  `{e['param']}` {e['old_value']}→{e['new_value']} WR={e['win_rate']}% PnL=${e['pnl']}\n"
    else:
        summary += "_No improvements found — current params are optimal for this window._"

    return summary


def main():
    parser = argparse.ArgumentParser(description="AutoTrader Research Loop")
    parser.add_argument("--once",    action="store_true", help="Single experiment then exit")
    parser.add_argument("--dry-run", action="store_true", help="Simulate without MT5")
    parser.add_argument("--hours",   type=int, default=BACKTEST_HOURS, help="Backtest window (hours)")
    parser.add_argument("--delay",   type=int, default=LOOP_DELAY_SECONDS, help="Seconds between experiments")
    parser.add_argument("--max-hours", type=float, default=0, help="Stop after N hours (0 = no limit)")
    parser.add_argument("--auto-stop", type=int, default=50, help="Stop if 0 keeps in last N experiments (0 = disable)")
    args = parser.parse_args()

    max_hours_str = f"{args.max_hours}h" if args.max_hours > 0 else "unlimited"
    auto_stop_str = f"after {args.auto_stop} consecutive discards" if args.auto_stop > 0 else "disabled"

    print("=" * 58)
    print("  AutoTrader Research Loop — BTCUSD Trend Strategy")
    print(f"  Model  : {CLAUDE_MODEL}")
    print(f"  Memory : {MEMORY_FILE}")
    print(f"  Log    : {CALIB_LOG}")
    print(f"  Window : {args.hours}h backtest")
    print(f"  Runtime: {max_hours_str}")
    print(f"  Auto-stop: {auto_stop_str}")
    print(f"  Mode   : {'DRY-RUN (simulated)' if args.dry_run else 'LIVE (MT5)'}")
    print("=" * 58)

    if not MEMORY_FILE.exists():
        print(f"❌  {MEMORY_FILE} not found.")
        sys.exit(1)

    if args.once:
        run_one_experiment(hours=args.hours, dry_run=args.dry_run)
        return

    # Send start notification
    telegram_notify(
        f"🔬 *AutoResearch started*\n"
        f"Window: `{args.hours}h` | Delay: `{args.delay}s`\n"
        f"Runtime: `{max_hours_str}` | Auto-stop: `{auto_stop_str}`"
    )

    start_time = time.time()
    deadline = start_time + (args.max_hours * 3600) if args.max_hours > 0 else None
    count = kept = 0
    consecutive_discards = 0

    try:
        while True:
            # Time limit check
            if deadline and time.time() >= deadline:
                elapsed_h = (time.time() - start_time) / 3600
                print(f"\n  ⏱  Max runtime ({args.max_hours}h) reached. Stopping.")
                summary = _generate_summary()
                telegram_notify(
                    f"⏱ *AutoResearch finished* (time limit: {args.max_hours}h)\n"
                    f"{count} experiments, {kept} kept ({kept/max(count,1)*100:.0f}%)\n\n"
                    f"{summary}"
                )
                break

            exp    = run_one_experiment(hours=args.hours, dry_run=args.dry_run)
            count += 1
            if exp["decision"] == "keep":
                kept += 1
                consecutive_discards = 0
            else:
                consecutive_discards += 1

            # Handle /stop command sent via Telegram
            if exp.get("stop_requested"):
                print(f"\n  🛑  Stop requested via Telegram. Shutting down.")
                summary = _generate_summary()
                telegram_notify(
                    f"🛑 *AutoResearch stopped* via /stop\n"
                    f"{count} experiments, {kept} kept ({kept/max(count,1)*100:.0f}%)\n\n"
                    f"{summary}"
                )
                break

            # Auto-stop on convergence
            if args.auto_stop > 0 and consecutive_discards >= args.auto_stop:
                print(f"\n  🎯  Convergence detected ({consecutive_discards} consecutive discards). Stopping.")
                summary = _generate_summary()
                telegram_notify(
                    f"🎯 *AutoResearch converged*\n"
                    f"{consecutive_discards} consecutive discards — params are optimal.\n"
                    f"{count} experiments, {kept} kept ({kept/max(count,1)*100:.0f}%)\n\n"
                    f"{summary}"
                )
                break

            keep_rate = kept / count * 100
            print(f"\n  📈  Session: {count} experiments | {kept} kept ({keep_rate:.0f}%) | "
                  f"streak: {consecutive_discards} discards")
            print(f"  💤  Next experiment in {args.delay}s...")
            time.sleep(args.delay)

    except KeyboardInterrupt:
        print(f"\n\n  🛑  Stopped. {count} experiments | {kept} kept.")
        summary = _generate_summary()
        telegram_notify(
            f"🛑 *AutoResearch stopped*\n"
            f"{count} experiments, {kept} kept ({kept/max(count,1)*100:.0f}%)\n\n"
            f"{summary}"
        )


if __name__ == "__main__":
    main()
