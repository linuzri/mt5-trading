"""
strategy_discovery.py — Strategy Discovery Pipeline for BTCUSD
Goes beyond parameter tuning to INVENT NEW STRATEGY LOGIC

Generates new strategy variants using Claude API, tests them in sandboxed environment,
and evolves the best performers. Uses existing backtest_mt5.py infrastructure.

Usage:
  python strategy_discovery.py --dry-run --once        # single test run
  python strategy_discovery.py --hours 168 --max-hours 8  # full discovery session
  python strategy_discovery.py --auto-stop 20         # stop after 20 failures
"""

import os
import sys
import json
import time
import argparse
import datetime
import importlib.util
from pathlib import Path
from typing import Dict, Any, Optional

# Fix Windows console encoding
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

# Load environment
from dotenv import load_dotenv
load_dotenv(Path(__file__).parent.parent / ".env")

import anthropic
import pandas as pd
import numpy as np

# ── Config ────────────────────────────────────────────────────────────────────

MEMORY_FILE = Path("MEMORY_DISCOVERY.md")
LOG_FILE = Path("strategy_discovery.jsonl")
VARIANTS_DIR = Path("variants")
CLAUDE_MODEL = "claude-sonnet-4-20250514"
BACKTEST_HOURS = 168
MAX_TOKENS = 4096

# Create variants directory
VARIANTS_DIR.mkdir(exist_ok=True)

# ── Anthropic client ──────────────────────────────────────────────────────────

client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))

# ── Memory management ─────────────────────────────────────────────────────────

def read_memory() -> str:
    """Read the discovery memory file."""
    if not MEMORY_FILE.exists():
        raise FileNotFoundError(f"{MEMORY_FILE} not found. Create it first.")
    return MEMORY_FILE.read_text(encoding="utf-8")


def write_memory(content: str):
    """Write updated memory file."""
    MEMORY_FILE.write_text(content, encoding="utf-8")


# ── Experiment logging ───────────────────────────────────────────────────────

def load_recent_experiments(n: int = 10) -> str:
    """Load last N experiments from strategy_discovery.jsonl as summary string."""
    if not LOG_FILE.exists():
        return "No strategy experiments yet."

    lines = [l for l in LOG_FILE.read_text().splitlines() if l.strip()]
    if not lines:
        return "No strategy experiments yet."

    recent = lines[-n:]
    rows = []
    for l in recent:
        try:
            e = json.loads(l)
            rows.append(
                f"  {e['variant_name']:<20} "
                f"WR={e['win_rate']}% PnL=${e['pnl']} DD={e['drawdown']}% "
                f"T={e['total_trades']} -> {e['decision'].upper()}"
            )
        except Exception:
            continue

    return "\n".join(rows) if rows else "No valid experiments."


def log_experiment(record: dict):
    """Log experiment to JSONL file."""
    record["timestamp"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")


# ── Telegram notifications ───────────────────────────────────────────────────

def telegram_notify(msg: str):
    """Send Telegram notification if configured."""
    token = os.environ.get("TELEGRAM_BOT_TOKEN_AUTORESEARCH")
    chat_id = os.environ.get("TELEGRAM_CHAT_ID_AUTORESEARCH")
    if not token or not chat_id:
        return
    try:
        import urllib.request
        data = json.dumps({
            "chat_id": chat_id,
            "text": f"🧬 *Strategy Discovery*\n{msg}",
            "parse_mode": "Markdown"
        }).encode()
        req = urllib.request.Request(
            f"https://api.telegram.org/bot{token}/sendMessage",
            data=data, headers={"Content-Type": "application/json"}
        )
        urllib.request.urlopen(req, timeout=5)
    except Exception as e:
        print(f"  ⚠ Telegram failed: {e}")


# ── Read source code for LLM context ─────────────────────────────────────────

def _read_source_functions() -> str:
    """Read the actual source code of backtest functions for LLM context."""
    bt_path = Path(__file__).parent / "backtest_mt5.py"
    if not bt_path.exists():
        return "(backtest_mt5.py not found)"
    src = bt_path.read_text(encoding="utf-8")
    # Extract the three key functions + helpers
    lines = src.split("\n")
    # Find calc_ema, calc_atr, compute_h4_trend, replay_signals, simulate_trades
    result_sections = []
    capture_fns = ["def calc_ema", "def calc_atr", "def compute_h4_trend", "def replay_signals", "def simulate_trades"]
    for fn_sig in capture_fns:
        start = None
        for i, line in enumerate(lines):
            if fn_sig in line:
                start = i
                break
        if start is not None:
            # Capture until next top-level def or section comment
            end = start + 1
            while end < len(lines):
                if end > start + 1 and (lines[end].startswith("def ") or lines[end].startswith("# ── ")):
                    break
                end += 1
            result_sections.append("\n".join(lines[start:end]))
    return "\n\n".join(result_sections)

_SOURCE_CODE_CACHE = None

def _get_source_code() -> str:
    global _SOURCE_CODE_CACHE
    if _SOURCE_CODE_CACHE is None:
        _SOURCE_CODE_CACHE = _read_source_functions()
    return _SOURCE_CODE_CACHE


# ── Strategy variant generation ──────────────────────────────────────────────

def generate_strategy_variant(memory: str) -> dict:
    """
    Ask Claude to generate a new strategy variant.
    Returns dict with variant_name, description, and generated_code.
    """
    recent_history = load_recent_experiments(10)
    source_code = _get_source_code()

    system = f"""You are a trading strategy inventor for BTCUSD trend-following.

Your job is to propose ONE NEW strategy variant that modifies the core trading logic - NOT just parameters.

## ACTUAL SOURCE CODE (you MUST base your modifications on this exact code)

```python
{source_code}
```

## CRITICAL CONSTRAINTS — READ CAREFULLY

1. **Pandas 3.0** — Do NOT use deprecated methods:
   - NO `fillna(method='ffill')` → use `.ffill()` instead
   - NO `fillna(method='bfill')` → use `.bfill()` instead
   - NO `Rolling.apply(axis=...)` — axis param not supported
   - NO `DataFrame.append()` → use `pd.concat()` instead
   
2. **Available imports ONLY:** pandas, numpy, datetime, math
   - NO `talib` / `ta-lib` — compute indicators manually
   - NO `scipy`, `sklearn`, or any other library
   
3. **Data columns in h1_df:** time, open, high, low, close, tick_volume, spread, real_volume
   - Volume data is in `tick_volume` column (NOT `volume`)
   - `time` column contains pandas Timestamps (datetime64)
   
4. **h4_trend_series** is a pandas Series indexed by Timestamp, values are strings ('bullish'/'bearish'/'neutral')
   - When comparing: `h4_trend_series[h4_trend_series.index <= h1_time]` — h1_time is a Timestamp, this works correctly
   
5. **calc_ema(series, period)** and **calc_atr(df, period)** are available in the module namespace
   - Use these helpers, don't reimplement them
   
6. **Signal output format** for replay_signals must match exactly:
   columns: time, bar_index, signal, entry_price, trend, atr, reason
   signal values: "buy" or "sell"
   
7. **Long-only mode:** The bot only takes BUY signals. You can still generate SELL signals (they'll be filtered), but focus on improving BUY logic.

8. **Don't be too restrictive!** The baseline generates ~17 trades per 168h. If your filter is too strict, you'll get 0 signals. Aim for 5-20 trades.

9. **MIN_SIGNAL_GAP = 4** must be preserved (4 H1 bars cooldown between signals).

## RESPONSE FORMAT

Respond with ONLY valid JSON:
{{
  "variant_name": "rsi_confirmation",
  "description": "Add RSI oversold confirmation for BUY signals",
  "function_to_modify": "replay_signals",
  "generated_code": "def replay_signals(h1_df, h4_trend_series, h1_ema_period, atr_period, min_atr):\\n    ...",
  "rationale": "RSI oversold levels could filter for higher-probability BUY entries"
}}

The generated_code must be a COMPLETE, EXECUTABLE function that replaces the original. Copy the original function and add your modifications to it."""

    # Build error summary from recent experiments to help LLM avoid repeated mistakes
    error_summary = ""
    if LOG_FILE.exists():
        lines = [l for l in LOG_FILE.read_text().splitlines() if l.strip()]
        errors = set()
        for l in lines[-20:]:
            try:
                e = json.loads(l)
                if e.get("error"):
                    errors.add(e["error"][:100])
            except Exception:
                pass
        if errors:
            error_summary = "\n\nCOMMON ERRORS FROM PAST RUNS (avoid these!):\n" + "\n".join(f"- {e}" for e in errors)

    user_content = (
        f"MEMORY_DISCOVERY.md:\n\n{memory}\n\n"
        f"RECENT EXPERIMENTS:\n{recent_history}\n"
        f"{error_summary}\n\n"
        f"Generate a new strategy variant. Avoid repeating recent experiments. "
        f"Base your code on the ACTUAL SOURCE CODE provided in the system prompt. JSON only."
    )

    response = client.messages.create(
        model=CLAUDE_MODEL,
        max_tokens=MAX_TOKENS,
        system=system,
        messages=[{"role": "user", "content": user_content}]
    )

    raw = response.content[0].text.strip()
    if "```" in raw:
        # Extract JSON from code blocks
        parts = raw.split("```")
        for part in parts:
            if part.startswith("json"):
                raw = part[4:].strip()
                break
            elif "{" in part and "}" in part:
                raw = part.strip()
                break

    return json.loads(raw)


# ── Sandboxed backtest execution ─────────────────────────────────────────────

def run_sandboxed_backtest(variant: dict, hours: int = 168, dry_run: bool = False) -> dict:
    """
    Execute backtest with dynamically loaded strategy variant.

    Creates an isolated namespace, loads the modified function, and runs backtest
    without touching the original files.
    """
    # Import backtest functions into local namespace
    from backtest_mt5 import (
        run_backtest, compute_h4_trend, replay_signals, simulate_trades,
        fetch_candles, calc_ema, calc_atr, connect_mt5, disconnect_mt5
    )

    # Create sandbox namespace with all required functions
    sandbox = {
        'pd': pd,
        'np': np,
        'calc_ema': calc_ema,
        'calc_atr': calc_atr,
        'fetch_candles': fetch_candles,
        'connect_mt5': connect_mt5,
        'disconnect_mt5': disconnect_mt5,
        'datetime': datetime,
        'UTC': datetime.timezone.utc,
        'timedelta': datetime.timedelta,
        'time': time,
        # Original functions as fallbacks
        '_original_compute_h4_trend': compute_h4_trend,
        '_original_replay_signals': replay_signals,
        '_original_simulate_trades': simulate_trades
    }

    # Execute the generated code in sandbox
    try:
        exec(variant["generated_code"], sandbox)
    except Exception as e:
        return {
            "error": f"Code execution failed: {str(e)}",
            "win_rate": 0.0,
            "pnl": -999.0,
            "drawdown": 99.9,
            "total_trades": 0
        }

    # Get the modified function
    function_name = variant["function_to_modify"]
    if function_name not in sandbox:
        return {
            "error": f"Modified function '{function_name}' not found in generated code",
            "win_rate": 0.0,
            "pnl": -999.0,
            "drawdown": 99.9,
            "total_trades": 0
        }

    modified_function = sandbox[function_name]

    # Monkey-patch the function in backtest_mt5 module temporarily
    import backtest_mt5
    original_function = getattr(backtest_mt5, function_name)

    try:
        # Replace function temporarily
        setattr(backtest_mt5, function_name, modified_function)

        # Run backtest with default parameters
        params = {}  # Empty params uses config.json defaults
        result = run_backtest("btcusd", params, hours=hours, dry_run=dry_run)

    except Exception as e:
        result = {
            "error": f"Backtest execution failed: {str(e)}",
            "win_rate": 0.0,
            "pnl": -999.0,
            "drawdown": 99.9,
            "total_trades": 0
        }
    finally:
        # Always restore original function
        setattr(backtest_mt5, function_name, original_function)

    return result


# ── Variant evaluation ───────────────────────────────────────────────────────

def evaluate_variant(memory: str, variant: dict, result: dict) -> dict:
    """
    Evaluate whether to keep or discard the strategy variant.
    Returns decision and optionally updated memory.
    """
    payload = {
        "variant": variant,
        "backtest_result": result
    }

    # Step 1: Decision only (safe JSON response)
    step1_system = """You are evaluating a trading strategy variant against baseline metrics.

Apply the decision rules from MEMORY_DISCOVERY.md strictly.

Respond with ONLY this JSON:
{"decision": "keep", "reason": "brief explanation"}

decision must be exactly "keep" or "discard"."""

    step1_response = client.messages.create(
        model=CLAUDE_MODEL,
        max_tokens=200,
        system=step1_system,
        messages=[{
            "role": "user",
            "content": (
                f"MEMORY_DISCOVERY.md:\n\n{memory}\n\n"
                f"Experiment result:\n{json.dumps(payload, indent=2)}\n\n"
                f"Apply decision rules. JSON only."
            )
        }]
    )

    raw1 = step1_response.content[0].text.strip()
    if "```" in raw1:
        parts = raw1.split("```")
        for part in parts:
            if part.startswith("json"):
                raw1 = part[4:].strip()
                break
            elif "{" in part and "}" in part:
                raw1 = part.strip()
                break

    verdict = json.loads(raw1)
    decision = verdict["decision"]
    reason = verdict.get("reason", "")

    # Step 2: If KEEP, update memory
    updated_memory = None
    if decision == "keep":
        step2_system = """You are updating MEMORY_DISCOVERY.md after a successful strategy variant.

Output ONLY the complete updated MEMORY_DISCOVERY.md text.
No explanation, no markdown fences, no preamble."""

        step2_response = client.messages.create(
            model=CLAUDE_MODEL,
            max_tokens=MAX_TOKENS * 2,
            system=step2_system,
            messages=[{
                "role": "user",
                "content": (
                    f"Current MEMORY_DISCOVERY.md:\n\n{memory}\n\n"
                    f"Strategy variant that was KEPT:\n{json.dumps(payload, indent=2)}\n\n"
                    f"Update:\n"
                    f"1. BASELINE METRICS - set new win_rate, pnl, drawdown from result\n"
                    f"2. BEST VARIANT - update with this variant's details\n"
                    f"3. EXPERIMENT COUNTERS - increment totals\n"
                    f"4. Add variant to SUCCESSFUL VARIANTS list\n\n"
                    f"Output full updated MEMORY_DISCOVERY.md now."
                )
            }]
        )
        updated_memory = step2_response.content[0].text.strip()

    return {
        "decision": decision,
        "reason": reason,
        "updated_memory": updated_memory
    }


# ── Save variant code ─────────────────────────────────────────────────────────

def save_variant_code(variant: dict, result: dict):
    """Save successful variant code to variants/ directory."""
    filename = f"{variant['variant_name']}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.py"
    filepath = VARIANTS_DIR / filename

    code_content = f'''"""
Strategy Variant: {variant["variant_name"]}
Description: {variant["description"]}
Generated: {datetime.datetime.now().isoformat()}

Backtest Results:
- Win Rate: {result["win_rate"]}%
- PnL: ${result["pnl"]}
- Drawdown: {result["drawdown"]}%
- Total Trades: {result["total_trades"]}

Rationale: {variant["rationale"]}
"""

{variant["generated_code"]}
'''

    filepath.write_text(code_content, encoding="utf-8")
    return str(filepath)


# ── Main experiment loop ─────────────────────────────────────────────────────

def run_one_experiment(hours: int = 168, dry_run: bool = False) -> dict:
    """Run a single strategy discovery experiment."""
    sep = "─" * 65
    print(f"\n{sep}")
    print(f"  🧬  Strategy Discovery: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # 1. Read memory
    memory = read_memory()
    print(f"  📖  MEMORY_DISCOVERY.md read ({len(memory):,} chars)")

    # 2. Generate variant
    print(f"  🎲  Generating strategy variant...")
    try:
        variant = generate_strategy_variant(memory)
        print(f"  🎲  Variant: {variant['variant_name']}")
        print(f"      Modifies: {variant['function_to_modify']}")
        print(f"      Idea: {variant['description']}")
    except Exception as e:
        print(f"  ❌  Variant generation failed: {e}")
        return {"error": "variant_generation_failed", "decision": "discard"}

    # 3. Run sandboxed backtest
    print(f"  ⏱   Running {hours}h sandboxed backtest {'(dry-run)' if dry_run else '(live MT5)'}...")
    t0 = time.time()
    result = run_sandboxed_backtest(variant, hours=hours, dry_run=dry_run)
    elapsed = time.time() - t0

    if "error" in result:
        print(f"  ❌  Backtest failed: {result['error']}")
        record = {
            "variant_name": variant.get("variant_name", "unknown"),
            "function_to_modify": variant.get("function_to_modify", "unknown"),
            "description": variant.get("description", "failed"),
            "error": result["error"],
            "decision": "discard",
            "reason": "execution_error"
        }
        log_experiment(record)
        return record

    print(f"  📊  WR={result['win_rate']}% | PnL=${result['pnl']} | "
          f"DD={result['drawdown']}% | Trades={result['total_trades']} ({elapsed:.1f}s)")

    # 4. Evaluate decision
    print(f"  ⚖   Evaluating against baseline...")
    outcome = evaluate_variant(memory, variant, result)
    decision = outcome["decision"]
    reason = outcome.get("reason", "")
    print(f"  {'✅ KEEP' if decision == 'keep' else '❌ DISCARD'}  {reason}")

    # 5. Update memory if kept
    variant_path = None
    if decision == "keep":
        if outcome.get("updated_memory"):
            write_memory(outcome["updated_memory"])
            print(f"  💾  MEMORY_DISCOVERY.md updated")

        # Save variant code
        variant_path = save_variant_code(variant, result)
        print(f"  💾  Variant saved: {variant_path}")

        # Extended validation if --validate was set
        validate_hours = getattr(args, 'validate', 0) if 'args' in dir() else 0
        # Check global _validate_hours set by main()
        validate_hours = globals().get('_validate_hours', 0)
        if validate_hours > 0 and variant_path:
            print(f"\n  🔬  Running extended validation ({validate_hours}h)...")
            passed, val_result = run_extended_validation(variant_path, validate_hours)
            if passed:
                print(f"  ✅  Extended validation PASSED")
                telegram_notify(
                    f"✅ *VALIDATED*: `{variant['variant_name']}`\n"
                    f"Extended {validate_hours}h: WR `{val_result.get('variant_wr', '?')}%` PnL `${val_result.get('variant_pnl', '?')}`\n"
                    f"vs Baseline: WR `{val_result.get('baseline_wr', '?')}%` PnL `${val_result.get('baseline_pnl', '?')}`\n"
                    f"_{variant['description']}_\n\n"
                    f"Ready for PR. Awaiting human approval."
                )
            else:
                print(f"  ❌  Extended validation FAILED — downgrading to discard")
                decision = "discard"
                reason = f"Failed extended {validate_hours}h validation: {val_result.get('reason', 'underperformed baseline')}"
                # Remove saved variant
                try:
                    os.remove(variant_path)
                    print(f"  🗑  Removed failed variant: {variant_path}")
                except Exception:
                    pass
                variant_path = None
                telegram_notify(
                    f"❌ *FAILED VALIDATION*: `{variant['variant_name']}`\n"
                    f"Passed initial but failed {validate_hours}h extended test.\n"
                    f"_{reason}_"
                )

        if decision == "keep":  # Still keep after validation
            # Notify success
            telegram_notify(
                f"*KEPT*: `{variant['variant_name']}`\n"
                f"WR: `{result['win_rate']}%` PnL: `${result['pnl']}` DD: `{result['drawdown']}%`\n"
                f"_{variant['description']}_"
            )
    else:
        # Notify failure
        telegram_notify(
            f"*DISCARD*: `{variant['variant_name']}`\n"
            f"WR: `{result['win_rate']}%` PnL: `${result['pnl']}`\n"
            f"_{reason}_"
        )

    # 6. Build record
    record = {
        "variant_name": variant["variant_name"],
        "function_to_modify": variant["function_to_modify"],
        "description": variant["description"],
        "rationale": variant["rationale"],
        "generated_code_lines": len(variant["generated_code"].split('\n')),
        **result,
        "decision": decision,
        "reason": reason,
        "variant_path": variant_path,
        "duration_s": round(elapsed, 1)
    }

    # 7. Log experiment
    log_experiment(record)
    print(f"  📝  Logged → {LOG_FILE}")

    return record


# ── Main program ──────────────────────────────────────────────────────────────

def generate_summary() -> str:
    """Generate summary of strategy discovery experiments."""
    if not LOG_FILE.exists():
        return "No experiments logged."

    lines = [l for l in LOG_FILE.read_text().splitlines() if l.strip()]
    if not lines:
        return "No experiments logged."

    exps = []
    for line in lines:
        try:
            exps.append(json.loads(line))
        except:
            continue

    if not exps:
        return "No valid experiments."

    kept = [e for e in exps if e["decision"] == "keep"]
    total = len(exps)
    keep_rate = len(kept) / total * 100 if total > 0 else 0

    summary = f"*Strategy Discovery Summary*\n"
    summary += f"Total: {total} | Kept: {len(kept)} ({keep_rate:.0f}%)\n\n"

    if kept:
        best = max(kept, key=lambda x: x.get("pnl", -999))
        summary += f"*Best:* `{best['variant_name']}`\n"
        summary += f"WR: `{best['win_rate']}%` PnL: `${best['pnl']}` DD: `{best['drawdown']}%`\n"
        summary += f"_{best['description']}_\n\n"
        summary += "*All successful variants:*\n"
        for e in kept:
            summary += f"  `{e['variant_name']}` WR={e['win_rate']}% PnL=${e['pnl']}\n"
    else:
        summary += "_No successful variants found — baseline strategy remains optimal._"

    return summary


def run_extended_validation(variant_path: str, hours: int) -> tuple:
    """
    Run extended backtest comparing variant vs baseline over a longer window.
    Returns (passed: bool, details: dict)
    """
    import subprocess
    try:
        result = subprocess.run(
            ["python", "validate_variant.py", str(variant_path), "--hours", str(hours)],
            capture_output=True, text=True, timeout=300, encoding="utf-8", errors="replace"
        )
        output = result.stdout
        print(output)

        # Parse results from output
        details = {}
        for line in output.split("\n"):
            line = line.strip()
            if line.startswith("WR=") and "BASELINE" not in line:
                pass  # handled below

        # Extract baseline and variant metrics
        in_baseline = False
        in_variant = False
        for line in output.split("\n"):
            if "BASELINE" in line:
                in_baseline = True
                in_variant = False
            elif "VARIANT" in line:
                in_variant = True
                in_baseline = False
            elif "WR=" in line:
                parts = line.strip()
                # Parse: WR=69.0% | PnL=$179.85 | Trades=71 | DD=0.24%
                try:
                    wr = float(parts.split("WR=")[1].split("%")[0])
                    pnl = float(parts.split("PnL=$")[1].split(" ")[0])
                    if in_baseline:
                        details["baseline_wr"] = wr
                        details["baseline_pnl"] = pnl
                    elif in_variant:
                        details["variant_wr"] = wr
                        details["variant_pnl"] = pnl
                except (IndexError, ValueError):
                    pass

        passed = "PASSES" in output
        if not passed:
            details["reason"] = "Variant did not outperform baseline over extended window"

        return passed, details

    except subprocess.TimeoutExpired:
        return False, {"reason": "Validation timed out"}
    except Exception as e:
        return False, {"reason": f"Validation error: {str(e)}"}


def main():
    """Main program entry point."""
    parser = argparse.ArgumentParser(description="Strategy Discovery Pipeline")
    parser.add_argument("--once", action="store_true", help="Single experiment then exit")
    parser.add_argument("--dry-run", action="store_true", help="Simulate without MT5")
    parser.add_argument("--hours", type=int, default=BACKTEST_HOURS, help="Backtest window (hours)")
    parser.add_argument("--delay", type=int, default=120, help="Seconds between experiments")
    parser.add_argument("--max-hours", type=float, default=0, help="Stop after N hours (0 = no limit)")
    parser.add_argument("--auto-stop", type=int, default=20, help="Stop after N consecutive failures")
    parser.add_argument("--validate", type=int, default=0, help="Extended validation hours for keepers (0 = skip)")
    args = parser.parse_args()

    # Set global so run_one_experiment can access it
    globals()['_validate_hours'] = args.validate

    print("=" * 65)
    print("  Strategy Discovery Pipeline — BTCUSD")
    print(f"  Model: {CLAUDE_MODEL}")
    print(f"  Memory: {MEMORY_FILE}")
    print(f"  Log: {LOG_FILE}")
    print(f"  Variants: {VARIANTS_DIR}/")
    print(f"  Window: {args.hours}h backtest")
    print(f"  Mode: {'DRY-RUN (simulated)' if args.dry_run else 'LIVE (MT5)'}")
    print("=" * 65)

    if not MEMORY_FILE.exists():
        print(f"❌  {MEMORY_FILE} not found. Create it first.")
        sys.exit(1)

    if args.once:
        run_one_experiment(hours=args.hours, dry_run=args.dry_run)
        return

    # Send start notification
    telegram_notify(
        f"🧬 *Strategy Discovery started*\n"
        f"Window: `{args.hours}h` | Mode: {'DRY-RUN' if args.dry_run else 'LIVE'}\n"
        f"Auto-stop: `{args.auto_stop}` failures"
    )

    start_time = time.time()
    deadline = start_time + (args.max_hours * 3600) if args.max_hours > 0 else None
    count = kept = consecutive_failures = 0

    try:
        while True:
            # Time limit check
            if deadline and time.time() >= deadline:
                print(f"\n  ⏱  Max runtime reached. Stopping.")
                break

            # Run experiment
            exp = run_one_experiment(hours=args.hours, dry_run=args.dry_run)
            count += 1

            if exp["decision"] == "keep":
                kept += 1
                consecutive_failures = 0
            else:
                consecutive_failures += 1

            # Auto-stop on convergence
            if args.auto_stop > 0 and consecutive_failures >= args.auto_stop:
                print(f"\n  🎯  Convergence detected ({consecutive_failures} consecutive failures).")
                break

            keep_rate = kept / count * 100
            print(f"\n  📈  Session: {count} experiments | {kept} kept ({keep_rate:.0f}%) | "
                  f"failures: {consecutive_failures}")

            if not args.once:
                print(f"  💤  Next experiment in {args.delay}s...")
                time.sleep(args.delay)

    except KeyboardInterrupt:
        print(f"\n\n  🛑  Stopped. {count} experiments | {kept} kept.")

    # Send final summary
    summary = generate_summary()
    telegram_notify(
        f"🧬 *Strategy Discovery finished*\n"
        f"{count} experiments, {kept} kept ({kept/max(count,1)*100:.0f}%)\n\n"
        f"{summary}"
    )


if __name__ == "__main__":
    main()