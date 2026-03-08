"""
deploy.py — AutoResearch Deploy Pipeline
Called by autotrader.py after a Telegram-approved KEEP experiment.

What it does:
  1. Applies new param to config.json or trend_strategy.py
  2. Commits and pushes to GitHub
  3. Restarts PM2 bot
  4. Waits 2 hours then verifies live performance
  5. Rolls back if live performance is worse than expected

Usage (standalone for testing):
  python deploy.py --param tp_atr_multiplier --value 1.75 --dry-run
  python deploy.py --param h4_ema_fast --value 15 --dry-run
"""

import os
import sys
import json
import time
import re
import subprocess
import argparse
import datetime
from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────────────────
BOT_DIR         = Path(__file__).parent
CONFIG_FILE     = BOT_DIR / "config.json"
TREND_FILE      = BOT_DIR / "trend_strategy.py"
TRADE_LOG       = BOT_DIR / "logs" / "trade_log.csv"
PM2_BOT_NAME    = "bot-btcusd"
VERIFY_WAIT_SEC = 7200   # 2 hours before live verification
MIN_LIVE_TRADES = 3      # need at least this many live trades to verify

# ── Params that live in config.json vs trend_strategy.py ─────────────────────
CONFIG_PARAMS   = {"sl_atr_multiplier", "tp_atr_multiplier"}
STRATEGY_PARAMS = {"h4_ema_fast", "h4_ema_slow", "h1_ema_period"}


# ── Telegram helpers ───────────────────────────────────────────────────────────

def _tg_send(msg: str) -> bool:
    token   = os.environ.get("TELEGRAM_BOT_TOKEN_AUTORESEARCH")
    chat_id = os.environ.get("TELEGRAM_CHAT_ID_AUTORESEARCH")
    if not token or not chat_id:
        return False
    try:
        import urllib.request
        data = json.dumps({
            "chat_id": chat_id, "text": msg, "parse_mode": "Markdown"
        }).encode()
        req = urllib.request.Request(
            f"https://api.telegram.org/bot{token}/sendMessage",
            data=data, headers={"Content-Type": "application/json"}
        )
        urllib.request.urlopen(req, timeout=5)
        return True
    except Exception as e:
        print(f"  ⚠ Telegram failed: {e}")
        return False


# ── config.json updater ───────────────────────────────────────────────────────

def apply_config_param(param: str, value: float, dry_run: bool = False) -> bool:
    """Update sl_atr_multiplier or tp_atr_multiplier in config.json."""
    if not CONFIG_FILE.exists():
        print(f"  ❌ {CONFIG_FILE} not found")
        return False

    with open(CONFIG_FILE) as f:
        cfg = json.load(f)

    old = cfg["dynamic_sltp"].get(param)
    cfg["dynamic_sltp"][param] = value
    cfg["dynamic_sltp"]["_autotrader_note"] = (
        f"Updated by AutoResearch {datetime.date.today()} "
        f"— {param}: {old} → {value}"
    )

    if dry_run:
        print(f"  [DRY] Would set config.json dynamic_sltp.{param} = {value}")
        return True

    with open(CONFIG_FILE, "w") as f:
        json.dump(cfg, f, indent=4)

    print(f"  ✅ config.json: {param} {old} → {value}")
    return True


# ── trend_strategy.py updater ─────────────────────────────────────────────────

def _read_current_ema_values() -> dict:
    """Read current EMA span values from trend_strategy.py."""
    content = TREND_FILE.read_text()
    spans   = re.findall(r"ewm\(span=(\d+)", content)
    spans   = [int(s) for s in spans]
    result  = {}
    if len(spans) >= 4:
        # Order in file: H4 fast, H4 slow (50), H1 entry, H1 slow (50)
        result["h4_ema_fast"]  = spans[0]
        result["h4_ema_slow"]  = spans[1]
        result["h1_ema_period"] = spans[2]
    return result


def apply_strategy_param(param: str, value: int, dry_run: bool = False) -> bool:
    """
    Update EMA span values in trend_strategy.py.
    Handles h4_ema_fast, h4_ema_slow, h1_ema_period.
    """
    if not TREND_FILE.exists():
        print(f"  ❌ {TREND_FILE} not found")
        return False

    content  = TREND_FILE.read_text()
    current  = _read_current_ema_values()
    old_val  = current.get(param)

    if old_val is None:
        print(f"  ❌ Could not detect current value of {param}")
        return False

    if old_val == value:
        print(f"  ℹ {param} already at {value}, no change needed")
        return True

    # Build EMA variable name from span value  e.g. span=15 → ema15
    old_varname = f"ema{old_val}"
    new_varname = f"ema{value}"

    # Which section to update
    if param == "h4_ema_fast":
        # Replace first occurrence of the fast EMA span in get_trend_direction()
        # Pattern: ewm(span=OLD in the H4 section (before H1 section)
        h4_section_end = content.find("def get_h1_entry_signal")
        h4_part   = content[:h4_section_end]
        rest_part = content[h4_section_end:]

        new_h4 = h4_part.replace(
            f"ewm(span={old_val}", f"ewm(span={value}", 1
        ).replace(old_varname, new_varname)

        new_content = new_h4 + rest_part

    elif param == "h4_ema_slow":
        # Replace ema50 span in H4 section only (first span=50)
        h4_section_end = content.find("def get_h1_entry_signal")
        h4_part   = content[:h4_section_end]
        rest_part = content[h4_section_end:]

        new_h4 = re.sub(
            rf"ewm\(span={old_val}",
            f"ewm(span={value}",
            h4_part, count=1
        ).replace(f"ema{old_val}", f"ema{value}")

        new_content = new_h4 + rest_part

    elif param == "h1_ema_period":
        # Replace entry EMA span in get_h1_entry_signal() only
        h1_start = content.find("def get_h1_entry_signal")
        h1_end   = content.find("def _calc_atr")
        h4_part   = content[:h1_start]
        h1_part   = content[h1_start:h1_end]
        rest_part = content[h1_end:]

        new_h1 = h1_part.replace(
            f"ewm(span={old_val}", f"ewm(span={value}", 1
        ).replace(old_varname, new_varname)

        new_content = h4_part + new_h1 + rest_part
    else:
        print(f"  ❌ Unknown strategy param: {param}")
        return False

    if dry_run:
        print(f"  [DRY] Would set trend_strategy.py {param}: {old_val} → {value} "
              f"({old_varname} → {new_varname})")
        return True

    TREND_FILE.write_text(new_content)
    print(f"  ✅ trend_strategy.py: {param} {old_val} → {value}")
    return True


# ── Git operations ─────────────────────────────────────────────────────────────

def git_commit_push(param: str, old_val, new_val,
                    wr: float, pnl: float, dry_run: bool = False) -> bool:
    """Commit changed files and push to GitHub."""
    try:
        repo_root = BOT_DIR.parent

        if dry_run:
            print(f"  [DRY] Would git commit: {param} {old_val}→{new_val}")
            return True

        msg = (
            f"perf(autotrader): {param} {old_val}→{new_val}\n\n"
            f"AutoResearch backtest result:\n"
            f"  win_rate={wr}%  pnl=${pnl}\n"
            f"  deployed {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}"
        )

        # Stage only the files we changed
        files_to_add = []
        if param in CONFIG_PARAMS:
            files_to_add.append(str(CONFIG_FILE.relative_to(repo_root)))
        if param in STRATEGY_PARAMS:
            files_to_add.append(str(TREND_FILE.relative_to(repo_root)))

        subprocess.run(
            ["git", "add"] + files_to_add,
            cwd=repo_root, check=True, capture_output=True
        )
        subprocess.run(
            ["git", "commit", "-m", msg],
            cwd=repo_root, check=True, capture_output=True
        )
        subprocess.run(
            ["git", "push", "origin", "main"],
            cwd=repo_root, check=True, capture_output=True
        )

        print(f"  ✅ Pushed to GitHub: {param} {old_val}→{new_val}")
        return True

    except subprocess.CalledProcessError as e:
        print(f"  ❌ Git error: {e.stderr.decode()}")
        return False


def git_revert(dry_run: bool = False) -> bool:
    """Revert last commit (rollback on bad live performance)."""
    try:
        repo_root = BOT_DIR.parent
        if dry_run:
            print("  [DRY] Would git revert HEAD")
            return True
        subprocess.run(
            ["git", "revert", "HEAD", "--no-edit"],
            cwd=repo_root, check=True, capture_output=True
        )
        subprocess.run(
            ["git", "push", "origin", "main"],
            cwd=repo_root, check=True, capture_output=True
        )
        print("  ✅ Rollback committed and pushed")
        return True
    except subprocess.CalledProcessError as e:
        print(f"  ❌ Git revert error: {e.stderr.decode()}")
        return False


# ── PM2 operations ────────────────────────────────────────────────────────────

def pm2_restart(dry_run: bool = False) -> bool:
    """Restart the live bot via PM2."""
    try:
        if dry_run:
            print(f"  [DRY] Would pm2 restart {PM2_BOT_NAME}")
            return True
        result = subprocess.run(
            ["pm2", "restart", PM2_BOT_NAME],
            capture_output=True, text=True
        )
        if result.returncode == 0:
            print(f"  ✅ PM2 restarted {PM2_BOT_NAME}")
            return True
        else:
            print(f"  ❌ PM2 error: {result.stderr}")
            return False
    except FileNotFoundError:
        print("  ❌ pm2 not found in PATH")
        return False


# ── Live performance verifier ─────────────────────────────────────────────────

def get_live_performance(since_timestamp: str) -> dict:
    """
    Read trade_log.csv and compute win rate + PnL for trades
    executed after since_timestamp.
    """
    if not TRADE_LOG.exists():
        return {"trades": 0, "win_rate": 0.0, "pnl": 0.0}

    try:
        import csv
        trades = []
        with open(TRADE_LOG, newline="") as f:
            reader = csv.DictReader(f, delimiter=";")
            for row in reader:
                ts = row.get("close_time") or row.get("open_time", "")
                if ts >= since_timestamp:
                    trades.append(row)

        if not trades:
            return {"trades": 0, "win_rate": 0.0, "pnl": 0.0}

        wins  = sum(1 for t in trades if float(t.get("profit", 0)) > 0)
        total = len(trades)
        pnl   = sum(float(t.get("profit", 0)) for t in trades)

        return {
            "trades":   total,
            "win_rate": round(wins / total * 100, 1) if total > 0 else 0.0,
            "pnl":      round(pnl, 2),
        }
    except Exception as e:
        print(f"  ⚠ Could not read trade log: {e}")
        return {"trades": 0, "win_rate": 0.0, "pnl": 0.0}


# ── Main deploy flow ──────────────────────────────────────────────────────────

def deploy(
    param:         str,
    new_value,
    old_value,
    backtest_wr:   float,
    backtest_pnl:  float,
    dry_run:       bool = False,
) -> bool:
    """
    Full deploy pipeline:
    1. Apply param to config/strategy file
    2. Git commit + push
    3. PM2 restart
    4. Wait VERIFY_WAIT_SEC
    5. Check live performance
    6. Rollback if bad

    Returns True if deploy succeeded and live perf was acceptable.
    """
    deploy_time = datetime.datetime.now(datetime.timezone.utc).isoformat()
    print(f"\n  🚀 DEPLOYING: {param} {old_value} → {new_value}")
    print(f"  {'[DRY RUN] ' if dry_run else ''}Target: WR={backtest_wr}% PnL=${backtest_pnl}")

    # Step 1: Apply param
    success = False
    if param in CONFIG_PARAMS:
        success = apply_config_param(param, float(new_value), dry_run)
    elif param in STRATEGY_PARAMS:
        success = apply_strategy_param(param, int(new_value), dry_run)
    else:
        print(f"  ❌ Unknown param: {param}")
        return False

    if not success:
        _tg_send(f"❌ *Deploy FAILED* — could not apply `{param}` to file")
        return False

    # Step 2: Git push
    if not git_commit_push(param, old_value, new_value,
                           backtest_wr, backtest_pnl, dry_run):
        _tg_send(f"❌ *Deploy FAILED* — git push error for `{param}`")
        return False

    # Step 3: PM2 restart
    if not pm2_restart(dry_run):
        _tg_send(f"⚠️ *Deploy WARNING* — `{param}` pushed but PM2 restart failed\n"
                 f"Run manually: `pm2 restart {PM2_BOT_NAME}`")

    _tg_send(
        f"🚀 *Deployed* `{param}` {old_value}→{new_value}\n"
        f"Bot restarted. Will verify in {VERIFY_WAIT_SEC//3600}h.\n"
        f"Backtest: WR=`{backtest_wr}%` PnL=`${backtest_pnl}`"
    )

    if dry_run:
        print("  [DRY] Skipping wait + verification")
        return True

    # Step 4: Wait for live trades
    print(f"  ⏳ Waiting {VERIFY_WAIT_SEC//3600}h for live performance data...")
    time.sleep(VERIFY_WAIT_SEC)

    # Step 5: Check live performance
    live = get_live_performance(deploy_time)
    print(f"  📊 Live: {live['trades']} trades | WR={live['win_rate']}% | PnL=${live['pnl']}")

    if live["trades"] < MIN_LIVE_TRADES:
        _tg_send(
            f"⏳ *Verification pending* — only {live['trades']} live trades after 2h\n"
            f"Not enough data to verify. Keeping change, will monitor."
        )
        return True

    # Step 6: Rollback check
    # Rollback if live win rate is more than 10 points below backtest
    wr_gap = backtest_wr - live["win_rate"]
    if wr_gap > 10.0 and live["pnl"] < 0:
        print(f"  ⚠ Live WR {live['win_rate']}% vs backtest {backtest_wr}% — gap={wr_gap:.1f}%")
        print(f"  🔄 Rolling back...")
        git_revert()
        pm2_restart()
        _tg_send(
            f"🔄 *ROLLED BACK* `{param}` {new_value}→{old_value}\n"
            f"Live WR `{live['win_rate']}%` vs backtest `{backtest_wr}%` (gap={wr_gap:.1f}%)\n"
            f"Live PnL: `${live['pnl']}`"
        )
        return False

    _tg_send(
        f"✅ *Verified* `{param}` {old_value}→{new_value}\n"
        f"Live: WR=`{live['win_rate']}%` PnL=`${live['pnl']}` ({live['trades']} trades)\n"
        f"Backtest was: WR=`{backtest_wr}%` PnL=`${backtest_pnl}`"
    )
    return True


# ── CLI for standalone testing ─────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AutoResearch Deploy Tool")
    parser.add_argument("--param",   required=True, help="Param name to deploy")
    parser.add_argument("--value",   required=True, help="New param value")
    parser.add_argument("--old",     default=None,  help="Old param value (for commit message)")
    parser.add_argument("--wr",      type=float, default=0.0, help="Backtest win rate")
    parser.add_argument("--pnl",     type=float, default=0.0, help="Backtest PnL")
    parser.add_argument("--dry-run", action="store_true", help="Simulate without writing files")
    args = parser.parse_args()

    val = float(args.value) if "." in args.value else int(args.value)
    old = float(args.old) if args.old and "." in args.old else (int(args.old) if args.old else "?")

    print("=" * 55)
    print(f"  AutoResearch Deploy")
    print(f"  param  : {args.param}")
    print(f"  change : {old} → {val}")
    print(f"  dry-run: {args.dry_run}")
    print("=" * 55)

    deploy(
        param        = args.param,
        new_value    = val,
        old_value    = old,
        backtest_wr  = args.wr,
        backtest_pnl = args.pnl,
        dry_run      = args.dry_run,
    )
