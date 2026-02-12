#!/usr/bin/env python3
"""
Auto-retrain scheduler for MT5 trading bots.
Checks model age, retrains if needed, validates accuracy, and restarts bots via PM2.

Usage:
    python auto_retrain.py                        # Retrain all bots if due
    python auto_retrain.py --bot btcusd           # Retrain specific bot
    python auto_retrain.py --dry-run              # Check what needs retraining
    python auto_retrain.py --force                # Force retrain regardless of age
    python auto_retrain.py --force --notify       # Force retrain and send Telegram summary
"""

import argparse
import datetime
import glob
import json
import logging
import os
import shutil
import subprocess
import sys
import time

# ── Configuration ──────────────────────────────────────────────────────────────

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
RETRAIN_INTERVAL_DAYS = 7
ACCURACY_DROP_THRESHOLD = 0.05  # Allow up to 5% drop
TRAINING_TIMEOUT_SECONDS = 600  # 10 minutes

BOT_CONFIG = {
    "btcusd": {"pm2_name": "bot-btcusd", "pm2_id": 0},
    "xauusd": {"pm2_name": "bot-xauusd", "pm2_id": 1},
    "eurusd": {"pm2_name": "bot-eurusd", "pm2_id": 2},
}

# ── Logging Setup ─────────────────────────────────────────────────────────────

log_file = os.path.join(ROOT_DIR, "retrain.log")
logger = logging.getLogger("auto_retrain")
logger.setLevel(logging.INFO)
formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S")

fh = logging.FileHandler(log_file, encoding="utf-8")
fh.setFormatter(formatter)
logger.addHandler(fh)

sh = logging.StreamHandler(open(sys.stdout.fileno(), mode='w', encoding='utf-8', closefd=False))
sh.setFormatter(formatter)
logger.addHandler(sh)


# ── Helper Functions ───────────────────────────────────────────────────────────

def find_metadata(bot_name: str) -> dict | None:
    """Find and load the metadata JSON for a bot."""
    models_dir = os.path.join(ROOT_DIR, bot_name, "models")
    pattern = os.path.join(models_dir, f"*_{bot_name}_metadata.json")
    files = glob.glob(pattern)
    if not files:
        return None
    with open(files[0], "r") as f:
        data = json.load(f)
    data["_path"] = files[0]
    data["_model_file"] = files[0].replace("_metadata.json", ".pkl")
    return data


def get_model_age_days(metadata: dict) -> float:
    """Return how many days since the model was trained."""
    trained_at = datetime.datetime.fromisoformat(metadata["trained_at"])
    now = datetime.datetime.now()
    return (now - trained_at).total_seconds() / 86400


def get_accuracy(metadata: dict) -> float:
    """Extract the test accuracy from metadata."""
    return metadata.get("performance_metrics", {}).get("test_metrics", {}).get("accuracy", 0.0)


def backup_model(bot_name: str, metadata: dict) -> str | None:
    """Back up the current model files to models/backup/. Returns backup dir."""
    models_dir = os.path.join(ROOT_DIR, bot_name, "models")
    backup_dir = os.path.join(models_dir, "backup")
    os.makedirs(backup_dir, exist_ok=True)

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_subdir = os.path.join(backup_dir, timestamp)
    os.makedirs(backup_subdir, exist_ok=True)

    # Back up all model-related files (pkl, metadata, scaler)
    for ext_pattern in [f"*_{bot_name}.pkl", f"*_{bot_name}_metadata.json", f"scaler_{bot_name}.pkl"]:
        for f in glob.glob(os.path.join(models_dir, ext_pattern)):
            shutil.copy2(f, backup_subdir)

    logger.info(f"  Backed up models to {backup_subdir}")
    return backup_subdir


def restore_backup(bot_name: str, backup_dir: str):
    """Restore model files from backup."""
    models_dir = os.path.join(ROOT_DIR, bot_name, "models")
    for f in os.listdir(backup_dir):
        src = os.path.join(backup_dir, f)
        dst = os.path.join(models_dir, f)
        shutil.copy2(src, dst)
    logger.info(f"  Restored models from {backup_dir}")


def run_training(bot_name: str) -> tuple[bool, str]:
    """Run train_ml_model.py --refresh for the bot. Returns (success, output)."""
    bot_dir = os.path.join(ROOT_DIR, bot_name)
    cmd = [sys.executable, "train_ml_model.py", "--refresh"]

    logger.info(f"  Running: {' '.join(cmd)} (cwd={bot_dir})")
    try:
        result = subprocess.run(
            cmd,
            cwd=bot_dir,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=TRAINING_TIMEOUT_SECONDS,
        )
        output = result.stdout + "\n" + result.stderr
        if result.returncode != 0:
            logger.error(f"  Training failed (exit code {result.returncode})")
            logger.error(f"  Output: {output[-2000:]}")
            return False, output
        return True, output
    except subprocess.TimeoutExpired:
        logger.error(f"  Training timed out after {TRAINING_TIMEOUT_SECONDS}s")
        return False, "TIMEOUT"
    except Exception as e:
        logger.error(f"  Training exception: {e}")
        return False, str(e)


def restart_pm2(bot_name: str) -> bool:
    """Restart the bot via PM2."""
    pm2_name = BOT_CONFIG[bot_name]["pm2_name"]
    try:
        result = subprocess.run(
            f"pm2 restart {pm2_name}",
            capture_output=True, text=True, timeout=30, shell=True,
        )
        if result.returncode == 0:
            logger.info(f"  PM2 restart {pm2_name}: OK")
            return True
        else:
            logger.error(f"  PM2 restart failed: {result.stderr}")
            return False
    except Exception as e:
        logger.error(f"  PM2 restart exception: {e}")
        return False


def update_retrain_history(bot_name: str, entry: dict):
    """Append an entry to the bot's retrain_history.json."""
    history_path = os.path.join(ROOT_DIR, bot_name, "models", "retrain_history.json")
    history = []
    if os.path.exists(history_path):
        with open(history_path, "r") as f:
            history = json.load(f)
    history.append(entry)
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)


# ── Main Retrain Logic ────────────────────────────────────────────────────────

def process_bot(bot_name: str, force: bool, dry_run: bool) -> dict:
    """Process a single bot. Returns a summary dict."""
    logger.info(f"{'='*60}")
    logger.info(f"Processing: {bot_name.upper()}")
    logger.info(f"{'='*60}")

    result = {
        "bot": bot_name,
        "status": "skipped",
        "old_accuracy": None,
        "new_accuracy": None,
        "age_days": None,
        "message": "",
    }

    # Load metadata
    metadata = find_metadata(bot_name)
    if not metadata:
        result["status"] = "error"
        result["message"] = "No metadata file found"
        logger.warning(f"  No metadata found for {bot_name}")
        return result

    age_days = get_model_age_days(metadata)
    old_accuracy = get_accuracy(metadata)
    result["age_days"] = round(age_days, 1)
    result["old_accuracy"] = round(old_accuracy * 100, 2)

    logger.info(f"  Model age: {age_days:.1f} days | Accuracy: {old_accuracy*100:.2f}%")
    logger.info(f"  Model type: {metadata.get('model_type', 'unknown')}")

    needs_retrain = age_days > RETRAIN_INTERVAL_DAYS or force
    if not needs_retrain:
        result["message"] = f"Model is {age_days:.1f} days old (threshold: {RETRAIN_INTERVAL_DAYS}d)"
        logger.info(f"  Skipping - not due for retraining")
        return result

    reason = "forced" if force and age_days <= RETRAIN_INTERVAL_DAYS else f"age {age_days:.1f}d > {RETRAIN_INTERVAL_DAYS}d"
    logger.info(f"  Retraining needed: {reason}")

    if dry_run:
        result["status"] = "dry-run"
        result["message"] = f"Would retrain ({reason})"
        logger.info(f"  [DRY-RUN] Would retrain this bot")
        return result

    # Back up
    backup_dir = backup_model(bot_name, metadata)

    # Train
    success, output = run_training(bot_name)
    if not success:
        restore_backup(bot_name, backup_dir)
        result["status"] = "train_failed"
        result["message"] = f"Training failed, backup restored"
        update_retrain_history(bot_name, {
            "date": datetime.datetime.now().isoformat(),
            "old_accuracy": old_accuracy,
            "new_accuracy": None,
            "status": "train_failed",
        })
        return result

    # Compare accuracy
    new_metadata = find_metadata(bot_name)
    if not new_metadata:
        restore_backup(bot_name, backup_dir)
        result["status"] = "metadata_missing"
        result["message"] = "New metadata not found after training, backup restored"
        return result

    new_accuracy = get_accuracy(new_metadata)
    result["new_accuracy"] = round(new_accuracy * 100, 2)
    logger.info(f"  Old accuracy: {old_accuracy*100:.2f}% → New accuracy: {new_accuracy*100:.2f}%")

    accuracy_change = new_accuracy - old_accuracy
    if new_accuracy < old_accuracy - ACCURACY_DROP_THRESHOLD:
        # Accuracy dropped too much - rollback
        restore_backup(bot_name, backup_dir)
        result["status"] = "rolled_back"
        result["message"] = (
            f"Accuracy dropped {accuracy_change*100:+.2f}% (>{ACCURACY_DROP_THRESHOLD*100}% threshold). "
            f"Rolled back to previous model."
        )
        logger.warning(f"  {result['message']}")
        update_retrain_history(bot_name, {
            "date": datetime.datetime.now().isoformat(),
            "old_accuracy": old_accuracy,
            "new_accuracy": new_accuracy,
            "status": "rolled_back",
        })
        return result

    # Accept new model
    result["status"] = "retrained"
    result["message"] = f"Accuracy: {old_accuracy*100:.2f}% → {new_accuracy*100:.2f}% ({accuracy_change*100:+.2f}%)"
    logger.info(f"  Accepted new model: {result['message']}")

    # Restart PM2
    if restart_pm2(bot_name):
        result["message"] += " | PM2 restarted"
    else:
        result["message"] += " | PM2 restart FAILED"

    update_retrain_history(bot_name, {
        "date": datetime.datetime.now().isoformat(),
        "old_accuracy": old_accuracy,
        "new_accuracy": new_accuracy,
        "status": "retrained",
    })

    return result


def format_summary(results: list[dict]) -> str:
    """Format results into a readable summary."""
    lines = ["🤖 MT5 Auto-Retrain Summary", f"📅 {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}", ""]

    status_emoji = {
        "retrained": "✅", "skipped": "⏭️", "dry-run": "🔍",
        "rolled_back": "⚠️", "train_failed": "❌", "error": "❌",
        "metadata_missing": "❌",
    }

    for r in results:
        emoji = status_emoji.get(r["status"], "❓")
        lines.append(f"{emoji} {r['bot'].upper()} [{r['status']}]")
        if r["age_days"] is not None:
            lines.append(f"   Age: {r['age_days']}d | Old acc: {r['old_accuracy']}%")
        if r["new_accuracy"] is not None:
            lines.append(f"   New acc: {r['new_accuracy']}%")
        if r["message"]:
            lines.append(f"   {r['message']}")
        lines.append("")

    return "\n".join(lines)


def send_telegram_notify(summary: str):
    """Try to send summary via Telegram using the bot's notification config."""
    # Look for telegram config in any bot's mt5_auth.json
    for bot_name in BOT_CONFIG:
        auth_path = os.path.join(ROOT_DIR, bot_name, "mt5_auth.json")
        if os.path.exists(auth_path):
            with open(auth_path, "r") as f:
                auth = json.load(f)
            token = auth.get("telegram_bot_token") or auth.get("telegram_token")
            chat_id = auth.get("telegram_chat_id")
            if token and chat_id:
                try:
                    import urllib.request
                    url = f"https://api.telegram.org/bot{token}/sendMessage"
                    data = json.dumps({"chat_id": chat_id, "text": summary, "parse_mode": "HTML"}).encode()
                    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
                    urllib.request.urlopen(req, timeout=10)
                    logger.info("Telegram notification sent")
                    return True
                except Exception as e:
                    logger.warning(f"Telegram send failed: {e}")
                    return False
    logger.warning("No Telegram config found in mt5_auth.json files")
    return False


# ── Entry Point ────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Auto-retrain MT5 trading bot ML models")
    parser.add_argument("--bot", choices=["btcusd", "xauusd", "eurusd", "all"], default="all",
                        help="Which bot to retrain (default: all)")
    parser.add_argument("--force", action="store_true", help="Retrain even if model is not due")
    parser.add_argument("--dry-run", action="store_true", help="Check what needs retraining without doing it")
    parser.add_argument("--notify", action="store_true", help="Send summary to Telegram")
    args = parser.parse_args()

    logger.info(f"Auto-retrain started | bot={args.bot} force={args.force} dry_run={args.dry_run}")

    bots = list(BOT_CONFIG.keys()) if args.bot == "all" else [args.bot]
    results = []

    for bot_name in bots:
        try:
            result = process_bot(bot_name, args.force, args.dry_run)
            results.append(result)
        except Exception as e:
            logger.error(f"Unexpected error processing {bot_name}: {e}", exc_info=True)
            results.append({
                "bot": bot_name, "status": "error", "old_accuracy": None,
                "new_accuracy": None, "age_days": None, "message": str(e),
            })

    summary = format_summary(results)
    try:
        print("\n" + summary)
    except UnicodeEncodeError:
        print("\n" + summary.encode("ascii", errors="replace").decode("ascii"))

    if args.notify:
        send_telegram_notify(summary)

    # Exit with error code if any bot failed
    if any(r["status"] in ("error", "train_failed", "metadata_missing") for r in results):
        sys.exit(1)


if __name__ == "__main__":
    main()
