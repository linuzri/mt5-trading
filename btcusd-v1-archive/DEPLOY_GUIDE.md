# AutoTrader Research Loop — Deployment Guide

Full end-to-end setup: automated param research → Telegram approval → live deploy → verification.

---

## Architecture Overview

```
autotrader.py (PM2)
    │
    ├─ Proposes mutation via Claude API
    ├─ Runs backtest against live MT5 data
    ├─ Evaluates keep/discard
    │
    └─ KEEP found?
         │
         ├─ Sends Telegram message to your phone
         │   "tp_atr_multiplier 1.75→2.0 — WR=54% PnL=$82
         │    /deploy or /skip"
         │
         ├─ Waits up to 30 min for your reply
         │
         ├─ /deploy → deploy.py runs:
         │     1. Updates config.json or trend_strategy.py
         │     2. git commit + push to GitHub
         │     3. pm2 restart bot-btcusd
         │     4. Waits 2h, checks live trade log
         │     5. Rolls back if live performance drops
         │
         └─ /skip → skip this keep, continue research
```

---

## Prerequisites

- Python 3.10+
- PM2 installed (`npm install -g pm2`)
- Git configured with push access to `linuzri/mt5-trading`
- MT5 terminal running
- Telegram bot token + chat ID (see Step 1)
- `ANTHROPIC_API_KEY` env var set

---

## Step 1 — Set Up Telegram Bot

You need a Telegram bot to receive notifications and send /deploy commands.

**1a. Create a bot**

1. Open Telegram → search `@BotFather`
2. Send `/newbot`
3. Give it a name: `MT5 AutoResearch`
4. Give it a username: `mt5_autotrader_bot` (must end in `_bot`)
5. BotFather replies with a token like: `7123456789:AAHxxxxxx`
6. **Save this token** — it's your `TELEGRAM_BOT_TOKEN`

**1b. Get your chat ID**

1. Open Telegram → search `@userinfobot`
2. Send `/start`
3. It replies with your user ID e.g. `123456789`
4. **Save this** — it's your `TELEGRAM_CHAT_ID`

**1c. Set environment variables**

Open PowerShell and run (replace with your actual values):

```powershell
[System.Environment]::SetEnvironmentVariable("TELEGRAM_BOT_TOKEN", "7123456789:AAHxxxxxx", "User")
[System.Environment]::SetEnvironmentVariable("TELEGRAM_CHAT_ID", "123456789", "User")
[System.Environment]::SetEnvironmentVariable("ANTHROPIC_API_KEY", "sk-ant-xxxx", "User")
```

Then **close and reopen PowerShell** for the variables to take effect.

**1d. Test Telegram is working**

```powershell
cd C:\Users\Nazri Hussain\projects\mt5-trading\btcusd
python telegram_gate.py
```

You should receive a Telegram message within 5 seconds. Reply `/skip` to confirm it works.

---

## Step 2 — Copy New Files

Copy these 4 files into `btcusd/`:

| File | Purpose |
|---|---|
| `autotrader.py` | Updated research loop with approval gate |
| `telegram_gate.py` | Polls Telegram for /deploy or /skip |
| `deploy.py` | Applies param, pushes GitHub, restarts PM2 |
| `summary.py` | Morning report — already working |

Copy `ecosystem.config.js` to the **repo root** (`mt5-trading/`):

```
mt5-trading/
├── ecosystem.config.js      ← here
├── btcusd/
│   ├── autotrader.py        ← updated
│   ├── telegram_gate.py     ← new
│   ├── deploy.py            ← new
│   ├── backtest_mt5.py      ← unchanged
│   ├── summary.py           ← unchanged
│   └── MEMORY.md            ← unchanged
```

---

## Step 3 — Test Each Component

**Before running anything automated, test each piece manually.**

```powershell
cd C:\Users\Nazri Hussain\projects\mt5-trading\btcusd
```

**3a. Test Telegram gate (60s timeout)**
```powershell
python telegram_gate.py
# You get a Telegram message — reply /skip
# Should print: Result: skip
```

**3b. Test deploy.py in dry-run (no files changed, no git push)**
```powershell
python deploy.py --param tp_atr_multiplier --value 1.75 --old 1.75 --dry-run
# Should print all steps with [DRY] prefix
# Nothing written to disk
```

**3c. Test autotrader.py single experiment with approval gate**
```powershell
python autotrader.py --hours 168 --once
# If it finds a KEEP: Telegram message arrives on your phone
# Reply /skip (we're just testing — don't deploy yet)
# If DISCARD: loop exits cleanly
```

---

## Step 4 — Set Up PM2 with ecosystem.config.js

**4a. Stop old bot-btcusd process**
```powershell
pm2 stop bot-btcusd
pm2 delete bot-btcusd
```

**4b. Start from ecosystem config (repo root)**
```powershell
cd C:\Users\Nazri Hussain\projects\mt5-trading
pm2 start ecosystem.config.js --only bot-btcusd
pm2 save
```

**4c. Verify bot started correctly**
```powershell
pm2 logs bot-btcusd --lines 20
# Should see same startup messages as before
```

---

## Step 5 — Start AutoTrader in PM2

AutoTrader is set to `autostart: false` in ecosystem.config.js.
This means it does NOT start automatically on boot — you start it manually when you want research to run.

**Start autotrader:**
```powershell
cd C:\Users\Nazri Hussain\projects\mt5-trading
pm2 start ecosystem.config.js --only autotrader
pm2 save
```

**Check it's running:**
```powershell
pm2 list
# Should see both bot-btcusd and autotrader as "online"

pm2 logs autotrader --lines 30
# Should see research loop output
```

**Stop autotrader (e.g. before manual bot changes):**
```powershell
pm2 stop autotrader
# OR — send /stop from your Telegram bot
```

---

## Step 6 — The Daily Workflow

**Morning (after overnight run):**
```powershell
cd C:\Users\Nazri Hussain\projects\mt5-trading\btcusd
python summary.py
```

**If a KEEP was deployed overnight:**
- Check Telegram history for the verification message
- If "✅ Verified" → no action needed
- If "🔄 ROLLED BACK" → check the logs

**To manually review experiments:**
```powershell
# See last 10 experiments
python -c "
import json
lines = open('calibration.jsonl').readlines()
for l in lines[-10:]:
    e = json.loads(l)
    d = '✅' if e['decision']=='keep' else '❌'
    dep = '🚀' if e.get('deployed') else ''
    print(f\"{d}{dep} {e['param']:<22} {e['old_value']}->{e['new_value']:<8} WR={e['win_rate']}% PnL=\${e['pnl']}\")
"
```

---

## Telegram Commands Reference

| Command | What it does |
|---|---|
| `/deploy` | Apply the pending KEEP to live bot |
| `/skip` | Discard the KEEP, continue research |
| `/stop` | Stop autotrader loop after current experiment |
| `/status` | Show current MEMORY.md baseline metrics |

---

## Rollback — Manual

If something goes wrong after a deploy:

```powershell
cd C:\Users\Nazri Hussain\projects\mt5-trading
git log --oneline -5        # find the bad commit
git revert HEAD --no-edit   # revert last commit
git push origin main
pm2 restart bot-btcusd
```

---

## Troubleshooting

**autotrader keeps restarting in PM2**
- Check logs: `pm2 logs autotrader --lines 50`
- Usually means MT5 not connected — open MT5 terminal first

**Telegram not receiving messages**
- Verify `TELEGRAM_BOT_TOKEN` and `TELEGRAM_CHAT_ID` are set
- Run: `python -c "import os; print(os.environ.get('TELEGRAM_BOT_TOKEN', 'NOT SET'))"`
- First, send any message to your bot on Telegram (bots can't message you until you message them first)

**deploy.py can't push to GitHub**
- Ensure git credentials are cached: `git push origin main` manually once
- On Windows: git credential manager should handle this automatically

**PM2 env vars not passed to Python**
- Set vars in ecosystem.config.js `env:` block explicitly, or
- Use a `.env` file and load with `python-dotenv`

---

## File Summary

```
btcusd/
├── autotrader.py       # Research loop (UPDATED — has approval gate)
├── telegram_gate.py    # Telegram /deploy /skip poller (NEW)
├── deploy.py           # Config update + git push + PM2 restart (NEW)
├── backtest_mt5.py     # MT5 backtest engine (unchanged)
├── summary.py          # Morning report (unchanged)
├── MEMORY.md           # Strategy memory (auto-updated by loop)
└── calibration.jsonl   # Experiment log (auto-updated by loop)

mt5-trading/
└── ecosystem.config.js # PM2 process definitions (NEW)
```
