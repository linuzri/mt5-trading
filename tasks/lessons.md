# Lessons Learned

## 2026-02-24: NEVER commit one-off scripts with secrets
- import_live_trades.py had hardcoded Supabase service_role key
- Claude Code agent did git add -A which staged EVERYTHING including throwaway scripts
- **Rule:** Always use git add <specific files> not git add -A. Review staged files before committing.
- **Rule:** One-off scripts with credentials go in a scripts/ folder that's in .gitignore, or use env vars.
- This is the SECOND time (first was Telegram token in ecosystem.config.js, Feb 19).

## 2026-02-24: MT5 position.time is broker server time, not UTC
- position.time returns Unix epoch offset by broker timezone (UTC+2 for Pepperstone)
- Must detect broker offset dynamically at startup and subtract before datetime conversion
- Same applies to any MT5 time field from positions/orders
