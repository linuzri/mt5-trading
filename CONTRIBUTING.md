# Contributing to mt5-trading

## Workflow Gates

Every PR to `main` must pass through these gates in order. No skipping.

### Gate 1: Spec First
Before writing any code, the PR description must contain:
- **What** — one-line summary of the change
- **Why** — what problem this solves or what it improves
- **Acceptance criteria** — numbered list of requirements that define "done"
- **Files affected** — list of files to be created, modified, or deleted

No spec = no PR. Write the spec before touching code.

### Gate 2: Failing Test First
Write tests that capture the expected behavior BEFORE implementing:
- New feature → write tests that fail because the feature doesn't exist yet
- Bug fix → write a test that reproduces the bug
- Refactor → ensure existing tests cover the behavior being changed

Exception: documentation-only, config value tweaks, or test-only PRs.

### Gate 3: Implementation
Now write the code to make the tests pass:
- Follow existing module patterns (check neighboring files)
- Config values go in `config.yaml`, not hardcoded
- No secrets in code — use env vars / `.env`
- Keep functions focused (single responsibility)

### Gate 4: Verify
Before requesting merge:
```bash
cd btcusd && python -m pytest tests/ -v
```
- All tests pass (existing + new)
- No test requires MT5 broker connection (mock everything)
- For trade logic changes: dry-run or log review showing expected behavior

### Gate 5: Review & Merge
- **Trivial changes** (docs, config values, <20 lines, test-only): auto-merge OK
- **Significant changes** (new features, core logic, 3+ files, risk/trade changes): two-stage review required (spec compliance + code quality)
- PR description must be complete before merge

## Branch Convention
- Feature branches: `feature/[short-name]`
- Bug fixes: `fix/[short-name]`
- Always branch from `main`, PR back to `main`
- Never push directly to `main`

## Test Requirements
- All new code must have tests
- Tests live in `btcusd/tests/`
- Mock MT5 using `sys.modules` patch (see `tests/conftest.py`)
- Run in <5s total, no external dependencies
- If a PR has no tests, it must explain why in the description

## What NOT to Do
- Don't change risk management params without explicit approval
- Don't touch `mt5_auth.json` or credentials
- Don't add dependencies without justification in the PR
- Don't refactor unrelated code in a feature PR
