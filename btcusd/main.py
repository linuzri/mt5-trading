"""BTCUSD v2 Trading Bot — entry point."""
import argparse
import logging
import os
import sys

import yaml

from core.engine import Engine
from core.market import Market
from notifications.telegram import Telegram
from risk.limits import Limits
from risk.sizing import Sizer
from strategy.ema_cross import EmaCrossStrategy
from utils.state import State

STRATEGY_MAP = {
    "ema_cross": EmaCrossStrategy,
}


def load_config(path: str) -> dict:
    with open(path) as f:
        raw = yaml.safe_load(f)
    # Interpolate ${VAR} env vars
    def interpolate(value):
        if isinstance(value, str) and value.startswith("${") and value.endswith("}"):
            var = value[2:-1]
            return os.environ.get(var, value)
        return value
    return {k: interpolate(v) for k, v in raw.items()}


def setup_logging(level: str) -> None:
    logging.basicConfig(
        format="%(asctime)s %(levelname)s %(message)s",
        level=getattr(logging, level.upper(), logging.INFO),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="BTCUSD v2 Trading Bot")
    parser.add_argument("--dry-run", action="store_true", help="Log signals without executing trades")
    parser.add_argument("--config", default="config.yaml", help="Path to config file")
    args = parser.parse_args()

    config = load_config(args.config)
    setup_logging(config.get("log_level", "INFO"))

    if args.dry_run:
        logging.info("DRY-RUN mode enabled — no orders will be placed")

    strategy_cls = STRATEGY_MAP.get(config["strategy"])
    if strategy_cls is None:
        sys.exit(f"Unknown strategy: {config['strategy']}")

    market = Market(config)
    strategy = strategy_cls(config)
    sizer = Sizer(config)
    limits = Limits(config)
    state = State(config.get("log_file", "logs/trades.jsonl").replace("trades.jsonl", "state.json"))
    telegram = Telegram(config)

    engine = Engine(
        config=config,
        market=market,
        strategy=strategy,
        sizer=sizer,
        limits=limits,
        state=state,
        telegram=telegram,
        dry_run=args.dry_run,
    )
    engine.run()


if __name__ == "__main__":
    main()
