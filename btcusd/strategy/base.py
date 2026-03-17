"""Abstract base class for trading strategies."""
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional

import pandas as pd


@dataclass
class Signal:
    direction: str           # "buy" or "sell"
    reason: str              # Human-readable reason
    sl_distance: float       # SL distance in price units
    tp_distance: float       # TP distance in price units
    confidence: float = 1.0  # 0-1, for future use
    metadata: dict = field(default_factory=dict)


class Strategy(ABC):
    def __init__(self, config: dict) -> None:
        self.config = config

    @property
    @abstractmethod
    def name(self) -> str: ...

    @abstractmethod
    def evaluate(self, candles: pd.DataFrame) -> Optional[Signal]:
        """Return Signal if entry condition met, None otherwise."""
        ...
