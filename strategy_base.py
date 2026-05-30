"""SPY Alpha v9 — Base dataclass shared across all strategies."""

from dataclasses import dataclass, field
from typing import Any, Dict, List

@dataclass
class StrategyOutput:
    strategy_name: str
    proposed_weights: Dict[str, float]
    confidence: float
    active_assets: List[str]
    strategy_metadata: Dict[str, Any] = field(default_factory=dict)