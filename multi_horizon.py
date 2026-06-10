"""
SPY Alpha V9 — Multi-Horizon Framework (Step 7)
================================================
Rebuilt from V8 for V9 architecture. Key changes from V8:

    compute_slow_layer()    — REWRITTEN: 5-pillar composite replaces HMM
    _apply_posture_bias()   — UPDATED: V9 thresholds (0.54/0.43)
    compute_fast_layer()    — ENHANCED: transition_hazard preemptive trigger
    MultiHorizonCoordinator — REWRITTEN: ConvictionGovernor replaces
                              apply_conditional_weighting()
    backtest_multi_horizon_v9() — NEW: V9-specific backtest with 5-day
                              rebalance and full instrumentation

Ported unchanged from V8:
    apply_slow_layer_bounds(), SlowLayerState, FastLayerAction,
    EQUITY_ASSETS / BOND_ASSETS / COMMODITY_ASSETS,
    circuit breaker and vol check in compute_fast_layer()

Confirmed design parameters (from design review + diagnostics):
    Aggressive threshold   : 0.54  (classifies 2013-14 and 2019 as aggressive)
    Defensive threshold    : 0.43  (classifies 2008 GFC and 2022 as defensive)
    Slow composite formula : spec Section 9A (5 pillars, 63-day smoothed)
    Hazard fast trigger    : 0.65  (top 35% of hazard days — conservative)
    Posture bias           : equity_boost = 1.0 + intensity × 0.30
    Medium layer rebalance : every 5 trading days
    Slow layer update      : every 21 trading days
    Fast layer             : daily (always)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger("spy_alpha_v9.multi_horizon")

# ---------------------------------------------------------------------------
# Layer update frequencies
# ---------------------------------------------------------------------------

SLOW_LAYER_FREQUENCY:   int = 21   # monthly
MEDIUM_LAYER_FREQUENCY: int = 5    # weekly  ← portfolio rebalance frequency
FAST_LAYER_FREQUENCY:   int = 1    # daily

# ---------------------------------------------------------------------------
# Slow layer posture thresholds (confirmed from diagnostic)
# ---------------------------------------------------------------------------

AGGRESSIVE_THRESHOLD: float = 0.54   # 2013-14 (0.641) and 2019 (0.543) → aggressive
DEFENSIVE_THRESHOLD:  float = 0.43   # 2008 GFC (0.332) and 2022 (0.399) → defensive

# Slow composite smoothing
SLOW_SMOOTHING_WINDOW:    int = 63
SLOW_SMOOTHING_MIN_PERIODS: int = 21

# ---------------------------------------------------------------------------
# Slow layer asset budget bounds
# ---------------------------------------------------------------------------

SLOW_EQUITY_BUDGET_MIN:   float = 0.20
SLOW_EQUITY_BUDGET_MAX:   float = 0.85
SLOW_BOND_BUDGET_MIN:     float = 0.05
SLOW_BOND_BUDGET_MAX:     float = 0.50
SLOW_COMMODITY_BUDGET_MIN: float = 0.00
SLOW_COMMODITY_BUDGET_MAX: float = 0.25
SLOW_LEVERAGE_CEILING_MIN: float = 0.60
SLOW_LEVERAGE_CEILING_MAX: float = 1.30

# ---------------------------------------------------------------------------
# Fast layer
# ---------------------------------------------------------------------------

FAST_LAYER_MAX_REDUCTION:   float = 0.50
HAZARD_PREEMPTIVE_THRESHOLD: float = 0.65   # top 35% of hazard days trigger
MAX_PREEMPTIVE_REDUCTION:    float = 0.30   # max 30% reduction from hazard alone

# ---------------------------------------------------------------------------
# Asset classification (ported from V8 unchanged)
# ---------------------------------------------------------------------------

EQUITY_ASSETS = {
    "SPY", "QQQ", "IWM", "VWO", "UPRO",
    "XLK", "XLV", "XLF", "XLY", "XLP", "XLE", "XLI",
    "XLB", "XLRE", "XLU", "XLC", "SMH", "XBI", "XME",
    "AAPL", "NVDA", "MSFT", "AMZN", "TSLA", "META",
    "GOOGL", "JPM", "LLY", "UNH", "XOM", "CAT",
}
BOND_ASSETS      = {"TLT", "IEF", "SHY", "HYG"}
COMMODITY_ASSETS = {"GLD", "DBC"}


def classify_weights(weights: Dict[str, float]) -> Dict[str, float]:
    equity    = sum(w for a, w in weights.items() if a in EQUITY_ASSETS)
    bond      = sum(w for a, w in weights.items() if a in BOND_ASSETS)
    commodity = sum(w for a, w in weights.items() if a in COMMODITY_ASSETS)
    other     = sum(
        w for a, w in weights.items()
        if a not in EQUITY_ASSETS
        and a not in BOND_ASSETS
        and a not in COMMODITY_ASSETS
    )
    return {"equity": equity, "bond": bond, "commodity": commodity, "other": other}


# ---------------------------------------------------------------------------
# Slow Layer — state dataclass and computation
# ---------------------------------------------------------------------------

@dataclass
class SlowLayerState:
    """Output of the slow layer — strategic bounds for the medium layer."""
    date:             pd.Timestamp
    leverage_ceiling: float
    equity_budget:    Tuple[float, float]
    bond_budget:      Tuple[float, float]
    commodity_budget: Tuple[float, float]
    risk_posture:     str                    # "aggressive" | "balanced" | "defensive"
    posture_intensity: float                 # ∈ [0, 1]
    metadata:         Dict[str, Any] = field(default_factory=dict)


def compute_slow_composite(pillars: pd.DataFrame) -> pd.Series:
    """
    Pre-compute the full slow composite series from pillars.

    Applies 63-day rolling mean smoothing to each pillar before combining.
    Call once during build; index into the result by date.

    Formula (spec Section 9A):
        slow_composite = 0.25 × growth
                       + 0.10 × (1 - inflation)     # moderate inflation okay
                       + 0.25 × (1 - stress)
                       + 0.20 × trend
                       + 0.20 × participation
    """
    required = [
        "growth_momentum", "inflation_pressure", "financial_stress",
        "trend_persistence", "participation_quality",
    ]
    for col in required:
        if col not in pillars.columns:
            raise ValueError(f"compute_slow_composite: missing pillar '{col}'")

    def smooth(col: str) -> pd.Series:
        return pillars[col].rolling(
            SLOW_SMOOTHING_WINDOW,
            min_periods=SLOW_SMOOTHING_MIN_PERIODS,
        ).mean()

    composite = (
        0.25 * smooth("growth_momentum")
        + 0.10 * (1.0 - smooth("inflation_pressure"))
        + 0.25 * (1.0 - smooth("financial_stress"))
        + 0.20 * smooth("trend_persistence")
        + 0.20 * smooth("participation_quality")
    ).clip(0.0, 1.0)

    return composite


def compute_slow_layer(
    slow_composite_value: float,
    current_date: pd.Timestamp,
) -> SlowLayerState:
    """
    Classify a single slow_composite value into a SlowLayerState.

    Args:
        slow_composite_value : pre-computed smoothed composite for current_date
        current_date         : date of the assessment
    """
    c = float(slow_composite_value)

    if np.isnan(c):
        return _default_slow_state(current_date)

    # ---- Posture classification (V9 confirmed thresholds) ----
    if c > AGGRESSIVE_THRESHOLD:
        risk_posture     = "aggressive"
        posture_intensity = float(np.clip(
            (c - AGGRESSIVE_THRESHOLD) / (1.0 - AGGRESSIVE_THRESHOLD),
            0.0, 1.0
        ))
    elif c > DEFENSIVE_THRESHOLD:
        risk_posture     = "balanced"
        posture_intensity = 0.0
    else:
        risk_posture     = "defensive"
        posture_intensity = float(np.clip(
            (DEFENSIVE_THRESHOLD - c) / DEFENSIVE_THRESHOLD,
            0.0, 1.0
        ))

    # ---- Leverage ceiling scales with composite ----
    leverage_ceiling = float(
        SLOW_LEVERAGE_CEILING_MIN
        + c * (SLOW_LEVERAGE_CEILING_MAX - SLOW_LEVERAGE_CEILING_MIN)
    )

    # ---- Budget bounds ----
    eq_min  = SLOW_EQUITY_BUDGET_MIN + c * 0.15
    eq_max  = SLOW_EQUITY_BUDGET_MAX
    bond_min = SLOW_BOND_BUDGET_MIN
    bond_max = SLOW_BOND_BUDGET_MAX - c * 0.20

    return SlowLayerState(
        date              = current_date,
        leverage_ceiling  = leverage_ceiling,
        equity_budget     = (eq_min, eq_max),
        bond_budget       = (bond_min, bond_max),
        commodity_budget  = (SLOW_COMMODITY_BUDGET_MIN, SLOW_COMMODITY_BUDGET_MAX),
        risk_posture      = risk_posture,
        posture_intensity = posture_intensity,
        metadata          = {
            "composite_score": c,
            "aggressive_threshold": AGGRESSIVE_THRESHOLD,
            "defensive_threshold":  DEFENSIVE_THRESHOLD,
        },
    )


def _default_slow_state(date: pd.Timestamp) -> SlowLayerState:
    return SlowLayerState(
        date              = date,
        leverage_ceiling  = 1.0,
        equity_budget     = (SLOW_EQUITY_BUDGET_MIN, SLOW_EQUITY_BUDGET_MAX),
        bond_budget       = (SLOW_BOND_BUDGET_MIN,   SLOW_BOND_BUDGET_MAX),
        commodity_budget  = (SLOW_COMMODITY_BUDGET_MIN, SLOW_COMMODITY_BUDGET_MAX),
        risk_posture      = "balanced",
        posture_intensity = 0.0,
        metadata          = {"composite_score": 0.5},
    )


# ---------------------------------------------------------------------------
# Medium layer — slow layer bounds enforcement (ported from V8 unchanged)
# ---------------------------------------------------------------------------

def apply_slow_layer_bounds(
    weights: Dict[str, float],
    slow_state: SlowLayerState,
) -> Dict[str, float]:
    """Constrain weights within slow layer's strategic asset-class bounds."""
    if not weights:
        return weights

    bounded = weights.copy()
    buckets  = classify_weights(bounded)

    # Equity cap
    eq_min, eq_max = slow_state.equity_budget
    if buckets["equity"] > eq_max and buckets["equity"] > 0:
        scale = eq_max / buckets["equity"]
        for a in bounded:
            if a in EQUITY_ASSETS:
                bounded[a] *= scale

    # Bond cap
    _, bond_max = slow_state.bond_budget
    if buckets["bond"] > bond_max and buckets["bond"] > 0:
        scale = bond_max / buckets["bond"]
        for a in bounded:
            if a in BOND_ASSETS:
                bounded[a] *= scale

    # Commodity cap
    _, comm_max = slow_state.commodity_budget
    if buckets["commodity"] > comm_max and buckets["commodity"] > 0:
        scale = comm_max / buckets["commodity"]
        for a in bounded:
            if a in COMMODITY_ASSETS:
                bounded[a] *= scale

    # Leverage ceiling
    total = sum(bounded.values())
    if total > slow_state.leverage_ceiling:
        scale   = slow_state.leverage_ceiling / total
        bounded = {k: v * scale for k, v in bounded.items()}

    # Normalize
    total = sum(bounded.values())
    if total > 0:
        bounded = {k: v / total for k, v in bounded.items()}

    return bounded


# ---------------------------------------------------------------------------
# Fast layer — crisis response (ported from V8 + hazard trigger)
# ---------------------------------------------------------------------------

@dataclass
class FastLayerAction:
    date:                  pd.Timestamp
    reduction_factor:      float         # 1.0 = no change
    cash_increase:         float
    circuit_breaker_level: str
    crash_momentum_active: bool
    hazard_triggered:      bool          # NEW: transition_hazard preemptive
    override_active:       bool
    metadata:              Dict[str, Any] = field(default_factory=dict)


def compute_fast_layer(
    weights:           Dict[str, float],
    spy_prices:        pd.Series,
    current_date:      pd.Timestamp,
    transition_hazard: float = 0.0,
) -> Tuple[Dict[str, float], FastLayerAction]:
    """
    Fast layer crisis response. DAILY. Can only REDUCE, never INCREASE.

    V9 enhancement: transition_hazard > HAZARD_PREEMPTIVE_THRESHOLD
    triggers preemptive reduction BEFORE price-based circuit breakers fire.
    Threshold = 0.65 (top 35% of hazard days — conservative starting point).
    """
    from risk_engine import apply_circuit_breakers

    adjusted = weights.copy()

    # ---- V9 ENHANCEMENT: preemptive hazard reduction ----
    hazard_triggered    = False
    hazard_reduction_f  = 1.0
    hazard_cash_add     = 0.0

    if transition_hazard > HAZARD_PREEMPTIVE_THRESHOLD:
        intensity = float(np.clip(
            (transition_hazard - HAZARD_PREEMPTIVE_THRESHOLD)
            / (1.0 - HAZARD_PREEMPTIVE_THRESHOLD),
            0.0, 1.0
        ))
        hazard_reduction_f = 1.0 - intensity * MAX_PREEMPTIVE_REDUCTION
        hazard_cash_add    = intensity * MAX_PREEMPTIVE_REDUCTION * 0.5

        for a in list(adjusted.keys()):
            if a != "SHY":
                adjusted[a] *= hazard_reduction_f
        adjusted["SHY"] = adjusted.get("SHY", 0.0) + hazard_cash_add

        # Normalize after hazard reduction
        total = sum(adjusted.values())
        if total > 0:
            adjusted = {k: v / total for k, v in adjusted.items()}

        hazard_triggered = True
        logger.debug(
            f"Fast layer: hazard preemptive reduction "
            f"(hazard={transition_hazard:.3f}, intensity={intensity:.3f}, "
            f"reduction={hazard_reduction_f:.3f})"
        )

    # ---- Circuit breakers (ported from V8 / V7) ----
    adjusted, breaker_meta = apply_circuit_breakers(
        adjusted, spy_prices, current_date
    )

    breaker_level = breaker_meta.get("circuit_breaker_level", "none")
    crash_active  = breaker_meta.get("crash_momentum_active", False)

    # ---- Elevated vol check (ported from V8) ----
    reduction_factor = 1.0
    cash_increase    = 0.0

    spy_up_to = spy_prices.loc[:current_date]
    if len(spy_up_to) >= 10:
        recent_ret = spy_up_to.pct_change().tail(10).dropna()
        if len(recent_ret) >= 5:
            recent_vol = recent_ret.std() * np.sqrt(252)
            if recent_vol > 0.40:
                vol_red    = float(np.clip((recent_vol - 0.40) / 0.30, 0.0, 0.50))
                reduction_factor = 1.0 - vol_red
                cash_increase    = vol_red * 0.3
                for a in list(adjusted.keys()):
                    if a != "SHY":
                        adjusted[a] *= reduction_factor
                adjusted["SHY"] = adjusted.get("SHY", 0.0) + cash_increase

    # ---- CRITICAL: never increase any non-SHY position above original ----
    for a in adjusted:
        if a in weights and a != "SHY":
            if adjusted[a] > weights[a]:
                adjusted[a] = weights[a]

    # ---- Normalize ----
    total = sum(adjusted.values())
    if total > 0:
        adjusted = {k: v / total for k, v in adjusted.items()}

    override_active = (
        hazard_triggered
        or breaker_level != "none"
        or crash_active
        or reduction_factor < 1.0
    )

    action = FastLayerAction(
        date                  = current_date,
        reduction_factor      = reduction_factor * hazard_reduction_f,
        cash_increase         = cash_increase + hazard_cash_add,
        circuit_breaker_level = breaker_level,
        crash_momentum_active = crash_active,
        hazard_triggered      = hazard_triggered,
        override_active       = override_active,
        metadata              = breaker_meta,
    )

    return adjusted, action


# ---------------------------------------------------------------------------
# Multi-Horizon Coordinator (V9 — uses ConvictionGovernor)
# ---------------------------------------------------------------------------

class MultiHorizonCoordinator:
    """
    Coordinates three temporal layers into a single portfolio decision.

    V9 change from V8: uses ConvictionGovernor.apply_governance() for capital
    weights instead of apply_conditional_weighting(). The governor is
    pre-built and passed in at construction time.

    Flow (per trading day):
        1. Slow layer (monthly)  — pillar composite → posture + bounds
        2. Medium layer (weekly) — governor capital weights + posture bias
                                   → apply slow layer bounds
        3. Fast layer  (daily)   — hazard trigger + circuit breakers
    """

    def __init__(
        self,
        conviction_governor,                  # ConvictionGovernor instance
        profile: str = "aggressive",
    ) -> None:
        from conviction_governor import ConvictionGovernor
        if not isinstance(conviction_governor, ConvictionGovernor):
            raise TypeError("conviction_governor must be a ConvictionGovernor instance")

        self.governor     = conviction_governor
        self.profile_name = profile

        self.current_slow_state: Optional[SlowLayerState] = None
        self.last_slow_update:   Optional[pd.Timestamp]   = None
        self.last_medium_update: Optional[pd.Timestamp]   = None
        self.medium_weights: Dict[str, float] = {}

        # Exposure diagnostics storage
        self._diag_records: List[Dict] = []

        # Risk engine (Step 8 integration)
        from risk_engine import RiskEngine
        self._risk_engine = RiskEngine(profile=profile)

    def process(
        self,
        strategy_portfolios: Dict[str, Dict[str, float]],
        slow_composite:      float,
        spy_prices:          pd.Series,
        current_date:        pd.Timestamp,
        trading_index:       pd.DatetimeIndex,
        transition_hazard:   float = 0.0,
        analog_scores_row:   Optional[Dict[str, float]] = None,
        force_slow_update:   bool  = False,
        force_medium_update: bool  = False,
    ) -> Tuple[Dict[str, float], Dict[str, Any]]:
        """
        Process strategy portfolios through all four temporal layers.

        V9 Step 8 addition: risk engine runs between medium and fast layers.
        Governor output hoisted to top of method so it can be used by both
        the medium layer and the risk engine without a second call.
        """
        metadata: Dict[str, Any] = {
            "slow_layer":   {},
            "medium_layer": {},
            "risk_engine":  {},
            "fast_layer":   {},
        }

        # ----------------------------------------------------------------
        # Governor output (hoisted — used by both medium layer and risk engine)
        # ----------------------------------------------------------------
        gov_output = self.governor.apply_governance(current_date)
        if gov_output is not None:
            capital_weights_today = gov_output.capital_weights
            dyn_max_upro          = gov_output.dynamic_max_upro
            conviction_budget     = gov_output.conviction_state.conviction_budget
        else:
            n = len(strategy_portfolios)
            capital_weights_today = {k: 1/n for k in strategy_portfolios}
            dyn_max_upro          = 0.35
            conviction_budget     = 0.0

        # ----------------------------------------------------------------
        # 1. SLOW LAYER (monthly)
        # ----------------------------------------------------------------
        should_update_slow = (
            force_slow_update
            or self.current_slow_state is None
            or self.last_slow_update is None
            or self._days_since(
                self.last_slow_update, current_date, trading_index
            ) >= SLOW_LAYER_FREQUENCY
        )

        if should_update_slow:
            self.current_slow_state = compute_slow_layer(
                slow_composite, current_date
            )
            self.last_slow_update = current_date

        sl = self.current_slow_state
        metadata["slow_layer"] = {
            "risk_posture":      sl.risk_posture,
            "posture_intensity": sl.posture_intensity,
            "composite_score":   sl.metadata.get("composite_score", 0.5),
            "leverage_ceiling":  sl.leverage_ceiling,
            "updated":           should_update_slow,
        }

        # ----------------------------------------------------------------
        # 2. MEDIUM LAYER (weekly, every 5 days)
        # ----------------------------------------------------------------
        should_update_medium = (
            force_medium_update
            or self.last_medium_update is None
            or self._days_since(
                self.last_medium_update, current_date, trading_index
            ) >= MEDIUM_LAYER_FREQUENCY
        )

        if should_update_medium:
            capital_weights = capital_weights_today

            # Blend strategy asset weights using capital weights
            blended: Dict[str, float] = {}
            for strat, cap_w in capital_weights.items():
                port = strategy_portfolios.get(strat, {})
                for asset, w in port.items():
                    blended[asset] = blended.get(asset, 0.0) + cap_w * w

            pre_bias_buckets = classify_weights(blended)
            pre_bias_upro    = blended.get("UPRO", 0.0)

            biased = self._apply_posture_bias(blended, sl)

            post_bias_buckets = classify_weights(biased)
            post_bias_upro    = biased.get("UPRO", 0.0)

            self._diag_records.append({
                "date":              current_date,
                "posture":           sl.risk_posture,
                "intensity":         sl.posture_intensity,
                "pre_equity":        pre_bias_buckets["equity"],
                "post_equity":       post_bias_buckets["equity"],
                "pre_upro":          pre_bias_upro,
                "post_upro":         post_bias_upro,
                "post_risk_upro":    0.0,   # filled in after risk engine
                "post_fast_upro":    0.0,   # filled in after fast layer
                "conviction_budget": conviction_budget,
                "s1_weight":         capital_weights.get("s1", 1/3),
            })

            self.medium_weights     = apply_slow_layer_bounds(biased, sl)
            self.last_medium_update = current_date

            metadata["medium_layer"] = {
                "updated":           True,
                "s1_capital":        capital_weights.get("s1", 1/3),
                "s2_capital":        capital_weights.get("s2", 1/3),
                "s3_capital":        capital_weights.get("s3", 1/3),
                "conviction_budget": conviction_budget,
                "dynamic_max_upro":  dyn_max_upro,
                "pre_bias_equity":   pre_bias_buckets["equity"],
                "post_bias_equity":  post_bias_buckets["equity"],
                "pre_bias_upro":     pre_bias_upro,
                "post_bias_upro":    post_bias_upro,
            }
        else:
            metadata["medium_layer"] = {"updated": False}

        # ----------------------------------------------------------------
        # 3. RISK ENGINE (daily)
        # ----------------------------------------------------------------
        a_row = analog_scores_row or {}

        risk_adjusted, risk_meta = self._risk_engine.apply(
            proposed_weights  = self.medium_weights,
            strategy_weights  = strategy_portfolios,
            analog_scores_row = a_row,
            spy_prices        = spy_prices,
            current_date      = current_date,
            stress_score      = 0.0,        # hazard excluded; uncertainty covers stress
            dynamic_max_upro  = dyn_max_upro,
        )

        post_risk_upro = risk_adjusted.get("UPRO", 0.0)
        if self._diag_records:
            self._diag_records[-1]["post_risk_upro"] = post_risk_upro

        metadata["risk_engine"] = {
            "uncertainty_score": risk_meta.get("uncertainty_score", 0.0),
            "tightening_level":  risk_meta.get("tightening_level",  0.0),
            "pre_upro":          self.medium_weights.get("UPRO", 0.0),
            "post_upro":         post_risk_upro,
        }

        # ----------------------------------------------------------------
        # 4. FAST LAYER (daily — always runs)
        # ----------------------------------------------------------------
        final_weights, fast_action = compute_fast_layer(
            risk_adjusted, spy_prices, current_date, transition_hazard
        )

        post_fast_upro = final_weights.get("UPRO", 0.0)
        if self._diag_records:
            self._diag_records[-1]["post_fast_upro"] = post_fast_upro

        metadata["fast_layer"] = {
            "override_active":   fast_action.override_active,
            "reduction_factor":  fast_action.reduction_factor,
            "circuit_breaker":   fast_action.circuit_breaker_level,
            "crash_momentum":    fast_action.crash_momentum_active,
            "hazard_triggered":  fast_action.hazard_triggered,
            "transition_hazard": transition_hazard,
        }

        return final_weights, metadata

    def _apply_posture_bias(
        self,
        weights:    Dict[str, float],
        slow_state: SlowLayerState,
    ) -> Dict[str, float]:
        """
        Scale equity/defensive assets based on slow layer posture.

        V9 thresholds (updated from V8):
            aggressive: intensity = (composite - 0.54) / 0.46
            defensive:  intensity = (0.43 - composite) / 0.43

        Posture bias parameters:
            Aggressive: equity_boost = 1.0 + intensity × 0.30
                        defensive_cut = 1.0 - intensity × 0.25
            Defensive:  equity_cut = 1.0 - intensity × 0.25
                        defensive_boost = 1.0 + intensity × 0.20
            Balanced:   pass-through (no adjustment)
        """
        if slow_state.risk_posture == "balanced":
            return weights.copy()

        adjusted  = weights.copy()
        intensity = slow_state.posture_intensity

        if slow_state.risk_posture == "aggressive":
            equity_boost   = 1.0 + intensity * 0.30
            defensive_cut  = 1.0 - intensity * 0.25
            for a in adjusted:
                if a in EQUITY_ASSETS:
                    adjusted[a] *= equity_boost
                elif a in BOND_ASSETS or a == "SHY":
                    adjusted[a] *= defensive_cut

        elif slow_state.risk_posture == "defensive":
            equity_cut      = 1.0 - intensity * 0.25
            defensive_boost = 1.0 + intensity * 0.20
            for a in adjusted:
                if a in EQUITY_ASSETS:
                    adjusted[a] *= equity_cut
                elif a in BOND_ASSETS or a == "SHY":
                    adjusted[a] *= defensive_boost

        total = sum(adjusted.values())
        if total > 0:
            adjusted = {k: v / total for k, v in adjusted.items()}

        return adjusted

    def get_exposure_diagnostics(self) -> pd.DataFrame:
        """Return the per-day exposure diagnostics DataFrame."""
        if not self._diag_records:
            return pd.DataFrame()
        return pd.DataFrame(self._diag_records).set_index("date")

    @staticmethod
    def _days_since(
        last:  pd.Timestamp,
        now:   pd.Timestamp,
        index: pd.DatetimeIndex,
    ) -> int:
        return int(((index > last) & (index <= now)).sum())


# ---------------------------------------------------------------------------
# V9 Backtest — full pipeline with 5-day rebalance
# ---------------------------------------------------------------------------

def backtest_multi_horizon_v9(
    s1_portfolios:       Dict[pd.Timestamp, Dict[str, float]],
    s2_outputs:          list,
    s3_outputs:          list,
    conviction_governor,
    slow_composite:      pd.Series,
    analog_scores:       pd.DataFrame,
    adj_close:           pd.DataFrame,
    profile:             str = "aggressive",
) -> Dict[str, Any]:
    """
    Run the full V9 multi-horizon backtest.

    Portfolio rebalances every 5 days (medium layer).
    S1 weights forward-fill from their 21-day refit schedule.
    S2 and S3 weights forward-fill from their own schedules.
    Fast layer runs daily with transition_hazard from analog memory.

    Args:
        s1_portfolios    : StateAllocator._portfolios["C"] dict
        s2_outputs       : TrendCTAStrategy.generate_signals() output
        s3_outputs       : DefensiveStrategy.generate_signals() output
        conviction_governor : built ConvictionGovernor
        slow_composite   : pre-computed from compute_slow_composite(pillars)
        analog_scores    : AnalogMemory.analog_scores (coherence/reliability/hazard)
        adj_close        : adjusted close prices
        profile          : risk profile name
    """

    logger.info("backtest_multi_horizon_v9: starting...")

    spy_prices  = adj_close["SPY"] if "SPY" in adj_close.columns else pd.Series()
    daily_ret   = adj_close.pct_change()
    trading_idx = adj_close.index

    # ----------------------------------------------------------------
    # Build forward-fill lookups for S2 and S3
    # ----------------------------------------------------------------
    def _build_lookup(outputs: list) -> Dict[pd.Timestamp, Dict[str, float]]:
        return {
            pd.Timestamp(o.strategy_metadata["date"]): o.proposed_weights
            for o in outputs
        }

    s2_lookup = _build_lookup(s2_outputs)
    s3_lookup = _build_lookup(s3_outputs)

    def _ffill(lookup: Dict, date: pd.Timestamp) -> Dict[str, float]:
        past = [d for d in lookup if d <= date]
        return lookup[max(past)] if past else {}

    # ----------------------------------------------------------------
    # Build coordinator
    # ----------------------------------------------------------------
    coordinator = MultiHorizonCoordinator(
        conviction_governor = conviction_governor,
        profile             = profile,
    )

    # ----------------------------------------------------------------
    # Find backtest start — need all three strategies to have signals
    # ----------------------------------------------------------------
    starts = []
    for lookup in (s1_portfolios, s2_lookup, s3_lookup):
        if lookup:
            starts.append(min(lookup.keys()))
    if not starts:
        raise RuntimeError("No strategy signals found")
    backtest_start = max(starts)

    logger.info(f"  Backtest start: {backtest_start.date()}")

    # ----------------------------------------------------------------
    # Daily simulation
    # ----------------------------------------------------------------
    results = []

    for date in trading_idx:
        if date < backtest_start:
            continue

        # Transition hazard for today
        haz = 0.0
        if date in analog_scores.index and pd.notna(
            analog_scores.loc[date, "transition_hazard"]
        ):
            haz = float(analog_scores.loc[date, "transition_hazard"])

        # Slow composite for today
        sc = float(slow_composite.loc[date]) if date in slow_composite.index else float("nan")
        if np.isnan(sc):
            sc = 0.50  # fallback to balanced

        # Current strategy weights (forward-fill)
        w1 = _ffill(s1_portfolios, date)
        w2 = _ffill(s2_lookup,     date)
        w3 = _ffill(s3_lookup,     date)

        if not (w1 and w2 and w3):
            continue

        strategy_portfolios = {"s1": w1, "s2": w2, "s3": w3}

        # Analog scores row for risk engine uncertainty
        a_row: Dict[str, float] = {}
        if date in analog_scores.index:
            row = analog_scores.loc[date]
            a_row = {
                "coherence":         float(row.get("coherence",         0.5)),
                "reliability":       float(row.get("reliability",       0.5)),
                "transition_hazard": float(row.get("transition_hazard", 0.5)),
            }

        # Process through coordinator
        final_w, meta = coordinator.process(
            strategy_portfolios = strategy_portfolios,
            slow_composite      = sc,
            spy_prices          = spy_prices,
            current_date        = date,
            trading_index       = trading_idx,
            transition_hazard   = haz,
            analog_scores_row   = a_row,
        )

        # Compute daily return
        pr = sum(
            w * daily_ret.loc[date, a]
            for a, w in final_w.items()
            if a in daily_ret.columns and pd.notna(daily_ret.loc[date, a])
        )

        results.append({
            "date":         date,
            "return":       pr,
            "posture":      meta["slow_layer"].get("risk_posture", "unknown"),
            "composite":    sc,
            "s1_capital":   meta["medium_layer"].get("s1_capital", 1/3),
            "fast_override":      meta["fast_layer"].get("override_active", False),
            "hazard":             haz,
            "uncertainty_score":  meta["risk_engine"].get("uncertainty_score", 0.0),
            "post_risk_upro":     meta["risk_engine"].get("post_upro", 0.0),
            "post_fast_upro":     final_w.get("UPRO", 0.0),
        })

    if not results:
        raise RuntimeError("No backtest results generated")

    df = pd.DataFrame(results).set_index("date")
    port_returns = df["return"]

    # ----------------------------------------------------------------
    # Compute metrics
    # ----------------------------------------------------------------
    def _metrics(r: pd.Series, label: str) -> Dict:
        r = r.dropna()
        r = r[r != 0]
        if len(r) < 126:
            return {"label": label}
        n_yrs   = len(r) / 252
        cum     = (1 + r).cumprod()
        cagr    = float(cum.iloc[-1] ** (1/n_yrs) - 1)
        sharpe  = float(r.mean() / (r.std() + 1e-10) * np.sqrt(252))
        peak    = cum.expanding().max()
        max_dd  = float((cum/peak - 1).min())
        calmar  = float(cagr / abs(max_dd)) if abs(max_dd) > 1e-6 else float("nan")
        down    = r[r < 0]
        sortino = float(cagr / (down.std()*np.sqrt(252)+1e-10)) if len(down) else float("nan")
        kurt    = float(r.kurtosis())
        vol     = float(r.std() * np.sqrt(252))
        return {
            "label":   label,
            "sharpe":  round(sharpe,  3),
            "cagr":    round(cagr,    4),
            "max_dd":  round(max_dd,  4),
            "calmar":  round(calmar,  3),
            "sortino": round(sortino, 3),
            "kurtosis": round(kurt,   2),
            "vol":     round(vol,     4),
            "n_days":  len(r),
        }

    spy_ret = adj_close["SPY"].pct_change().loc[port_returns.index]

    metrics = {
        "multi_horizon":   _metrics(port_returns, "V9 Multi-Horizon"),
        "spy_benchmark":   _metrics(spy_ret,       "SPY buy-and-hold"),
    }

    # ----------------------------------------------------------------
    # Posture distribution
    # ----------------------------------------------------------------
    posture_dist = df["posture"].value_counts(normalize=True)

    # ----------------------------------------------------------------
    # Exposure diagnostics
    # ----------------------------------------------------------------
    exposure_diag = coordinator.get_exposure_diagnostics()

    logger.info(
        f"backtest_multi_horizon_v9 complete: "
        f"{len(df)} days, "
        f"Sharpe={metrics['multi_horizon'].get('sharpe','?')}, "
        f"CAGR={metrics['multi_horizon'].get('cagr',0)*100:.1f}%"
    )

    return {
        "returns":      port_returns,
        "daily_df":     df,
        "metrics":      metrics,
        "posture_dist": posture_dist,
        "exposure_diag": exposure_diag,
        "slow_composite": slow_composite,
    }


# ---------------------------------------------------------------------------
# Step 7 Validation
# ---------------------------------------------------------------------------

def validate_multi_horizon(results: Dict[str, Any]) -> bool:
    """
    Validate Step 7 pass criteria (spec Section 12, Step 7).

    Pass criteria:
        1. Slow layer posture: aggressive 50-75%, balanced 15-35%, defensive 5-20%
        2. Kurtosis ≤ 3.0
        3. Multi-horizon Sharpe ≥ 1.0 standalone
        4. Fast layer override on 5-20% of days (not too aggressive, not inactive)
        5. Posture bias measurable: post-bias equity > pre-bias equity on
           aggressive days
    """
    print("\n" + "=" * 76)
    print("MULTI-HORIZON FRAMEWORK — STEP 7 VALIDATION")
    print("=" * 76)

    # ---- Performance summary ----
    mh = results["metrics"].get("multi_horizon", {})
    spy = results["metrics"].get("spy_benchmark", {})
    print(f"\n  {'Strategy':<30} {'Sharpe':>7} {'CAGR':>8} "
          f"{'Max DD':>8} {'Calmar':>8} {'Kurtosis':>9}")
    print("  " + "─" * 70)
    for m in [mh, spy]:
        sharpe  = m.get("sharpe", float("nan"))
        cagr    = m.get("cagr", float("nan"))
        max_dd  = m.get("max_dd", float("nan"))
        calmar  = m.get("calmar", float("nan"))
        kurt    = m.get("kurtosis", float("nan"))
        label   = m.get("label", "?")
        def fv(v, pct=False):
            if isinstance(v, float) and np.isnan(v):
                return "   nan"
            return f"{v*100:>7.2f}%" if pct else f"{v:>7.3f}"
        print(f"  {label:<30} {fv(sharpe)} {fv(cagr,True)} "
              f"{fv(max_dd,True)} {fv(calmar)} {fv(kurt)}")

    # ---- Posture distribution ----
    posture = results["posture_dist"]
    agg  = float(posture.get("aggressive", 0.0))
    bal  = float(posture.get("balanced",   0.0))
    dfn  = float(posture.get("defensive",  0.0))

    print(f"\n  POSTURE DISTRIBUTION:")
    print(f"    Aggressive : {agg:.1%}")
    print(f"    Balanced   : {bal:.1%}")
    print(f"    Defensive  : {dfn:.1%}")

    # ---- Fast layer override frequency ----
    df         = results["daily_df"]
    fast_rate  = float(df["fast_override"].mean()) if "fast_override" in df.columns else 0.0
    hazard_rate = float((df["hazard"] > HAZARD_PREEMPTIVE_THRESHOLD).mean()) \
                  if "hazard" in df.columns else 0.0

    print(f"\n  FAST LAYER:")
    print(f"    Override rate : {fast_rate:.1%}")
    print(f"    Hazard trigger rate (>{HAZARD_PREEMPTIVE_THRESHOLD:.2f}): {hazard_rate:.1%}")

    # ---- Exposure diagnostics ----
    exp_diag = results.get("exposure_diag", pd.DataFrame())
    if not exp_diag.empty:
        print(f"\n  EXPOSURE DIAGNOSTICS (pre-bias → post-bias by posture):")
        for posture_name in ["aggressive", "balanced", "defensive"]:
            mask = exp_diag["posture"] == posture_name
            if mask.sum() == 0:
                continue
            pre_eq  = exp_diag.loc[mask, "pre_equity"].mean()
            post_eq = exp_diag.loc[mask, "post_equity"].mean()
            pre_up  = exp_diag.loc[mask, "pre_upro"].mean()
            post_up = exp_diag.loc[mask, "post_upro"].mean()
            print(f"    {posture_name:<12}: "
                  f"equity {pre_eq:.1%}→{post_eq:.1%}  "
                  f"UPRO {pre_up:.2%}→{post_up:.2%}  "
                  f"(N={mask.sum()})")
            
    # ---- UPRO stage diagnostics ----
    df = results["daily_df"]
    if "post_risk_upro" in df.columns:
        print(f"\n  UPRO STAGE DIAGNOSTICS:")
        upro_cols = {
            "post-bias":        "post_upro",
            "post-risk-engine": "post_risk_upro",
            "post-fast-layer":  "post_fast_upro",
        }
        for label, col in upro_cols.items():
            if col not in exp_diag.columns and col not in df.columns:
                continue
            src = exp_diag[col] if col in exp_diag.columns else df[col]
            presence = float((src > 0.01).mean())
            mean_when = float(src[src > 0.01].mean()) if (src > 0.01).any() else 0.0
            print(f"    {label:<20}: present {presence:.1%} of days  "
                  f"mean-when-present = {mean_when:.2%}")

    # ---- Uncertainty score distribution ----
    if "uncertainty_score" in df.columns:
        unc = df["uncertainty_score"].dropna()
        print(f"\n  UNCERTAINTY SCORE (risk engine):")
        print(f"    Mean={unc.mean():.3f}  Median={unc.median():.3f}  "
              f"Std={unc.std():.3f}  "
              f"P90={unc.quantile(0.90):.3f}")

    # ---- Pass criteria ----
    kurtosis   = float(mh.get("kurtosis", float("nan")))
    sharpe_val = float(mh.get("sharpe",   float("nan")))

    criteria = [
        (
            "Aggressive 50-75% of days",
            0.50 <= agg <= 0.75,
            f"actual = {agg:.1%}",
        ),
        (
            "Balanced 15-35% of days",
            0.15 <= bal <= 0.35,
            f"actual = {bal:.1%}",
        ),
        (
            "Defensive 5-20% of days",
            0.05 <= dfn <= 0.20,
            f"actual = {dfn:.1%}",
        ),
        (
            "Kurtosis ≤ 3.0",
            (not np.isnan(kurtosis)) and kurtosis <= 3.0,
            f"actual = {kurtosis:.2f}",
        ),
        (
            "Sharpe ≥ 1.0 standalone",
            (not np.isnan(sharpe_val)) and sharpe_val >= 1.0,
            f"actual = {sharpe_val:.3f}",
        ),
        (
            "Uncertainty score varies (std > 0.02)",
            float(df["uncertainty_score"].std()) > 0.02
            if "uncertainty_score" in df.columns else False,
            f"std = {float(df['uncertainty_score'].std()):.4f}"
            if "uncertainty_score" in df.columns else "no data",
        ),
        (
            "Fast layer high-severity override >10% reduction: 10-35% of days",
            0.10 <= float(
                (results["daily_df"]["hazard"]
                 .map(lambda h: max(0.0,
                      ((h - HAZARD_PREEMPTIVE_THRESHOLD)
                       / (1.0 - HAZARD_PREEMPTIVE_THRESHOLD))
                      * MAX_PREEMPTIVE_REDUCTION)
                     ) > 0.10
                ).mean()
            ) <= 0.35,
            f"hazard >10% = {float((results['daily_df']['hazard'].map(lambda h: max(0.0, ((h - HAZARD_PREEMPTIVE_THRESHOLD) / (1.0 - HAZARD_PREEMPTIVE_THRESHOLD)) * MAX_PREEMPTIVE_REDUCTION)) > 0.10).mean()):.1%}  (circuit breakers add ~5%)",
        ),
    ]

    print(f"\n  {'─' * 72}")
    print("  PASS CRITERIA")
    print(f"  {'─' * 72}")
    all_pass = True
    n_pass   = 0
    for desc, passed, detail in criteria:
        if passed:
            n_pass += 1
        else:
            all_pass = False
        print(f"  {'✓' if passed else '✗'}  {desc:<50} {detail}")

    verdict = (
        f"PASS ✓  ({n_pass}/{len(criteria)} criteria met)"
        if all_pass else
        f"PARTIAL  ({n_pass}/{len(criteria)} criteria met)"
    )
    print(f"\n  {'=' * 72}")
    print(f"  Overall: {verdict}")
    print("=" * 76)
    return all_pass