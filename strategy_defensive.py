"""
SPY Alpha V9 — Strategy 3: Defensive / Tail Protection
strategy_defensive.py
=========================================================
Modified from V8. V9 replaces observable stress computation
and HMM crisis-type labels with continuous state pillar inputs.

Activation formula (spec Section 6):
    defensive_activation = max(financial_stress, transition_hazard)

Asset selection (spec Section 6):
    inflation_pressure > 0.60 → {"GLD": 0.40, "SHY": 0.60}
    inflation_pressure < 0.30 → {"TLT": 0.40, "SHY": 0.60}
    otherwise                 → {"SHY": 0.70, "TLT": 0.15, "GLD": 0.15}

When activation < ACTIVATION_THRESHOLD: hold SHY only (dormant).
When activation ≥ threshold: blend from SHY toward base weights
by activation intensity.

No HMM dependency. No FRED fetching. No observable stress computation.
All inputs come from the continuous state engine and analog memory.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from strategy_base import StrategyOutput

logger = logging.getLogger("spy_alpha_v9.strategy_defensive")

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Activation threshold — below this, strategy holds SHY only
ACTIVATION_THRESHOLD: float = 0.35

# Inflation pressure pillar thresholds (from spec)
INFLATION_HIGH: float = 0.60   # above → inflationary regime → favor GLD
INFLATION_LOW:  float = 0.30   # below → deflationary regime → favor TLT

# Base weights per inflation regime (directly from spec)
BASE_WEIGHTS_INFLATION: Dict[str, float] = {"GLD": 0.40, "SHY": 0.60}
BASE_WEIGHTS_DEFLATION: Dict[str, float] = {"TLT": 0.40, "SHY": 0.60}
BASE_WEIGHTS_MIXED:     Dict[str, float] = {"SHY": 0.70, "TLT": 0.15, "GLD": 0.15}

# Dormant state — low or no activation
DORMANT_WEIGHTS: Dict[str, float] = {"SHY": 1.0}

# Default rebalance frequency
REBALANCE_EVERY: int = 5

# Validation episodes: (name, start, end, expected_lead_asset, note)
VALIDATION_EPISODES: List[Tuple[str, str, str, str, str]] = [
    ("2008 GFC",       "2008-09-01", "2009-03-31",
     "TLT", "activation high, deflationary"),
    ("2013-14 Bull",   "2013-01-01", "2014-12-31",
     "SHY", "activation low, dormant"),
    ("2019 Bull",      "2019-01-01", "2019-12-31",
     "SHY", "activation low, dormant"),
    ("Mar 2020 COVID", "2020-02-15", "2020-04-30",
     "TLT", "activation high, deflationary"),
    ("2021-22 Infla",  "2021-06-01", "2022-12-31",
     "GLD", "activation elevated, inflationary"),
]


# ---------------------------------------------------------------------------
# Helper — weight blending
# ---------------------------------------------------------------------------

def _blend_weights(
    dormant: Dict[str, float],
    target:  Dict[str, float],
    alpha:   float,
) -> Dict[str, float]:
    """
    Linear blend between dormant and target weight dicts.
    alpha=0 → dormant only. alpha=1 → target only.
    """
    all_assets = set(dormant) | set(target)
    return {
        a: (1.0 - alpha) * dormant.get(a, 0.0)
           + alpha       * target.get(a,  0.0)
        for a in all_assets
    }


# ---------------------------------------------------------------------------
# DefensiveStrategy
# ---------------------------------------------------------------------------

class DefensiveStrategy:
    """
    Strategy 3: Stress-activated defensive positioning.

    Dormant (100% SHY) when activation < ACTIVATION_THRESHOLD.
    Deploys TLT/GLD/SHY allocation when activation ≥ threshold.
    Asset split determined by inflation_pressure pillar.

    Expected standalone behavior:
        - Low or negative Sharpe (cost of protection, not alpha engine)
        - Strong positive returns during 2008 GFC and March 2020
        - GLD outperformance vs TLT during 2022 inflation bear
        - Negative or near-zero crisis correlation with Strategy 1
    """

    def __init__(
        self,
        activation_threshold: float = ACTIVATION_THRESHOLD,
        inflation_high:  float = INFLATION_HIGH,
        inflation_low:   float = INFLATION_LOW,
        rebalance_every: int   = REBALANCE_EVERY,
    ) -> None:
        self.activation_threshold = activation_threshold
        self.inflation_high       = inflation_high
        self.inflation_low        = inflation_low
        self.rebalance_every      = rebalance_every

        # Populated by build()
        self._financial_stress:   Optional[pd.Series]    = None
        self._inflation_pressure: Optional[pd.Series]    = None
        self._transition_hazard:  Optional[pd.Series]    = None
        self._activation:         Optional[pd.Series]    = None
        self._adj_close:          Optional[pd.DataFrame] = None

    # ---------------------------------------------------------------- #
    # Public: build                                                     #
    # ---------------------------------------------------------------- #

    def build(
        self,
        pillars:       pd.DataFrame,
        analog_scores: pd.DataFrame,
        adj_close:     pd.DataFrame,
    ) -> None:
        """
        Pre-compute activation series from pillars and analog memory.

        Args:
            pillars       : StateEngine output — must contain
                            financial_stress and inflation_pressure
            analog_scores : AnalogMemory output — must contain
                            transition_hazard
            adj_close     : adjusted close prices (stored for backtest)
        """
        logger.info("Strategy 3 (Defensive): Building signals...")

        # Validate required columns
        for col in ("financial_stress", "inflation_pressure"):
            if col not in pillars.columns:
                raise ValueError(
                    f"strategy_defensive.build(): "
                    f"'{col}' not found in pillars. "
                    f"Available: {list(pillars.columns)}"
                )
        if "transition_hazard" not in analog_scores.columns:
            raise ValueError(
                "strategy_defensive.build(): "
                "'transition_hazard' not found in analog_scores."
            )

        self._financial_stress   = pillars["financial_stress"]
        self._inflation_pressure = pillars["inflation_pressure"]
        self._adj_close          = adj_close

        # Align transition_hazard to pillar index via forward fill
        self._transition_hazard = (
            analog_scores["transition_hazard"]
            .reindex(pillars.index, method="ffill")
        )

        # Core spec formula: activation = max(financial_stress, transition_hazard)
        combined = pd.concat(
            [self._financial_stress, self._transition_hazard],
            axis=1
        )
        combined.columns = ["financial_stress", "transition_hazard"]
        self._activation = combined.max(axis=1)

        n_total     = int(self._activation.notna().sum())
        n_activated = int((self._activation > self.activation_threshold).sum())

        logger.info(
            f"Strategy 3 built: {n_total} days, "
            f"{n_activated} activated "
            f"({n_activated / max(n_total, 1) * 100:.1f}%), "
            f"mean activation: {self._activation.mean():.3f}"
        )

    # ---------------------------------------------------------------- #
    # Public: generate_signals                                          #
    # ---------------------------------------------------------------- #

    def generate_signals(
        self,
        rebalance_dates: Optional[pd.DatetimeIndex] = None,
    ) -> List[StrategyOutput]:
        """
        Generate StrategyOutput for each rebalance date.

        Args:
            rebalance_dates : if provided, signals generated on these dates
                              (for alignment with Strategy 1 refit schedule).
                              If None, generates on own schedule.
        """
        if self._activation is None:
            raise RuntimeError(
                "Call build() before generate_signals()"
            )

        available = self._activation.dropna().index

        if rebalance_dates is not None:
            dates = rebalance_dates.intersection(available)
        else:
            if len(available) == 0:
                logger.warning("No valid activation dates — check pillar data.")
                return []
            dates = available[::self.rebalance_every]

        outputs: List[StrategyOutput] = []
        for date in dates:
            out = self._generate_single_signal(date)
            if out is not None:
                outputs.append(out)

        n_activated = sum(
            1 for o in outputs
            if o.strategy_metadata.get("stress_activated", False)
        )
        logger.info(
            f"Strategy 3: {len(outputs)} signals, "
            f"{n_activated} activated "
            f"({n_activated / max(len(outputs), 1) * 100:.1f}%)"
        )
        return outputs

    # ---------------------------------------------------------------- #
    # Private: single signal computation                                #
    # ---------------------------------------------------------------- #

    def _generate_single_signal(
        self,
        date: pd.Timestamp,
    ) -> Optional[StrategyOutput]:
        """Compute a single StrategyOutput for one rebalance date."""
        if date not in self._activation.index:
            return None

        activation         = float(self._activation.loc[date])
        financial_stress   = float(self._financial_stress.loc[date]) \
                             if date in self._financial_stress.index else 0.0
        inflation_pressure = float(self._inflation_pressure.loc[date]) \
                             if date in self._inflation_pressure.index else 0.5
        transition_hazard  = float(self._transition_hazard.loc[date]) \
                             if date in self._transition_hazard.index else 0.0

        if any(np.isnan(v)
               for v in [activation, financial_stress, inflation_pressure]):
            return None

        # ---- Inflation regime and base weights (spec Section 6) ----
        if inflation_pressure > self.inflation_high:
            base_weights     = BASE_WEIGHTS_INFLATION.copy()
            inflation_regime = "inflationary"
        elif inflation_pressure < self.inflation_low:
            base_weights     = BASE_WEIGHTS_DEFLATION.copy()
            inflation_regime = "deflationary"
        else:
            base_weights     = BASE_WEIGHTS_MIXED.copy()
            inflation_regime = "mixed"

        # ---- Activation intensity and weight construction ----
        stress_activated = activation > self.activation_threshold

        if stress_activated:
            # Intensity: 0 at threshold → 1 at full activation
            intensity = float(np.clip(
                (activation - self.activation_threshold)
                / (1.0 - self.activation_threshold),
                0.0, 1.0
            ))
            # Smooth blend from SHY-only toward spec base weights
            weights    = _blend_weights(DORMANT_WEIGHTS, base_weights, intensity)
            confidence = intensity
        else:
            weights    = DORMANT_WEIGHTS.copy()
            intensity  = 0.0
            confidence = float(
                activation / max(self.activation_threshold, 1e-6)
            ) * 0.1

        # Clean and normalize
        weights = {k: v for k, v in weights.items() if v > 1e-6}
        total   = sum(weights.values())
        if total > 0:
            weights = {k: v / total for k, v in weights.items()}

        return StrategyOutput(
            strategy_name     = "defensive",
            proposed_weights  = weights,
            confidence        = float(np.clip(confidence, 0.0, 1.0)),
            active_assets     = [k for k, v in weights.items() if v > 0.01],
            strategy_metadata = {
                "date":               date.strftime("%Y-%m-%d"),
                "activation":         round(activation,         4),
                "financial_stress":   round(financial_stress,   4),
                "transition_hazard":  round(transition_hazard,  4),
                "inflation_pressure": round(inflation_pressure, 4),
                "inflation_regime":   inflation_regime,
                "stress_activated":   stress_activated,
                "intensity":          round(intensity,          4),
            },
        )

    # ---------------------------------------------------------------- #
    # Public: validate                                                  #
    # ---------------------------------------------------------------- #

    def validate(self) -> bool:
        """
        Validate strategy behavior against known historical episodes.

        Pass criteria:
            1. Mean activation during 2008 GFC > 0.50
            2. Mean activation during 2013-14 bull < 0.30
            3. Mean activation during March 2020 > 0.60
            4. 2008 GFC: TLT weight > GLD weight (deflationary)
            5. 2021-22: GLD weight > TLT weight (inflationary)
        """
        if self._activation is None:
            logger.error("validate(): call build() first.")
            return False

        print("\n" + "=" * 76)
        print("STRATEGY 3: DEFENSIVE — VALIDATION")
        print("=" * 76)

        n_total     = int(self._activation.notna().sum())
        n_activated = int((self._activation > self.activation_threshold).sum())
        print(f"\n  Full period: {n_total} days")
        print(f"  Mean activation        : {self._activation.mean():.3f}")
        print(f"  Days activated (>{self.activation_threshold:.2f}): "
              f"{n_activated} ({n_activated/max(n_total,1)*100:.1f}%)")

        # ---- Episode analysis ----
        print(f"\n  {'─' * 74}")
        print("  EPISODE ANALYSIS")
        print(f"\n  {'Episode':<20} {'Dates':<24} {'MeanAct':>8} "
              f"{'TLT%':>7} {'GLD%':>7} {'SHY%':>7}  Regime")
        print(f"  {'─'*20} {'─'*24} {'─'*8} "
              f"{'─'*7} {'─'*7} {'─'*7}  {'─'*16}")

        episode_results: Dict[str, Dict] = {}

        for name, start, end, expected_lead, note in VALIDATION_EPISODES:
            mask = (
                (self._activation.index >= start) &
                (self._activation.index <= end)    &
                self._activation.notna()
            )
            if mask.sum() == 0:
                print(f"  {name:<20} (no data in window)")
                continue

            mean_act = float(self._activation[mask].mean())

            # Collect weights by generating signals over episode dates
            episode_dates = self._activation[mask].index
            tlt_ws, gld_ws, shy_ws = [], [], []
            for date in episode_dates[::self.rebalance_every]:
                out = self._generate_single_signal(date)
                if out:
                    tlt_ws.append(out.proposed_weights.get("TLT", 0.0))
                    gld_ws.append(out.proposed_weights.get("GLD", 0.0))
                    shy_ws.append(out.proposed_weights.get("SHY", 0.0))

            mean_tlt = float(np.mean(tlt_ws)) if tlt_ws else 0.0
            mean_gld = float(np.mean(gld_ws)) if gld_ws else 0.0
            mean_shy = float(np.mean(shy_ws)) if shy_ws else 0.0

            episode_results[name] = dict(
                mean_act=mean_act,
                mean_tlt=mean_tlt,
                mean_gld=mean_gld,
                mean_shy=mean_shy,
            )

            print(
                f"  {name:<20} {start+' → '+end:<24} "
                f"{mean_act:>8.3f} "
                f"{mean_tlt:>6.1%} "
                f"{mean_gld:>6.1%} "
                f"{mean_shy:>6.1%}  "
                f"{note}"
            )

        # ---- Pass / Fail criteria ----
        def _get(ep: str, key: str, default: float) -> float:
            return episode_results.get(ep, {}).get(key, default)

        criteria = [
            (
                "2008 GFC: TLT+GLD weight > 20% (deployed)",
                _get("2008 GFC", "mean_tlt", 0.0)
                + _get("2008 GFC", "mean_gld", 0.0) > 0.20,
                f"TLT+GLD = {_get('2008 GFC','mean_tlt',0)+_get('2008 GFC','mean_gld',0):.1%}",
            ),
            (
                "2013-14 Bull: SHY weight > 85% (dormant)",
                _get("2013-14 Bull", "mean_shy", 0.0) > 0.85,
                f"SHY = {_get('2013-14 Bull', 'mean_shy', 0.0):.1%}",
            ),
            (
                "Mar 2020: TLT+GLD weight > 20% (deployed)",
                _get("Mar 2020 COVID", "mean_tlt", 0.0)
                + _get("Mar 2020 COVID", "mean_gld", 0.0) > 0.20,
                f"TLT+GLD = {_get('Mar 2020 COVID','mean_tlt',0)+_get('Mar 2020 COVID','mean_gld',0):.1%}",
            ),
            (
                "2008 GFC: TLT > GLD (deflationary crisis)",
                _get("2008 GFC", "mean_tlt", 0.0) >
                _get("2008 GFC", "mean_gld", 0.0),
                f"TLT={_get('2008 GFC','mean_tlt',0):.1%}  "
                f"GLD={_get('2008 GFC','mean_gld',0):.1%}",
            ),
            (
                "2021-22: GLD > TLT (inflationary environment)",
                _get("2021-22 Infla", "mean_gld", 0.0) >
                _get("2021-22 Infla", "mean_tlt", 0.0),
                f"GLD={_get('2021-22 Infla','mean_gld',0):.1%}  "
                f"TLT={_get('2021-22 Infla','mean_tlt',0):.1%}",
            ),
        ]

        print(f"\n  {'─' * 74}")
        print("  PASS CRITERIA")
        print(f"  {'─' * 74}")

        all_pass  = True
        n_passing = 0
        for desc, passed, detail in criteria:
            if passed:
                n_passing += 1
            else:
                all_pass = False
            print(f"  {'✓' if passed else '✗'}  {desc:<50} {detail}")

        # ---- Standalone backtest ----
        if self._adj_close is not None:
            print(f"\n  {'─' * 74}")
            print("  STANDALONE BACKTEST")
            _run_standalone_metrics(self)

        verdict = (
            f"PASS ✓  ({n_passing}/{len(criteria)} criteria met)"
            if all_pass else
            f"PARTIAL  ({n_passing}/{len(criteria)} criteria met)"
            f" — review failures above"
        )
        print(f"\n  {'=' * 74}")
        print(f"  Overall: {verdict}")
        print("=" * 76)
        return all_pass


# ---------------------------------------------------------------------------
# Standalone backtest helpers
# ---------------------------------------------------------------------------

def _run_standalone_metrics(strategy: DefensiveStrategy) -> None:
    """Compute and print standalone performance metrics."""
    adj_close = strategy._adj_close
    activation = strategy._activation

    available_dates = activation.dropna().index
    if len(available_dates) == 0:
        print("  No activation data for backtest.")
        return

    dates   = available_dates[::strategy.rebalance_every]
    outputs = [
        out for date in dates
        for out in [strategy._generate_single_signal(date)]
        if out is not None
    ]

    if not outputs:
        print("  No signals generated.")
        return

    sig_dates  = [pd.Timestamp(o.strategy_metadata["date"]) for o in outputs]
    def_assets = [a for a in ("TLT", "GLD", "SHY") if a in adj_close.columns]
    daily_ret  = adj_close[def_assets].pct_change()
    spy_ret    = adj_close["SPY"].pct_change() \
                 if "SPY" in adj_close.columns \
                 else pd.Series(0.0, index=adj_close.index)

    port_returns: List[float] = []
    current_w: Dict[str, float] = {}
    sig_idx = 0

    for date in daily_ret.index:
        while sig_idx < len(sig_dates) and sig_dates[sig_idx] <= date:
            current_w = outputs[sig_idx].proposed_weights
            sig_idx  += 1
        if not current_w:
            port_returns.append(0.0)
            continue
        pr = sum(
            w * daily_ret.loc[date, a]
            for a, w in current_w.items()
            if a in daily_ret.columns and pd.notna(daily_ret.loc[date, a])
        )
        port_returns.append(float(pr))

    port = pd.Series(port_returns, index=daily_ret.index)
    spy  = spy_ret.reindex(port.index)

    first = port[port != 0].index[0] if (port != 0).any() else port.index[0]
    port  = port.loc[first:].dropna()
    spy   = spy.loc[first:].dropna()
    n_yrs = len(port) / 252

    cum    = (1 + port).cumprod()
    cagr   = float(cum.iloc[-1] ** (1 / n_yrs) - 1)
    sharpe = float(port.mean() / (port.std() + 1e-10) * np.sqrt(252))
    peak   = cum.expanding().max()
    max_dd = float((cum / peak - 1).min())
    corr   = float(port.rolling(63).corr(spy).dropna().mean())

    print(f"\n  Summary:  CAGR={cagr:.1%}  Sharpe={sharpe:.2f}  "
          f"MaxDD={max_dd:.1%}  Mean 63d corr w/SPY={corr:.3f}")
    print(f"  (Low/negative Sharpe is expected — "
          f"value is in crisis protection, not standalone alpha)")

    crises = [
        ("2008 GFC",  "2008-09-01", "2009-03-31"),
        ("Mar 2020",  "2020-02-15", "2020-04-30"),
        ("2022 Bear", "2022-01-01", "2022-12-31"),
    ]
    print(f"\n  Crisis period returns:")
    print(f"  {'Episode':<16} {'Strategy':>10} {'SPY':>10} {'Excess':>10}")
    print(f"  {'─'*50}")
    for name, s, e in crises:
        mask = (port.index >= s) & (port.index <= e)
        if mask.sum() > 5:
            cr = float((1 + port[mask]).prod() - 1)
            sr = float((1 + spy[mask]).prod()  - 1)
            print(f"  {name:<16} {cr:>9.1%} {sr:>9.1%} {cr-sr:>9.1%}")
        else:
            print(f"  {name:<16} (insufficient data)")


def backtest_defensive_standalone(
    pillars:       pd.DataFrame,
    analog_scores: pd.DataFrame,
    adj_close:     pd.DataFrame,
) -> DefensiveStrategy:
    """
    Build and validate Strategy 3. Returns the built strategy object.
    """
    strategy = DefensiveStrategy()
    strategy.build(
        pillars       = pillars,
        analog_scores = analog_scores,
        adj_close     = adj_close,
    )
    strategy.validate()
    return strategy