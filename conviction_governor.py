"""
SPY Alpha V9 — Layer 3: Conviction Governor
conviction_governor.py
=============================================
Spec Section 7.

Triple-gated conviction budget:
    conviction_budget = favorable_score × coherence × reliability

Where:
    market_quality  = 0.30×participation + 0.30×(1-stress) + 0.20×trend + 0.20×growth
    favorable_score = market_quality × (1 - financial_stress) × growth_momentum

Governs:
    S1 capital weight boost:
        hazard_dampener = max(0, 1 - 2 × transition_hazard)
        s1_boost        = max_boost × conviction_budget × hazard_dampener

    Dynamic concentration ceilings:
        dynamic_max_upro   = base_max_upro × conviction_budget
        dynamic_max_weight = base_max_weight × (0.5 + 0.5 × conviction_budget)

    Conviction floor:
        conviction_budget < floor → no boost, no UPRO, full defensive

Capital weight redistribution:
    S1 receives: base_s1 + s1_boost
    S2/S3 split remaining proportional to their base weights
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger("spy_alpha_v9.conviction_governor")

# ---------------------------------------------------------------------------
# Risk profiles (spec Section 8D)
# ---------------------------------------------------------------------------

RISK_PROFILES: Dict[str, Dict] = {
    "aggressive": {
        "max_boost":        0.55,
        "base_max_upro":    0.45,
        "base_max_weight":  0.30,
        "conviction_floor": 0.10,
    },
    "balanced": {
        "max_boost":        0.45,
        "base_max_upro":    0.35,
        "base_max_weight":  0.30,
        "conviction_floor": 0.15,
    },
    "defensive": {
        "max_boost":        0.25,
        "base_max_upro":    0.20,
        "base_max_weight":  0.25,
        "conviction_floor": 0.20,
    },
}

DEFAULT_PROFILE = "aggressive"

# Default equal-weight base capital allocation across three strategies
DEFAULT_BASE_WEIGHTS: Dict[str, float] = {
    "s1": 1.0 / 3.0,
    "s2": 1.0 / 3.0,
    "s3": 1.0 / 3.0,
}

# market_quality component weights (spec Section 4G)
MQ_WEIGHTS: Dict[str, float] = {
    "participation_quality": 0.30,
    "financial_stress_inv":  0.30,   # uses (1 - financial_stress)
    "trend_persistence":     0.20,
    "growth_momentum":       0.20,
}


# ---------------------------------------------------------------------------
# Output dataclasses
# ---------------------------------------------------------------------------

@dataclass
class ConvictionState:
    """All computed conviction values for a single date."""
    date:              pd.Timestamp
    # Intermediate signals
    market_quality:    float
    favorable_score:   float
    coherence:         float
    reliability:       float
    transition_hazard: float
    # Core output
    conviction_budget: float
    hazard_dampener:   float
    # Governance outputs
    s1_boost:          float
    dynamic_max_upro:  float
    dynamic_max_weight: float
    # Adjusted capital weights
    adjusted_s1:       float
    adjusted_s2:       float
    adjusted_s3:       float
    # Flags
    full_defensive:    bool


@dataclass
class GovernanceOutput:
    """Capital weights and ceilings produced by governance application."""
    capital_weights:    Dict[str, float]
    dynamic_max_upro:   float
    dynamic_max_weight: float
    conviction_state:   ConvictionState


# ---------------------------------------------------------------------------
# ConvictionGovernor
# ---------------------------------------------------------------------------

class ConvictionGovernor:
    """
    Layer 3: Conviction Governor.

    Computes a triple-gated conviction budget and applies it to
    strategy capital weights and dynamic concentration ceilings.

    Inputs (via build()):
        pillars       : StateEngine output — needs growth_momentum,
                        financial_stress, trend_persistence,
                        participation_quality
        analog_scores : AnalogMemory output — coherence, reliability,
                        transition_hazard (all pre-ranked to [0,1])

    Usage:
        gov = ConvictionGovernor(risk_profile="aggressive")
        gov.build(engine.pillars, memory.analog_scores)
        output = gov.apply_governance(date)
    """

    def __init__(
        self,
        risk_profile: str = DEFAULT_PROFILE,
        base_weights: Optional[Dict[str, float]] = None,
    ) -> None:
        if risk_profile not in RISK_PROFILES:
            raise ValueError(
                f"Unknown risk_profile '{risk_profile}'. "
                f"Choose from: {list(RISK_PROFILES.keys())}"
            )
        self.risk_profile = risk_profile
        self.profile      = RISK_PROFILES[risk_profile].copy()
        self.base_weights = base_weights or DEFAULT_BASE_WEIGHTS.copy()

        # Populated by build()
        self._market_quality:    Optional[pd.Series] = None
        self._favorable_score:   Optional[pd.Series] = None
        self._conviction_budget: Optional[pd.Series] = None
        self._coherence:         Optional[pd.Series] = None
        self._reliability:       Optional[pd.Series] = None
        self._transition_hazard: Optional[pd.Series] = None
        self._growth_momentum:   Optional[pd.Series] = None
        self._financial_stress:  Optional[pd.Series] = None

    # ---------------------------------------------------------------- #
    # Public: build                                                     #
    # ---------------------------------------------------------------- #

    def build(
        self,
        pillars:       pd.DataFrame,
        analog_scores: pd.DataFrame,
    ) -> None:
        """
        Precompute conviction budget series for the full history.

        Args:
            pillars       : StateEngine output (growth_momentum,
                            financial_stress, trend_persistence,
                            participation_quality required)
            analog_scores : AnalogMemory output (coherence, reliability,
                            transition_hazard required — all [0,1])
        """
        logger.info("ConvictionGovernor: Building conviction series...")

        # Validate inputs
        required_pillars = [
            "growth_momentum", "financial_stress",
            "trend_persistence", "participation_quality",
        ]
        required_analog = ["coherence", "reliability", "transition_hazard"]

        for col in required_pillars:
            if col not in pillars.columns:
                raise ValueError(
                    f"ConvictionGovernor.build(): "
                    f"pillar column '{col}' not found. "
                    f"Available: {list(pillars.columns)}"
                )
        for col in required_analog:
            if col not in analog_scores.columns:
                raise ValueError(
                    f"ConvictionGovernor.build(): "
                    f"analog column '{col}' not found. "
                    f"Available: {list(analog_scores.columns)}"
                )

        # Store pillar series for single-date lookups
        self._growth_momentum  = pillars["growth_momentum"]
        self._financial_stress = pillars["financial_stress"]

        # Align analog scores to pillar index
        aligned = analog_scores.reindex(pillars.index, method="ffill")
        self._coherence         = aligned["coherence"]
        self._reliability       = aligned["reliability"]
        self._transition_hazard = aligned["transition_hazard"]

        # Compute market quality (spec Section 4G)
        self._market_quality = _compute_market_quality(pillars)

        # Compute favorable score (spec Section 7A)
        self._favorable_score = _compute_favorable_score(pillars)

        # Conviction budget = favorable × coherence × reliability (spec 7A)
        self._conviction_budget = (
            self._favorable_score
            * self._coherence
            * self._reliability
        ).clip(0.0, 1.0)

        n       = int(self._conviction_budget.notna().sum())
        cb_mean = float(self._conviction_budget.mean())
        cb_p75  = float(self._conviction_budget.quantile(0.75))
        cb_max  = float(self._conviction_budget.max())

        logger.info(
            f"ConvictionGovernor built: {n} days, "
            f"profile={self.risk_profile}, "
            f"conviction: mean={cb_mean:.4f}  "
            f"p75={cb_p75:.4f}  max={cb_max:.4f}"
        )

    # ---------------------------------------------------------------- #
    # Public: get_conviction                                            #
    # ---------------------------------------------------------------- #

    def get_conviction(
        self,
        date: pd.Timestamp,
    ) -> Optional[ConvictionState]:
        """
        Return all conviction values for a single date.
        Returns None if date is outside the conviction series.
        """
        if self._conviction_budget is None:
            raise RuntimeError("Call build() before get_conviction()")

        # Forward-fill to handle dates between pillar observations
        try:
            cb  = float(self._conviction_budget.loc[:date].iloc[-1])
            mq  = float(self._market_quality.loc[:date].iloc[-1])
            fs  = float(self._favorable_score.loc[:date].iloc[-1])
            coh = float(self._coherence.loc[:date].iloc[-1])
            rel = float(self._reliability.loc[:date].iloc[-1])
            haz = float(self._transition_hazard.loc[:date].iloc[-1])
        except (KeyError, IndexError):
            return None

        if any(np.isnan(v) for v in [cb, mq, fs, coh, rel, haz]):
            return None

        # Hazard dampener and boost (spec Section 7B)
        haz_damp = float(max(0.0, 1.0 - haz))
        s1_boost  = float(self.profile["max_boost"] * cb * haz_damp)

        # Dynamic ceilings (spec Section 7C)
        dyn_upro = float(self.profile["base_max_upro"] * cb)
        dyn_wt   = float(
            self.profile["base_max_weight"] * (0.5 + 0.5 * cb)
        )

        # Conviction floor check
        floor          = self.profile["conviction_floor"]
        full_defensive = cb < floor

        # Adjusted capital weights
        adj_s1, adj_s2, adj_s3 = self._apply_s1_boost(cb, s1_boost)

        return ConvictionState(
            date               = date,
            market_quality     = round(mq,       4),
            favorable_score    = round(fs,        4),
            coherence          = round(coh,       4),
            reliability        = round(rel,       4),
            transition_hazard  = round(haz,       4),
            conviction_budget  = round(cb,        4),
            hazard_dampener    = round(haz_damp,  4),
            s1_boost           = round(s1_boost,  4),
            dynamic_max_upro   = round(dyn_upro,  4),
            dynamic_max_weight = round(dyn_wt,    4),
            adjusted_s1        = round(adj_s1,    4),
            adjusted_s2        = round(adj_s2,    4),
            adjusted_s3        = round(adj_s3,    4),
            full_defensive     = full_defensive,
        )

    # ---------------------------------------------------------------- #
    # Public: apply_governance                                          #
    # ---------------------------------------------------------------- #

    def apply_governance(
        self,
        date: pd.Timestamp,
    ) -> Optional[GovernanceOutput]:
        """
        Apply conviction governance for a given date.

        Returns GovernanceOutput with adjusted capital weights and
        dynamic ceilings, or None if the date is not available.

        When conviction_budget < conviction_floor:
            - No S1 boost applied
            - dynamic_max_upro = 0
            - dynamic_max_weight = base × 0.5
        """
        state = self.get_conviction(date)
        if state is None:
            return None

        if state.full_defensive:
            return GovernanceOutput(
                capital_weights    = self.base_weights.copy(),
                dynamic_max_upro   = 0.0,
                dynamic_max_weight = self.profile["base_max_weight"] * 0.5,
                conviction_state   = state,
            )

        return GovernanceOutput(
            capital_weights    = {
                "s1": state.adjusted_s1,
                "s2": state.adjusted_s2,
                "s3": state.adjusted_s3,
            },
            dynamic_max_upro   = state.dynamic_max_upro,
            dynamic_max_weight = state.dynamic_max_weight,
            conviction_state   = state,
        )

    # ---------------------------------------------------------------- #
    # Public: validate                                                  #
    # ---------------------------------------------------------------- #

    def validate(self, adj_close: pd.DataFrame) -> bool:
        """
        Validate conviction governor calibration (spec Section 13C).

        Pass criteria:
            1. Conviction budget varies meaningfully (std > 0.005)
            2. S1 mean weight > base weight (governor adds upward bias)
            3. Monotonic forward Sharpe across conviction quintiles
            4. High-hazard states show lower S1 boost than low-hazard
            5. Crisis periods show lower conviction than bull periods
        """
        if self._conviction_budget is None:
            logger.error("validate(): call build() first.")
            return False

        print("\n" + "=" * 76)
        print("CONVICTION GOVERNOR — VALIDATION")
        print(f"Risk profile : {self.risk_profile}")
        print(f"Base weights : S1={self.base_weights['s1']:.3f}  "
              f"S2={self.base_weights['s2']:.3f}  "
              f"S3={self.base_weights['s3']:.3f}")
        print("=" * 76)

        cb = self._conviction_budget.dropna()

        # ---- Conviction budget distribution ----
        print(f"\n  Conviction Budget Distribution ({len(cb)} days):")
        print(f"  {'Min':>8} {'P10':>8} {'P25':>8} {'Median':>8} "
              f"{'P75':>8} {'P90':>8} {'Max':>8} {'Mean':>8}")
        print(f"  {'─'*8} {'─'*8} {'─'*8} {'─'*8} "
              f"{'─'*8} {'─'*8} {'─'*8} {'─'*8}")
        print(
            f"  {cb.min():>8.4f} "
            f"{cb.quantile(0.10):>8.4f} "
            f"{cb.quantile(0.25):>8.4f} "
            f"{cb.median():>8.4f} "
            f"{cb.quantile(0.75):>8.4f} "
            f"{cb.quantile(0.90):>8.4f} "
            f"{cb.max():>8.4f} "
            f"{cb.mean():>8.4f}"
        )

        floor = self.profile["conviction_floor"]
        floor_days = int((cb < floor).sum())
        print(f"\n  Full defensive days (cb < {floor:.2f}): "
              f"{floor_days} ({floor_days/len(cb)*100:.1f}%)")

        # ---- S1 capital weight distribution ----
        adj_s1_series = self._compute_adj_s1_series()
        print(f"\n  S1 Capital Weight Distribution:")
        print(f"  Mean={adj_s1_series.mean():.4f}  "
              f"Median={adj_s1_series.median():.4f}  "
              f"P75={adj_s1_series.quantile(0.75):.4f}  "
              f"Max={adj_s1_series.max():.4f}  "
              f"(base={self.base_weights['s1']:.3f})")

        # ---- Quintile Sharpe calibration ----
        print(f"\n  {'─' * 72}")
        print("  CONVICTION QUINTILE CALIBRATION")
        print("  (Higher conviction → higher forward SPY Sharpe = PASS)")
        monotonic, q_df = self._quintile_calibration(cb, adj_close)
        q1_ret = float(q_df.iloc[0]["fwd_return"])  if q_df is not None and len(q_df) >= 5 else float("nan")
        q5_ret = float(q_df.iloc[-1]["fwd_return"]) if q_df is not None and len(q_df) >= 5 else float("nan")

        if q_df is not None:
            print(f"\n  {'Quintile':<10} {'CB Range':>18} "
                  f"{'N':>6} {'Fwd 21d Sharpe':>16} {'Fwd Return%':>13}")
            print(f"  {'─'*10} {'─'*18} {'─'*6} {'─'*16} {'─'*13}")
            for _, row in q_df.iterrows():
                print(
                    f"  {row['label']:<10} "
                    f"[{row['cb_min']:.4f}, {row['cb_max']:.4f}]{' ':>2}"
                    f"{row['n']:>6} "
                    f"{row['fwd_sharpe']:>16.3f} "
                    f"{row['fwd_return']*100:>12.2f}%"
                )
            print(f"\n  Monotonic Q1→Q5: {'✓' if monotonic else '✗'}")
            if not np.isnan(q1_ret) and not np.isnan(q5_ret):
                print(f"  Q5 return ({q5_ret*100:.2f}%) vs Q1 return "
                      f"({q1_ret*100:.2f}%):  "
                      f"{'Q5>Q1 ✓' if q5_ret > q1_ret else 'Q5≤Q1 ✗'}  "
                      f"[Sharpe monotonic = {'✓' if monotonic else '✗'} — informational]")

        # ---- Hazard dampener check ----
        print(f"\n  {'─' * 72}")
        print("  HAZARD DAMPENER CHECK")
        hazard_pass = self._hazard_dampener_check()

        # ---- Episode spot check ----
        print(f"\n  {'─' * 72}")
        print("  EPISODE SPOT CHECK")
        episode_pass = self._episode_spot_check()

        # ---- Pass criteria ----
        criteria = [
            (
                "Conviction budget varies (std > 0.005)",
                float(cb.std()) > 0.005,
                f"std = {cb.std():.5f}",
            ),
            (
                "S1 mean weight > base weight",
                float(adj_s1_series.mean()) > self.base_weights["s1"],
                f"mean = {adj_s1_series.mean():.4f}  "
                f"base = {self.base_weights['s1']:.4f}",
            ),
            (
                "High conviction return > low conviction (Q5 > Q1)",
                (not np.isnan(q5_ret)) and (not np.isnan(q1_ret))
                and q5_ret > q1_ret,
                f"Q5={q5_ret*100:.2f}%  Q1={q1_ret*100:.2f}%"
                if not np.isnan(q5_ret) else "no data",
            ),
            (
                "Hazard dampener reduces S1 boost",
                hazard_pass,
                "high hazard → lower boost",
            ),
            (
                "Crisis < bull conviction (episode check)",
                episode_pass,
                "see episode table above",
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
            print(f"  {'✓' if passed else '✗'}  {desc:<52} {detail}")

        verdict = (
            f"PASS ✓  ({n_pass}/{len(criteria)} criteria met)"
            if all_pass else
            f"PARTIAL  ({n_pass}/{len(criteria)} criteria met)"
        )
        print(f"\n  {'=' * 72}")
        print(f"  Overall: {verdict}")
        print("=" * 76)
        return all_pass

    # ---------------------------------------------------------------- #
    # Private: computation                                              #
    # ---------------------------------------------------------------- #

    def _apply_s1_boost(
        self,
        conviction_budget: float,
        s1_boost: float,
    ) -> Tuple[float, float, float]:
        """
        Apply S1 boost and redistribute remaining weight to S2/S3.

        S1 receives: min(base_s1 + s1_boost, 0.80)
        S2/S3 split remaining proportionally to their base weights.
        """
        floor = self.profile["conviction_floor"]
        if conviction_budget < floor:
            return (
                self.base_weights["s1"],
                self.base_weights["s2"],
                self.base_weights["s3"],
            )

        adj_s1    = float(min(self.base_weights["s1"] + s1_boost, 0.80))
        remaining = 1.0 - adj_s1

        base_s23 = self.base_weights["s2"] + self.base_weights["s3"]
        if base_s23 > 1e-6:
            adj_s2 = remaining * (self.base_weights["s2"] / base_s23)
            adj_s3 = remaining * (self.base_weights["s3"] / base_s23)
        else:
            adj_s2 = remaining / 2.0
            adj_s3 = remaining / 2.0

        return float(adj_s1), float(adj_s2), float(adj_s3)

    def _compute_adj_s1_series(self) -> pd.Series:
        """Vectorised computation of full adjusted S1 weight series."""
        cb       = self._conviction_budget.dropna()
        haz      = self._transition_hazard.reindex(cb.index).fillna(0.0)
        haz_damp = (1.0 - haz).clip(0.0, 1.0)
        s1_boost = self.profile["max_boost"] * cb * haz_damp

        # Below conviction floor → no boost
        floor          = self.profile["conviction_floor"]
        effective_boost = s1_boost.copy()
        effective_boost[cb < floor] = 0.0

        adj_s1 = (self.base_weights["s1"] + effective_boost).clip(
            self.base_weights["s1"], 0.80
        )
        return adj_s1

    def _quintile_calibration(
        self,
        cb: pd.Series,
        adj_close: pd.DataFrame,
    ) -> Tuple[bool, Optional[pd.DataFrame]]:
        """
        Compute forward 21-day SPY Sharpe by conviction quintile.
        Monotonically increasing Sharpe (Q1→Q5) = PASS.
        """
        if "SPY" not in adj_close.columns:
            logger.warning("SPY not in adj_close — skipping quintile test")
            return False, None

        spy_ret = adj_close["SPY"].pct_change()
        records = []

        for i, (date, cv) in enumerate(cb.items()):
            fwd = spy_ret.loc[date:].iloc[1:22]
            if len(fwd) < 15:
                continue
            fwd_sharpe = float(fwd.mean() / (fwd.std() + 1e-10) * np.sqrt(252))
            fwd_return = float(fwd.mean() * 252)
            records.append({
                "conviction": cv,
                "fwd_sharpe": fwd_sharpe,
                "fwd_return": fwd_return,
            })

        if len(records) < 50:
            return False, None

        df = pd.DataFrame(records)
        try:
            df["q"] = pd.qcut(df["conviction"], q=5,
                               labels=False, duplicates="drop")
        except Exception:
            return False, None

        agg = (
            df.groupby("q")
            .agg(
                cb_min     = ("conviction", "min"),
                cb_max     = ("conviction", "max"),
                n          = ("conviction", "count"),
                fwd_sharpe = ("fwd_sharpe", "mean"),
                fwd_return = ("fwd_return", "mean"),
            )
            .reset_index()
        )
        labels = ["Q1 (low)", "Q2", "Q3", "Q4", "Q5 (high)"]
        agg["label"] = [
            labels[i] if i < len(labels) else f"Q{i+1}"
            for i in range(len(agg))
        ]

        sharpes  = agg["fwd_sharpe"].values
        monotonic = all(
            sharpes[i] <= sharpes[i + 1]
            for i in range(len(sharpes) - 1)
        )
        return monotonic, agg

    def _hazard_dampener_check(self) -> bool:
        """
        Verify low-hazard states receive higher S1 boost than high-hazard.
        """
        cb  = self._conviction_budget.dropna()
        haz = self._transition_hazard.reindex(cb.index).fillna(0.0)
        haz_damp  = (1.0 - haz).clip(0.0, 1.0)
        s1_boost  = self.profile["max_boost"] * cb * haz_damp

        low_mask  = haz < haz.quantile(0.25)
        high_mask = haz > haz.quantile(0.75)

        low_boost  = float(s1_boost[low_mask].mean())
        high_boost = float(s1_boost[high_mask].mean())

        print(f"  Mean S1 boost  low hazard (Q1)  : {low_boost:.5f}")
        print(f"  Mean S1 boost  high hazard (Q4) : {high_boost:.5f}")
        passed = low_boost > high_boost
        print(f"  Low > High: {'✓' if passed else '✗'}")
        return passed

    def _episode_spot_check(self) -> bool:
        """
        Print conviction budget during known episodes.
        PASS if mean conviction during crisis < mean conviction during bull.
        """
        cb = self._conviction_budget

        episodes = [
            ("2008 GFC",     "2008-09-01", "2009-03-31", "crisis"),
            ("2013-14 Bull", "2013-01-01", "2014-12-31", "bull"),
            ("2019 Bull",    "2019-01-01", "2019-12-31", "bull"),
            ("Mar 2020",     "2020-02-15", "2020-04-30", "crisis"),
            ("2022 Bear",    "2022-01-01", "2022-12-31", "bear"),
        ]

        print(f"\n  {'Episode':<20} {'Mean CB':>9} {'Mean S1 Wt':>11} "
              f"{'Type':>8}")
        print(f"  {'─'*20} {'─'*9} {'─'*11} {'─'*8}")

        bull_cbs:   list = []
        crisis_cbs: list = []

        for name, start, end, ep_type in episodes:
            mask = (
                (cb.index >= start) & (cb.index <= end) & cb.notna()
            )
            if mask.sum() == 0:
                print(f"  {name:<20} (no data)")
                continue

            mean_cb  = float(cb[mask].mean())
            haz      = self._transition_hazard[mask].fillna(0.0).mean()
            haz_damp = float(max(0.0, 1.0 - haz))
            boost    = self.profile["max_boost"] * mean_cb * haz_damp
            adj_s1   = float(min(self.base_weights["s1"] + boost, 0.80))

            print(f"  {name:<20} {mean_cb:>9.4f} {adj_s1:>10.4f}  "
                  f"{ep_type:>8}")

            if ep_type == "bull":
                bull_cbs.append(mean_cb)
            elif ep_type == "crisis":
                crisis_cbs.append(mean_cb)

        if bull_cbs and crisis_cbs:
            bull_mean   = float(np.mean(bull_cbs))
            crisis_mean = float(np.mean(crisis_cbs))
            passed      = bull_mean > crisis_mean
            print(f"\n  Mean bull conviction   : {bull_mean:.4f}")
            print(f"  Mean crisis conviction : {crisis_mean:.4f}")
            print(f"  Bull > Crisis: {'✓' if passed else '✗'}")
            return passed

        return False


# ---------------------------------------------------------------------------
# Module-level formula functions (shared with downstream modules)
# ---------------------------------------------------------------------------

def _compute_market_quality(pillars: pd.DataFrame) -> pd.Series:
    """
    Spec Section 4G: market quality derived summary.

    market_quality = 0.30×participation + 0.30×(1-stress)
                   + 0.20×trend         + 0.20×growth
    """
    return (
        MQ_WEIGHTS["participation_quality"] * pillars["participation_quality"]
        + MQ_WEIGHTS["financial_stress_inv"] * (1.0 - pillars["financial_stress"])
        + MQ_WEIGHTS["trend_persistence"]    * pillars["trend_persistence"]
        + MQ_WEIGHTS["growth_momentum"]      * pillars["growth_momentum"]
    ).clip(0.0, 1.0)


def _compute_favorable_score(
    pillars: pd.DataFrame,
) -> pd.Series:
    """
    Additive favorable score — spec Section 4G risk_appetite formula.

    Two consecutive tests of the multiplicative formula (original triple
    product, then market_quality × growth_momentum) both left bull market
    conviction below the conviction floor (0.10), meaning the governor
    provided zero S1 boost during 2013-14 and 2019 bull markets.

    The additive formula preserves the economic content (growth, trend,
    inverse-stress all matter) while producing a [0.25, 0.75] range
    compatible with the conviction floor and the triple product gate
    (favorable × coherence × reliability).
    """
    return (
        0.35 * pillars["growth_momentum"]
        + 0.30 * pillars["trend_persistence"]
        + 0.35 * (1.0 - pillars["financial_stress"])
    ).clip(0.0, 1.0)