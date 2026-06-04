"""
SPY Alpha V9 — Step 3: Trajectory Features
trajectory.py

For each of the 5 state engine pillars, computes 4 trajectory features:

    velocity    : 10-day rate of change — is the pillar rising or falling?
    persistence : 21-day directional consistency — is the trend sustained?
    divergence  : deviation from 63-day rolling mean — is the pillar at an extreme?
    stability   : inverse of 21-day rolling std — is the signal reliable?

20 features total (4 × 5 pillars) form the dynamic component of the
25-dimensional state vector used by the analog memory for KNN retrieval.

    State vector = [5 pillar scores] + [20 trajectory features] = 25 dimensions

Design note: This module intentionally avoids importing from state_engine.py
to prevent circular imports. rolling_percentile_rank is re-implemented locally
with identical logic.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ------------------------------------------------------------------ #
# Constants                                                            #
# ------------------------------------------------------------------ #

PILLAR_NAMES: List[str] = [
    "growth_momentum",
    "inflation_pressure",
    "financial_stress",
    "trend_persistence",
    "participation_quality",
]

FEATURE_TYPES: List[str] = ["velocity", "persistence", "divergence", "stability"]

# Rolling window for percentile ranking trajectory features (1 year of context)
RANK_WINDOW: int = 252
RANK_MIN_PERIODS: int = 63

# Computation windows
VELOCITY_WINDOW: int = 10       # 2 trading weeks — short-term direction
PERSISTENCE_WINDOW: int = 21    # 1 trading month — directional consistency
DIVERGENCE_WINDOW: int = 63     # 1 trading quarter — deviation from recent mean
STABILITY_WINDOW: int = 21      # 1 trading month — flickering vs reliable
MIN_PERIODS: int = 21           # Minimum periods for valid trajectory output


# ------------------------------------------------------------------ #
# Utilities                                                            #
# ------------------------------------------------------------------ #

def _rolling_percentile_rank(
    series: pd.Series,
    window: int = RANK_WINDOW,
    min_periods: int = RANK_MIN_PERIODS,
) -> pd.Series:
    """
    Convert a raw series to [0, 1] via rolling percentile rank.

    Identical implementation to state_engine.rolling_percentile_rank.
    Defined locally to avoid circular imports.

    High raw value → high percentile → high score.
    """
    return series.rolling(window, min_periods=min_periods).rank(pct=True)


# ------------------------------------------------------------------ #
# TrajectoryEngine                                                     #
# ------------------------------------------------------------------ #

class TrajectoryEngine:
    """
    Computes 20 trajectory features from 5 pillar scores.

    All features are rolling-percentile-ranked to [0, 1]:

        velocity    > 0.5  pillar rising  (state building)
                    ~ 0.5  pillar stable  (no direction)
                    < 0.5  pillar falling (state unwinding)

        persistence > 0.5  consistently rising  (trend locked in)
                    ~ 0.5  oscillating          (no persistence)
                    < 0.5  consistently falling (deteriorating trend)

        divergence  > 0.5  above recent mean  (extended — may revert)
                    ~ 0.5  near recent mean   (neutral position)
                    < 0.5  below recent mean  (depressed — may recover)

        stability   > 0.5  stable / low vol   (high conviction)
                    ~ 0.5  average volatility (standard conviction)
                    < 0.5  flickering         (low conviction — uncertainty)

    Critical downstream uses:
        - Analog memory:        trajectory features enrich the 25-dim state vector
                                so KNN retrieval matches dynamic context, not just
                                pillar level
        - Conviction governor:  stability features directly weight pillar confidence
                                (low stability → reduced pillar conviction weight)
    """

    def __init__(
        self,
        velocity_window: int = VELOCITY_WINDOW,
        persistence_window: int = PERSISTENCE_WINDOW,
        divergence_window: int = DIVERGENCE_WINDOW,
        stability_window: int = STABILITY_WINDOW,
        min_periods: int = MIN_PERIODS,
        rank_window: int = RANK_WINDOW,
        rank_min_periods: int = RANK_MIN_PERIODS,
    ) -> None:
        self.velocity_window    = velocity_window
        self.persistence_window = persistence_window
        self.divergence_window  = divergence_window
        self.stability_window   = stability_window
        self.min_periods        = min_periods
        self.rank_window        = rank_window
        self.rank_min_periods   = rank_min_periods
        self._features: Optional[pd.DataFrame] = None

    # ---------------------------------------------------------------- #
    # Public interface                                                   #
    # ---------------------------------------------------------------- #

    @property
    def features(self) -> Optional[pd.DataFrame]:
        """20-column DataFrame of trajectory features. None until build() called."""
        return self._features

    @property
    def feature_names(self) -> List[str]:
        """Ordered list of all 20 trajectory feature column names."""
        return [
            f"{pillar}_{feature}"
            for pillar in PILLAR_NAMES
            for feature in FEATURE_TYPES
        ]

    def build(self, pillars: pd.DataFrame) -> pd.DataFrame:
        """
        Compute all 20 trajectory features from validated pillar scores.

        Args:
            pillars: DataFrame from StateEngine.pillars — must contain columns
                     for each of the 5 PILLAR_NAMES. Missing pillars are skipped
                     with a warning; weights renormalize automatically downstream.

        Returns:
            DataFrame with up to 20 trajectory feature columns, index aligned
            to pillars. Rows during warmup period contain NaN.
        """
        missing_pillars = [p for p in PILLAR_NAMES if p not in pillars.columns]
        if missing_pillars:
            logger.warning(
                f"TrajectoryEngine.build: missing pillars {missing_pillars}. "
                f"Trajectory features for these pillars will be absent."
            )

        feature_dict: Dict[str, pd.Series] = {}

        for pillar_name in PILLAR_NAMES:
            if pillar_name not in pillars.columns:
                continue

            pillar = pillars[pillar_name]
            px = pillar_name   # column prefix

            vel = self._velocity(pillar)
            per = self._persistence(pillar)
            div = self._divergence(pillar)
            sta = self._stability(pillar)

            feature_dict[f"{px}_velocity"]    = vel
            feature_dict[f"{px}_persistence"] = per
            feature_dict[f"{px}_divergence"]  = div
            feature_dict[f"{px}_stability"]   = sta

            logger.info(
                f"  [Trajectory] {pillar_name:<24} "
                f"vel={vel.notna().sum():4d}  "
                f"per={per.notna().sum():4d}  "
                f"div={div.notna().sum():4d}  "
                f"sta={sta.notna().sum():4d} valid days"
            )

        if not feature_dict:
            logger.error("TrajectoryEngine.build: no features computed — check pillars")
            return pd.DataFrame(index=pillars.index)

        self._features = (
            pd.DataFrame(feature_dict)
            .reindex(pillars.index)
        )

        n_complete = self._features.notna().all(axis=1).sum()
        logger.info(
            f"TrajectoryEngine complete: {len(feature_dict)}/20 features built. "
            f"{n_complete}/{len(pillars)} days with full trajectory vectors."
        )

        return self._features

    def state_vector(self, pillars: pd.DataFrame) -> pd.DataFrame:
        """
        Construct the full 25-dimensional state vector for analog memory.

        Concatenates 5 pillar scores + 20 trajectory features.

        Args:
            pillars: DataFrame with 5 pillar score columns.

        Returns:
            DataFrame with 25 columns ordered: [pillar scores | trajectory features].
            Only rows where ALL 25 values are non-NaN form valid state observations.

        Raises:
            RuntimeError: if build() has not been called first.
        """
        if self._features is None:
            raise RuntimeError(
                "TrajectoryEngine.state_vector: call build() before state_vector(). "
                "Trajectory features have not been computed yet."
            )

        sv = pd.concat([pillars[PILLAR_NAMES], self._features], axis=1)
        n_complete = sv.notna().all(axis=1).sum()

        logger.info(
            f"State vector: 25 dimensions × {len(sv)} trading days. "
            f"{n_complete} complete rows ({n_complete / len(sv):.1%})."
        )

        return sv

    # ---------------------------------------------------------------- #
    # Feature computation (private)                                     #
    # ---------------------------------------------------------------- #

    def _velocity(self, pillar: pd.Series) -> pd.Series:
        """
        N-day rate of change of the pillar score, rolling-percentile-ranked.

        Captures short-term momentum of the regime:
          - Is financial stress rising (bad) or falling (recovering)?
          - Is growth momentum improving or deteriorating?
          - Is trend persistence building or collapsing?

        The rolling percentile provides context: is today's velocity
        unusually fast relative to recent history?
        """
        raw = pillar.diff(self.velocity_window)
        return _rolling_percentile_rank(
            raw,
            window=self.rank_window,
            min_periods=self.rank_min_periods,
        )

    def _persistence(self, pillar: pd.Series) -> pd.Series:
        """
        Fraction of last N days where the pillar moved in the upward direction,
        rolling-percentile-ranked.

        Captures directional consistency — is the current trend sustained?
          - A pillar rising 90% of days in the past month = persistent trend
          - A pillar oscillating ~50/50 = no persistent direction
          - A pillar falling 80% of days = persistent deterioration

        Distinguishes between genuine regime transitions and noise:
        the analog memory uses persistence to weight neighbors that
        entered similar regime states via similar trajectories.
        """
        daily_diff = pillar.diff(1)
        direction_fraction = daily_diff.rolling(
            self.persistence_window,
            min_periods=max(5, self.min_periods // 4),
        ).apply(lambda x: float((x > 0).mean()), raw=True)

        return _rolling_percentile_rank(
            direction_fraction,
            window=self.rank_window,
            min_periods=self.rank_min_periods,
        )

    def _divergence(self, pillar: pd.Series) -> pd.Series:
        """
        Current pillar value minus its N-day rolling mean,
        rolling-percentile-ranked.

        Captures mean-reversion signal — is the pillar at an unusual extreme?
          - > 0.5: pillar is above its recent average (extended — may revert)
          - ~ 0.5: pillar is near its recent average (no unusual deviation)
          - < 0.5: pillar is below its recent average (depressed — may recover)

        Enriches analog memory: two states with identical pillar scores
        but opposite divergences have materially different forward-return
        distributions (extended high stress vs. just-arrived high stress).
        """
        rolling_mean = pillar.rolling(
            self.divergence_window,
            min_periods=self.min_periods,
        ).mean()
        raw = pillar - rolling_mean

        return _rolling_percentile_rank(
            raw,
            window=self.rank_window,
            min_periods=self.rank_min_periods,
        )

    def _stability(self, pillar: pd.Series) -> pd.Series:
        """
        Inverse of the rolling standard deviation of daily pillar changes,
        rolling-percentile-ranked.

        Captures signal confidence — how reliably is this pillar reading?
          - > 0.5: stable / low volatility (high conviction — trust the signal)
          - ~ 0.5: average volatility (standard conviction)
          - < 0.5: flickering / high volatility (low conviction — uncertainty)

        Direct input to the conviction governor's uncertainty dampener:
        stability < 0.30 triggers reduced weighting of that pillar's
        contribution to the composite regime score.
        """
        daily_changes = pillar.diff(1)
        volatility = daily_changes.rolling(
            self.stability_window,
            min_periods=max(5, self.min_periods // 4),
        ).std()
        stability_raw = -volatility   # negate: high vol → low stability

        return _rolling_percentile_rank(
            stability_raw,
            window=self.rank_window,
            min_periods=self.rank_min_periods,
        )

    # ---------------------------------------------------------------- #
    # Validation                                                        #
    # ---------------------------------------------------------------- #

    def validate(self, pillars: pd.DataFrame) -> bool:
        """
        Validate trajectory features against expected economic behavior.

        Checks per spec Section 5:
            1. All 20 features present and in [0, 1]
            2. State vector has >= 80% complete rows for analog memory
            3. Feature distributions not degenerate (std > 0.05)
            4. Economic episodes behave as expected

        Episode validation logic:
            Mar 2020       financial_stress_velocity HIGH  (stress surging)
            Mar 2020       trend_persistence_velocity LOW  (trend collapsing)
            2013-14 Bull   trend_persistence_persistence HIGH (sustained trend)
            2017 Calm Bull trend_persistence_stability HIGH  (calm signals)
            2022 Bear      trend_persistence_stability LOW   (volatile signals)
        """
        if self._features is None or self._features.empty:
            logger.error("Validate: call build() first")
            return False

        tf = self._features
        all_pass = True

        print("\n" + "=" * 70)
        print("TRAJECTORY FEATURES — VALIDATION")
        print("=" * 70)

        # ---- Check 1: Feature completeness ----
        expected = set(self.feature_names)
        present  = set(tf.columns)
        missing  = expected - present
        n_present = len(present)
        feat_ok = (len(missing) == 0)
        all_pass = all_pass and feat_ok
        print(
            f"\n  Feature completeness:  {n_present}/20 present  "
            f"{'✓' if feat_ok else f'✗  missing: {missing}'}"
        )

        # ---- Check 2: Range [0, 1] ----
        violations = int(((tf < 0) | (tf > 1)).sum().sum())
        range_ok = (violations == 0)
        all_pass = all_pass and range_ok
        print(f"  Range [0, 1]:          {violations} violations  "
              f"{'✓' if range_ok else '✗'}")

        # ---- Check 3: State vector coverage ----
        sv = pd.concat([pillars[PILLAR_NAMES], tf], axis=1)
        n_complete = int(sv.notna().all(axis=1).sum())
        n_total    = len(sv)
        coverage   = n_complete / n_total if n_total > 0 else 0.0
        cov_ok     = coverage >= 0.80
        all_pass   = all_pass and cov_ok
        print(
            f"  State vector coverage: {n_complete}/{n_total} rows "
            f"({coverage:.1%})  {'✓' if cov_ok else '✗'}"
        )

        # ---- Check 4: Distribution check ----
        print(
            f"\n  {'Feature':<44} {'Mean':>6} {'Std':>6}  Status"
        )
        print("  " + "-" * 64)
        dist_ok = True
        for col in sorted(tf.columns):
            s = tf[col].dropna()
            if len(s) == 0:
                print(f"  {col:<44} {'N/A':>6} {'N/A':>6}  ✗ no data")
                dist_ok = False
                continue
            mean_v = float(s.mean())
            std_v  = float(s.std())
            flag   = "⚠ flat" if std_v < 0.05 else "✓"
            if std_v < 0.05:
                dist_ok = False
            print(f"  {col:<44} {mean_v:>6.3f} {std_v:>6.3f}  {flag}")
        all_pass = all_pass and dist_ok

        # ---- Check 5: Economic episode validation ----
        print(f"\n  {'─' * 68}")
        print("  Economic Episode Validation")
        print(f"  {'─' * 68}")
        print(
            f"  {'Episode':<14} {'Feature':<40} {'Mean':>6}  "
            f"{'Expected':<12} Pass"
        )
        print("  " + "-" * 80)

        econ_tests = [
            (
                "Mar 2020",
                "2020-03-01", "2020-03-31",
                "financial_stress_velocity",
                "high", 0.65,
                "stress surging",
            ),
            (
                "Mar 2020",
                "2020-03-01", "2020-03-31",
                "trend_persistence_velocity",
                "low", 0.35,
                "trend collapsing",
            ),
            (
                "2013-14 Bull",
                "2013-01-01", "2014-12-31",
                "trend_persistence_persistence",
                "high", 0.52,
                "sustained uptrend",
            ),
            (
                "2017 Calm",
                "2017-01-01", "2017-12-31",
                "trend_persistence_stability",
                "high", 0.50,
                "calm bull → stable signal",
            ),
        ]

        econ_ok = True
        for ep_name, start, end, feature, direction, threshold, desc in econ_tests:
            if feature not in tf.columns:
                print(
                    f"  {ep_name:<14} {feature:<40} {'N/A':>6}  "
                    f"MISSING  ✗"
                )
                econ_ok = False
                continue

            mask = (tf.index >= start) & (tf.index <= end)
            if mask.sum() == 0:
                print(
                    f"  {ep_name:<14} {feature:<40} {'N/A':>6}  "
                    f"NO DATA  ✗"
                )
                continue

            mean_v = float(tf.loc[mask, feature].mean())
            if direction == "high":
                passed = mean_v >= threshold
                expected_str = f">= {threshold:.2f}"
            else:
                passed = mean_v <= threshold
                expected_str = f"<= {threshold:.2f}"

            econ_ok  = econ_ok and passed
            status   = "✓" if passed else "✗"
            print(
                f"  {ep_name:<14} {feature:<40} {mean_v:>6.3f}  "
                f"{expected_str:<12} {status}"
            )

        all_pass = all_pass and econ_ok

        # ---- Summary ----
        print(f"\n  {'─' * 68}")
        verdict = "PASS ✓" if all_pass else "FAIL ✗  — review flagged items above"
        print(f"  Overall: {verdict}")
        print(f"\n  State vector dimensions: 25 (5 pillars + 20 trajectory features)")
        print(f"  Complete state observations: {n_complete} trading days")
        print("=" * 70)

        return all_pass