"""
SPY Alpha V9 — Step 4: Analog Memory
analog_memory.py — Phase 4A: ETF-era (2005-present) only.

For each trading day, finds K historically similar states using strict
walk-forward KNN retrieval and computes three outputs:

    coherence         [0,1]  Tight neighbor cluster → well-defined state
    reliability       [0,1]  Consistent forward outcomes → predictable state
    transition_hazard [0,1]  Analogs preceded stress spikes → elevated risk

These three signals feed the Conviction Governor (Step 6), which uses
them to scale confidence in regime-based allocation decisions.

Architecture decisions (per spec Section 5 + other AI review):
    - Walk-forward retrieval: day t searches only days in [0, t-purge-1]
    - Full pairwise distance matrix (scipy cdist) for efficiency
    - Calibrated hazard threshold from empirical stress-change distribution
    - Reliability: directional agreement + IQR-based magnitude consistency
      (avoids CV instability near zero means flagged in architecture review)
    - Pre-ETF proxy construction deferred to Phase 4C after 4B validates

Phase 4B validation: coherence and reliability must show monotonic
relationships with forward return distributions. This is the first
true proof-of-thesis test for the entire V9 architecture.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist

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

TRAJECTORY_TYPES: List[str] = [
    "velocity",
    "persistence",
    "divergence",
    "stability",
]

DERIVED_SUMMARY_NAMES: List[str] = [
    "risk_appetite",
    "market_quality",
    "transition_instability",
]

# 28-dimensional state vector column order (spec Section 5B)
STATE_VECTOR_COLUMNS: List[str] = (
    PILLAR_NAMES
    + [f"{p}_{t}" for p in PILLAR_NAMES for t in TRAJECTORY_TYPES]
    + DERIVED_SUMMARY_NAMES
)

# Dimension weights for KNN distance (spec Section 5B)
_W: Dict[str, float] = {}
for _p in PILLAR_NAMES:
    _W[_p] = 1.0
for _p in PILLAR_NAMES:
    _W[f"{_p}_velocity"]    = 0.5
    _W[f"{_p}_persistence"] = 0.3
    _W[f"{_p}_divergence"]  = 0.5
    _W[f"{_p}_stability"]   = 0.3
for _s in DERIVED_SUMMARY_NAMES:
    _W[_s] = 0.2

DIMENSION_WEIGHTS: np.ndarray = np.array(
    [_W[c] for c in STATE_VECTOR_COLUMNS], dtype=np.float64
)

# KNN / retrieval
DEFAULT_K:              int   = 30
DEFAULT_PURGE:          int   = 21    # days to exclude around query date
HAZARD_PERCENTILE:      float = 75.0  # percentile for hazard threshold calibration

# Rolling window for percentile-ranking analog scores
RANK_WINDOW:       int = 252
RANK_MIN_PERIODS:  int = 63


# ------------------------------------------------------------------ #
# Local utility                                                        #
# ------------------------------------------------------------------ #

def _pct_rank(
    series: pd.Series,
    window: int = RANK_WINDOW,
    min_periods: int = RANK_MIN_PERIODS,
) -> pd.Series:
    """
    Rolling [0,1] percentile rank.
    Defined locally to avoid circular import from state_engine.py.
    """
    return series.rolling(window, min_periods=min_periods).rank(pct=True)


# ------------------------------------------------------------------ #
# AnalogMemory                                                         #
# ------------------------------------------------------------------ #

class AnalogMemory:
    """
    Step 4: Analog Memory — Phase 4A (ETF-era, 2005-present).

    Finds K historically similar market states for each trading day
    using strict walk-forward KNN retrieval and computes coherence,
    reliability, and transition hazard scores.

    Usage:
        memory = AnalogMemory(k=30, purge_window=21)
        analog_scores = memory.build(
            state_vector = engine.state_vector,   # 25-dim from StateEngine
            pillars      = engine.pillars,         # 5-col pillar DataFrame
            raw_close    = engine._raw_close,      # for SPY forward returns
        )
        memory.validate(engine.pillars, engine._raw_close)
    """

    def __init__(
        self,
        k:                  int   = DEFAULT_K,
        purge_window:       int   = DEFAULT_PURGE,
        hazard_percentile:  float = HAZARD_PERCENTILE,
    ) -> None:
        self.k                 = k
        self.purge_window      = purge_window
        self.hazard_percentile = hazard_percentile

        # Populated by build()
        self.analog_scores:       Optional[pd.DataFrame] = None
        self._full_sv:            Optional[pd.DataFrame] = None
        self._knn_indices:        Optional[Dict[int, np.ndarray]] = None
        self._forward_outcomes:   Optional[pd.DataFrame] = None
        self._hazard_threshold:   Optional[float] = None
        self._distance_matrix:    Optional[np.ndarray] = None
        self._valid_index:        Optional[pd.DatetimeIndex] = None

    # ---------------------------------------------------------------- #
    # Public: build                                                     #
    # ---------------------------------------------------------------- #

    def build(
        self,
        state_vector: pd.DataFrame,
        pillars:      pd.DataFrame,
        raw_close:    pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Build analog memory for all valid ETF-era trading days.

        Args:
            state_vector : 25-dim DataFrame (5 pillars + 20 trajectory features)
                           from StateEngine.state_vector
            pillars      : 5-col pillar DataFrame from StateEngine.pillars
            raw_close    : raw price DataFrame — must contain 'SPY'

        Returns:
            DataFrame[coherence, reliability, transition_hazard]
            indexed by trading day, values rolling-percentile-ranked to [0,1].
            Reindexed to match the full pillars index (NaN during warmup).
        """
        logger.info("=" * 62)
        logger.info("AnalogMemory.build() — Phase 4A (ETF-era only)")
        logger.info("=" * 62)

        # ---- 1. Derived state summaries ----
        logger.info("Step 1/8: Computing derived state summaries...")
        derived = self._compute_derived_summaries(pillars, state_vector)

        # ---- 2. Assemble 28-dim full state vector ----
        logger.info("Step 2/8: Assembling 28-dim full state vector...")
        full_sv = self._assemble_full_state_vector(state_vector, derived)
        sv_complete = full_sv.dropna()
        self._full_sv    = full_sv
        self._valid_index = sv_complete.index
        N = len(sv_complete)
        logger.info(
            f"  28-dim state vector: {N} complete rows "
            f"({N / len(pillars):.1%} of full sample)"
        )

        # ---- 3. Forward outcomes ----
        logger.info("Step 3/8: Computing forward outcomes (21-day SPY return, stress change)...")
        outcomes = self._compute_forward_outcomes(pillars, raw_close)
        self._forward_outcomes = outcomes

        # ---- 4. Calibrate hazard threshold ----
        logger.info("Step 4/8: Calibrating hazard threshold...")
        self._hazard_threshold = self._calibrate_hazard_threshold(pillars)

        # ---- 5. Build pairwise distance matrix ----
        logger.info("Step 5/8: Building pairwise distance matrix...")
        W = sv_complete.values * np.sqrt(DIMENSION_WEIGHTS)
        mem_mb = N * N * 8 / 1e6
        logger.info(f"  cdist({N}×{N}, 28-dim weighted) — ~{mem_mb:.0f} MB")
        self._distance_matrix = cdist(W, W, metric="euclidean")
        logger.info("  Distance matrix complete.")

        # ---- 6. Walk-forward KNN ----
        logger.info("Step 6/8: Walk-forward KNN retrieval...")
        self._knn_indices = self._run_walk_forward_knn(N)
        n_valid_queries = len(self._knn_indices)
        logger.info(
            f"  {n_valid_queries}/{N} query days have >= {self.k} valid neighbors."
        )

        # ---- 7. Compute raw analog scores ----
        logger.info("Step 7/8: Computing coherence, reliability, hazard...")
        raw_scores = self._compute_all_scores(sv_complete, outcomes)

        # ---- 8. Percentile-rank to [0, 1] ----
        # coherence_raw = mean pairwise distance (low = tightly clustered = good)
        # → negate before ranking so HIGH coherence score = tight cluster
        logger.info("Step 8/8: Percentile-ranking analog scores...")
        self.analog_scores = pd.DataFrame({
            "coherence":          _pct_rank(-raw_scores["coherence_raw"]),
            "reliability":        _pct_rank(raw_scores["reliability_raw"]),
            "transition_hazard":  _pct_rank(raw_scores["hazard_raw"]),
        }, index=raw_scores.index).reindex(pillars.index)

        n_scored = int(self.analog_scores.notna().all(axis=1).sum())
        logger.info(
            f"AnalogMemory complete: {n_scored} days with full analog scores "
            f"({n_scored / len(pillars):.1%} of sample)."
        )
        return self.analog_scores

    # ---------------------------------------------------------------- #
    # Private: construction                                             #
    # ---------------------------------------------------------------- #

    def _compute_derived_summaries(
        self,
        pillars:      pd.DataFrame,
        state_vector: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Three derived summaries added to the state vector before KNN search.
        Per spec Section 4G.

        risk_appetite          : blends growth, trend, (1-stress)
        market_quality         : blends participation, (1-stress), trend, growth
        transition_instability : velocity magnitude weighted by inverse stability
                                 (high value = multiple pillars changing fast
                                  with unreliable readings)
        """
        gm = pillars["growth_momentum"]
        fs = pillars["financial_stress"]
        tp = pillars["trend_persistence"]
        pq = pillars["participation_quality"]

        risk_appetite = (
            0.35 * gm
            + 0.30 * tp
            + 0.35 * (1.0 - fs)
        ).clip(0.0, 1.0)

        market_quality = (
            0.30 * pq
            + 0.30 * (1.0 - fs)
            + 0.20 * tp
            + 0.20 * gm
        ).clip(0.0, 1.0)

        # transition_instability:
        # Weighted mean of |velocity_i - 0.5| where weight_i = 1 / stability_i
        # velocity is percentile-ranked with 0.5 = stable → |vel - 0.5| = magnitude
        # low stability → high weight → unstable pillar amplifies the instability signal
        eps = 1e-4
        vel_cols = [f"{p}_velocity" for p in PILLAR_NAMES if f"{p}_velocity" in state_vector.columns]
        sta_cols = [f"{p}_stability" for p in PILLAR_NAMES if f"{p}_stability" in state_vector.columns]

        if vel_cols and sta_cols:
            abs_vel = (state_vector[vel_cols] - 0.5).abs()
            abs_vel.columns = [c.replace("_velocity", "") for c in vel_cols]

            inv_sta = 1.0 / (state_vector[sta_cols] + eps)
            inv_sta.columns = [c.replace("_stability", "") for c in sta_cols]

            common = abs_vel.columns.intersection(inv_sta.columns)
            numerator   = (abs_vel[common] * inv_sta[common]).sum(axis=1, min_count=1)
            denominator = inv_sta[common].sum(axis=1, min_count=1)
            ti_raw = (numerator / denominator.replace(0, np.nan))
            transition_instability = _pct_rank(ti_raw)
        else:
            logger.warning(
                "  Derived: velocity/stability columns not found in state_vector. "
                "transition_instability set to NaN."
            )
            transition_instability = pd.Series(np.nan, index=pillars.index)

        derived = pd.DataFrame({
            "risk_appetite":          risk_appetite,
            "market_quality":         market_quality,
            "transition_instability": transition_instability,
        }, index=pillars.index)

        logger.info(
            f"  risk_appetite μ={risk_appetite.mean():.3f}  "
            f"market_quality μ={market_quality.mean():.3f}  "
            f"transition_instability μ={transition_instability.dropna().mean():.3f}"
        )
        return derived

    def _assemble_full_state_vector(
        self,
        state_vector: pd.DataFrame,
        derived:      pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Concatenate 25-dim state_vector + 3 derived summaries → 28-dim.
        Column order follows STATE_VECTOR_COLUMNS (canonical, consistent
        with spec Section 5B).
        """
        full = pd.concat([state_vector, derived], axis=1)
        available = [c for c in STATE_VECTOR_COLUMNS if c in full.columns]
        missing   = [c for c in STATE_VECTOR_COLUMNS if c not in full.columns]
        if missing:
            logger.warning(f"  Full state vector missing columns: {missing}")
        return full[available]

    def _compute_forward_outcomes(
        self,
        pillars:   pd.DataFrame,
        raw_close: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Forward 21-day outcomes for reliability and hazard computation.

        forward_21d_spy_return    : SPY[t+21] / SPY[t] - 1
        forward_21d_stress_change : financial_stress[t+21] - financial_stress[t]

        Last 21 rows will have NaN (no future data). These rows remain valid
        analogs (their state vectors are used for earlier queries) but cannot
        contribute to outcome-based scores.
        """
        if "SPY" in raw_close.columns:
            spy = raw_close["SPY"].dropna()
            fwd_spy = (spy.shift(-21) / spy - 1.0).reindex(pillars.index)
        else:
            logger.warning("  'SPY' not in raw_close — forward returns set to NaN.")
            fwd_spy = pd.Series(np.nan, index=pillars.index)

        if "financial_stress" in pillars.columns:
            stress = pillars["financial_stress"]
            fwd_stress = (stress.shift(-21) - stress).reindex(pillars.index)
        else:
            logger.warning("  'financial_stress' not in pillars — hazard set to NaN.")
            fwd_stress = pd.Series(np.nan, index=pillars.index)

        outcomes = pd.DataFrame({
            "forward_21d_spy_return":    fwd_spy,
            "forward_21d_stress_change": fwd_stress,
        }, index=pillars.index)

        logger.info(
            f"  {outcomes['forward_21d_spy_return'].notna().sum()} SPY forward returns, "
            f"{outcomes['forward_21d_stress_change'].notna().sum()} stress changes computed."
        )
        return outcomes

    def _calibrate_hazard_threshold(self, pillars: pd.DataFrame) -> float:
        """
        Set transition hazard threshold at the {hazard_percentile}th percentile
        of POSITIVE 21-day financial_stress increases.

        Per other AI recommendation: the original 0.15 was arbitrary and
        needed empirical calibration. 75th percentile of positive increases
        ensures the threshold represents a genuinely significant stress spike,
        not routine noise.
        """
        stress = pillars["financial_stress"].dropna()
        chg_21d = (stress.shift(-21) - stress).dropna()
        positive = chg_21d[chg_21d > 0]

        if len(positive) < 20:
            threshold = 0.15
            logger.warning(
                f"  Only {len(positive)} positive stress changes — "
                "using fallback threshold 0.15."
            )
        else:
            threshold = float(positive.quantile(self.hazard_percentile / 100.0))
            logger.info(
                f"  Hazard threshold: {threshold:.4f} "
                f"({self.hazard_percentile:.0f}th pctile of "
                f"{len(positive)} positive 21d stress increases)"
            )
        return threshold

    # ---------------------------------------------------------------- #
    # Private: KNN retrieval                                            #
    # ---------------------------------------------------------------- #

    def _run_walk_forward_knn(self, N: int) -> Dict[int, np.ndarray]:
        """
        Strict walk-forward KNN for all N query dates.

        For query at position i (date t):
            Candidate set = positions [0, i - purge_window - 1]

        This guarantees:
          1. No future states are ever used as analogs (walk-forward)
          2. The most recent purge_window days are excluded (prevents
             autocorrelation contamination from very recent neighbors)

        Uses np.argpartition for O(N) top-K selection (faster than argsort
        when K << N).

        Returns:
            Dict mapping query position i → array of K neighbor positions
            sorted by ascending distance. Positions index into sv_complete.
        """
        D = self._distance_matrix
        knn: Dict[int, np.ndarray] = {}
        min_needed = self.k + self.purge_window

        for i in range(N):
            max_cand = i - self.purge_window - 1  # last valid candidate (inclusive)
            if max_cand < self.k - 1:
                continue  # not enough history yet

            row = D[i, :max_cand + 1]  # distances to all valid candidates

            # O(N) partial sort: get K smallest indices, then sort them
            part = np.argpartition(row, self.k - 1)[:self.k]
            knn[i] = part[np.argsort(row[part])]  # sorted K neighbors

        logger.info(
            f"  First valid query: position {min_needed} "
            f"({min_needed} days of history required)."
        )
        return knn

    # ---------------------------------------------------------------- #
    # Private: analog score computation                                 #
    # ---------------------------------------------------------------- #

    def _compute_all_scores(
        self,
        sv_complete: pd.DataFrame,
        outcomes:    pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Single-pass computation of coherence, reliability, and hazard
        for all valid query positions.

        Pre-aligns outcome arrays to sv_complete to enable fast numpy indexing
        inside the query loop.
        """
        D         = self._distance_matrix
        threshold = self._hazard_threshold

        # Pre-align outcomes to sv_complete index for efficient positional indexing
        fwd_spy_arr = (
            outcomes["forward_21d_spy_return"]
            .reindex(sv_complete.index)
            .values.astype(np.float64)
        )
        fwd_str_arr = (
            outcomes["forward_21d_stress_change"]
            .reindex(sv_complete.index)
            .values.astype(np.float64)
        )

        coherence_d:   Dict = {}
        reliability_d: Dict = {}
        hazard_d:      Dict = {}

        for i, date in enumerate(sv_complete.index):
            if i not in self._knn_indices:
                continue

            nbr = self._knn_indices[i]  # K neighbor positions in sv_complete

            coherence_d[date]   = self._coherence(D, nbr)
            reliability_d[date] = self._reliability(fwd_spy_arr, nbr)
            hazard_d[date]      = self._hazard(fwd_str_arr, nbr, threshold)

        raw = pd.DataFrame({
            "coherence_raw":   pd.Series(coherence_d),
            "reliability_raw": pd.Series(reliability_d),
            "hazard_raw":      pd.Series(hazard_d),
        })

        logger.info(
            f"  Raw scores — "
            f"coherence: {raw['coherence_raw'].mean():.4f} ± {raw['coherence_raw'].std():.4f} | "
            f"reliability: {raw['reliability_raw'].mean():.4f} ± {raw['reliability_raw'].std():.4f} | "
            f"hazard: {raw['hazard_raw'].mean():.4f} ± {raw['hazard_raw'].std():.4f}"
        )
        return raw

    def _coherence(self, D: np.ndarray, nbr: np.ndarray) -> float:
        """
        Mean pairwise distance among K neighbors in the weighted state space.

        Low value → neighbors are tightly clustered → state is well-defined.
        High value → neighbors are scattered → ambiguous / transitional state.

        Negated in build() before ranking so HIGH coherence score = tight cluster.
        """
        k = len(nbr)
        if k < 2:
            return np.nan
        sub = D[np.ix_(nbr, nbr)]
        upper = sub[np.triu_indices(k, k=1)]
        return float(upper.mean())

    def _reliability(self, fwd_returns: np.ndarray, nbr: np.ndarray) -> float:
        """
        Reliability of forward outcomes for the K neighbors.

        Revised formulation (avoids CV instability near zero means):

        Component 1 — Directional agreement [0, 1]:
            How strongly do K neighbors agree on return direction?
            max(positive_frac, negative_frac) in [0.5, 1.0] → rescaled to [0, 1].
            0 = 50/50 split (random), 1 = unanimous agreement.

        Component 2 — IQR-based magnitude consistency [0, 1]:
            IQR / (median absolute return + epsilon) measures spread relative
            to magnitude robustly. Transformed with exp(-iqr_cv) so that
            low spread → high consistency.

        Final: 0.5 * directional + 0.5 * magnitude
        """
        rets = fwd_returns[nbr]
        valid = rets[~np.isnan(rets)]
        if len(valid) < 3:
            return np.nan

        # Component 1: directional agreement
        pos_frac = float((valid > 0).mean())
        # Rescale: max(p, 1-p) in [0.5, 1.0] → (max - 0.5) * 2 in [0, 1]
        directional = (max(pos_frac, 1.0 - pos_frac) - 0.5) * 2.0

        # Component 2: IQR-based magnitude consistency
        iqr    = float(np.percentile(valid, 75) - np.percentile(valid, 25))
        med_ab = float(np.median(np.abs(valid))) + 1e-6
        iqr_cv = iqr / med_ab
        # exp(-iqr_cv): 0 spread → 1.0, moderate spread → 0.37, high spread → near 0
        # Cap iqr_cv at 3 to prevent numerical instability
        magnitude = float(np.exp(-min(iqr_cv, 3.0)))

        return 0.5 * directional + 0.5 * magnitude

    def _hazard(
        self,
        fwd_stress_changes: np.ndarray,
        nbr:                np.ndarray,
        threshold:          float,
    ) -> float:
        """
        Fraction of K neighbors that preceded a significant stress spike
        (21-day financial_stress increase > calibrated threshold).

        High hazard → similar historical states often preceded stress surges.
        """
        changes = fwd_stress_changes[nbr]
        valid   = changes[~np.isnan(changes)]
        if len(valid) < 3:
            return np.nan
        return float((valid > threshold).mean())

    # ---------------------------------------------------------------- #
    # Public: Phase 4B monotonicity validation                         #
    # ---------------------------------------------------------------- #

    def validate(
        self,
        pillars:   pd.DataFrame,
        raw_close: pd.DataFrame,
    ) -> bool:
        """
        Phase 4B: Monotonicity validation — the first true proof-of-thesis
        test for the V9 architecture.

        If analog memory adds genuine predictive value, then:
          1. High coherence days → tighter forward return distributions
          2. High reliability days → tighter forward return distributions
          3. High hazard days → higher actual stress spike rate

        Bucket each score into deciles and verify the expected monotonic
        relationship holds from bottom to top.

        Per other AI: if these tests fail, V9 becomes a more complicated V8.
        If they pass, the core thesis (market states repeat usefully) is validated.
        """
        if self.analog_scores is None or self._forward_outcomes is None:
            logger.error("validate(): call build() first.")
            return False

        sc   = self.analog_scores
        outs = self._forward_outcomes

        # Common valid index: need analog scores AND outcomes
        common = (
            sc.dropna().index
            .intersection(outs["forward_21d_spy_return"].dropna().index)
            .intersection(outs["forward_21d_stress_change"].dropna().index)
        )

        if len(common) < 200:
            logger.error(f"validate(): only {len(common)} common obs — too few.")
            return False

        sc_c   = sc.loc[common]
        spy_c  = outs.loc[common, "forward_21d_spy_return"]
        str_c  = outs.loc[common, "forward_21d_stress_change"]
        N_DECILES = 10
        all_pass = True

        print("\n" + "=" * 72)
        print("ANALOG MEMORY — PHASE 4B MONOTONICITY VALIDATION")
        print("=" * 72)
        print(f"  Common observations: {len(common)}")
        print(f"  K={self.k}, purge={self.purge_window}d, "
              f"hazard threshold={self._hazard_threshold:.4f}")

        # ---------------------------------------------------------------- #
        # Test 1: Coherence → forward return distribution tightness        #
        # High coherence should predict tighter (lower std) forward returns #
        # ---------------------------------------------------------------- #
        print("\n  ── Test 1: Coherence ──────────────────────────────────────────")
        print("  Expected: forward return std DECREASES as coherence increases")
        print(f"  {'Decile':>7} {'Coh Mean':>10} {'Fwd Ret Std':>13} {'Fwd Ret Mean':>14} {'N':>6}")
        print("  " + "─" * 54)

        try:
            coh_bins = pd.qcut(
                sc_c["coherence"], N_DECILES, labels=False, duplicates="drop"
            )
        except Exception as e:
            logger.warning(f"  Coherence decile binning failed: {e}")
            coh_bins = None

        coh_stds: List[float] = []
        if coh_bins is not None:
            for d in sorted(coh_bins.dropna().unique()):
                idx = coh_bins[coh_bins == d].index
                if len(idx) < 20:
                    continue
                c_mean = sc_c.loc[idx, "coherence"].mean()
                r_std  = spy_c.loc[idx].std()
                r_mean = spy_c.loc[idx].mean()
                coh_stds.append(r_std)
                print(
                    f"  {int(d)+1:>7} {c_mean:>10.3f} "
                    f"{r_std:>13.4f} {r_mean:>14.4f} {len(idx):>6}"
                )

        coh_ok = False
        if len(coh_stds) >= 4:
            low3  = np.mean(coh_stds[:3])
            high3 = np.mean(coh_stds[-3:])
            coh_ok = high3 < low3
            status = "✓" if coh_ok else "✗"
            pct_imp = (low3 - high3) / low3 * 100
            print(
                f"\n  Return std — bottom-3 deciles: {low3:.4f} | "
                f"top-3 deciles: {high3:.4f} | "
                f"improvement: {pct_imp:.1f}%  {status}"
            )
        else:
            print("  ✗ Insufficient deciles for monotonicity assessment")
        all_pass = all_pass and coh_ok

        # ---------------------------------------------------------------- #
        # Test 2: Reliability → forward return distribution tightness      #
        # High reliability should also predict tighter forward returns       #
        # ---------------------------------------------------------------- #
        print("\n  ── Test 2: Reliability ────────────────────────────────────────")
        print("  Expected: forward return std DECREASES as reliability increases")
        print(f"  {'Decile':>7} {'Rel Mean':>10} {'Fwd Ret Std':>13} {'Fwd Ret Mean':>14} {'N':>6}")
        print("  " + "─" * 54)

        try:
            rel_bins = pd.qcut(
                sc_c["reliability"], N_DECILES, labels=False, duplicates="drop"
            )
        except Exception as e:
            logger.warning(f"  Reliability decile binning failed: {e}")
            rel_bins = None

        rel_stds: List[float] = []
        if rel_bins is not None:
            for d in sorted(rel_bins.dropna().unique()):
                idx = rel_bins[rel_bins == d].index
                if len(idx) < 20:
                    continue
                r_mean_s = sc_c.loc[idx, "reliability"].mean()
                r_std    = spy_c.loc[idx].std()
                r_mean   = spy_c.loc[idx].mean()
                rel_stds.append(r_std)
                print(
                    f"  {int(d)+1:>7} {r_mean_s:>10.3f} "
                    f"{r_std:>13.4f} {r_mean:>14.4f} {len(idx):>6}"
                )

        rel_ok = False
        if len(rel_stds) >= 4:
            low3  = np.mean(rel_stds[:3])
            high3 = np.mean(rel_stds[-3:])
            rel_ok = high3 < low3
            status = "✓" if rel_ok else "✗"
            pct_imp = (low3 - high3) / low3 * 100
            print(
                f"\n  Return std — bottom-3 deciles: {low3:.4f} | "
                f"top-3 deciles: {high3:.4f} | "
                f"improvement: {pct_imp:.1f}%  {status}"
            )
        else:
            print("  ✗ Insufficient deciles for monotonicity assessment")
        all_pass = all_pass and rel_ok

        # ---------------------------------------------------------------- #
        # Test 3: Transition hazard → actual stress spike rate              #
        # High hazard should predict higher actual stress spikes             #
        # ---------------------------------------------------------------- #
        print("\n  ── Test 3: Transition Hazard ──────────────────────────────────")
        print("  Expected: actual stress spike rate INCREASES as hazard increases")
        print(f"  {'Decile':>7} {'Haz Mean':>10} {'Spike Rate':>12} {'N':>6}")
        print("  " + "─" * 38)

        try:
            haz_bins = pd.qcut(
                sc_c["transition_hazard"], N_DECILES, labels=False, duplicates="drop"
            )
        except Exception as e:
            logger.warning(f"  Hazard decile binning failed: {e}")
            haz_bins = None

        haz_rates: List[float] = []
        if haz_bins is not None:
            for d in sorted(haz_bins.dropna().unique()):
                idx = haz_bins[haz_bins == d].index
                if len(idx) < 20:
                    continue
                h_mean  = sc_c.loc[idx, "transition_hazard"].mean()
                sp_rate = (str_c.loc[idx] > self._hazard_threshold).mean()
                haz_rates.append(sp_rate)
                print(
                    f"  {int(d)+1:>7} {h_mean:>10.3f} "
                    f"{sp_rate:>12.3f} {len(idx):>6}"
                )

        haz_ok = False
        if len(haz_rates) >= 4:
            low3  = np.mean(haz_rates[:3])
            high3 = np.mean(haz_rates[-3:])
            haz_ok = high3 > low3
            status = "✓" if haz_ok else "✗"
            print(
                f"\n  Spike rate — bottom-3: {low3:.3f} | "
                f"top-3: {high3:.3f}  {status}"
            )
        else:
            print("  ✗ Insufficient deciles for assessment")
        all_pass = all_pass and haz_ok

        # ---------------------------------------------------------------- #
        # Test 4: Coverage                                                  #
        # ---------------------------------------------------------------- #
        print("\n  ── Test 4: Coverage (min 50 obs per decile) ───────────────────")
        for name, bins in [
            ("coherence",         coh_bins),
            ("reliability",       rel_bins),
            ("transition_hazard", haz_bins),
        ]:
            if bins is None:
                print(f"  {name:<22}: binning failed  ✗")
                all_pass = False
                continue
            min_ct = int(bins.value_counts().min())
            cov_ok = min_ct >= 50
            all_pass = all_pass and cov_ok
            status = "✓" if cov_ok else "✗"
            print(f"  {name:<22}: min decile count = {min_ct}  {status}")

        # ---- Summary ----
        print("\n  " + "=" * 68)
        verdict = (
            "PASS ✓  — analog memory adds predictive value"
            if all_pass else
            "FAIL ✗  — review monotonicity failures above"
        )
        print(f"  Overall: {verdict}")
        if all_pass:
            print(
                "\n  Phase 4B complete. Analog memory validated."
                "\n  Proceed to Phase 4C (pre-ETF proxy construction) "
                "or Step 5 (Strategy Allocator)."
            )
        else:
            print(
                "\n  Monotonicity failure indicates the analog memory is not"
                "\n  adding genuine predictive value. Review coherence/reliability"
                "\n  formulations before proceeding."
            )
        print("=" * 72)

        return all_pass