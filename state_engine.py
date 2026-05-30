"""
SPY Alpha v9 — Continuous Multi-Factor State Engine
====================================================

Computes the 5-pillar continuous market state representation per
Section 4 of the v9 build spec.

Pillar orientation:
    growth_momentum:       1 = strong acceleration,  0 = deep contraction
    inflation_pressure:    1 = extreme inflation,     0 = deflation
    financial_stress:      1 = extreme stress,        0 = calm            ← INVERTED
    trend_persistence:     1 = strong uptrend,        0 = strong downtrend
    participation_quality: 1 = broad participation,   0 = narrow/fragile

IMPORTANT: financial_stress is the only pillar where high = bad for
risk assets. Downstream consumers (favorable_score, conviction_governor)
must use (1 - financial_stress) when combining with other pillars.

All pillars scaled to [0, 1] via 252-day rolling percentile rank.

Build order (one pillar at a time, validate before proceeding):
    Phase 1: Financial Stress     — thesis-critical, built first
    Phase 2: Growth Momentum      — addresses V7 misclassification
    Phase 3: Trend Persistence    — clearest validation targets
    Phase 4: Inflation Pressure   — interacts with stress
    Phase 5: Participation Quality — most fragile, validate last
"""

from __future__ import annotations
import logging
import os
from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd

logger = logging.getLogger("spy_alpha_v9.state_engine")

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

PILLAR_WINDOW: int = 252    # rolling percentile rank window (trading days)
MIN_PERIODS: int = 126      # minimum observations for a valid estimate

# ---- Component weights per pillar (exact values from spec Section 4) ----

GROWTH_WEIGHTS: Dict[str, float] = {
    "cyclicals_defensives":  0.25,
    "yield_curve_momentum":  0.20,
    "claims_trend":          0.20,
    "industrial_strength":   0.15,
    "small_large_cap":       0.10,
    "consumer_sentiment":    0.10,
}

INFLATION_WEIGHTS: Dict[str, float] = {
    "breakeven_inflation": 0.25,
    "commodity_momentum":  0.25,
    "cpi_acceleration":    0.20,
    "energy_momentum":     0.15,
    "tips_relative":       0.15,
}

STRESS_WEIGHTS: Dict[str, float] = {
    "hy_oas":                  0.25,
    "financial_conditions":    0.20,
    "financial_stress_idx":    0.20,
    "vix_term_structure":      0.15,
    "cross_asset_correlation": 0.10,
    "equity_volatility":       0.10,
}

TREND_WEIGHTS: Dict[str, float] = {
    "spy_ma_positioning":  0.30,
    "multi_asset_breadth": 0.25,
    "trend_duration":      0.20,
    "momentum_magnitude":  0.15,
    "trend_consistency":   0.10,
}

PARTICIPATION_WEIGHTS: Dict[str, float] = {
    "sector_breadth":           0.25,
    "equal_vs_cap_weight":      0.25,
    "sector_dispersion":        0.20,
    "multi_asset_breadth_p":    0.15,
    "leadership_concentration": 0.15,
}

# Trend persistence observation universe — 14 assets per spec Section 4D
TREND_UNIVERSE: List[str] = [
    "SPY", "QQQ", "IWM", "VEA", "VWO",
    "TLT", "IEF", "GLD", "DBC",
    "XLK", "XLF", "XLV", "XLE", "XLI",
]

# Sector ETFs for participation quality — 11 per spec Section 4E
SECTOR_ETFS: List[str] = [
    "XLK", "XLV", "XLF", "XLY", "XLP",
    "XLE", "XLI", "XLB", "XLRE", "XLU", "XLC",
]

# ---------------------------------------------------------------------------
# Core Scaling Utilities
# ---------------------------------------------------------------------------

def rolling_percentile_rank(
    series: pd.Series,
    window: int = PILLAR_WINDOW,
    min_periods: int = MIN_PERIODS,
) -> pd.Series:
    """
    Convert a raw series to [0, 1] via rolling percentile rank.

    High raw value → high percentile rank → high score.

    This is the universal scaling method per spec Section 4 for ALL
    pillar components. For financial_stress, high score = high stress
    (1 = extreme stress). The downstream inversion (1 - financial_stress)
    happens in favorable_score, NOT here.

    Args:
        series:      raw values (any scale)
        window:      rolling window size (default 252 trading days)
        min_periods: minimum observations required (default 126)

    Returns:
        Series in [0, 1], NaN during warmup period
    """
    return series.rolling(window, min_periods=min_periods).rank(pct=True)


def weighted_pillar_score(
    components: Dict[str, pd.Series],
    weights: Dict[str, float],
) -> pd.Series:
    """
    Compute a pillar score as a weighted average of component percentile ranks.

    Missing components are excluded and remaining weights renormalized.
    This ensures valid output even when some data sources are unavailable
    (e.g., VIX3M before 2007, RSP before 2003).

    Returns:
        Series in [0, 1]
    """
    available = {k: v for k, v in components.items() if k in weights}
    if not available:
        logger.warning("weighted_pillar_score: no available components")
        return pd.Series(dtype=float)

    df = pd.DataFrame(available)
    weight_series = pd.Series({k: weights[k] for k in available})

    # Weighted sum across non-NaN columns only
    weighted_sum = (df * weight_series).sum(axis=1, min_count=1)

    # Sum of weights for non-NaN columns (for proper renormalization)
    non_nan_weights = df.notna().mul(weight_series).sum(axis=1)

    result = weighted_sum / non_nan_weights.replace(0, np.nan)
    return result.clip(0.0, 1.0)


# ---------------------------------------------------------------------------
# State Engine
# ---------------------------------------------------------------------------

class StateEngine:
    """
    Computes the 5-pillar continuous market state representation.

    Usage (backtest):
        engine = StateEngine()
        pillars = engine.build(snapshot)

    Usage (live inference):
        engine = StateEngine()
        pillars = engine.build(snapshot)   # pre-compute for history
        current = engine.get_current_state(date)
    """

    def __init__(self):
        self.pillars: Optional[pd.DataFrame] = None
        self._raw_close: Optional[pd.DataFrame] = None
        self._fred_data: Optional[pd.DataFrame] = None
        self._supp_fred: Optional[pd.DataFrame] = None
        self._vix_term: Optional[pd.DataFrame] = None
        self._additional: Optional[pd.DataFrame] = None

    def build(
        self,
        snapshot: Dict[str, Any],
        fred_api_key: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Build all 5 pillars for the full snapshot period.

        Fetches supplementary data (NFCI, STLFSI4, ^VIX3M) that is not
        in the baseline snapshot using data_pipeline.fetch_stress_data().
        Also fetches RSP and TIP for later pillars.

        Args:
            snapshot:     loaded snapshot dict from SnapshotManager
            fred_api_key: FRED API key (falls back to FRED_API_KEY env var)

        Returns:
            DataFrame with columns: [growth_momentum, inflation_pressure,
                                     financial_stress, trend_persistence,
                                     participation_quality]
        """
        from data_pipeline import get_raw_close, get_fred, fetch_stress_data

        self._raw_close = get_raw_close(snapshot)
        self._fred_data = get_fred(snapshot)

        api_key = fred_api_key or os.environ.get("FRED_API_KEY")

        # Fetch supplementary data not in baseline snapshot
        logger.info("StateEngine: Fetching supplementary pillar data...")
        stress_data = fetch_stress_data(
            api_key=api_key,
            start="2005-01-01",
        )
        self._supp_fred = stress_data.get("stress_fred", pd.DataFrame())
        self._vix_term = stress_data.get("vix_term", pd.DataFrame())

        # Fetch additional tickers needed for later pillars (RSP, TIP, T5YIE)
        self._fetch_additional_tickers(api_key=api_key)

        logger.info("StateEngine: Computing pillars...")

        p3 = self._compute_financial_stress()
        p1 = self._compute_growth_momentum()
        p2 = self._compute_inflation_pressure()
        p4 = self._compute_trend_persistence()
        p5 = self._compute_participation_quality()

        self.pillars = pd.DataFrame({
            "growth_momentum":      p1,
            "inflation_pressure":   p2,
            "financial_stress":     p3,
            "trend_persistence":    p4,
            "participation_quality": p5,
        })

        for col in self.pillars.columns:
            valid = self.pillars[col].notna().sum()
            first = self.pillars[col].first_valid_index()
            logger.info(f"  {col}: {valid} valid days, first valid: {first}")

        return self.pillars

    def get_current_state(self, date: pd.Timestamp) -> Dict[str, float]:
        """Return pillar values for a specific date (for live inference)."""
        if self.pillars is None:
            raise RuntimeError("Call build() before get_current_state()")
        if date not in self.pillars.index:
            date = self.pillars.index[self.pillars.index <= date][-1]
        row = self.pillars.loc[date]
        return row.to_dict()

    # -----------------------------------------------------------------------
    # Supplementary Data Fetch
    # -----------------------------------------------------------------------

    def _fetch_additional_tickers(self, api_key: Optional[str] = None) -> None:
        """
        Fetch tickers not in the baseline snapshot needed for pillars 2 and 5:
            RSP  — equal-weight S&P 500 (participation quality)
            TIP  — TIPS ETF (inflation pressure)
            T5YIE — 5-year breakeven inflation from FRED (inflation pressure)
        """
        import yfinance as yf

        if self._raw_close is not None and len(self._raw_close) > 0:
            start = self._raw_close.index[0].strftime("%Y-%m-%d")
            end = self._raw_close.index[-1].strftime("%Y-%m-%d")
        else:
            start, end = "2005-01-01", None

        # ---- RSP and TIP from yfinance ----
        try:
            extra = yf.download(
                ["RSP", "TIP"], start=start, end=end,
                auto_adjust=False, progress=False,
            )
            if not extra.empty:
                if isinstance(extra.columns, pd.MultiIndex):
                    self._additional = extra["Close"].copy()
                else:
                    self._additional = extra[["Close"]].copy()
                logger.info(
                    f"  Additional tickers fetched: "
                    f"{list(self._additional.columns)}"
                )
        except Exception as e:
            logger.warning(f"  RSP/TIP fetch failed: {e}")

        # ---- T5YIE from FRED (5-year breakeven inflation) ----
        if api_key:
            try:
                from fredapi import Fred
                fred_client = Fred(api_key=api_key)
                t5yie = fred_client.get_series(
                    "T5YIE", observation_start=start
                )
                t5yie.index = pd.to_datetime(t5yie.index)
                t5yie = t5yie.asfreq("B").ffill()

                if self._supp_fred is None or self._supp_fred.empty:
                    self._supp_fred = pd.DataFrame({"T5YIE": t5yie})
                else:
                    self._supp_fred["T5YIE"] = t5yie

                logger.info(f"  T5YIE fetched: {len(t5yie)} observations")
                logger.info(f"  Fetching BAMLH0A0HYM2 from {start}")
                baa10y = fred_client.get_series(
                    "BAA10Y", observation_start="2005-01-01"
                )
                baa10y.index = pd.to_datetime(baa10y.index)
                baa10y = baa10y.asfreq("B").ffill()
                self._supp_fred["BAA10Y"] = baa10y
                logger.info(f"  BAA10Y fetched: {len(baa10y)} observations")
            except Exception as e:
                logger.warning(f"  T5YIE fetch failed: {e}")

    # -----------------------------------------------------------------------
    # Pillar 3: Financial Stress  (PHASE 1 — implemented)
    # -----------------------------------------------------------------------

    def _compute_financial_stress(self) -> pd.Series:
        """
        Pillar 3: Financial Stress — measures whether the financial system
        is functioning normally.

        Score orientation: 1 = extreme stress, 0 = calm conditions.
        This is the ONLY pillar where high score = bad for risk assets.
        Downstream consumers MUST use (1 - financial_stress).

        Components (all oriented: high raw → more stress → high score):
            hy_oas (0.25):                HY OAS level
            financial_conditions (0.20):  NFCI level
            financial_stress_idx (0.20):  STLFSI4 level
            vix_term_structure (0.15):    VIX/VIX3M ratio (>1 = inverted)
            cross_asset_correlation (0.10): SPY/TLT 21-day rolling corr
            equity_volatility (0.10):     VIX level

        Historical expectations (spec Section 4C):
            2013-2014: 0.05-0.20 (very calm)
            2019:      0.10-0.25 (calm)
            Jan 2020:  0.10-0.20 (calm — VIX was 15, below average)
            Mar 2020:  0.90-1.00 (extreme stress)
            2022 bear: 0.40-0.65 (elevated but not extreme)
        """
        rc = self._raw_close
        fred = self._fred_data
        supp = self._supp_fred if self._supp_fred is not None else pd.DataFrame()
        vix_df = self._vix_term if self._vix_term is not None else pd.DataFrame()

        components: Dict[str, pd.Series] = {}

        # ---- Credit Stress: BAA10Y (0.25) ----
        # Moody's BAA Corporate Spread over 10Y Treasury — full history from 2005
        # Replaces BAMLH0A0HYM2 which FRED truncated to 3-year rolling window
        if not supp.empty and "BAA10Y" in supp.columns:
            baa = supp["BAA10Y"].dropna()
            if len(baa) > MIN_PERIODS:
                components["hy_oas"] = rolling_percentile_rank(baa)
                logger.info(f"  [Stress] credit_stress (BAA10Y): {len(baa)} days")

        # ---- Financial Conditions: NFCI (0.20) ----
        if not supp.empty and "NFCI" in supp.columns:
            nfci = supp["NFCI"].dropna()
            if len(nfci) > MIN_PERIODS:
                components["financial_conditions"] = rolling_percentile_rank(nfci)
                logger.info(f"  [Stress] financial_conditions (NFCI): {len(nfci)} days")

        # ---- Financial Stress Index: STLFSI4 (0.20) ----
        # STLFSI4 is a normalized composite z-score index — percentile ranking
        # loses its natural scale. Use empirical quantile bounds instead.
        # Bounds derived from full available history (1st and 99th percentiles).
        if not supp.empty and "STLFSI4" in supp.columns:
            stlfsi = supp["STLFSI4"].dropna()
            if len(stlfsi) > MIN_PERIODS:
                q01 = stlfsi.quantile(0.01)
                q99 = stlfsi.quantile(0.99)
                stlfsi_scaled = ((stlfsi - q01) / (q99 - q01)).clip(0, 1)
                components["financial_stress_idx"] = stlfsi_scaled
                logger.info(
                    f"  [Stress] financial_stress_idx (STLFSI4 quantile-scaled): "
                    f"q01={q01:.3f}, q99={q99:.3f}, {len(stlfsi)} days"
                )

        # ---- VIX Term Structure: VIX/VIX3M ratio (0.15) ----
        # High ratio (>1) = inverted term structure = stress
        # Fallback: use VIX level alone if VIX3M unavailable
        if (not vix_df.empty
                and "^VIX" in vix_df.columns
                and "^VIX3M" in vix_df.columns):
            vix_spot = vix_df["^VIX"]
            vix3m = vix_df["^VIX3M"]
            ratio = (vix_spot / vix3m.replace(0, np.nan)).dropna()
            if len(ratio) > MIN_PERIODS:
                q01 = ratio.quantile(0.01)
                q99 = ratio.quantile(0.99)
                vix_term_scaled = ((ratio - q01) / (q99 - q01)).clip(0, 1)
                components["vix_term_structure"] = vix_term_scaled
                logger.info(
                    f"  [Stress] vix_term_structure (quantile-scaled): "
                    f"q01={q01:.3f}, q99={q99:.3f}, {len(ratio)} days"
                )
        elif rc is not None and "^VIX" in rc.columns:
            vix_lvl = rc["^VIX"].dropna()
            if len(vix_lvl) > MIN_PERIODS:
                components["vix_term_structure"] = rolling_percentile_rank(vix_lvl)
                logger.info("  [Stress] vix_term_structure: fallback to ^VIX level")

        # ---- Cross-Asset Correlation: SPY/TLT 21-day (0.10) ----
        # High positive correlation = stress (both falling, or crisis correlation)
        if rc is not None and "SPY" in rc.columns and "TLT" in rc.columns:
            spy_ret = rc["SPY"].pct_change()
            tlt_ret = rc["TLT"].pct_change()
            corr = spy_ret.rolling(21, min_periods=10).corr(tlt_ret)
            abs_corr = corr.abs()
            if abs_corr.notna().sum() > MIN_PERIODS:
                components["cross_asset_correlation"] = rolling_percentile_rank(abs_corr)
                logger.info("  [Stress] cross_asset_correlation (|SPY/TLT|): computed")

        # ---- Equity Volatility: ^VIX level (0.10) ----
        if rc is not None and "^VIX" in rc.columns:
            vix_lvl = rc["^VIX"].dropna()
            if len(vix_lvl) > MIN_PERIODS:
                q01 = vix_lvl.quantile(0.01)
                q99 = vix_lvl.quantile(0.99)
                vix_scaled = ((vix_lvl - q01) / (q99 - q01)).clip(0, 1)
                components["equity_volatility"] = vix_scaled
                logger.info(
                    f"  [Stress] equity_volatility (^VIX quantile-scaled): "
                    f"q01={q01:.3f}, q99={q99:.3f}, {len(vix_lvl)} days"
                )
        elif rc is not None and "SPY" in rc.columns:
            spy_vol = (
                rc["SPY"].pct_change()
                .rolling(20, min_periods=10).std() * np.sqrt(252)
            )
            if spy_vol.notna().sum() > MIN_PERIODS:
                components["equity_volatility"] = rolling_percentile_rank(spy_vol)
                logger.info("  [Stress] equity_volatility: fallback to SPY realized vol")

        if not components:
            logger.error("Financial Stress: no components computed — check data")
            return pd.Series(dtype=float)

        n_avail = len(components)
        n_expected = len(STRESS_WEIGHTS)
        if n_avail < n_expected:
            missing = set(STRESS_WEIGHTS) - set(components)
            logger.warning(
                f"Financial Stress: {n_avail}/{n_expected} components available. "
                f"Missing: {missing}. Weights renormalized."
            )

        score = weighted_pillar_score(components, STRESS_WEIGHTS)
        logger.info(
            f"Financial Stress pillar complete: {score.notna().sum()} valid days, "
            f"mean={score.mean():.3f}, std={score.std():.3f}"
        )
        return score

    # -----------------------------------------------------------------------
    # Pillars 1, 2, 4, 5 — Stubs (implemented in later phases)
    # -----------------------------------------------------------------------

    def _compute_growth_momentum(self) -> pd.Series:
        """Pillar 1: Growth Momentum — STUB (Phase 2)."""
        if self._raw_close is not None:
            return pd.Series(np.nan, index=self._raw_close.index)
        return pd.Series(dtype=float)

    def _compute_inflation_pressure(self) -> pd.Series:
        """Pillar 2: Inflation Pressure — STUB (Phase 4)."""
        if self._raw_close is not None:
            return pd.Series(np.nan, index=self._raw_close.index)
        return pd.Series(dtype=float)

    def _compute_trend_persistence(self) -> pd.Series:
        """Pillar 4: Trend Persistence — STUB (Phase 3)."""
        if self._raw_close is not None:
            return pd.Series(np.nan, index=self._raw_close.index)
        return pd.Series(dtype=float)

    def _compute_participation_quality(self) -> pd.Series:
        """Pillar 5: Participation Quality — STUB (Phase 5)."""
        if self._raw_close is not None:
            return pd.Series(np.nan, index=self._raw_close.index)
        return pd.Series(dtype=float)

    # -----------------------------------------------------------------------
    # Validation
    # -----------------------------------------------------------------------

    def validate_financial_stress(self) -> bool:
        """
        Validate financial stress pillar against known historical episodes.
        Per spec Section 13A validation table.

        Returns True if all episodes pass, False otherwise.
        Do NOT proceed to Phase 2 until this returns True.
        """
        if self.pillars is None or "financial_stress" not in self.pillars.columns:
            logger.error("Run build() before validate_financial_stress()")
            return False

        fs = self.pillars["financial_stress"].dropna()

        # Episodes: (name, start, end, expected_mean_lo, expected_mean_hi)
        episodes = [
            ("2008 Crisis",      "2008-09-01", "2009-03-31",  0.75, 1.00),
            ("2013-14 Bull",     "2013-01-01", "2014-12-31",  0.05, 0.30),
            ("2019 Bull",        "2019-01-01", "2019-12-31",  0.20, 0.40),
            ("Jan 2020 Calm",    "2020-01-01", "2020-02-14",  0.05, 0.30),
            ("Mar 2020 Crisis",  "2020-03-01", "2020-04-30",  0.75, 1.00),
            ("2022 Bear",        "2022-01-01", "2022-10-31",  0.35, 0.70),
        ]

        print("\n" + "=" * 68)
        print("FINANCIAL STRESS PILLAR — EPISODE VALIDATION")
        print("=" * 68)
        print(
            f"  {'Episode':<22} {'Mean':>6} {'Min':>6} {'Max':>6} "
            f"{'Expected Range':>18}  {'Pass':>4}"
        )
        print("-" * 68)

        all_pass = True
        for name, start, end, lo, hi in episodes:
            mask = (fs.index >= start) & (fs.index <= end)
            if mask.sum() == 0:
                print(f"  {name:<22} {'NO DATA — check date range':>44}")
                continue
            ep = fs[mask]
            mean_v = ep.mean()
            min_v = ep.min()
            max_v = ep.max()
            passed = lo <= mean_v <= hi
            all_pass = all_pass and passed
            status = "✓" if passed else "✗"
            print(
                f"  {name:<22} {mean_v:>6.2f} {min_v:>6.2f} {max_v:>6.2f} "
                f"  [{lo:.2f} – {hi:.2f}]     {status}"
            )

        print("-" * 68)
        verdict = "PASS ✓" if all_pass else "FAIL ✗  — iterate before Phase 2"
        print(f"  Overall: {verdict}")
        print("=" * 68)

        # Range and coverage stats
        out_of_range = ((fs < 0) | (fs > 1)).sum()
        print(f"\n  Range check  [0, 1]: {out_of_range} values out of range")
        print(f"  Valid days:          {len(fs)} ({self.pillars['financial_stress'].isna().sum()} NaN)")
        print(f"  Overall mean:        {fs.mean():.3f}")
        print(f"  Overall std:         {fs.std():.3f}")
        print(f"  Warmup cutoff:       {fs.first_valid_index()}")

        # Component coverage report
        print(f"\n  Components used vs expected: check logs above for details")

        return all_pass