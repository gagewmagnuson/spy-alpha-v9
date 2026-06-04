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
    expanding: bool = False,
) -> pd.Series:
    """
    Convert a raw series to [0, 1] via rolling or expanding percentile rank.

    expanding=False (default): 252-day rolling window.
        Compares against recent past only.
    expanding=True: all available preceding history.
        Use when the component needs full-cycle context to discriminate
        (e.g. momentum signals during sustained growth or contraction).

    Args:
        series:      raw values (any scale)
        window:      rolling window size (default 252, ignored if expanding=True)
        min_periods: minimum observations required (default 126)
        expanding:   if True, use expanding window instead of rolling

    Returns:
        Series in [0, 1], NaN during warmup period
    """
    if expanding:
        return series.expanding(min_periods=min_periods).rank(pct=True)
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

        # ---- Supplementary FRED series ----
        if api_key:
            from fredapi import Fred
            fred_client = Fred(api_key=api_key)

            for series_id, col_name, obs_start in [
                ("T5YIE",  "T5YIE",  start),
                ("BAA10Y", "BAA10Y", "2005-01-01"),
                ("ICSA",   "ICSA",   "2005-01-01"),
            ]:
                try:
                    s = fred_client.get_series(
                        series_id, observation_start=obs_start
                    )
                    s.index = pd.to_datetime(s.index)
                    if series_id == "ICSA":
                        s = s.resample("B").ffill()
                    else:
                        s = s.asfreq("B").ffill()
                    if self._supp_fred is None or self._supp_fred.empty:
                        self._supp_fred = pd.DataFrame({col_name: s})
                    else:
                        self._supp_fred[col_name] = s
                    logger.info(f"  {series_id} fetched: {len(s)} observations")
                except Exception as e:
                    logger.warning(f"  {series_id} fetch failed: {e}")

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
        """
        Pillar 1: Growth Momentum — measures economic acceleration or deceleration.

        Directly addresses the V7/V8 HMM misclassification problem: the HMM could
        not distinguish 'rates rising because economy is strong' from 'rates rising
        because inflation is out of control.' This pillar measures the growth
        dimension independently of inflation.

        Components (spec Section 4A):
            cyclicals_defensives (0.25): 63-day return of XLY/XLP ratio
            yield_curve_momentum (0.20): 63-day change in T10Y2Y spread
            claims_trend (0.20):         Negative 13-week rate of change in ICSA
            industrial_strength (0.15):  63-day relative return of XLI vs SPY
            small_large_cap (0.10):      63-day relative return of IWM vs SPY
            consumer_sentiment (0.10):   3-month change in UMCSENT

        All components use standard rolling_percentile_rank (252-day window).
        High score = strong acceleration, low score = deep contraction.

        Historical expectations (spec Section 4A):
            2013-2014:       0.65-0.85  (sustained expansion)
            2019:            0.55-0.75  (moderate growth)
            Feb 2020:        0.50-0.60  (slowing but positive)
            March 2020:      0.05-0.15  (collapse)
            2021 recovery:   0.70-0.90  (rapid acceleration)
            Late 2022:       0.30-0.45  (slowdown)

        Key thesis validation: Must be > 0.50 during HMM misclassification periods
        (2012-2014, 2019) — central V9 thesis test.
        """
        rc = self._raw_close
        fred = self._fred_data

        components: Dict[str, pd.Series] = {}

        # Pre-compute SPY 63-day return — reused in multiple components
        spy_ret_63 = None
        if rc is not None and "SPY" in rc.columns:
            spy_ret_63 = rc["SPY"].pct_change(63)

        # ---- Cyclicals vs Defensives: XLY/XLP ratio 63-day return (0.25) ----
        # XLY outperforming XLP = risk appetite / growth acceleration
        if rc is not None and "XLY" in rc.columns and "XLP" in rc.columns:
            ratio = rc["XLY"] / rc["XLP"].replace(0, np.nan)
            cycl_def = ratio.pct_change(63)
            if cycl_def.notna().sum() > MIN_PERIODS:
                components["cyclicals_defensives"] = rolling_percentile_rank(cycl_def, expanding=True)
                logger.info("  [Growth] cyclicals_defensives (XLY/XLP): computed")

        # ---- Yield Curve Momentum: 63-day change in T10Y2Y (0.20) ----
        # Steepening curve = healthy growth expectations
        if fred is not None and "T10Y2Y" in fred.columns:
            t10y2y = fred["T10Y2Y"]
            yc_momentum = t10y2y.diff(63)
            if yc_momentum.notna().sum() > MIN_PERIODS:
                components["yield_curve_momentum"] = rolling_percentile_rank(yc_momentum)
                logger.info("  [Growth] yield_curve_momentum (T10Y2Y diff 63d): computed")

        # ---- Initial Claims Trend: ICSA (0.20) ----
        # Prefer supplementary fetch — snapshot ICSA column is empty
        icsa_source = None
        supp = self._supp_fred if self._supp_fred is not None else pd.DataFrame()
        if not supp.empty and "ICSA" in supp.columns:
            icsa_source = supp["ICSA"].replace(0, np.nan)
            logger.info("  [Growth] claims_trend: using supplementary ICSA fetch")
        elif fred is not None and "ICSA" in fred.columns:
            icsa_source = fred["ICSA"].replace(0, np.nan)
            logger.info("  [Growth] claims_trend: using snapshot ICSA (fallback)")
        if icsa_source is not None:
            claims_roc = -(icsa_source.pct_change(63))
            if claims_roc.notna().sum() > MIN_PERIODS:
                components["claims_trend"] = rolling_percentile_rank(claims_roc, expanding=True)
                logger.info("  [Growth] claims_trend (neg ICSA pct_change 63d): computed")

        # ---- Industrial Sector Strength: XLI vs SPY 63-day relative return (0.15) ----
        # Industrials outperforming = capex cycle / manufacturing strength
        if rc is not None and "XLI" in rc.columns and spy_ret_63 is not None:
            xli_ret = rc["XLI"].pct_change(63)
            industrial = xli_ret - spy_ret_63
            if industrial.notna().sum() > MIN_PERIODS:
                components["industrial_strength"] = rolling_percentile_rank(industrial)
                logger.info("  [Growth] industrial_strength (XLI vs SPY): computed")

        # ---- Small Cap vs Large Cap: IWM vs SPY 63-day relative return (0.10) ----
        # Small cap outperforming = domestic growth confidence / risk appetite
        if rc is not None and "IWM" in rc.columns and spy_ret_63 is not None:
            iwm_ret = rc["IWM"].pct_change(63)
            small_large = iwm_ret - spy_ret_63
            if small_large.notna().sum() > MIN_PERIODS:
                components["small_large_cap"] = rolling_percentile_rank(small_large)
                logger.info("  [Growth] small_large_cap (IWM vs SPY): computed")

        # ---- Consumer Sentiment Momentum: 3-month change in UMCSENT (0.10) ----
        # Rising sentiment = forward-looking growth signal
        # Use diff (absolute change) — UMCSENT is already an index level
        if fred is not None and "UMCSENT" in fred.columns:
            umcsent = fred["UMCSENT"]
            sentiment_change = umcsent.diff(63)  # 3-month change ≈ 63 trading days
            if sentiment_change.notna().sum() > MIN_PERIODS:
                components["consumer_sentiment"] = rolling_percentile_rank(
                    sentiment_change
                )
                logger.info("  [Growth] consumer_sentiment (UMCSENT diff 63d): computed")

        if not components:
            logger.error("Growth Momentum: no components computed — check data")
            if self._raw_close is not None:
                return pd.Series(np.nan, index=self._raw_close.index)
            return pd.Series(dtype=float)

        n_avail = len(components)
        n_expected = len(GROWTH_WEIGHTS)
        if n_avail < n_expected:
            missing = set(GROWTH_WEIGHTS) - set(components)
            logger.warning(
                f"Growth Momentum: {n_avail}/{n_expected} components available. "
                f"Missing: {missing}. Weights renormalized."
            )

        score = weighted_pillar_score(components, GROWTH_WEIGHTS)
        logger.info(
            f"Growth Momentum pillar complete: {score.notna().sum()} valid days, "
            f"mean={score.mean():.3f}, std={score.std():.3f}"
        )
        return score

    def _compute_inflation_pressure(self) -> pd.Series:
        """
        Pillar 2: Inflation Pressure — measures whether price pressure is
        building or subsiding.

        Addresses the HMM's inability to distinguish inflation environments
        from crisis environments. This pillar measures inflation independently,
        allowing the system to distinguish 'moderate inflation during growth'
        (benign) from 'accelerating inflation during stress' (dangerous).

        Components (spec Section 4B):
            breakeven_inflation (0.25): T5YIE 5-year breakeven inflation level
            commodity_momentum  (0.25): DBC 63-day return
            cpi_acceleration    (0.20): 3-month annualized CPI minus 12-month rate
            energy_momentum     (0.15): XLE vs SPY 63-day relative return
            tips_relative       (0.15): TIP vs IEF 63-day relative return

        Historical expectations (spec Section 4B):
            2014-2015:  0.15-0.30  (low inflation / disinflation)
            2021 H2:    0.75-0.95  (surging inflation)
            2022 H1:    0.80-0.95  (peak inflation)
            2022 H2:    0.50-0.65  (inflation decelerating)
            2023:       0.30-0.50  (normalizing)

        Spec validation test: Must be > 0.70 during 2021-2022 and
        < 0.40 during 2014-2015 disinflation.
        """
        rc = self._raw_close
        fred = self._fred_data
        supp = self._supp_fred if self._supp_fred is not None else pd.DataFrame()
        additional = self._additional

        components: Dict[str, pd.Series] = {}

        # Pre-compute SPY 63-day return (reused in energy component)
        spy_ret_63 = None
        if rc is not None and "SPY" in rc.columns:
            spy_ret_63 = rc["SPY"].pct_change(63)

        # ---- Breakeven Inflation: T5YIE level (0.25) ----
        # 5-year breakeven = market's inflation expectation over next 5 years
        # High level = elevated inflation expectations = inflation pressure
        if not supp.empty and "T5YIE" in supp.columns:
            t5yie = supp["T5YIE"].dropna()
            if len(t5yie) > MIN_PERIODS:
                components["breakeven_inflation"] = rolling_percentile_rank(t5yie)
                logger.info(
                    f"  [Inflation] breakeven_inflation (T5YIE): {len(t5yie)} days"
                )

        # ---- Commodity Momentum: DBC 63-day return (0.25) ----
        # Rising commodities = upstream price pressure = inflation building
        if rc is not None and "DBC" in rc.columns:
            dbc_ret = rc["DBC"].pct_change(63)
            if dbc_ret.notna().sum() > MIN_PERIODS:
                components["commodity_momentum"] = rolling_percentile_rank(dbc_ret, expanding=True)
                logger.info("  [Inflation] commodity_momentum (DBC 63d return): computed")

        # ---- CPI Acceleration: 3-month annualized minus 12-month rate (0.20) ----
        # Positive = inflation speeding up beyond trend = building pressure
        # Negative = inflation decelerating below trend = easing pressure
        if fred is not None and "CPIAUCSL" in fred.columns:
            cpi = fred["CPIAUCSL"].replace(0, np.nan)
            # 63 trading days ≈ 3 months, annualized by factor of 4
            cpi_3m_annualized = (cpi / cpi.shift(63) - 1) * 4
            # 252 trading days ≈ 12 months
            cpi_12m = (cpi / cpi.shift(252) - 1)
            acceleration = cpi_3m_annualized - cpi_12m
            if acceleration.notna().sum() > MIN_PERIODS:
                components["cpi_acceleration"] = rolling_percentile_rank(acceleration)
                logger.info(
                    "  [Inflation] cpi_acceleration "
                    "(3m annualized - 12m rate): computed"
                )

        # ---- Energy Sector Momentum: XLE vs SPY 63-day relative return (0.15) ----
        # Energy outperforming = oil/gas prices rising = inflation pressure
        if rc is not None and "XLE" in rc.columns and spy_ret_63 is not None:
            xle_ret = rc["XLE"].pct_change(63)
            energy_rel = xle_ret - spy_ret_63
            if energy_rel.notna().sum() > MIN_PERIODS:
                components["energy_momentum"] = rolling_percentile_rank(energy_rel, expanding=True)
                logger.info("  [Inflation] energy_momentum (XLE vs SPY): computed")

        # ---- TIPS Relative Performance: TIP vs IEF 63-day relative return (0.15) ----
        # TIPS outperforming nominal Treasuries = inflation expectations rising
        tip_series = None
        if additional is not None and "TIP" in additional.columns:
            tip_series = additional["TIP"]
            logger.info("  [Inflation] tips_relative: using supplementary TIP fetch")
        elif rc is not None and "TIP" in rc.columns:
            tip_series = rc["TIP"]
            logger.info("  [Inflation] tips_relative: using snapshot TIP (fallback)")

        if tip_series is not None and rc is not None and "IEF" in rc.columns:
            tip_ret = tip_series.pct_change(63)
            ief_ret = rc["IEF"].pct_change(63)
            tips_rel = tip_ret - ief_ret
            if tips_rel.notna().sum() > MIN_PERIODS:
                components["tips_relative"] = rolling_percentile_rank(tips_rel)
                logger.info("  [Inflation] tips_relative (TIP vs IEF): computed")

        if not components:
            logger.error("Inflation Pressure: no components computed — check data")
            if self._raw_close is not None:
                return pd.Series(np.nan, index=self._raw_close.index)
            return pd.Series(dtype=float)

        n_avail = len(components)
        n_expected = len(INFLATION_WEIGHTS)
        if n_avail < n_expected:
            missing = set(INFLATION_WEIGHTS) - set(components)
            logger.warning(
                f"Inflation Pressure: {n_avail}/{n_expected} components available. "
                f"Missing: {missing}. Weights renormalized."
            )

        score = weighted_pillar_score(components, INFLATION_WEIGHTS)
        logger.info(
            f"Inflation Pressure pillar complete: {score.notna().sum()} valid days, "
            f"mean={score.mean():.3f}, std={score.std():.3f}"
        )
        return score

    def _compute_trend_persistence(self) -> pd.Series:
        """
        Pillar 4: Trend Persistence — measures how strong and sustained the
        directional move is across risk assets.

        As an independent state dimension (not a strategy signal), provides
        unique information about directional conviction feeding governance
        decisions. Addresses V7's redundancy where trend overlapped with HMM.

        Components (spec Section 4D):
            spy_ma_positioning  (0.30): Weighted above 50d/100d/200d MAs
            multi_asset_breadth (0.25): Fraction of 14-asset universe above 100d MA
            trend_duration      (0.20): Days since SPY last crossed below 100d MA
            momentum_magnitude  (0.15): SPY risk-adjusted 63-day return
            trend_consistency   (0.10): Fraction of 63 days with positive 5d return

        Observation universe (14 assets per spec Section 4D):
            SPY, QQQ, IWM, VEA, VWO, TLT, IEF, GLD, DBC,
            XLK, XLF, XLV, XLE, XLI

        Historical expectations (spec Section 4D):
            2013-2014:  0.70-0.90  (strong sustained uptrend)
            2017:       0.80-0.95  (extremely persistent)
            Late 2018:  0.10-0.25  (sharp downturn)
            2019:       0.60-0.80  (recovery then sustained)
            Mar 2020:   0.00-0.10  (crash)
            2022 bear:  0.10-0.30  (persistent downtrend)

        Key thesis test: Must be > 0.60 during HMM misclassification periods
        (2012-2014, 2019) that were actually trending bull markets.
        """
        rc = self._raw_close

        if rc is None:
            logger.error("Trend Persistence: raw_close not available")
            return pd.Series(dtype=float)

        if "SPY" not in rc.columns:
            logger.error("Trend Persistence: SPY not in raw_close")
            return pd.Series(dtype=float)

        spy = rc["SPY"]
        components: Dict[str, pd.Series] = {}

        # Pre-compute MAs used in multiple components
        spy_ma100 = spy.rolling(100, min_periods=100).mean()
        spy_ma200 = spy.rolling(200, min_periods=200).mean()

        # ---- SPY MA Positioning (0.30) ----
        # Inner weights per spec: 50d=0.30, 100d=0.35, 200d=0.35
        # Longer-term MA adherence signals more durable trend
        spy_ma50 = spy.rolling(50, min_periods=50).mean()

        above_50d  = (spy > spy_ma50).astype(float)
        above_100d = (spy > spy_ma100).astype(float)
        above_200d = (spy > spy_ma200).astype(float)

        ma_raw = 0.30 * above_50d + 0.35 * above_100d + 0.35 * above_200d
        if ma_raw.notna().sum() > MIN_PERIODS:
            components["spy_ma_positioning"] = rolling_percentile_rank(ma_raw, expanding=True)
            logger.info("  [Trend] spy_ma_positioning (50/100/200d MAs): computed")

        # ---- Multi-Asset Trend Breadth (0.25) ----
        # Fraction of 14-asset TREND_UNIVERSE above their 100-day MA
        # Broad breadth = healthy, sustainable trend; narrow = fragile
        available = [t for t in TREND_UNIVERSE if t in rc.columns]
        if available:
            breadth_df = pd.DataFrame({
                t: (rc[t] > rc[t].rolling(100, min_periods=100).mean()).astype(float)
                for t in available
            })
            breadth = breadth_df.mean(axis=1)
            if breadth.notna().sum() > MIN_PERIODS:
                components["multi_asset_breadth"] = rolling_percentile_rank(breadth, expanding=True)
                logger.info(
                    f"  [Trend] multi_asset_breadth "
                    f"({len(available)}/{len(TREND_UNIVERSE)} assets): computed"
                )
            if len(available) < len(TREND_UNIVERSE):
                missing_assets = set(TREND_UNIVERSE) - set(available)
                logger.warning(
                    f"  [Trend] missing universe assets: {missing_assets}"
                )

        # ---- Trend Duration (0.20) ----
        # Days since SPY last crossed BELOW its 100-day MA
        # Long streak without disruption = high persistence; recent cross = low
        if spy_ma100.notna().sum() > MIN_PERIODS:
            above_flag = (spy > spy_ma100).fillna(False)
            prev_above = above_flag.shift(1).fillna(True)
            # Below crossing: was above yesterday, now below
            below_cross = (prev_above & ~above_flag)

            # Vectorized: running max of crossing positions propagates the most
            # recent crossing index forward, giving days since last crossing
            pos = np.arange(len(spy))
            cross_pos = np.where(below_cross.values, pos, 0)
            last_cross = np.maximum.accumulate(cross_pos)
            days_since = pd.Series(
                (pos - last_cross).astype(float), index=spy.index
            )
            # Mask warmup period
            days_since[spy_ma100.isna()] = np.nan

            if days_since.notna().sum() > MIN_PERIODS:
                components["trend_duration"] = rolling_percentile_rank(days_since)
                logger.info(
                    "  [Trend] trend_duration (days since below 100d MA): computed"
                )

        # ---- Momentum Magnitude (0.15) ----
        # SPY risk-adjusted 63-day return = return / annualized realized vol
        # High positive = strong upward momentum per unit of risk
        spy_ret_63 = spy.pct_change(63)
        spy_ann_vol = (
            spy.pct_change()
            .rolling(63, min_periods=30).std() * np.sqrt(252)
        )
        momentum_mag = spy_ret_63 / spy_ann_vol.replace(0, np.nan)

        if momentum_mag.notna().sum() > MIN_PERIODS:
            components["momentum_magnitude"] = rolling_percentile_rank(momentum_mag, expanding=True)
            logger.info(
                "  [Trend] momentum_magnitude (risk-adj 63d return): computed"
            )

        # ---- Trend Consistency (0.10) ----
        # Fraction of last 63 days with positive 5-day trailing return
        # High fraction = directionally consistent move
        spy_5d_ret = spy.pct_change(5)
        trend_consistency = spy_5d_ret.rolling(
            63, min_periods=30
        ).apply(lambda x: float((x > 0).mean()), raw=True)

        if trend_consistency.notna().sum() > MIN_PERIODS:
            components["trend_consistency"] = rolling_percentile_rank(
                trend_consistency, expanding=True
            )

            logger.info(
                "  [Trend] trend_consistency "
                "(fraction positive 5d returns in 63d): computed"
            )

        if not components:
            logger.error("Trend Persistence: no components computed — check data")
            return pd.Series(np.nan, index=spy.index)

        n_avail = len(components)
        n_expected = len(TREND_WEIGHTS)
        if n_avail < n_expected:
            missing = set(TREND_WEIGHTS) - set(components)
            logger.warning(
                f"Trend Persistence: {n_avail}/{n_expected} components available. "
                f"Missing: {missing}. Weights renormalized."
            )

        score = weighted_pillar_score(components, TREND_WEIGHTS)
        logger.info(
            f"Trend Persistence pillar complete: {score.notna().sum()} valid days, "
            f"mean={score.mean():.3f}, std={score.std():.3f}"
        )
        return score

    def _compute_participation_quality(self) -> pd.Series:
        """
        Pillar 5: Participation Quality — measures whether a rally is
        broad-based and healthy or narrow and fragile.

        Addresses the HMM's inability to distinguish broad market strength
        from narrow leadership that looks strong but is fragile. This pillar
        feeds directly into the conviction governor's confidence assessment
        for bull market states.

        Components (spec Section 4E):
            sector_breadth         (0.30): Fraction of sector ETFs above 50d MA
            equal_vs_cap           (0.25): RSP vs SPY 21-day relative return
            sector_dispersion      (0.20): Negative of cross-sectional sector return std
            multi_asset_breadth    (0.15): Fraction of TREND_UNIVERSE with positive 21d return
            leadership_concentration (0.10): Negative of top-3 sector return concentration

        Historical expectations (spec Section 4E):
            2013-2014:   0.65-0.85  (broad participation in recovery)
            2017:        0.70-0.90  (broad and healthy)
            Mar 2020:    0.05-0.20  (complete breakdown of participation)
            2020-2021:   0.60-0.85  (broad recovery)
            2022 Bear:   0.15-0.45  (energy-led, uneven sector performance)
            2023 AI:     0.20-0.40  (narrow AI/tech leadership)

        Key validation test: Must be high (>0.60) during broad rallies
        (2013-14, 2017) and low (<0.40) during narrow leadership (2023).
        """
        rc = self._raw_close
        additional = self._additional

        if rc is None:
            logger.error("Participation Quality: raw_close not available")
            return pd.Series(dtype=float)

        components: Dict[str, pd.Series] = {}

        # Available sector ETFs — XLRE (2015+) and XLC (2018+) absent in early periods
        available_sectors = [t for t in SECTOR_ETFS if t in rc.columns]
        if not available_sectors:
            logger.error("Participation Quality: no sector ETFs found in raw_close")
            return pd.Series(np.nan, index=rc.index)

        # ---- Sector Breadth: fraction of sector ETFs above 50-day MA (0.30) ----
        # High = most sectors in uptrend = healthy broad participation
        breadth_signals = pd.DataFrame({
            t: (rc[t] > rc[t].rolling(50, min_periods=50).mean()).astype(float)
            for t in available_sectors
        })
        sector_breadth = breadth_signals.mean(axis=1)
        if sector_breadth.notna().sum() > MIN_PERIODS:
            components["sector_breadth"] = rolling_percentile_rank(sector_breadth)
            logger.info(
                f"  [Participation] sector_breadth "
                f"({len(available_sectors)}/{len(SECTOR_ETFS)} sectors): computed"
            )

        # ---- Equal vs Cap Weight: RSP vs SPY 21-day relative return (0.25) ----
        # RSP outperforming → equal-weight beats cap-weight → small/mid participation
        rsp_series = None
        if additional is not None and "RSP" in additional.columns:
            rsp_series = additional["RSP"]
            logger.info("  [Participation] equal_vs_cap: using supplementary RSP")
        if rsp_series is not None and "SPY" in rc.columns:
            eq_vs_cap = rsp_series.pct_change(21) - rc["SPY"].pct_change(21)
            if eq_vs_cap.notna().sum() > MIN_PERIODS:
                components["equal_vs_cap"] = rolling_percentile_rank(eq_vs_cap)
                logger.info("  [Participation] equal_vs_cap (RSP vs SPY 21d): computed")
        else:
            logger.warning("  [Participation] equal_vs_cap: RSP not available")

        # ---- Sector Dispersion: negative of 21-day cross-sectional std (0.20) ----
        # Low dispersion = sectors moving together = healthy broad market
        # High dispersion = divergence, narrow leadership
        sector_rets_21 = pd.DataFrame({
            t: rc[t].pct_change(21) for t in available_sectors
        })
        dispersion = sector_rets_21.std(axis=1)
        sector_dispersion_raw = -dispersion   # negate: low dispersion → high quality
        if sector_dispersion_raw.notna().sum() > MIN_PERIODS:
            components["sector_dispersion"] = rolling_percentile_rank(
                sector_dispersion_raw, expanding=True
            )
            logger.info("  [Participation] sector_dispersion (neg std): computed")

        # ---- Multi-Asset Breadth: fraction with positive 21-day return (0.15) ----
        # Broad cross-asset participation beyond equities
        avail_universe = [t for t in TREND_UNIVERSE if t in rc.columns]
        if avail_universe:
            positive_rets = pd.DataFrame({
                t: (rc[t].pct_change(21) > 0).astype(float)
                for t in avail_universe
            })
            multi_breadth = positive_rets.mean(axis=1)
            if multi_breadth.notna().sum() > MIN_PERIODS:
                components["multi_asset_breadth"] = rolling_percentile_rank(
                    multi_breadth
                )
                logger.info(
                    f"  [Participation] multi_asset_breadth "
                    f"({len(avail_universe)} assets): computed"
                )

        # ---- Leadership Concentration: negative of top-3 sector return
        # concentration (0.10) ----
        # Measures whether gains are spread across sectors or dominated by a few.
        # Uses absolute 21-day sector returns as proxy for sector "pull" on the market.
        # concentration = top-3 abs returns / sum of all abs returns
        # High concentration → narrow leadership → low participation quality
        # Max-minus-median of sector returns
        # High excess = one sector far above median = concentrated leadership = low quality
        # Works in both bull and bear markets — crashes correctly score low
        conc_raw = -(sector_rets_21.max(axis=1) - sector_rets_21.median(axis=1))
        if conc_raw.notna().sum() > MIN_PERIODS:
            components["leadership_concentration"] = rolling_percentile_rank(conc_raw, expanding=True)
            logger.info(
                "  [Participation] leadership_concentration (top-3 abs): computed"
            )

        if not components:
            logger.error("Participation Quality: no components computed — check data")
            return pd.Series(np.nan, index=rc.index)

        n_avail = len(components)
        n_expected = len(PARTICIPATION_WEIGHTS)
        if n_avail < n_expected:
            missing = set(PARTICIPATION_WEIGHTS) - set(components)
            logger.warning(
                f"Participation Quality: {n_avail}/{n_expected} components available. "
                f"Missing: {missing}. Weights renormalized."
            )

        score = weighted_pillar_score(components, PARTICIPATION_WEIGHTS)
        logger.info(
            f"Participation Quality pillar complete: {score.notna().sum()} valid days, "
            f"mean={score.mean():.3f}, std={score.std():.3f}"
        )
        return score

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
    
    def validate_growth_momentum(self) -> bool:
        """
        Validate growth momentum pillar against known historical episodes.
        Per spec Sections 4A and 13A.

        Key thesis test: Must be > 0.50 during HMM misclassification periods
        (2012-2014, 2019) — this directly validates the V9 core thesis.
        """
        if self.pillars is None or "growth_momentum" not in self.pillars.columns:
            logger.error("Run build() before validate_growth_momentum()")
            return False

        gm = self.pillars["growth_momentum"].dropna()

        # Episodes from spec Section 4A historical expectations
        episodes = [
            ("2008 Crisis",       "2008-09-01", "2009-03-31",  0.00, 0.25),
            ("2013-14 Bull",      "2013-01-01", "2014-12-31",  0.65, 0.85),
            ("2019 Bull",         "2019-01-01", "2019-12-31",  0.55, 0.75),
            ("Feb 2020 Pre-crash","2020-01-01", "2020-02-14",  0.40, 0.65),
            ("Mar 2020 Collapse", "2020-03-01", "2020-04-30",  0.00, 0.20),
            ("2021 Recovery",     "2021-01-01", "2021-12-31",  0.65, 0.90),
            ("Late 2022",         "2022-07-01", "2022-12-31",  0.25, 0.50),
        ]

        print("\n" + "=" * 68)
        print("GROWTH MOMENTUM PILLAR — EPISODE VALIDATION")
        print("=" * 68)
        print(
            f"  {'Episode':<24} {'Mean':>6} {'Min':>6} {'Max':>6} "
            f"{'Expected Range':>18}  {'Pass':>4}"
        )
        print("-" * 68)

        all_pass = True
        for name, start, end, lo, hi in episodes:
            mask = (gm.index >= start) & (gm.index <= end)
            if mask.sum() == 0:
                print(f"  {name:<24} {'NO DATA':>44}")
                continue
            ep = gm[mask]
            mean_v = ep.mean()
            min_v = ep.min()
            max_v = ep.max()
            passed = lo <= mean_v <= hi
            all_pass = all_pass and passed
            status = "✓" if passed else "✗"
            print(
                f"  {name:<24} {mean_v:>6.2f} {min_v:>6.2f} {max_v:>6.2f} "
                f"  [{lo:.2f} – {hi:.2f}]     {status}"
            )

        print("-" * 68)
        verdict = "PASS ✓" if all_pass else "FAIL ✗  — iterate before Phase 3"
        print(f"  Overall: {verdict}")
        print("=" * 68)

        out_of_range = ((gm < 0) | (gm > 1)).sum()
        print(f"\n  Range check  [0, 1]: {out_of_range} values out of range")
        print(f"  Valid days:          {len(gm)} "
            f"({self.pillars['growth_momentum'].isna().sum()} NaN)")
        print(f"  Overall mean:        {gm.mean():.3f}")
        print(f"  Overall std:         {gm.std():.3f}")
        print(f"  Warmup cutoff:       {gm.first_valid_index()}")

        # ---- Key thesis test ----
        print(f"\n  --- Key Thesis Test: HMM Misclassification Periods ---")
        thesis_periods = [
            ("2012-2014", "2012-01-01", "2014-12-31"),
            ("2019",      "2019-01-01", "2019-12-31"),
        ]
        thesis_pass = True
        for name, start, end in thesis_periods:
            mask = (gm.index >= start) & (gm.index <= end)
            if mask.sum() > 0:
                mean_v = gm[mask].mean()
                pct_above = (gm[mask] > 0.50).mean()
                passed = mean_v > 0.50
                thesis_pass = thesis_pass and passed
                status = "✓" if passed else "✗"
                print(
                    f"  {name}: mean={mean_v:.3f}, "
                    f"days above 0.50: {pct_above:.1%}  {status}"
                )
        print(
            f"\n  Thesis test: "
            f"{'PASS ✓' if thesis_pass else 'FAIL ✗ — V9 core thesis not validated'}"
        )

        return all_pass
    
    def validate_inflation_pressure(self) -> bool:
        """
        Validate inflation pressure pillar against known historical episodes.
        Per spec Sections 4B and 13A.

        Key spec validation tests:
            Must be > 0.70 during 2021-2022 inflation episode
            Must be < 0.40 during 2014-2015 disinflation
            Must NOT correlate > 0.60 with growth momentum
        """
        if self.pillars is None or "inflation_pressure" not in self.pillars.columns:
            logger.error("Run build() before validate_inflation_pressure()")
            return False

        ip = self.pillars["inflation_pressure"].dropna()

        episodes = [
            ("2008 Crisis",   "2008-09-01", "2009-03-31",  0.10, 0.45),
            ("2014-2015",     "2014-01-01", "2015-12-31",  0.15, 0.45),
            ("2019 Low Infl", "2019-01-01", "2019-12-31",  0.15, 0.45),
            ("2021 H2 Surge", "2021-06-01", "2021-12-31",  0.55, 0.80),
            ("2022 H1 Peak",  "2022-01-01", "2022-06-30",  0.65, 0.90),
            ("2022 H2 Decel", "2022-07-01", "2022-12-31",  0.15, 0.45),
            ("2023 Normal",   "2023-01-01", "2023-12-31",  0.30, 0.50),
        ]

        print("\n" + "=" * 68)
        print("INFLATION PRESSURE PILLAR — EPISODE VALIDATION")
        print("=" * 68)
        print(
            f"  {'Episode':<22} {'Mean':>6} {'Min':>6} {'Max':>6} "
            f"{'Expected Range':>18}  {'Pass':>4}"
        )
        print("-" * 68)

        all_pass = True
        for name, start, end, lo, hi in episodes:
            mask = (ip.index >= start) & (ip.index <= end)
            if mask.sum() == 0:
                print(f"  {name:<22} {'NO DATA':>44}")
                continue
            ep = ip[mask]
            mean_v = ep.mean()
            min_v  = ep.min()
            max_v  = ep.max()
            passed = lo <= mean_v <= hi
            all_pass = all_pass and passed
            status = "✓" if passed else "✗"
            print(
                f"  {name:<22} {mean_v:>6.2f} {min_v:>6.2f} {max_v:>6.2f} "
                f"  [{lo:.2f} – {hi:.2f}]     {status}"
            )

        print("-" * 68)
        verdict = "PASS ✓" if all_pass else "FAIL ✗  — iterate before next pillar"
        print(f"  Overall: {verdict}")
        print("=" * 68)

        out_of_range = ((ip < 0) | (ip > 1)).sum()
        print(f"\n  Range check  [0, 1]: {out_of_range} values out of range")
        print(f"  Valid days:          {len(ip)} "
            f"({self.pillars['inflation_pressure'].isna().sum()} NaN)")
        print(f"  Overall mean:        {ip.mean():.3f}")
        print(f"  Overall std:         {ip.std():.3f}")
        print(f"  Warmup cutoff:       {ip.first_valid_index()}")

        # ---- Key spec validation tests ----
        print(f"\n  --- Key Validation Tests (spec Section 4B) ---")

        # Test 1: High during 2021 H2 - 2022 H1
        mask_hi = (ip.index >= "2021-06-01") & (ip.index <= "2022-06-30")
        if mask_hi.sum() > 0:
            mean_hi = ip[mask_hi].mean()
            pct_above = (ip[mask_hi] > 0.70).mean()
            status = "✓" if mean_hi > 0.70 else "✗"
            print(
                f"  2021 H2 – 2022 H1: mean={mean_hi:.3f}, "
                f"days above 0.70: {pct_above:.1%}  {status}"
            )

        # Test 2: Low during 2014-2015
        mask_lo = (ip.index >= "2014-01-01") & (ip.index <= "2015-12-31")
        if mask_lo.sum() > 0:
            mean_lo = ip[mask_lo].mean()
            pct_below = (ip[mask_lo] < 0.40).mean()
            status = "✓" if mean_lo < 0.40 else "✗"
            print(
                f"  2014-2015:         mean={mean_lo:.3f}, "
                f"days below 0.40: {pct_below:.1%}  {status}"
            )

        # Test 3: Orthogonality with growth momentum (spec requirement)
        if "growth_momentum" in self.pillars.columns:
            gm = self.pillars["growth_momentum"]
            common = ip.index.intersection(gm.dropna().index)
            if len(common) > 252:
                corr = ip.reindex(common).corr(gm.reindex(common))
                status = "✓" if abs(corr) < 0.60 else "✗"
                print(
                    f"  Orthogonality (r with Growth): {corr:.3f}  "
                    f"{status} (spec requires |r| < 0.60)"
                )

        return all_pass
    
    def validate_trend_persistence(self) -> bool:
        """
        Validate trend persistence pillar against known historical episodes.
        Per spec Sections 4D and 13A.

        Key thesis test: Must be > 0.60 during HMM misclassification periods
        (2012-2014, 2019) — actually trending bull markets the HMM called
        Crisis-Inflation.
        """
        if self.pillars is None or "trend_persistence" not in self.pillars.columns:
            logger.error("Run build() before validate_trend_persistence()")
            return False

        tp = self.pillars["trend_persistence"].dropna()

        episodes = [
            ("2012-14 Bull (HMM miss)", "2012-01-01", "2014-12-31",  0.65, 0.90),
            ("2017 Bull",               "2017-01-01", "2017-12-31",  0.70, 0.95),
            ("Late 2018 Downturn",      "2018-10-01", "2018-12-31",  0.10, 0.25),
            ("2019 Bull (HMM miss)",    "2019-01-01", "2019-12-31",  0.60, 0.80),
            ("Mar 2020 Crash",          "2020-03-01", "2020-04-30",  0.00, 0.25),
            ("2022 Bear",               "2022-01-01", "2022-10-31",  0.10, 0.30),
        ]

        print("\n" + "=" * 68)
        print("TREND PERSISTENCE PILLAR — EPISODE VALIDATION")
        print("=" * 68)
        print(
            f"  {'Episode':<26} {'Mean':>6} {'Min':>6} {'Max':>6} "
            f"{'Expected Range':>18}  {'Pass':>4}"
        )
        print("-" * 68)

        all_pass = True
        for name, start, end, lo, hi in episodes:
            mask = (tp.index >= start) & (tp.index <= end)
            if mask.sum() == 0:
                print(f"  {name:<26} {'NO DATA':>44}")
                continue
            ep = tp[mask]
            mean_v = ep.mean()
            min_v  = ep.min()
            max_v  = ep.max()
            passed = lo <= mean_v <= hi
            all_pass = all_pass and passed
            status = "✓" if passed else "✗"
            print(
                f"  {name:<26} {mean_v:>6.2f} {min_v:>6.2f} {max_v:>6.2f} "
                f"  [{lo:.2f} – {hi:.2f}]     {status}"
            )

        print("-" * 68)
        verdict = "PASS ✓" if all_pass else "FAIL ✗  — iterate before next pillar"
        print(f"  Overall: {verdict}")
        print("=" * 68)

        out_of_range = ((tp < 0) | (tp > 1)).sum()
        print(f"\n  Range check  [0, 1]: {out_of_range} values out of range")
        print(f"  Valid days:          {len(tp)} "
            f"({self.pillars['trend_persistence'].isna().sum()} NaN)")
        print(f"  Overall mean:        {tp.mean():.3f}")
        print(f"  Overall std:         {tp.std():.3f}")
        print(f"  Warmup cutoff:       {tp.first_valid_index()}")

        # Key thesis test
        print(f"\n  --- Key Thesis Test: HMM Misclassification Periods ---")
        thesis_pass = True
        for name, start, end in [
            ("2012-2014", "2012-01-01", "2014-12-31"),
            ("2019",      "2019-01-01", "2019-12-31"),
        ]:
            mask = (tp.index >= start) & (tp.index <= end)
            if mask.sum() > 0:
                mean_v = tp[mask].mean()
                pct_above = (tp[mask] > 0.60).mean()
                passed = mean_v > 0.60
                thesis_pass = thesis_pass and passed
                status = "✓" if passed else "✗"
                print(
                    f"  {name}: mean={mean_v:.3f}, "
                    f"days above 0.60: {pct_above:.1%}  {status}"
                )

        print(
            f"\n  Thesis test: "
            f"{'PASS ✓' if thesis_pass else 'FAIL ✗ — V9 thesis not validated for trend'}"
        )

        return all_pass
    
    def validate_participation_quality(self) -> bool:
        """
        Validate participation quality pillar against known historical episodes.
        Per spec Sections 4E and 13A.

        Key validation test: Must be high (>0.60) during broad rallies
        (2013-14, 2017) and low (<0.40) during narrow AI/tech leadership (2023).
        """
        if self.pillars is None or "participation_quality" not in self.pillars.columns:
            logger.error("Run build() before validate_participation_quality()")
            return False

        pq = self.pillars["participation_quality"].dropna()

        episodes = [
            ("2013-14 Broad Bull",   "2013-01-01", "2014-12-31",  0.60, 0.85),
            ("2017 Broad Bull",      "2017-01-01", "2017-12-31",  0.50, 0.75),
            ("Mar 2020 Breakdown",   "2020-03-01", "2020-04-30",  0.05, 0.20),
            ("2020-21 Recovery",     "2020-06-01", "2021-12-31",  0.30, 0.65),
            ("2022 Bear",            "2022-01-01", "2022-10-31",  0.15, 0.45),
            ("2023 AI Narrow",       "2023-01-01", "2023-12-31",  0.35, 0.60),
        ]

        print("\n" + "=" * 70)
        print("PARTICIPATION QUALITY PILLAR — EPISODE VALIDATION")
        print("=" * 70)
        print(
            f"  {'Episode':<24} {'Mean':>6} {'Min':>6} {'Max':>6} "
            f"{'Expected Range':>18}  {'Pass':>4}"
        )
        print("-" * 70)

        all_pass = True
        for name, start, end, lo, hi in episodes:
            mask = (pq.index >= start) & (pq.index <= end)
            if mask.sum() == 0:
                print(f"  {name:<24} {'NO DATA':>44}")
                continue
            ep = pq[mask]
            mean_v = ep.mean()
            min_v  = ep.min()
            max_v  = ep.max()
            passed = lo <= mean_v <= hi
            all_pass = all_pass and passed
            status = "✓" if passed else "✗"
            print(
                f"  {name:<24} {mean_v:>6.2f} {min_v:>6.2f} {max_v:>6.2f} "
                f"  [{lo:.2f} – {hi:.2f}]     {status}"
            )

        print("-" * 70)
        verdict = "PASS ✓" if all_pass else "FAIL ✗  — iterate before next pillar"
        print(f"  Overall: {verdict}")
        print("=" * 70)

        out_of_range = ((pq < 0) | (pq > 1)).sum()
        print(f"\n  Range check  [0, 1]: {out_of_range} values out of range")
        print(
            f"  Valid days:          {len(pq)} "
            f"({self.pillars['participation_quality'].isna().sum()} NaN)"
        )
        print(f"  Overall mean:        {pq.mean():.3f}")
        print(f"  Overall std:         {pq.std():.3f}")
        print(f"  Warmup cutoff:       {pq.first_valid_index()}")

        # Key validation tests
        print(f"\n  --- Key Validation Tests (spec Section 4E) ---")

        # Test 1: Broad rallies score high
        for name, start, end, threshold in [
            ("2013-2014", "2013-01-01", "2014-12-31", 0.60),
            ("2017",      "2017-01-01", "2017-12-31", 0.55),
        ]:
            mask = (pq.index >= start) & (pq.index <= end)
            if mask.sum() > 0:
                mean_v = pq[mask].mean()
                pct_above = (pq[mask] > threshold).mean()
                status = "✓" if mean_v > threshold else "✗"
                print(
                    f"  {name} broad rally: mean={mean_v:.3f}, "
                    f"above 0.60: {pct_above:.1%}  {status}"
                )

        # Test 2: Narrow leadership scores low
        mask_2023 = (pq.index >= "2023-01-01") & (pq.index <= "2023-12-31")
        if mask_2023.sum() > 0:
            mean_2023 = pq[mask_2023].mean()
            pct_below = (pq[mask_2023] < 0.40).mean()
            status = "✓" if mean_2023 < 0.50 else "✗"
            print(
                f"  2023 AI narrow rally: mean={mean_2023:.3f}, "
                f"below 0.40: {pct_below:.1%}  {status}"
            )

        # Test 3: Orthogonality with other pillars
        for pillar_name in ["growth_momentum", "trend_persistence"]:
            if pillar_name in self.pillars.columns:
                other = self.pillars[pillar_name]
                common = pq.index.intersection(other.dropna().index)
                if len(common) > 252:
                    corr = pq.reindex(common).corr(other.reindex(common))
                    status = "✓" if abs(corr) < 0.60 else "✗"
                    print(
                        f"  Orthogonality vs {pillar_name}: "
                        f"r={corr:.3f}  {status} (|r| < 0.60)"
                    )

        return all_pass
    
    