"""
SPY Alpha V9 — Combined Strategy Validation
validate_combined.py
=============================================
Builds all three strategies plus conviction governor and compares:
    S1 standalone (excess_return baseline)
    S1+S2+S3 equal weight (33/33/33)
    S1+S2+S3 conviction-governed
    SPY buy-and-hold

Answers: did adding S2, S3, and the Conviction Governor improve
on the S1 standalone baseline of Sharpe 1.298 / CAGR 30.04%?
"""

from __future__ import annotations

import logging
import time
import numpy as np
import pandas as pd
from typing import Dict, List, Optional

logging.basicConfig(level=logging.WARNING)

FRED_API_KEY  = "55a0b587c09627fe956afaf6cb6d2bf7"
SNAPSHOT_NAME = "baseline_v7"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def compute_metrics(returns: pd.Series, label: str) -> Dict:
    r = returns.dropna()
    r = r[r != 0]
    if len(r) < 126:
        return {"label": label, "sharpe": np.nan, "cagr": np.nan,
                "max_dd": np.nan, "calmar": np.nan, "sortino": np.nan}
    n_yrs   = len(r) / 252
    cum     = (1 + r).cumprod()
    cagr    = float(cum.iloc[-1] ** (1 / n_yrs) - 1)
    sharpe  = float(r.mean() / (r.std() + 1e-10) * np.sqrt(252))
    peak    = cum.expanding().max()
    max_dd  = float((cum / peak - 1).min())
    calmar  = float(cagr / abs(max_dd)) if abs(max_dd) > 1e-6 else np.nan
    down    = r[r < 0]
    sortino = float(cagr / (down.std() * np.sqrt(252) + 1e-10)) \
              if len(down) > 0 else np.nan
    return {
        "label":   label,
        "sharpe":  round(sharpe,  3),
        "cagr":    round(cagr,    4),
        "max_dd":  round(max_dd,  4),
        "calmar":  round(calmar,  3) if not np.isnan(calmar)  else np.nan,
        "sortino": round(sortino, 3) if not np.isnan(sortino) else np.nan,
    }


def print_table(results: List[Dict]) -> None:
    print(f"\n  {'Strategy':<40} {'Sharpe':>7} {'CAGR':>8} "
          f"{'Max DD':>8} {'Calmar':>8} {'Sortino':>8}")
    print("  " + "─" * 76)
    for r in results:
        def f(v, w=8, pct=False):
            if isinstance(v, float) and np.isnan(v):
                return f"{'nan':>{w}}"
            return f"{v*100:{w}.2f}%" if pct else f"{v:{w}.3f}"
        print(f"  {r['label']:<40} "
              f"{f(r['sharpe'], 7)} "
              f"{f(r['cagr'], 7, True)} "
              f"{f(r['max_dd'], 7, True)} "
              f"{f(r['calmar'], 8)} "
              f"{f(r['sortino'], 8)}")


def simulate_daily_returns(
    portfolios:  Dict[pd.Timestamp, Dict[str, float]],
    adj_close:   pd.DataFrame,
) -> pd.Series:
    """Forward-fill rebalance weights and compute daily portfolio returns."""
    daily_ret   = adj_close.pct_change()
    refit_dates = sorted(portfolios.keys())
    if not refit_dates:
        return pd.Series(dtype=float)

    results  = []
    current  = {}
    ridx     = 0

    for date in daily_ret.index:
        while ridx < len(refit_dates) and refit_dates[ridx] <= date:
            current = portfolios[refit_dates[ridx]]
            ridx   += 1
        if not current or date < refit_dates[0]:
            results.append(np.nan)
            continue
        pr = sum(
            w * daily_ret.loc[date, a]
            for a, w in current.items()
            if a in daily_ret.columns and pd.notna(daily_ret.loc[date, a])
        )
        results.append(float(pr))

    return pd.Series(results, index=daily_ret.index)


def ffill_weights(
    portfolios: Dict[pd.Timestamp, Dict[str, float]],
    date:       pd.Timestamp,
) -> Dict[str, float]:
    """Return the most recent portfolio weights on or before date."""
    past = [d for d in portfolios if d <= date]
    return portfolios[max(past)] if past else {}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    total_start = time.time()

    # ----------------------------------------------------------------
    # Infrastructure
    # ----------------------------------------------------------------
    print("=" * 76)
    print("COMBINED STRATEGY VALIDATION")
    print("=" * 76)
    print("\n[1/6] Building engine + memory...")

    from data_pipeline import SnapshotManager, get_adj_close
    from state_engine   import StateEngine
    from analog_memory  import AnalogMemory

    snap   = SnapshotManager().load_snapshot(SNAPSHOT_NAME)
    engine = StateEngine()
    engine.build(snap, fred_api_key=FRED_API_KEY)

    memory = AnalogMemory(k=30, purge_window=21)
    memory.build(
        state_vector = engine.state_vector,
        pillars      = engine.pillars,
        raw_close    = engine._raw_close,
    )
    adj_close = get_adj_close(snap)
    print(f"     Done in {(time.time()-total_start)/60:.1f} min")

    # ----------------------------------------------------------------
    # Strategy 1
    # ----------------------------------------------------------------
    print("\n[2/6] Building Strategy 1 (StateAllocator)...")
    t = time.time()

    from strategy_state_allocator import StateAllocator
    allocator = StateAllocator(n_top=12, min_train=504, refit_every=21)
    allocator.build(
        pillars       = engine.pillars,
        state_vector  = engine.state_vector,
        analog_scores = memory.analog_scores,
        raw_close     = engine._raw_close,
        adj_close     = adj_close,
    )
    s1_portfolios = allocator._portfolios.get("C", {})
    print(f"     Done in {(time.time()-t)/60:.1f} min  "
          f"({len(s1_portfolios)} rebalance dates)")

    # ----------------------------------------------------------------
    # Strategy 2
    # ----------------------------------------------------------------
    print("\n[3/6] Building Strategy 2 (TrendCTA)...")

    from strategy_trend import TrendCTAStrategy
    s2_strat = TrendCTAStrategy()
    s2_strat.build(snap)
    s2_outputs  = s2_strat.generate_signals(snap)
    s2_portfolios = {
        pd.Timestamp(o.strategy_metadata["date"]): o.proposed_weights
        for o in s2_outputs
    }
    print(f"     Done — {len(s2_portfolios)} signal dates")

    # ----------------------------------------------------------------
    # Strategy 3
    # ----------------------------------------------------------------
    print("\n[4/6] Building Strategy 3 (Defensive)...")

    from strategy_defensive import DefensiveStrategy
    s3_strat = DefensiveStrategy()
    s3_strat.build(
        pillars       = engine.pillars,
        analog_scores = memory.analog_scores,
        adj_close     = adj_close,
    )
    s3_outputs = s3_strat.generate_signals()
    s3_portfolios = {
        pd.Timestamp(o.strategy_metadata["date"]): o.proposed_weights
        for o in s3_outputs
    }
    print(f"     Done — {len(s3_portfolios)} signal dates")

    # ----------------------------------------------------------------
    # Conviction Governor
    # ----------------------------------------------------------------
    print("\n[5/6] Building Conviction Governor...")

    from conviction_governor import ConvictionGovernor
    gov = ConvictionGovernor(risk_profile="aggressive")
    gov.build(engine.pillars, memory.analog_scores)
    print("     Done")

    # ----------------------------------------------------------------
    # Build blended portfolios on S1 rebalance schedule
    # ----------------------------------------------------------------
    print("\n[6/6] Constructing blended portfolios...")

    eq_portfolios  = {}
    gov_portfolios = {}

    for date in sorted(s1_portfolios.keys()):
        w1 = s1_portfolios[date]
        w2 = ffill_weights(s2_portfolios, date)
        w3 = ffill_weights(s3_portfolios, date)

        if not w1 or not w2 or not w3:
            continue

        # Equal-weight blend (1/3 each strategy)
        eq = {}
        for a, w in w1.items():
            eq[a] = eq.get(a, 0.0) + w / 3
        for a, w in w2.items():
            eq[a] = eq.get(a, 0.0) + w / 3
        for a, w in w3.items():
            eq[a] = eq.get(a, 0.0) + w / 3
        eq_portfolios[date] = eq

        # Conviction-governed blend
        gov_out = gov.apply_governance(date)
        cw = gov_out.capital_weights if gov_out else {"s1": 1/3, "s2": 1/3, "s3": 1/3}

        gv = {}
        for a, w in w1.items():
            gv[a] = gv.get(a, 0.0) + w * cw.get("s1", 1/3)
        for a, w in w2.items():
            gv[a] = gv.get(a, 0.0) + w * cw.get("s2", 1/3)
        for a, w in w3.items():
            gv[a] = gv.get(a, 0.0) + w * cw.get("s3", 1/3)
        gov_portfolios[date] = gv

    print(f"     Done — {len(eq_portfolios)} blended rebalance dates")

    # ----------------------------------------------------------------
    # Simulate and compare
    # ----------------------------------------------------------------
    print("\nSimulating daily returns...")

    s1_ret  = allocator.daily_returns["model_C"] \
              if "model_C" in allocator.daily_returns.columns \
              else simulate_daily_returns(s1_portfolios, adj_close)
    s2_ret  = simulate_daily_returns(s2_portfolios, adj_close)
    s3_ret  = simulate_daily_returns(s3_portfolios, adj_close)
    eq_ret  = simulate_daily_returns(eq_portfolios,  adj_close)
    gov_ret = simulate_daily_returns(gov_portfolios,  adj_close)
    spy_ret = adj_close["SPY"].pct_change()

    # Align evaluation window to the latest first valid date
    starts = []
    for r in [s1_ret, eq_ret, gov_ret]:
        nz = r.dropna()
        nz = nz[nz != 0]
        if len(nz):
            starts.append(nz.index[0])
    eval_start = max(starts) if starts else None

    print("\n" + "=" * 76)
    print("RESULTS")
    print("=" * 76)
    if eval_start:
        print(f"  Evaluation from: {eval_start.date()}")

    results = [
        compute_metrics(s1_ret.loc[eval_start:]  if eval_start else s1_ret,
                        "S1 standalone (baseline)"),
        compute_metrics(s2_ret.loc[eval_start:]  if eval_start else s2_ret,
                        "S2 standalone (TrendCTA)"),
        compute_metrics(s3_ret.loc[eval_start:]  if eval_start else s3_ret,
                        "S3 standalone (Defensive)"),
        compute_metrics(eq_ret.loc[eval_start:]  if eval_start else eq_ret,
                        "S1+S2+S3 equal weight (33/33/33)"),
        compute_metrics(gov_ret.loc[eval_start:] if eval_start else gov_ret,
                        "S1+S2+S3 conviction-governed"),
        compute_metrics(spy_ret.loc[eval_start:] if eval_start else spy_ret,
                        "SPY buy-and-hold"),
    ]

    print_table(results)

    print(f"\n  Total runtime: {(time.time()-total_start)/60:.1f} min")
    print("=" * 76)