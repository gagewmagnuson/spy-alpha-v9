"""
SPY Alpha V9 — Target and Weighting Comparison
target_weight_comparison.py
===============================================
Tests 5 combinations of target function and weighting method
to identify the correct Strategy 1 configuration.

Combinations:
  1. tail_aware    + inverse_vol  — current baseline (proven architecturally wrong)
  2. raw_return    + inverse_vol  — V8-style target, current weighting
  3. excess_return + inverse_vol  — alpha target, current weighting
  4. raw_return    + score_prop   — closest to V8 architecture
  5. excess_return + score_prop   — most aggressive combination

Decisive metric: UPRO vs SHY score in confirmed bull states
(growth_momentum > 0.65, financial_stress < 0.30).
UPRO must beat SHY in the winning combination.

Run: python3 target_weight_comparison.py
"""

from __future__ import annotations

import logging
import time
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logging.basicConfig(
    level=logging.WARNING,  # suppress INFO noise during comparison runs
    format="%(levelname)s  %(message)s",
)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

FRED_API_KEY   = "55a0b587c09627fe956afaf6cb6d2bf7"
SNAPSHOT_NAME  = "baseline_v7"

BULL_GROWTH_MIN: float = 0.65
BULL_STRESS_MAX: float = 0.30

COMBINATIONS: List[Tuple[str, str, str]] = [
    ("tail_aware",    "inverse_vol", "Baseline  (tail_aware  + inverse_vol)"),
    ("raw_return",    "inverse_vol", "V8 target (raw_return  + inverse_vol)"),
    ("excess_return", "inverse_vol", "Excess    (excess_ret  + inverse_vol)"),
    ("raw_return",    "score_prop",  "V8 arch   (raw_return  + score_prop) "),
    ("excess_return", "score_prop",  "Aggressive(excess_ret  + score_prop) "),
]

# ---------------------------------------------------------------------------
# Bull-state analysis (Possibility B — the decisive test)
# ---------------------------------------------------------------------------

def _bull_state_analysis(
    predictions_c: pd.DataFrame,
    pillars: pd.DataFrame,
) -> Dict[str, float]:
    """
    For Model C predictions, compute UPRO vs SHY scores in bull states.

    Returns dict with:
        shy_bull_mean   : SHY mean score during bull states
        upro_bull_mean  : UPRO mean score during bull states
        upro_beats_pct  : % of bull dates where UPRO score > SHY score
        n_bull_dates    : number of bull state refit dates
    """
    result = {
        "shy_bull_mean":  np.nan,
        "upro_bull_mean": np.nan,
        "upro_beats_pct": np.nan,
        "n_bull_dates":   0,
    }

    if predictions_c is None or predictions_c.empty:
        return result
    if "UPRO" not in predictions_c.columns or "SHY" not in predictions_c.columns:
        return result

    pred_dates = predictions_c.index
    aligned    = pillars.reindex(pred_dates, method="ffill")

    if "growth_momentum" not in aligned.columns or \
       "financial_stress" not in aligned.columns:
        return result

    gm   = aligned["growth_momentum"]
    fs   = aligned["financial_stress"]
    bull = (gm > BULL_GROWTH_MIN) & (fs < BULL_STRESS_MAX)

    bull_dates = pred_dates[bull.values]
    result["n_bull_dates"] = int(len(bull_dates))

    if len(bull_dates) == 0:
        return result

    bp = predictions_c.loc[bull_dates, ["SHY", "UPRO"]]
    result["shy_bull_mean"]  = float(bp["SHY"].mean())
    result["upro_bull_mean"] = float(bp["UPRO"].mean())
    result["upro_beats_pct"] = float((bp["UPRO"] > bp["SHY"]).mean() * 100)

    return result


# ---------------------------------------------------------------------------
# Single combination runner
# ---------------------------------------------------------------------------

def _run_single_combination(
    target_type:   str,
    weighting_type: str,
    label:         str,
    engine,
    memory,
    adj_close:     pd.DataFrame,
) -> Dict:
    """
    Instantiate StateAllocator with given params, run build(), extract results.
    Returns a flat dict of all comparison metrics.
    """
    from strategy_state_allocator import StateAllocator

    print(f"\n  Building: {label}")
    print(f"  target_type={target_type!r}  weighting_type={weighting_type!r}")
    t0 = time.time()

    allocator = StateAllocator(
        n_top          = 12,
        min_train      = 504,
        refit_every    = 21,
        target_type    = target_type,
        weighting_type = weighting_type,
    )

    allocator.build(
        pillars       = engine.pillars,
        state_vector  = engine.state_vector,
        analog_scores = memory.analog_scores,
        raw_close     = engine._raw_close,
        adj_close     = adj_close,
    )

    elapsed = time.time() - t0
    print(f"  Build complete in {elapsed/60:.1f} min")

    # Extract Model C portfolio metrics
    m = allocator.metrics
    row: Dict = {
        "label":         label,
        "target_type":   target_type,
        "weighting_type": weighting_type,
    }

    if m is not None and "model_C" in m.index:
        r = m.loc["model_C"]
        row["sharpe"]   = float(r.get("Sharpe",  np.nan))
        row["cagr"]     = float(r.get("CAGR",    np.nan))
        row["max_dd"]   = float(r.get("Max_DD",  np.nan))
        row["calmar"]   = float(r.get("Calmar",  np.nan))
        row["sortino"]  = float(r.get("Sortino", np.nan))
    else:
        row.update({"sharpe": np.nan, "cagr": np.nan,
                    "max_dd": np.nan, "calmar": np.nan, "sortino": np.nan})

    # Equal weight benchmark (same across all runs, just record once)
    if m is not None and "equal_weight" in m.index:
        ew = m.loc["equal_weight"]
        row["ew_cagr"]   = float(ew.get("CAGR",   np.nan))
        row["ew_sharpe"] = float(ew.get("Sharpe",  np.nan))
    else:
        row["ew_cagr"]   = np.nan
        row["ew_sharpe"] = np.nan

    # UPRO effective weight
    portfolios = allocator._portfolios.get("C", {})
    if portfolios:
        upro_weights = [
            w.get("UPRO", 0.0)
            for w in portfolios.values()
        ]
        shy_weights = [
            w.get("SHY", 0.0)
            for w in portfolios.values()
        ]
        row["upro_eff_weight"] = float(np.mean(upro_weights) * 100)
        row["shy_eff_weight"]  = float(np.mean(shy_weights)  * 100)
    else:
        row["upro_eff_weight"] = np.nan
        row["shy_eff_weight"]  = np.nan

    # Possibility B — bull state analysis
    predictions_c = allocator.predictions.get("C", pd.DataFrame())
    bull = _bull_state_analysis(predictions_c, engine.pillars)
    row["shy_bull_score"]  = bull["shy_bull_mean"]
    row["upro_bull_score"] = bull["upro_bull_mean"]
    row["upro_beats_pct"]  = bull["upro_beats_pct"]
    row["n_bull_dates"]    = bull["n_bull_dates"]

    return row


# ---------------------------------------------------------------------------
# Summary printer
# ---------------------------------------------------------------------------

def _print_summary(results: List[Dict]) -> None:
    """Print the full comparison table."""

    print("\n\n" + "=" * 76)
    print("TARGET & WEIGHTING COMPARISON — FULL RESULTS")
    print("=" * 76)

    # Performance table
    print(f"\n  {'Combination':<42} {'Sharpe':>7} {'CAGR':>8} "
          f"{'Max DD':>8} {'Calmar':>8} {'Sortino':>8}")
    print("  " + "─" * 74)
    for r in results:
        print(
            f"  {r['label']:<42} "
            f"{r['sharpe']:>7.3f} "
            f"{r['cagr']:>8.2%} "
            f"{r['max_dd']:>8.2%} "
            f"{r['calmar']:>8.3f} "
            f"{r['sortino']:>8.3f}"
        )

    # Equal weight reference
    ew_cagr   = results[0]["ew_cagr"]
    ew_sharpe = results[0]["ew_sharpe"]
    print(f"\n  {'Equal weight (reference)':<42} "
          f"{ew_sharpe:>7.3f} {ew_cagr:>8.2%}")

    # Effective weight table
    print(f"\n\n  {'Combination':<42} {'UPRO Eff%':>10} {'SHY Eff%':>10}")
    print("  " + "─" * 62)
    for r in results:
        print(
            f"  {r['label']:<42} "
            f"{r['upro_eff_weight']:>9.2f}% "
            f"{r['shy_eff_weight']:>9.2f}%"
        )

    # Possibility B — THE DECISIVE TABLE
    print(f"\n\n  POSSIBILITY B — UPRO vs SHY in BULL states "
          f"(growth>{BULL_GROWTH_MIN}, stress<{BULL_STRESS_MAX})")
    print(f"  Bull dates: {results[0]['n_bull_dates']} refit dates")
    print(f"\n  {'Combination':<42} {'SHY score':>10} {'UPRO score':>11} "
          f"{'UPRO>SHY%':>10}  Verdict")
    print("  " + "─" * 76)
    for r in results:
        upro_pct = r["upro_beats_pct"]
        if np.isnan(upro_pct):
            verdict = "N/A"
        elif upro_pct > 60:
            verdict = "PASS ✓  state engine routing correctly"
        elif upro_pct > 40:
            verdict = "MARGINAL"
        else:
            verdict = "FAIL ✗  reward bias persists"
        print(
            f"  {r['label']:<42} "
            f"{r['shy_bull_score']:>10.4f} "
            f"{r['upro_bull_score']:>11.4f} "
            f"{upro_pct:>9.1f}%  {verdict}"
        )

    # Recommendation
    print(f"\n\n  {'─' * 74}")
    print("  RECOMMENDATION:")

    # Find best on Calmar (most balanced return/risk metric)
    valid = [r for r in results if not np.isnan(r["calmar"])]
    if valid:
        best_calmar = max(valid, key=lambda x: x["calmar"])
        # Find first passing Possibility B
        passing_b = [r for r in results
                     if not np.isnan(r.get("upro_beats_pct", np.nan))
                     and r["upro_beats_pct"] > 60]

        print(f"  Best Calmar ratio  : {best_calmar['label'].strip()}")
        if passing_b:
            best_passing = max(passing_b, key=lambda x: x["calmar"])
            print(f"  Passes Possibility B + best Calmar: "
                  f"{best_passing['label'].strip()}")
        else:
            print("  WARNING: No combination passes Possibility B (UPRO>SHY in >60% of bull dates)")
            print("  Review results — a different target formulation may be needed.")

    print("=" * 76 + "\n")


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def run_comparison(
    engine,
    memory,
    adj_close: pd.DataFrame,
) -> List[Dict]:
    """
    Run all 5 combinations and print full comparison.
    Can be called from a run script with already-built engine and memory.
    """
    print("\n" + "=" * 76)
    print("TARGET & WEIGHTING COMPARISON")
    print(f"Running {len(COMBINATIONS)} combinations. "
          f"Expected runtime: 50-90 minutes.")
    print("=" * 76)

    results = []
    total_start = time.time()

    for i, (target_type, weighting_type, label) in enumerate(COMBINATIONS, 1):
        print(f"\n[{i}/{len(COMBINATIONS)}]", end="")
        row = _run_single_combination(
            target_type    = target_type,
            weighting_type = weighting_type,
            label          = label,
            engine         = engine,
            memory         = memory,
            adj_close      = adj_close,
        )
        results.append(row)
        elapsed = (time.time() - total_start) / 60
        remaining = elapsed / i * (len(COMBINATIONS) - i)
        print(f"  Elapsed: {elapsed:.1f} min  |  "
              f"Est. remaining: {remaining:.0f} min")

    _print_summary(results)
    return results


if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.WARNING)

    print("Loading data and building engine + memory (one-time)...")
    t0 = time.time()

    from data_pipeline import SnapshotManager, get_adj_close
    from state_engine import StateEngine
    from analog_memory import AnalogMemory

    sm   = SnapshotManager()
    snap = sm.load_snapshot(SNAPSHOT_NAME)

    engine = StateEngine()
    engine.build(snap, fred_api_key=FRED_API_KEY)

    memory = AnalogMemory(k=30, purge_window=21)
    memory.build(
        state_vector = engine.state_vector,
        pillars      = engine.pillars,
        raw_close    = engine._raw_close,
    )

    adj_close = get_adj_close(snap)

    print(f"Engine + memory built in {(time.time()-t0)/60:.1f} min\n")

    run_comparison(engine, memory, adj_close)