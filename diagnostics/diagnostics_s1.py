"""
SPY Alpha V9 — Strategy 1 Diagnostic Analysis
==============================================
Run AFTER StateAllocator.build() and validate().
No model modifications. Diagnosis only.

Usage (add to your existing run script after allocator.validate()):

    from diagnostics_s1 import run_strategy1_diagnostics
    run_strategy1_diagnostics(allocator, adj_close)

Produces 6 diagnostic reports:
    1. Weight distribution when selected (selection freq vs actual weight)
    2. Return attribution by asset (annualized contribution to CAGR)
    3. Drawdown contribution by asset (losses during portfolio drawdown days)
    4. Holdings and weights during crisis periods (2008, Mar 2020, 2022)
    5. Portfolio turnover per rebalance
    6. Reward score distribution by asset (mean predicted score across refits)
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Crisis windows for Diagnostic 4
# ---------------------------------------------------------------------------

CRISIS_WINDOWS: Dict[str, Tuple[str, str]] = {
    "2008 GFC":         ("2008-09-01", "2009-03-31"),
    "Mar 2020 COVID":   ("2020-02-15", "2020-04-30"),
    "2022 Bear Market": ("2022-01-01", "2022-12-31"),
}


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def run_strategy1_diagnostics(allocator, adj_close: pd.DataFrame) -> None:
    """
    Run all 6 diagnostics on a built StateAllocator.

    Args:
        allocator  : StateAllocator instance after build() has been called
        adj_close  : adjusted close prices (same DataFrame passed to build())
    """
    portfolios  = allocator._portfolios.get("C", {})
    predictions = allocator.predictions.get("C", pd.DataFrame())

    if not portfolios:
        print("ERROR: No Model C portfolios found. Run allocator.build() first.")
        return

    print("\n" + "=" * 76)
    print("STRATEGY 1 — DIAGNOSTIC ANALYSIS (Model C)")
    print("=" * 76)

    _diag_1_weight_distribution(portfolios)
    _diag_2_return_attribution(portfolios, adj_close)
    _diag_3_drawdown_contribution(portfolios, adj_close)
    _diag_4_crisis_snapshots(portfolios)
    _diag_5_turnover(portfolios)
    _diag_6_reward_scores(predictions)

    print("\n" + "=" * 76)
    print("DIAGNOSTIC ANALYSIS COMPLETE")
    print("=" * 76 + "\n")


# ---------------------------------------------------------------------------
# Diagnostic 1 — Weight distribution when selected
# ---------------------------------------------------------------------------

def _diag_1_weight_distribution(portfolios: Dict) -> None:
    """
    Selection frequency vs average weight when selected vs effective weight.

    Key question: Is UPRO getting 15% weights or 1.5% weights?
    These two situations look identical in selection-frequency-only reporting.
    """
    print("\n" + "─" * 76)
    print("DIAGNOSTIC 1: Weight Distribution When Selected")
    print("─" * 76)

    n_refits = len(portfolios)
    sel_counts: Dict[str, int]   = {}
    weight_sum: Dict[str, float] = {}

    for weights in portfolios.values():
        for asset, w in weights.items():
            sel_counts[asset]  = sel_counts.get(asset, 0) + 1
            weight_sum[asset]  = weight_sum.get(asset, 0.0) + w

    rows: List[Tuple] = []
    for asset in sel_counts:
        n      = sel_counts[asset]
        sel_pct  = n / n_refits * 100
        avg_w_sel  = weight_sum[asset] / n * 100          # avg weight when selected
        eff_w    = weight_sum[asset] / n_refits * 100     # avg weight across ALL refits
        rows.append((asset, sel_pct, avg_w_sel, eff_w, n))

    rows.sort(key=lambda x: -x[3])  # sort by effective weight

    print(f"\n  {'Asset':<10} {'Sel%':>7} {'AvgW% | Selected':>17} {'EffW% | Always':>15} {'N':>6}")
    print(f"  {'─'*10} {'─'*7} {'─'*17} {'─'*15} {'─'*6}")

    for asset, sel_pct, avg_w_sel, eff_w, n in rows:
        print(f"  {asset:<10} {sel_pct:>6.1f}%"
              f" {avg_w_sel:>16.2f}%"
              f" {eff_w:>14.2f}%"
              f" {n:>6d}")

    print(f"\n  Total rebalance dates evaluated: {n_refits}")

    # Highlight UPRO and SHY specifically
    print(f"\n  KEY ASSETS:")
    for asset, sel_pct, avg_w_sel, eff_w, n in rows:
        if asset in ("UPRO", "SHY", "TLT", "SPY", "QQQ"):
            print(f"  {asset:<6} selected {sel_pct:.1f}% of refits | "
                  f"avg weight when selected = {avg_w_sel:.2f}% | "
                  f"effective weight = {eff_w:.2f}%")


# ---------------------------------------------------------------------------
# Diagnostic 2 — Return attribution by asset
# ---------------------------------------------------------------------------

def _diag_2_return_attribution(
    portfolios: Dict,
    adj_close: pd.DataFrame,
) -> None:
    """
    Which assets contributed the most to the portfolio's CAGR?

    Computed as: sum(weight_t * asset_return_t) per asset, annualized.
    """
    print("\n" + "─" * 76)
    print("DIAGNOSTIC 2: Return Attribution by Asset")
    print("─" * 76)

    daily_ret    = adj_close.pct_change()
    refit_dates  = sorted(portfolios.keys())
    all_dates    = adj_close.index

    # Build daily weight for each asset via forward-fill
    current_w: Dict[str, float] = {}
    refit_idx = 0
    contrib_sum: Dict[str, float] = {}
    n_days_active = 0

    for date in all_dates:
        # Advance to the latest refit date that has passed
        while refit_idx < len(refit_dates) and refit_dates[refit_idx] <= date:
            current_w = portfolios[refit_dates[refit_idx]]
            refit_idx += 1

        if not current_w:
            continue

        n_days_active += 1
        for asset, w in current_w.items():
            if asset not in daily_ret.columns:
                continue
            r = daily_ret.loc[date, asset]
            if pd.notna(r):
                contrib_sum[asset] = contrib_sum.get(asset, 0.0) + w * r

    if n_days_active == 0:
        print("  No active trading days found.")
        return

    n_years = n_days_active / 252
    contrib_ann = {
        asset: (total / n_years) * 100
        for asset, total in contrib_sum.items()
    }

    total_ann = sum(contrib_ann.values())
    sorted_contrib = sorted(contrib_ann.items(), key=lambda x: -x[1])

    print(f"\n  Annualized return contribution over {n_years:.1f} years:")
    print(f"\n  {'Asset':<10} {'Contrib% (ann)':>15} {'% of Total Return':>18}")
    print(f"  {'─'*10} {'─'*15} {'─'*18}")

    for asset, c in sorted_contrib:
        pct_of_total = (c / total_ann * 100) if abs(total_ann) > 1e-6 else 0.0
        print(f"  {asset:<10} {c:>14.3f}% {pct_of_total:>17.1f}%")

    print(f"  {'─'*44}")
    print(f"  {'TOTAL':<10} {total_ann:>14.3f}%")
    print(f"\n  Note: Total should match Model C CAGR from validate() report.")


# ---------------------------------------------------------------------------
# Diagnostic 3 — Drawdown contribution by asset
# ---------------------------------------------------------------------------

def _diag_3_drawdown_contribution(
    portfolios: Dict,
    adj_close: pd.DataFrame,
) -> None:
    """
    Which assets caused losses on days when the portfolio was in drawdown?

    Identifies the portfolio's worst 10% of days (by portfolio return)
    and shows which assets drove those losses.
    """
    print("\n" + "─" * 76)
    print("DIAGNOSTIC 3: Drawdown Contribution by Asset")
    print("─" * 76)

    daily_ret   = adj_close.pct_change()
    refit_dates = sorted(portfolios.keys())
    all_dates   = adj_close.index

    current_w: Dict[str, float] = {}
    refit_idx = 0

    port_daily: List[Tuple[pd.Timestamp, float, Dict[str, float]]] = []

    for date in all_dates:
        while refit_idx < len(refit_dates) and refit_dates[refit_idx] <= date:
            current_w = portfolios[refit_dates[refit_idx]]
            refit_idx += 1

        if not current_w:
            continue

        day_total = 0.0
        day_asset_contribs: Dict[str, float] = {}

        for asset, w in current_w.items():
            if asset not in daily_ret.columns:
                continue
            r = daily_ret.loc[date, asset]
            if pd.notna(r):
                c = w * r
                day_total += c
                day_asset_contribs[asset] = c

        port_daily.append((date, day_total, day_asset_contribs))

    if not port_daily:
        print("  No portfolio data.")
        return

    port_returns = pd.Series(
        [x[1] for x in port_daily],
        index=[x[0] for x in port_daily]
    )

    # Identify worst 10% of days by portfolio return
    threshold = port_returns.quantile(0.10)
    bad_days  = port_returns[port_returns <= threshold].index

    print(f"\n  Worst 10% of days threshold: {threshold:.4f} ({threshold*100:.2f}%)")
    print(f"  Number of bad days: {len(bad_days)}")

    bad_contribs: Dict[str, float] = {}
    for date, total, asset_contribs in port_daily:
        if date in bad_days:
            for asset, c in asset_contribs.items():
                bad_contribs[asset] = bad_contribs.get(asset, 0.0) + c

    sorted_bad = sorted(bad_contribs.items(), key=lambda x: x[1])  # most negative first

    total_bad = sum(bad_contribs.values())

    print(f"\n  Asset contributions on worst 10% of portfolio days:")
    print(f"\n  {'Asset':<10} {'Total Contrib':>14} {'% of Losses':>12}")
    print(f"  {'─'*10} {'─'*14} {'─'*12}")

    for asset, c in sorted_bad:
        pct = (c / total_bad * 100) if abs(total_bad) > 1e-6 else 0.0
        print(f"  {asset:<10} {c*100:>13.3f}% {pct:>11.1f}%")

    print(f"  {'─'*37}")
    print(f"  {'TOTAL':<10} {total_bad*100:>13.3f}%")


# ---------------------------------------------------------------------------
# Diagnostic 4 — Crisis snapshots
# ---------------------------------------------------------------------------

def _diag_4_crisis_snapshots(portfolios: Dict) -> None:
    """
    Average holdings and weights during known crisis periods.

    If the allocator holds SHY/TLT heavily during crises → correct behavior.
    If it holds UPRO during crises → reward function is not penalizing properly.
    """
    print("\n" + "─" * 76)
    print("DIAGNOSTIC 4: Holdings and Weights During Crisis Periods")
    print("─" * 76)

    refit_dates = sorted(portfolios.keys())

    for crisis_name, (start_str, end_str) in CRISIS_WINDOWS.items():
        start = pd.Timestamp(start_str)
        end   = pd.Timestamp(end_str)

        crisis_refits = [d for d in refit_dates if start <= d <= end]

        print(f"\n  {crisis_name}  ({start_str} → {end_str})")

        if not crisis_refits:
            print(f"    No rebalance dates in this window "
                  f"(backtest may not cover this period)")
            continue

        asset_weights: Dict[str, List[float]] = {}
        for d in crisis_refits:
            for asset, w in portfolios[d].items():
                if asset not in asset_weights:
                    asset_weights[asset] = []
                asset_weights[asset].append(w)

        rows = [
            (asset, np.mean(ws) * 100, len(ws) / len(crisis_refits) * 100)
            for asset, ws in asset_weights.items()
        ]
        rows.sort(key=lambda x: -x[1])

        print(f"    Rebalance dates in window: {len(crisis_refits)}")
        print(f"\n    {'Asset':<10} {'AvgW%':>7} {'Sel%':>6}")
        print(f"    {'─'*10} {'─'*7} {'─'*6}")

        for asset, avg_w, sel_pct in rows:
            print(f"    {asset:<10} {avg_w:>6.2f}%  {sel_pct:>5.1f}%")


# ---------------------------------------------------------------------------
# Diagnostic 5 — Portfolio turnover
# ---------------------------------------------------------------------------

def _diag_5_turnover(portfolios: Dict) -> None:
    """
    Average one-way turnover per rebalance.

    High turnover erodes returns and indicates instability.
    Expected: 20-40% per rebalance (21 days) for an active strategy.
    """
    print("\n" + "─" * 76)
    print("DIAGNOSTIC 5: Portfolio Turnover")
    print("─" * 76)

    refit_dates = sorted(portfolios.keys())

    if len(refit_dates) < 2:
        print("  Insufficient rebalance dates.")
        return

    turnovers: List[float] = []

    for i in range(1, len(refit_dates)):
        prev = portfolios[refit_dates[i - 1]]
        curr = portfolios[refit_dates[i]]
        all_assets = set(prev.keys()) | set(curr.keys())
        one_way = sum(
            abs(curr.get(a, 0.0) - prev.get(a, 0.0))
            for a in all_assets
        ) / 2.0
        turnovers.append(one_way)

    t = np.array(turnovers)
    refits_per_year = 252 / 21

    print(f"\n  One-way turnover per rebalance (~21 trading days):")
    print(f"  Mean   : {np.mean(t)*100:>6.1f}%")
    print(f"  Median : {np.median(t)*100:>6.1f}%")
    print(f"  Max    : {np.max(t)*100:>6.1f}%")
    print(f"  Min    : {np.min(t)*100:>6.1f}%")
    print(f"\n  Annualized (~{refits_per_year:.0f} refits/yr): "
          f"{np.mean(t) * refits_per_year * 100:.0f}%/yr")
    print(f"\n  Note: V8 hard constraint was max 40% per rebalance. "
          f"Values above 60% suggest instability.")


# ---------------------------------------------------------------------------
# Diagnostic 6 — Reward score distribution by asset
# ---------------------------------------------------------------------------

def _diag_6_reward_scores(predictions: pd.DataFrame) -> None:
    """
    Mean predicted reward score per asset across all refit dates.

    This is the root-cause diagnostic.
    If SHY has the highest mean score regardless of state conditions,
    the reward function has a systematic defensive bias that the state
    features are not overriding.
    """
    print("\n" + "─" * 76)
    print("DIAGNOSTIC 6: Reward Score Distribution by Asset")
    print("─" * 76)

    if predictions is None or predictions.empty:
        print("  No prediction data available.")
        return

    n_refits = len(predictions)
    mean_scores = predictions.mean(axis=0).sort_values(ascending=False)
    std_scores  = predictions.std(axis=0)
    min_scores  = predictions.min(axis=0)
    max_scores  = predictions.max(axis=0)

    print(f"\n  Mean predicted reward score across {n_refits} refit dates:")
    print(f"  (Higher mean = model consistently rates this asset as attractive)")
    print(f"  (Low std = model always predicts similar score regardless of state)")
    print(f"\n  {'Asset':<10} {'Mean':>8} {'Std':>8} {'Min':>8} {'Max':>8}")
    print(f"  {'─'*10} {'─'*8} {'─'*8} {'─'*8} {'─'*8}")

    for asset in mean_scores.index:
        m  = mean_scores[asset]
        s  = std_scores.get(asset, np.nan)
        mn = min_scores.get(asset, np.nan)
        mx = max_scores.get(asset, np.nan)
        print(f"  {asset:<10} {m:>8.4f} {s:>8.4f} {mn:>8.4f} {mx:>8.4f}")

    # Score range diagnostic
    top_asset    = mean_scores.index[0]
    top_score    = mean_scores.iloc[0]
    bottom_asset = mean_scores.index[-1]
    bottom_score = mean_scores.iloc[-1]
    score_range  = top_score - bottom_score

    print(f"\n  OBSERVATIONS:")
    print(f"  Highest mean score : {top_asset:<8} ({top_score:.4f})")
    print(f"  Lowest  mean score : {bottom_asset:<8} ({bottom_score:.4f})")
    print(f"  Score range (top − bottom) : {score_range:.4f}")

    if score_range < 0.3:
        print("  WARNING: Very narrow score range — model is not differentiating "
              "between assets. Reward function may have a systematic bias.")
    elif score_range < 0.6:
        print("  CAUTION: Moderate score range — some differentiation present "
              "but may be insufficient for meaningful concentration.")
    else:
        print("  Score range is adequate for differentiation.")

    # Check if defensive assets dominate top scores
    defensive = {"SHY", "TLT", "GLD"}
    top_5 = set(mean_scores.index[:5])
    defensive_in_top5 = top_5 & defensive

    if len(defensive_in_top5) >= 3:
        print(f"\n  WARNING: Defensive assets dominate top-5 scores: {defensive_in_top5}")
        print("  This suggests the tail-aware reward function's drawdown penalty "
              "(2x coefficient) is systematically favoring low-vol assets.")
    elif len(defensive_in_top5) >= 2:
        print(f"\n  NOTE: {defensive_in_top5} in top-5 scores — monitor after "
              "Conviction Governor is integrated.")
    else:
        print(f"\n  Defensive asset dominance: not detected in top-5 scores.")