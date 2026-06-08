"""
SPY Alpha V9 — Reward Decomposition Diagnostic
diagnostics_reward.py
================================================
Investigates three root causes of SHY dominance in Strategy 1:

  Possibility A — Reward function bias
    Decomposes realized reward components (Sharpe, Sortino, Drawdown, Tail)
    for SHY, SPY, QQQ, TLT, UPRO across all historical 63-day windows.
    Answers: which component makes SHY score 2.014 vs UPRO's 0.999?

  Possibility B — Correct state reading
    Compares SHY vs UPRO predicted scores conditional on market state pillars.
    Answers: does UPRO beat SHY during confirmed bull states?
    If yes: state engine routes correctly, mean is dragged by defensive periods.
    If no: reward function is biased regardless of state.

  Possibility C — Weight construction compression
    Measures score ratio vs weight ratio for UPRO vs SPY at every refit.
    Answers: how much does inverse-vol weighting further suppress UPRO
    beyond what the reward scoring already penalizes?

Usage (add to run script after allocator.validate()):

    from diagnostics_reward import run_reward_decomposition
    run_reward_decomposition(
        allocator  = allocator,
        adj_close  = adj_close,
        pillars    = engine.pillars,
        raw_close  = engine._raw_close,
    )
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

FORWARD_WINDOW: int = 63

FOCUS_ASSETS: List[str] = ["SHY", "SPY", "QQQ", "TLT", "UPRO"]

REWARD_WEIGHTS: Dict[str, float] = {
    "differential_sharpe": 1.0,
    "sortino_component":   0.5,
    "drawdown_penalty":    2.0,
    "tail_risk_penalty":   1.0,
}

# Possibility B state thresholds
BULL_GROWTH_MIN:  float = 0.65
BULL_STRESS_MAX:  float = 0.30
BEAR_STRESS_MIN:  float = 0.60


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def run_reward_decomposition(
    allocator,
    adj_close: pd.DataFrame,
    pillars: Optional[pd.DataFrame] = None,
    raw_close: Optional[pd.DataFrame] = None,
) -> None:
    """
    Run all three root-cause diagnostics.

    Args:
        allocator  : built StateAllocator (after build() has been called)
        adj_close  : adjusted close prices (same DataFrame passed to build())
        pillars    : engine.pillars — required for Possibility B
        raw_close  : engine._raw_close — required for vol analysis in Possibility C
    """
    print("\n" + "=" * 76)
    print("REWARD DECOMPOSITION DIAGNOSTIC")
    print("Investigating three root causes of SHY dominance:")
    print("  A) Reward function bias")
    print("  B) Correct state reading (most states genuinely defensive)")
    print("  C) Weight construction compression (inverse-vol suppression)")
    print("=" * 76)

    _possibility_a(adj_close)
    _possibility_c(allocator, raw_close)

    if pillars is not None:
        _possibility_b(allocator, pillars)
    else:
        print("\n  [Possibility B skipped — pass pillars=engine.pillars to enable]")

    print("\n" + "=" * 76)
    print("REWARD DECOMPOSITION COMPLETE")
    print("=" * 76 + "\n")


# ---------------------------------------------------------------------------
# Core: single-window reward component decomposition
# ---------------------------------------------------------------------------

def _decompose_components(
    asset_returns: pd.Series,
    spy_returns: pd.Series,
) -> Dict[str, float]:
    """
    Decompose reward into its four components for a single 63-day window.
    Returns all components and the final reward score.
    """
    port  = asset_returns.dropna()
    bench = spy_returns.reindex(port.index).dropna()

    null = {k: np.nan for k in
            ["diff_sharpe", "sortino_comp", "dd_penalty", "tail_penalty", "reward"]}

    if len(port) < 21:
        return null

    # ---- Differential Sharpe ----
    ps = port.mean()  / (port.std()  + 1e-10) * np.sqrt(252)
    bs = bench.mean() / (bench.std() + 1e-10) * np.sqrt(252)
    diff_sharpe = float(ps - bs)

    # ---- Sortino component ----
    downside = port[port < 0]
    if len(downside) > 0 and downside.std() > 0:
        ds_vol = float(downside.std() * np.sqrt(252))
    elif port.std() > 0:
        ds_vol = float(port.std() * np.sqrt(252))
    else:
        ds_vol = 0.01
    sortino      = float(np.clip((port.mean() * 252) / ds_vol, -10.0, 10.0))
    sortino_comp = float(np.clip(sortino / 2.0, -2.0, 2.0))

    # ---- Drawdown penalty ----
    cum    = (1.0 + port).cumprod()
    peak   = cum.expanding().max()
    max_dd = float(abs((cum / peak - 1.0).min()))
    dd_pen = float(max(max_dd - 0.10, 0.0))

    # ---- Tail risk penalty ----
    tail_pen = 0.0
    if len(port) > 10:
        kurt = port.kurtosis()
        if pd.notna(kurt):
            tail_pen = float(np.clip(max(float(kurt) - 3.0, 0.0) / 10.0, 0.0, 2.0))

    w = REWARD_WEIGHTS
    reward = (
        w["differential_sharpe"] * diff_sharpe
        + w["sortino_component"]  * sortino_comp
        - w["drawdown_penalty"]   * dd_pen
        - w["tail_risk_penalty"]  * tail_pen
    )

    return {
        "diff_sharpe":  diff_sharpe,
        "sortino_comp": sortino_comp,
        "dd_penalty":   dd_pen,
        "tail_penalty": tail_pen,
        "reward":       reward,
    }


# ---------------------------------------------------------------------------
# Possibility A: Reward function component decomposition
# ---------------------------------------------------------------------------

def _possibility_a(adj_close: pd.DataFrame) -> None:
    """
    For each focus asset, compute realized reward components across all
    historical 63-day forward windows and show distribution statistics.

    The cross-asset contribution table directly answers whether SHY's
    dominance comes from the drawdown penalty or from genuine Sharpe advantage.
    """
    print("\n" + "─" * 76)
    print("POSSIBILITY A: Reward Component Decomposition")
    print("Realized components across all 63-day forward windows in history")
    print("─" * 76)

    available = [a for a in FOCUS_ASSETS if a in adj_close.columns]
    if "SPY" not in available:
        print("  ERROR: SPY required as benchmark but not found in adj_close.")
        return

    daily_ret = adj_close.pct_change()
    spy_ret   = daily_ret["SPY"]
    dates     = daily_ret.index.tolist()
    N         = len(dates)

    print(f"\n  Computing components for {N - FORWARD_WINDOW} windows × "
          f"{len(available)} assets...")

    all_components: Dict[str, List[Dict]] = {a: [] for a in available}

    for i in range(N - FORWARD_WINDOW):
        s_win = spy_ret.iloc[i: i + FORWARD_WINDOW]
        for asset in available:
            if asset not in daily_ret.columns:
                continue
            p_win = daily_ret[asset].iloc[i: i + FORWARD_WINDOW]
            c     = _decompose_components(p_win, s_win)
            if not np.isnan(c["reward"]):
                all_components[asset].append(c)

    # Per-asset distribution table
    comp_cols = ["diff_sharpe", "sortino_comp", "dd_penalty", "tail_penalty", "reward"]
    coeff_label = {
        "diff_sharpe":  "(×+1.0)",
        "sortino_comp": "(×+0.5)",
        "dd_penalty":   "(×-2.0)",
        "tail_penalty": "(×-1.0)",
        "reward":       "  FINAL",
    }

    for asset in available:
        rows = all_components[asset]
        if not rows:
            print(f"\n  {asset}: insufficient data")
            continue
        df = pd.DataFrame(rows)
        n  = len(df)
        print(f"\n  {asset}  ({n} windows):")
        print(f"  {'Component':<16} {'Coeff':>8} {'Mean':>8} {'Median':>8} "
              f"{'P25':>8} {'P75':>8}")
        print(f"  {'─'*16} {'─'*8} {'─'*8} {'─'*8} {'─'*8} {'─'*8}")
        for col in comp_cols:
            s    = df[col].dropna()
            lbl  = coeff_label[col]
            print(f"  {col:<16} {lbl:>8} {s.mean():>8.4f} {s.median():>8.4f} "
                  f"{s.quantile(0.25):>8.4f} {s.quantile(0.75):>8.4f}")

    # Cross-asset weighted contribution table — the key diagnostic
    print(f"\n  {'─' * 76}")
    print("  WEIGHTED CONTRIBUTION TABLE  (mean × coefficient, sign applied)")
    print("  This shows exactly which component drives each asset's final score.")
    print(f"\n  {'Asset':<8} {'Sharpe×1.0':>11} {'Sortino×0.5':>12} "
          f"{'DD×-2.0':>10} {'Tail×-1.0':>11} {'= Score':>9}")
    print(f"  {'─'*8} {'─'*11} {'─'*12} {'─'*10} {'─'*11} {'─'*9}")

    scores = {}
    for asset in available:
        rows = all_components[asset]
        if not rows:
            continue
        df  = pd.DataFrame(rows)
        sh  =  df["diff_sharpe"].mean()  * REWARD_WEIGHTS["differential_sharpe"]
        so  =  df["sortino_comp"].mean() * REWARD_WEIGHTS["sortino_component"]
        dd  = -df["dd_penalty"].mean()   * REWARD_WEIGHTS["drawdown_penalty"]
        tp  = -df["tail_penalty"].mean() * REWARD_WEIGHTS["tail_risk_penalty"]
        tot = sh + so + dd + tp
        scores[asset] = tot
        print(f"  {asset:<8} {sh:>+11.4f} {so:>+12.4f} {dd:>+10.4f} {tp:>+11.4f} {tot:>+9.4f}")

    # Interpretation
    print(f"\n  OBSERVATIONS:")
    if "SHY" in scores and "UPRO" in scores:
        gap = scores["SHY"] - scores["UPRO"]
        print(f"  SHY score: {scores['SHY']:+.4f}  |  UPRO score: {scores['UPRO']:+.4f}  "
              f"|  Gap: {gap:+.4f}")

        # Which component is responsible for most of the gap?
        if all_components.get("SHY") and all_components.get("UPRO"):
            shy_df  = pd.DataFrame(all_components["SHY"])
            upro_df = pd.DataFrame(all_components["UPRO"])
            gaps = {
                "Sharpe":   (shy_df["diff_sharpe"].mean()  - upro_df["diff_sharpe"].mean())
                             * REWARD_WEIGHTS["differential_sharpe"],
                "Sortino":  (shy_df["sortino_comp"].mean() - upro_df["sortino_comp"].mean())
                             * REWARD_WEIGHTS["sortino_component"],
                "DD pen":  -(shy_df["dd_penalty"].mean()   - upro_df["dd_penalty"].mean())
                             * REWARD_WEIGHTS["drawdown_penalty"],
                "Tail pen":-(shy_df["tail_penalty"].mean() - upro_df["tail_penalty"].mean())
                             * REWARD_WEIGHTS["tail_risk_penalty"],
            }
            largest = max(gaps, key=lambda k: abs(gaps[k]))
            print(f"\n  Component gap breakdown (SHY − UPRO, weighted):")
            for comp, g in gaps.items():
                pct = g / gap * 100 if abs(gap) > 1e-6 else 0
                bar = "█" * int(abs(pct) / 5)
                sign = "SHY advantage" if g > 0 else "UPRO advantage"
                print(f"    {comp:<12} {g:>+8.4f}  ({pct:>+6.1f}%)  {bar}  {sign}")
            print(f"\n  Largest gap driver: {largest} ({gaps[largest]:+.4f})")
            if largest == "DD pen":
                print("  → Possibility A likely CONFIRMED: drawdown penalty is the primary bias.")
                print(f"    The 2.0× coefficient penalizes UPRO's higher drawdown much more")
                print(f"    than it penalizes SHY's near-zero drawdown.")
            elif largest == "Sharpe":
                print("  → SHY genuinely has higher differential Sharpe over the full period.")
                print("    Reward function may be correct; problem is downstream construction.")
            else:
                print(f"  → {largest} drives the gap. Review coefficient for that component.")


# ---------------------------------------------------------------------------
# Possibility C: Weight construction compression
# ---------------------------------------------------------------------------

def _possibility_c(
    allocator,
    raw_close: Optional[pd.DataFrame],
) -> None:
    """
    Measures how much the inverse-vol weighting further suppresses UPRO
    after the model has already predicted its reward score.

    Compression factor = weight_ratio / score_ratio
    If < 1.0 → vol weighting applies additional compression beyond score differences.
    """
    print("\n" + "─" * 76)
    print("POSSIBILITY C: Weight Construction Compression")
    print("Score ratio vs weight ratio for UPRO vs SPY at each shared refit date")
    print("─" * 76)

    predictions = allocator.predictions.get("C", pd.DataFrame())
    portfolios  = allocator._portfolios.get("C", {})

    if predictions.empty or not portfolios:
        print("  No data available. Run allocator.build() first.")
        return

    refit_dates = sorted(portfolios.keys())

    score_ratios:   List[float] = []
    weight_ratios:  List[float] = []
    upro_scores:    List[float] = []
    spy_scores:     List[float] = []
    upro_weights:   List[float] = []
    spy_weights:    List[float] = []
    shared_dates:   List[pd.Timestamp] = []

    for date in refit_dates:
        if date not in predictions.index:
            continue
        row = predictions.loc[date]
        w   = portfolios[date]

        s_upro = row.get("UPRO", np.nan)
        s_spy  = row.get("SPY",  np.nan)
        w_upro = w.get("UPRO",   0.0)
        w_spy  = w.get("SPY",    0.0)

        # Only include dates where both were selected and both have valid scores
        if np.isnan(s_upro) or np.isnan(s_spy):
            continue
        if w_upro == 0.0 or w_spy == 0.0:
            continue
        if s_spy <= 0:
            continue

        score_ratios.append(s_upro / s_spy)
        weight_ratios.append(w_upro / w_spy)
        upro_scores.append(s_upro)
        spy_scores.append(s_spy)
        upro_weights.append(w_upro * 100)
        spy_weights.append(w_spy  * 100)
        shared_dates.append(date)

    if not score_ratios:
        print("  No refit dates where both UPRO and SPY were selected.")
        return

    sr  = np.array(score_ratios)
    wr  = np.array(weight_ratios)
    cf  = wr / sr  # compression factor

    print(f"\n  Shared refit dates (both UPRO and SPY selected): {len(sr)}")

    print(f"\n  {'Metric':<35} {'Mean':>8} {'Median':>8} {'P25':>8} {'P75':>8}")
    print(f"  {'─'*35} {'─'*8} {'─'*8} {'─'*8} {'─'*8}")

    def row(label, arr):
        print(f"  {label:<35} {arr.mean():>8.4f} {np.median(arr):>8.4f} "
              f"{np.percentile(arr,25):>8.4f} {np.percentile(arr,75):>8.4f}")

    row("Score ratio  (UPRO score / SPY score)", sr)
    row("Weight ratio (UPRO wt   / SPY wt   )", wr)
    row("Compression  (weight / score ratio  )", cf)

    print(f"\n  Absolute values (mean):")
    print(f"    UPRO predicted score  : {np.mean(upro_scores):.4f}")
    print(f"    SPY  predicted score  : {np.mean(spy_scores):.4f}")
    print(f"    UPRO weight (selected): {np.mean(upro_weights):.2f}%")
    print(f"    SPY  weight (selected): {np.mean(spy_weights):.2f}%")

    # Vol analysis
    if raw_close is not None and \
       "UPRO" in raw_close.columns and "SPY" in raw_close.columns:

        print(f"\n  Trailing 21-day annualized vol (on dates where both selected):")
        upro_vols:  List[float] = []
        spy_vols:   List[float] = []
        vol_ratios: List[float] = []

        daily_raw = raw_close.pct_change()

        for date in shared_dates:
            hist = daily_raw.loc[:date].tail(21)
            if len(hist) < 10:
                continue
            v_u = float(hist["UPRO"].std() * np.sqrt(252)) \
                  if "UPRO" in hist.columns else np.nan
            v_s = float(hist["SPY"].std()  * np.sqrt(252)) \
                  if "SPY"  in hist.columns else np.nan
            if not np.isnan(v_u) and not np.isnan(v_s) and v_s > 0:
                upro_vols.append(v_u)
                spy_vols.append(v_s)
                vol_ratios.append(v_u / v_s)

        if vol_ratios:
            vr = np.array(vol_ratios)
            print(f"    UPRO mean ann vol : {np.mean(upro_vols):.1%}")
            print(f"    SPY  mean ann vol : {np.mean(spy_vols):.1%}")
            print(f"    Vol ratio (UPRO/SPY): mean={vr.mean():.2f}x  "
                  f"median={np.median(vr):.2f}x")

            # Theoretical weight ratio if vol alone determined weights
            # raw_w ∝ score / vol → weight_ratio = (score_U/vol_U) / (score_S/vol_S)
            #                                     = score_ratio / vol_ratio
            predicted_wr = sr.mean() / vr.mean()
            actual_wr    = wr.mean()
            residual     = actual_wr - predicted_wr

            print(f"\n  Weight ratio decomposition:")
            print(f"    Predicted by score/vol alone : {predicted_wr:.4f}")
            print(f"    Actual weight ratio          : {actual_wr:.4f}")
            print(f"    Residual                     : {residual:+.4f}")

            if abs(residual) < 0.04:
                print(f"    → Vol weighting FULLY explains compression (residual < 0.04)")
            else:
                print(f"    → Additional factors beyond vol weighting (cap enforcement, etc.)")

    # Conclusion
    print(f"\n  CONCLUSION:")
    mean_cf = cf.mean()
    if mean_cf < 0.60:
        print(f"  Possibility C is SIGNIFICANT (compression factor={mean_cf:.3f})")
        print(f"  After the model predicts UPRO's score, the weight construction")
        print(f"  applies an additional {(1-mean_cf)*100:.0f}% reduction relative to SPY.")
        print(f"  Even if Possibility A is fixed (reward equalized), UPRO will still")
        print(f"  be structurally underweighted by inverse-vol sizing.")
        print(f"  The Conviction Governor's dynamic_max_upro is the correct architectural fix.")
    elif mean_cf < 0.85:
        print(f"  Possibility C is MODERATE (compression factor={mean_cf:.3f})")
        print(f"  Partial vol-driven suppression present.")
    else:
        print(f"  Possibility C is MINOR (compression factor={mean_cf:.3f})")
        print(f"  Weight construction tracks score ratios closely.")


# ---------------------------------------------------------------------------
# Possibility B: State-conditional score analysis
# ---------------------------------------------------------------------------

def _possibility_b(allocator, pillars: pd.DataFrame) -> None:
    """
    The single most diagnostic question: does UPRO score higher than SHY
    during confirmed bull states (growth_momentum high, financial_stress low)?

    If yes → state engine routes correctly, SHY's overall mean is dragged
             by the majority of mixed/defensive periods. Fix = Conviction Governor.
    If no  → reward function biases toward SHY regardless of state conditions.
             Fix = reward function recalibration.
    """
    print("\n" + "─" * 76)
    print("POSSIBILITY B: State-Conditional Score Analysis")
    print("Does UPRO beat SHY in confirmed bull states?")
    print("─" * 76)

    predictions = allocator.predictions.get("C", pd.DataFrame())
    if predictions.empty:
        print("  No prediction data.")
        return

    focus = [a for a in ["SHY", "TLT", "SPY", "QQQ", "UPRO"]
             if a in predictions.columns]

    pred_dates = predictions.index
    aligned    = pillars.reindex(pred_dates, method="ffill")

    required = ["growth_momentum", "financial_stress"]
    missing  = [c for c in required if c not in aligned.columns]
    if missing:
        print(f"  Missing pillar columns: {missing}")
        print(f"  Available: {list(aligned.columns)}")
        return

    gm = aligned["growth_momentum"]
    fs = aligned["financial_stress"]

    bull_mask  = (gm > BULL_GROWTH_MIN) & (fs < BULL_STRESS_MAX)
    bear_mask  = fs > BEAR_STRESS_MIN
    mixed_mask = ~bull_mask & ~bear_mask

    states = {
        f"BULL  (growth>{BULL_GROWTH_MIN}, stress<{BULL_STRESS_MAX})": bull_mask,
        f"BEAR  (stress>{BEAR_STRESS_MIN})":                           bear_mask,
        f"MIXED (all other)":                                          mixed_mask,
    }

    print(f"\n  Total refit dates: {len(pred_dates)}")
    print(f"  {'State':<45} {'N dates':>8} {'% of total':>11}")
    print(f"  {'─'*45} {'─'*8} {'─'*11}")
    for name, mask in states.items():
        n = int(mask.sum())
        print(f"  {name:<45} {n:>8} {n/len(pred_dates)*100:>10.1f}%")

    # Per-state score ranking
    for state_name, mask in states.items():
        state_dates = pred_dates[mask.values]

        print(f"\n  ── {state_name}  ({len(state_dates)} dates) ──")

        if len(state_dates) == 0:
            print("  (no dates in this state)")
            continue

        sp = predictions.loc[state_dates, focus]
        means = sp.mean().sort_values(ascending=False)

        print(f"  {'Asset':<8} {'Mean':>9} {'Median':>9} {'P25':>9} {'P75':>9}  Rank")
        print(f"  {'─'*8} {'─'*9} {'─'*9} {'─'*9} {'─'*9}  ────")
        for rank, asset in enumerate(means.index, 1):
            col = sp[asset].dropna()
            print(f"  {asset:<8} {col.mean():>9.4f} {col.median():>9.4f} "
                  f"{col.quantile(0.25):>9.4f} {col.quantile(0.75):>9.4f}  {rank}")

        if "SHY" in means.index and "UPRO" in means.index:
            shy_mean   = float(sp["SHY"].mean())
            upro_mean  = float(sp["UPRO"].mean())
            gap        = shy_mean - upro_mean
            upro_wins  = float((sp["UPRO"] > sp["SHY"]).mean() * 100)
            print(f"\n  SHY mean={shy_mean:.4f}  |  UPRO mean={upro_mean:.4f}  "
                  f"|  Gap={gap:+.4f}  |  UPRO beats SHY: {upro_wins:.1f}% of dates")

    # Final interpretation focused on bull state
    print(f"\n  {'─' * 76}")
    print("  KEY FINDING:")

    bull_dates = pred_dates[bull_mask.values]
    if len(bull_dates) > 0 and "UPRO" in predictions.columns and "SHY" in predictions.columns:
        bp        = predictions.loc[bull_dates, ["SHY", "UPRO"]]
        upro_pct  = float((bp["UPRO"] > bp["SHY"]).mean() * 100)
        shy_bull  = float(bp["SHY"].mean())
        upro_bull = float(bp["UPRO"].mean())

        print(f"  In confirmed BULL states ({len(bull_dates)} dates):")
        print(f"    SHY  mean score : {shy_bull:.4f}")
        print(f"    UPRO mean score : {upro_bull:.4f}")
        print(f"    UPRO beats SHY  : {upro_pct:.1f}% of bull dates")

        if upro_pct > 60:
            print(f"\n  → Possibility B CONFIRMED ✓")
            print(f"    The state engine correctly routes toward UPRO during bull conditions.")
            print(f"    SHY's higher overall mean is driven by mixed/bear periods dominating.")
            print(f"    Primary fix: Conviction Governor (concentrate S1 capital during bull states).")
            print(f"    Secondary: monitor reward function but do not change before Step 6.")
        elif upro_pct > 40:
            print(f"\n  → Possibility B PARTIAL")
            print(f"    State engine provides weak directional routing toward UPRO.")
            print(f"    Both the reward function (A) and downstream governance (B) need attention.")
        else:
            print(f"\n  → Possibility B NOT CONFIRMED")
            print(f"    UPRO still loses to SHY in {100-upro_pct:.1f}% of bull states.")
            print(f"    The reward function is biased regardless of state conditions.")
            print(f"    Reducing the drawdown_penalty coefficient (2.0 → 1.0) is warranted")
            print(f"    before proceeding to Strategy 3.")
    else:
        if len(bull_dates) == 0:
            print("  No bull state dates found — check pillar threshold values.")