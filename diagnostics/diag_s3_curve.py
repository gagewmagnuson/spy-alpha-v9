"""
Strategy 3 Activation → Allocation Curve
diag_s3_curve.py
------------------------------------------
Purely analytical — no data loading required.
Shows exactly how proposed_weights change with activation level
for each of the three inflation regimes.

Answers the question: is 0.55 activation already effectively dormant?
"""

ACTIVATION_THRESHOLD = 0.35

REGIMES = {
    "Deflationary  (inflation_pressure < 0.30)":
        {"TLT": 0.40, "SHY": 0.60, "GLD": 0.00},
    "Mixed         (0.30 ≤ inflation ≤ 0.60)  ":
        {"SHY": 0.70, "TLT": 0.15, "GLD": 0.15},
    "Inflationary  (inflation_pressure > 0.60)":
        {"GLD": 0.40, "SHY": 0.60, "TLT": 0.00},
}

DORMANT = {"SHY": 1.00, "TLT": 0.00, "GLD": 0.00}

# Activation levels to evaluate
# Marked reference points from the validation output
LEVELS = [
    (0.20,  "below threshold"),
    (0.35,  "AT threshold"),
    (0.40,  ""),
    (0.50,  ""),
    (0.551, "← 2013-14 Bull mean"),
    (0.589, "← 2021-22 mean"),
    (0.608, "← 2019 Bull mean"),
    (0.65,  ""),
    (0.70,  ""),
    (0.764, "← Mar 2020 mean"),
    (0.804, "← 2008 GFC mean"),
    (0.90,  ""),
    (1.00,  "maximum"),
]


def blend(dormant, target, alpha):
    all_assets = set(dormant) | set(target)
    return {
        a: (1.0 - alpha) * dormant.get(a, 0.0)
           + alpha        * target.get(a,  0.0)
        for a in sorted(all_assets)
    }


def compute_weights(activation, base_weights):
    if activation <= ACTIVATION_THRESHOLD:
        intensity = 0.0
        weights   = DORMANT.copy()
    else:
        intensity = (activation - ACTIVATION_THRESHOLD) / (1.0 - ACTIVATION_THRESHOLD)
        intensity = min(intensity, 1.0)
        weights   = blend(DORMANT, base_weights, intensity)
    return weights, intensity


print("\n" + "=" * 76)
print("STRATEGY 3 — ACTIVATION → ALLOCATION CURVE")
print(f"ACTIVATION_THRESHOLD = {ACTIVATION_THRESHOLD}")
print("=" * 76)

for regime_name, base in REGIMES.items():
    print(f"\n  Regime: {regime_name}")
    print(f"  Base weights: SHY={base.get('SHY',0):.0%}  "
          f"TLT={base.get('TLT',0):.0%}  "
          f"GLD={base.get('GLD',0):.0%}")
    print(f"\n  {'Activation':>11} {'Intensity':>10} "
          f"{'SHY%':>8} {'TLT%':>8} {'GLD%':>8}  Note")
    print(f"  {'─'*11} {'─'*10} {'─'*8} {'─'*8} {'─'*8}  {'─'*22}")

    for level, note in LEVELS:
        w, intensity = compute_weights(level, base)
        activated = "ACTIVATED" if level > ACTIVATION_THRESHOLD else "dormant  "
        print(
            f"  {level:>11.3f} {intensity:>10.3f} "
            f"{w.get('SHY',0):>7.1%} "
            f"{w.get('TLT',0):>7.1%} "
            f"{w.get('GLD',0):>7.1%}  "
            f"{note}"
        )

print(f"\n\n  KEY QUESTION: Is 87% SHY during 2013-14 'dormant enough'?")
print(f"  At activation=0.551, mixed regime:")
w, intensity = compute_weights(0.551, REGIMES["Mixed         (0.30 ≤ inflation ≤ 0.60)  "])
print(f"    SHY={w.get('SHY',0):.1%}  TLT={w.get('TLT',0):.1%}  "
      f"GLD={w.get('GLD',0):.1%}  intensity={intensity:.3f}")

print(f"\n  At activation=0.551, deflationary regime:")
w, intensity = compute_weights(0.551, REGIMES["Deflationary  (inflation_pressure < 0.30)"])
print(f"    SHY={w.get('SHY',0):.1%}  TLT={w.get('TLT',0):.1%}  "
      f"GLD={w.get('GLD',0):.1%}  intensity={intensity:.3f}")

print(f"\n  Compare to fully active 2008 GFC (activation=0.804), deflationary:")
w, intensity = compute_weights(0.804, REGIMES["Deflationary  (inflation_pressure < 0.30)"])
print(f"    SHY={w.get('SHY',0):.1%}  TLT={w.get('TLT',0):.1%}  "
      f"GLD={w.get('GLD',0):.1%}  intensity={intensity:.3f}")

print("\n" + "=" * 76)