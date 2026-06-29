"""
dprime_balance.py — D′ balance engine (depletion & reconstitution).

Two models are implemented:

1. **"bellenger"** — Simple linear depletion per segment.
   D′_bal = D′_prev − max(0, (v_eq − CS) × Δt)
   When v_eq < CS: D′_bal = min(D′_0, D′_prev + (CS − v_eq) × Δt)
   (Bellenger et al., 2026, Eq. 1)

2. **"skiba"** — ODE-based model with exponential reconstitution.
   Based on Skiba et al. (2015) first-order kinetics, as used by
   Kirby et al. (2021).
   - Depletion: D′_exp accumulates as (v_eq − CS) × Δt
   - Reconstitution: exponential recovery with time constant τ = D′_0 / (CS − v_eq)

References
----------
- Bellenger et al. (2026): IJSPP — Eq. 1 (modified-for-running W′ expenditure)
- Kirby et al. (2021): J Appl Physiol 131:1532–1542 — Eqs. 1a/1b
- Skiba et al. (2015): Eur J Appl Physiol 115:703–713
- Skiba et al. (2012): Med Sci Sports Exerc 44:1526–1532
"""

from __future__ import annotations
import math
from typing import Literal


def update_dprime_balance(
    d_prime_balance: float,
    d_prime_0: float,
    v_eq: float,
    cs: float,
    segment_duration_s: float,
    method: Literal["bellenger", "skiba"] = "bellenger",
    # Skiba-specific accumulated expenditure tracking
    d_prime_exp_accum: float = 0.0,
) -> tuple[float, float]:
    """Update D′ balance after one segment.

    Parameters
    ----------
    d_prime_balance : float
        D′ balance entering this segment (metres).
    d_prime_0 : float
        Initial (full) D′ value (metres).
    v_eq : float
        Equivalent solo speed for this segment (m/s).  This already accounts
        for drafting, shoes, etc.
    cs : float
        Critical Speed (m/s).
    segment_duration_s : float
        Duration of the segment (seconds) = segment_length / v_actual.
    method : 'bellenger' or 'skiba'
        Which D′ balance model to use.
    d_prime_exp_accum : float
        Accumulated D′ expenditure (used only by 'skiba' method).

    Returns
    -------
    (new_d_prime_balance, new_d_prime_exp_accum) : tuple of floats
        Updated D′ balance and accumulated expenditure.

    Notes
    -----
    • When v_eq > CS the runner is in the **severe** domain; D′ depletes.
    • When v_eq ≤ CS the runner is in the **heavy/moderate** domain; D′
      reconstitutes (partially) toward D′_0.
    • D′ balance is clamped to [0, D′_0].  A balance of 0 means the runner
      can sustain at most CS for the remainder of the race.
    """
    dt = segment_duration_s

    if method == "bellenger":
        return _bellenger_update(d_prime_balance, d_prime_0, v_eq, cs, dt)
    elif method == "skiba":
        return _skiba_update(
            d_prime_balance, d_prime_0, v_eq, cs, dt, d_prime_exp_accum
        )
    else:
        raise ValueError(f"Unknown method '{method}'. Use 'bellenger' or 'skiba'.")


# ══════════════════════════════════════════════════════════════════════
# Bellenger (simple linear) model
# ══════════════════════════════════════════════════════════════════════

def _bellenger_update(
    d_bal: float, d0: float, v_eq: float, cs: float, dt: float
) -> tuple[float, float]:
    """
    Bellenger et al. (2026) Eq. 1:
        D′_BALANCE = D′(start) − ([S − CS] × u)

    When v_eq > CS  →  D′ is spent.
    When v_eq ≤ CS  →  D′ reconstitutes linearly, capped at D′_0.
    """
    delta_v = v_eq - cs  # positive → depletion, negative → reconstitution
    d_change = delta_v * dt  # metres of D′ spent (positive) or recovered (negative)
    new_bal = d_bal - d_change
    # Clamp
    new_bal = max(0.0, min(new_bal, d0))
    return new_bal, 0.0  # exp_accum not used in this model


# ══════════════════════════════════════════════════════════════════════
# Skiba (ODE / exponential) model
# ══════════════════════════════════════════════════════════════════════

def _skiba_update(
    d_bal: float, d0: float, v_eq: float, cs: float, dt: float,
    d_exp_accum: float,
) -> tuple[float, float]:
    """
    Skiba et al. (2015) / Kirby et al. (2021) Eqs. 1a–1b:
        D′_balance = D′_0  −  D′_exp · exp(−D_CS · t / D′_0)
        D′_exp     = D_CS · t + D′_prior

    When v_eq > CS:
        D_CS = v_eq − CS  (positive → depletion with exponential decay)
    When v_eq ≤ CS:
        Exponential reconstitution toward D′_0 with time constant τ = D′_0 / |D_CS|
    """
    d_cs = v_eq - cs  # signed: positive = above CS

    if d_cs > 0:
        # --- DEPLETION (severe domain) ---
        # Accumulate expenditure
        new_exp = d_cs * dt + d_exp_accum
        # Exponential decay of expenditure signal
        exponent = -d_cs * dt / d0 if d0 > 0 else 0.0
        # Guard against extreme exponents
        exponent = max(exponent, -50.0)
        new_bal = d0 - new_exp * math.exp(exponent)
    else:
        # --- RECONSTITUTION (heavy/moderate domain) ---
        # Time constant τ = D′_0 / |D_CS|  (Skiba et al., 2015)
        d_cs_abs = abs(d_cs) if abs(d_cs) > 1e-6 else 1e-6
        tau = d0 / d_cs_abs
        # Exponential recovery toward D′_0
        deficit = d0 - d_bal
        new_bal = d0 - deficit * math.exp(-dt / tau)
        # Reset expenditure accumulator during recovery
        new_exp = max(0.0, d_exp_accum - d_cs_abs * dt)

    # Clamp
    new_bal = max(0.0, min(new_bal, d0))
    new_exp = max(0.0, new_exp)
    return new_bal, new_exp


# ══════════════════════════════════════════════════════════════════════
# Within-race predicted finish time (Bellenger Eq. 3 / Kirby Eq. 2)
# ══════════════════════════════════════════════════════════════════════

def within_race_predicted_time(
    elapsed_time_s: float,
    remaining_distance_m: float,
    d_prime_balance: float,
    cs: float,
    max_speed: float | None = None,
) -> float:
    """Predict remaining race time from current D′ balance.

    predicted_finish = elapsed  +  (remaining_dist − D′_balance) / CS

    If ``max_speed`` is provided, a speed-capped version is also checked and
    the slower (more conservative) estimate is returned — see Bellenger Eq. 4.
    """
    if cs <= 0:
        return float("inf")

    # Standard prediction
    time_remaining = (remaining_distance_m - d_prime_balance) / cs
    pred = elapsed_time_s + max(0.0, time_remaining)

    # Speed-capped prediction (Bellenger Eq. 4)
    if max_speed is not None and max_speed > 0:
        time_at_max = remaining_distance_m / max_speed
        pred_capped = elapsed_time_s + time_at_max
        pred = max(pred, pred_capped)  # use the slower (more realistic) estimate

    return pred
