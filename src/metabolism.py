"""
metabolism.py — VO₂–velocity models and equivalent-speed solver.

This module provides the full metabolic chain:

    actual speed  →  VO₂_treadmill (intrinsic cost)
                 →  VO₂_air (aerodynamic cost, scaled by air density)
                 →  apply shoe RE delta (intrinsic component only)
                 →  apply drag reduction (aero component only)
                 →  total drafted VO₂
                 →  solve for equivalent solo speed  (v_eq)

The equivalent solo speed v_eq is the speed at which a runner WITHOUT any
shoe or drafting benefit would have the same metabolic rate.  v_eq is then
used in the D′-balance engine so that CS and D′ (which were derived from
historical races with typical shoe/drafting conditions) remain the
bioenergetic reference.

Key equations
-------------
Batliner et al. (2018) treadmill VO₂:
    VO₂_tm(v) = 1.5355·v² + 1.5374·v + 15.661   [ml O₂/kg/min]

Pugh (1971) air-resistance VO₂:
    VO₂_air(v) = (3.54 · A_p / mass) · (ρ/ρ₀) · v³   [ml O₂/kg/min]

Combined (Kipp et al., 2019):
    VO₂_total(v) = VO₂_tm(v) + VO₂_air(v)

References
----------
- Kipp et al. (2019): Front Physiol 10:79
- Batliner et al. (2018): Sports Med Int Open 2:E1–E8
- Pugh (1971): J Physiol 213:255–276
"""

from __future__ import annotations
import numpy as np
from scipy.optimize import brentq
from typing import Dict, Any

from .config import (
    BATLINER_A,
    BATLINER_B,
    BATLINER_C,
    PUGH_K,
    STANDARD_AIR_DENSITY,
    estimate_frontal_area,
)


# ══════════════════════════════════════════════════════════════════════
# VO₂ component functions
# ══════════════════════════════════════════════════════════════════════

def vo2_treadmill(v: float) -> float:
    """Intrinsic (treadmill) metabolic rate at speed *v* (m/s).

    Returns VO₂ in ml O₂ / kg / min.
    Uses the Batliner et al. (2018) quadratic regression.
    """
    return BATLINER_A * v**2 + BATLINER_B * v + BATLINER_C


def vo2_air_resistance(
    v: float,
    frontal_area_m2: float,
    mass_kg: float,
    air_density: float = STANDARD_AIR_DENSITY,
) -> float:
    """Metabolic cost of overcoming air resistance at speed *v*.

    Returns VO₂ in ml O₂ / kg / min.
    Scales by actual air-density ratio to the Pugh (1971) reference.
    """
    rho_ratio = air_density / STANDARD_AIR_DENSITY
    c_air = PUGH_K * frontal_area_m2 / mass_kg
    return c_air * rho_ratio * v**3


def vo2_total_overground(
    v: float,
    frontal_area_m2: float,
    mass_kg: float,
    air_density: float = STANDARD_AIR_DENSITY,
) -> float:
    """Total overground VO₂ = treadmill cost + air-resistance cost.

    Returns VO₂ in ml O₂ / kg / min.
    """
    return (vo2_treadmill(v)
            + vo2_air_resistance(v, frontal_area_m2, mass_kg, air_density))


# ══════════════════════════════════════════════════════════════════════
# Drafted VO₂ and equivalent-speed solver
# ══════════════════════════════════════════════════════════════════════

def drafted_vo2(
    v: float,
    r_drag: float,
    frontal_area_m2: float,
    mass_kg: float,
    air_density: float = STANDARD_AIR_DENSITY,
    shoe_re_delta_pct: float = 0.0,
) -> float:
    """Effective metabolic rate when running at speed *v* with drafting + shoe.

    Steps
    -----
    1. Treadmill VO₂ is reduced by ``shoe_re_delta_pct`` (positive = benefit,
       e.g. 4.0 means 4 % lower intrinsic cost — as with Nike Vaporfly).
    2. Air-resistance VO₂ is reduced by ``r_drag`` (fractional drag reduction
       from the aerodynamics module).
    3. The two components are summed.

    Parameters
    ----------
    v : float – actual running speed (m/s)
    r_drag : float – fractional drag reduction [0, 1)
    frontal_area_m2 : float – runner's projected frontal area (m²)
    mass_kg : float – runner mass (kg)
    air_density : float – ambient air density (kg/m³)
    shoe_re_delta_pct : float – shoe RE improvement as positive % (default 0)

    Returns
    -------
    VO₂ in ml O₂ / kg / min
    """
    vo2_tm = vo2_treadmill(v) * (1.0 - shoe_re_delta_pct / 100.0)
    vo2_air = vo2_air_resistance(v, frontal_area_m2, mass_kg, air_density)
    vo2_air_draft = vo2_air * (1.0 - r_drag)
    return vo2_tm + vo2_air_draft


def solve_equivalent_speed(
    vo2_target: float,
    frontal_area_m2: float,
    mass_kg: float,
    air_density: float = STANDARD_AIR_DENSITY,
    v_bounds: tuple = (0.5, 12.0),
) -> float:
    """Find the solo overground speed that produces ``vo2_target``.

    Solves:  vo2_total_overground(v_eq) = vo2_target
    for v_eq ∈ ``v_bounds`` using Brent's method (guaranteed convergence
    on a monotone bracket).

    Parameters
    ----------
    vo2_target : float – target VO₂ (ml O₂ / kg / min)
    frontal_area_m2, mass_kg, air_density : runner / environment params
    v_bounds : (v_lo, v_hi) – search bracket (m/s)

    Returns
    -------
    v_eq : float – equivalent solo speed (m/s)

    Raises
    ------
    ValueError if no solution exists within bounds.
    """
    f = lambda v: vo2_total_overground(v, frontal_area_m2, mass_kg, air_density) - vo2_target
    f_lo = f(v_bounds[0])
    f_hi = f(v_bounds[1])

    if f_lo * f_hi > 0:
        # No sign change — target may be below the minimum VO₂ or above maximum
        if vo2_target <= vo2_total_overground(v_bounds[0], frontal_area_m2, mass_kg, air_density):
            return v_bounds[0]
        raise ValueError(
            f"Cannot find v_eq for VO₂ = {vo2_target:.2f} within "
            f"[{v_bounds[0]}, {v_bounds[1]}] m/s."
        )

    return float(brentq(f, v_bounds[0], v_bounds[1], xtol=1e-6))


def compute_segment_metabolics(
    v_actual: float,
    r_drag: float,
    frontal_area_m2: float,
    mass_kg: float,
    air_density: float = STANDARD_AIR_DENSITY,
    shoe_re_delta_pct: float = 0.0,
) -> Dict[str, float]:
    """Full metabolic calculation for a single segment.

    Returns a dict with detailed breakdown useful for diagnostics and plots.
    """
    # Baseline (solo, no shoe, no drafting)
    vo2_tm_base = vo2_treadmill(v_actual)
    vo2_air_base = vo2_air_resistance(v_actual, frontal_area_m2, mass_kg, air_density)
    vo2_total_base = vo2_tm_base + vo2_air_base

    # With shoe and drafting
    vo2_tm_shoe = vo2_tm_base * (1.0 - shoe_re_delta_pct / 100.0)
    vo2_air_drafted = vo2_air_base * (1.0 - r_drag)
    vo2_effective = vo2_tm_shoe + vo2_air_drafted

    # Savings
    vo2_saving = vo2_total_base - vo2_effective
    re_improvement_pct = (vo2_saving / vo2_total_base) * 100.0 if vo2_total_base > 0 else 0.0

    # Equivalent solo speed (no shoe, no drafting) that matches the effective
    # VO₂.  This is the speed used for D′ balance calculations.
    v_eq = solve_equivalent_speed(
        vo2_effective, frontal_area_m2, mass_kg, air_density
    )

    return {
        "v_actual":           v_actual,
        "v_eq":               v_eq,
        "vo2_treadmill":      vo2_tm_base,
        "vo2_air_base":       vo2_air_base,
        "vo2_total_base":     vo2_total_base,
        "vo2_air_drafted":    vo2_air_drafted,
        "vo2_effective":      vo2_effective,
        "vo2_saving":         vo2_saving,
        "re_improvement_pct": re_improvement_pct,
        "r_drag":             r_drag,
        "shoe_re_delta_pct":  shoe_re_delta_pct,
    }
