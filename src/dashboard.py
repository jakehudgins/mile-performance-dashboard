"""
dashboard.py — Metabolic-savings computation for the performance dashboard.

Given a baseline even pace, computes per-segment metabolic savings from
drafting and shoe interventions, then converts those savings into an
equivalent faster finish time.
"""

from __future__ import annotations
import numpy as np
import pandas as pd
from typing import Dict, Any, List

from .segments import generate_segments
from .aerodynamics import drag_reduction_prior, curve_offset_multiplier
from .metabolism import compute_segment_metabolics, vo2_total_overground, solve_equivalent_speed
from .config import STANDARD_AIR_DENSITY, estimate_frontal_area


def compute_segment_savings(
    baseline_speed_ms: float,
    event_distance_m: float,
    segment_length_m: float,
    drafting_plan: pd.DataFrame,
    shoe_re_pct: float = 0.0,
    air_density: float = STANDARD_AIR_DENSITY,
    height_m: float = 1.84,
    mass_kg: float = 72.6,
    frontal_area_m2: float | None = None,
) -> Dict[str, Any]:
    """Compute per-segment metabolic savings and equivalent time.

    Parameters
    ----------
    baseline_speed_ms : float
        Even-pace baseline speed (m/s). All segments run at this speed.
    event_distance_m : float
        Race distance (e.g. 1609.34 for mile).
    segment_length_m : float
        Default segment length (100 m).
    drafting_plan : DataFrame
        Must have columns: pos_of_n, gap_m, is_curve_offset.
        One row per segment.
    shoe_re_pct : float
        Running economy improvement from shoes (%).
    air_density : float
        Ambient air density (kg/m³).
    height_m, mass_kg : float
        Athlete anthropometrics.
    frontal_area_m2 : float or None
        Projected frontal area. Estimated if None.

    Returns
    -------
    dict with keys:
        segments_df : DataFrame with per-segment breakdown
        baseline_finish_s : float
        equivalent_finish_s : float
        time_saved_s : float
        pct_improvement : float
        drafting_contribution_s : float
        shoe_contribution_s : float
    """
    seg_df = generate_segments(event_distance_m, segment_length_m)
    n_seg = len(seg_df)

    if frontal_area_m2 is None:
        frontal_area_m2 = estimate_frontal_area(height_m, mass_kg)

    plan = drafting_plan.reset_index(drop=True)

    rows = []
    total_baseline_time = 0.0
    total_equivalent_time = 0.0
    total_drafting_saved = 0.0
    total_shoe_saved = 0.0

    for i in range(n_seg):
        seg_len = seg_df.loc[i, "length_m"]
        is_bend = seg_df.loc[i, "is_bend"]
        pct_bend = seg_df.loc[i, "pct_bend"]

        pos_of_n_str = str(plan.loc[i, "pos_of_n"])
        gap_m = float(plan.loc[i, "gap_m"]) if pd.notna(plan.loc[i, "gap_m"]) else 1.3
        is_offset = int(plan.loc[i, "is_curve_offset"]) if pd.notna(plan.loc[i, "is_curve_offset"]) else 0

        # Drafting drag reduction
        r_drag = drag_reduction_prior(pos_of_n_str, gap_m)
        if is_offset and is_bend:
            r_drag *= curve_offset_multiplier(1)

        # Full metabolic calculation at baseline speed
        metab = compute_segment_metabolics(
            v_actual=baseline_speed_ms,
            r_drag=r_drag,
            frontal_area_m2=frontal_area_m2,
            mass_kg=mass_kg,
            air_density=air_density,
            shoe_re_delta_pct=shoe_re_pct,
        )

        v_eq = metab["v_eq"]  # equivalent solo speed (lower than baseline due to savings)

        # Baseline segment time (no interventions)
        baseline_seg_time = seg_len / baseline_speed_ms

        # Equivalent segment time: same metabolic cost runs v_eq solo,
        # but we have interventions so we're actually running baseline_speed
        # at lower cost. The "free speed" means we could run faster at same cost.
        # Time savings: we run at baseline_speed but metabolically it costs
        # only what v_eq would cost solo. The savings let us run faster.
        # Equivalent faster speed at same VO2 cost as baseline:
        vo2_at_baseline_no_help = vo2_total_overground(
            baseline_speed_ms, frontal_area_m2, mass_kg, air_density
        )
        vo2_effective = metab["vo2_effective"]  # actual cost with interventions

        # The savings are: we're spending vo2_effective instead of vo2_at_baseline_no_help
        # This means we have "budget" to go faster. Find the speed that costs
        # vo2_at_baseline_no_help with our interventions still active:
        # vo2_tm(v_fast)*(1-shoe/100) + vo2_air(v_fast)*(1-r_drag) = vo2_at_baseline_no_help
        # This is the speed we COULD run at same metabolic cost as unassisted baseline.
        try:
            v_faster = _solve_assisted_speed(
                vo2_target=vo2_at_baseline_no_help,
                r_drag=r_drag,
                shoe_re_pct=shoe_re_pct,
                frontal_area_m2=frontal_area_m2,
                mass_kg=mass_kg,
                air_density=air_density,
            )
        except Exception:
            v_faster = baseline_speed_ms

        equivalent_seg_time = seg_len / v_faster
        seg_time_saved = baseline_seg_time - equivalent_seg_time

        # Decompose: drafting-only contribution
        try:
            v_draft_only = _solve_assisted_speed(
                vo2_target=vo2_at_baseline_no_help,
                r_drag=r_drag,
                shoe_re_pct=0.0,
                frontal_area_m2=frontal_area_m2,
                mass_kg=mass_kg,
                air_density=air_density,
            )
        except Exception:
            v_draft_only = baseline_speed_ms

        draft_saved = baseline_seg_time - (seg_len / v_draft_only)
        shoe_saved = seg_time_saved - draft_saved

        total_baseline_time += baseline_seg_time
        total_equivalent_time += equivalent_seg_time
        total_drafting_saved += draft_saved
        total_shoe_saved += shoe_saved

        rows.append({
            "segment_idx": i + 1,
            "start_m": seg_df.loc[i, "start_m"],
            "end_m": seg_df.loc[i, "end_m"],
            "length_m": seg_len,
            "is_bend": is_bend,
            "pct_bend": pct_bend,
            "pos_of_n": pos_of_n_str,
            "gap_m": gap_m,
            "r_drag": r_drag,
            "baseline_speed_ms": baseline_speed_ms,
            "equivalent_speed_ms": v_faster,
            "baseline_seg_time_s": baseline_seg_time,
            "equivalent_seg_time_s": equivalent_seg_time,
            "time_saved_s": seg_time_saved,
            "drafting_saved_s": draft_saved,
            "shoe_saved_s": shoe_saved,
            "re_improvement_pct": metab["re_improvement_pct"],
        })

    segments_df = pd.DataFrame(rows)
    total_saved = total_baseline_time - total_equivalent_time

    return {
        "segments_df": segments_df,
        "baseline_finish_s": total_baseline_time,
        "equivalent_finish_s": total_equivalent_time,
        "time_saved_s": total_saved,
        "pct_improvement": 100.0 * total_saved / total_baseline_time if total_baseline_time > 0 else 0.0,
        "drafting_contribution_s": total_drafting_saved,
        "shoe_contribution_s": total_shoe_saved,
    }


def _solve_assisted_speed(
    vo2_target: float,
    r_drag: float,
    shoe_re_pct: float,
    frontal_area_m2: float,
    mass_kg: float,
    air_density: float,
    v_bounds: tuple = (0.5, 12.0),
) -> float:
    """Find the speed where assisted VO2 equals target.

    Solves: vo2_tm(v)*(1-shoe/100) + vo2_air(v)*(1-r_drag) = vo2_target
    """
    from scipy.optimize import brentq
    from .metabolism import vo2_treadmill, vo2_air_resistance
    from .config import BATLINER_A, BATLINER_B, BATLINER_C, PUGH_K

    shoe_factor = 1.0 - shoe_re_pct / 100.0
    drag_factor = 1.0 - r_drag

    def residual(v):
        vo2_tm = (BATLINER_A * v**2 + BATLINER_B * v + BATLINER_C) * shoe_factor
        vo2_air = (PUGH_K * frontal_area_m2 / mass_kg) * (air_density / STANDARD_AIR_DENSITY) * v**3 * drag_factor
        return vo2_tm + vo2_air - vo2_target

    return brentq(residual, v_bounds[0], v_bounds[1])


def build_drafting_plan(
    n_segments: int,
    zone_config: Dict[str, Any] | None = None,
    per_segment_overrides: Dict[int, Dict[str, Any]] | None = None,
    event_distance_m: float = 1609.34,
    segment_length_m: float = 100.0,
    no_draft_final_m: float = 400.0,
) -> pd.DataFrame:
    """Build a drafting plan from zone presets + per-segment overrides.

    Parameters
    ----------
    n_segments : int
        Total segment count.
    zone_config : dict or None
        Keys: 'formation', 'gap_m'. Applied to all segments before final zone.
    per_segment_overrides : dict or None
        Keys are 0-based segment indices, values are dicts with
        'pos_of_n', 'gap_m', 'is_curve_offset'.
    no_draft_final_m : float
        Final distance with no drafting.

    Returns
    -------
    DataFrame with columns: pos_of_n, gap_m, is_curve_offset
    """
    seg_df = generate_segments(event_distance_m, segment_length_m)
    draft_cutoff = event_distance_m - no_draft_final_m

    # Defaults
    default_formation = "1of1"
    default_gap = 0.0

    if zone_config:
        default_formation = zone_config.get("formation", "1of1")
        default_gap = zone_config.get("gap_m", 1.3)

    rows = []
    for i in range(n_segments):
        start_m = seg_df.loc[i, "start_m"]
        if start_m >= draft_cutoff:
            row = {"pos_of_n": "1of1", "gap_m": 0.0, "is_curve_offset": 0}
        else:
            row = {"pos_of_n": default_formation, "gap_m": default_gap, "is_curve_offset": 0}
        rows.append(row)

    df = pd.DataFrame(rows)

    # Apply per-segment overrides
    if per_segment_overrides:
        for idx, overrides in per_segment_overrides.items():
            for col, val in overrides.items():
                if col in df.columns and 0 <= idx < n_segments:
                    df.loc[idx, col] = val

    return df


def format_time(seconds: float) -> str:
    """Format seconds as M:SS.ss"""
    mins = int(seconds // 60)
    secs = seconds - mins * 60
    return f"{mins}:{secs:05.2f}"


def calculate_air_density(temp_c: float, rh_pct: float, pressure_hpa: float) -> float:
    """Calculate air density from temperature, relative humidity, and pressure.

    Uses the ideal gas law with humidity correction:
        rho = (p_d / (R_d * T)) + (p_v / (R_v * T))

    Parameters
    ----------
    temp_c : float
        Temperature in degrees Celsius.
    rh_pct : float
        Relative humidity in percent (0-100).
    pressure_hpa : float
        Barometric pressure in hectopascals (hPa / mbar).

    Returns
    -------
    float : Air density in kg/m³.
    """
    T = temp_c + 273.15
    p = pressure_hpa * 100.0

    # Saturation vapor pressure (Magnus formula)
    p_sat = 611.2 * np.exp((17.67 * temp_c) / (temp_c + 243.5))
    p_v = (rh_pct / 100.0) * p_sat
    p_d = p - p_v

    R_d = 287.058  # J/(kg·K)
    R_v = 461.495  # J/(kg·K)

    rho = (p_d / (R_d * T)) + (p_v / (R_v * T))
    return rho


def calculate_air_density(temp_c: float, rh_pct: float, pressure_hpa: float) -> float:
    """Calculate air density from temperature, relative humidity, and pressure.

    Uses the ideal gas law with humidity correction:
        rho = (p_d / (R_d * T)) + (p_v / (R_v * T))

    where p_v is vapor pressure from the Magnus formula.

    Parameters
    ----------
    temp_c : float
        Temperature in degrees Celsius.
    rh_pct : float
        Relative humidity in percent (0-100).
    pressure_hpa : float
        Barometric pressure in hectopascals (hPa / mbar).

    Returns
    -------
    float : Air density in kg/m³.
    """
    T = temp_c + 273.15  # Kelvin
    p = pressure_hpa * 100.0  # Pa

    # Saturation vapor pressure (Magnus formula)
    p_sat = 611.2 * np.exp((17.67 * temp_c) / (temp_c + 243.5))  # Pa
    p_v = (rh_pct / 100.0) * p_sat  # Actual vapor pressure
    p_d = p - p_v  # Dry air partial pressure

    R_d = 287.058  # J/(kg·K) specific gas constant for dry air
    R_v = 461.495  # J/(kg·K) specific gas constant for water vapor

    rho = (p_d / (R_d * T)) + (p_v / (R_v * T))
    return rho
