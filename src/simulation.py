"""
simulation.py — End-to-end race simulation engine.

Orchestrates:
    segments  ×  race_script  ×  aerodynamics  ×  metabolism  ×  D′ balance

to produce a rich per-segment results table and predicted finish time.
"""

from __future__ import annotations
import pandas as pd
import numpy as np
from typing import Dict, Any, Literal

from .segments import generate_segments
from .aerodynamics import drag_reduction_prior, curve_offset_multiplier, parse_pos_of_n
from .metabolism import compute_segment_metabolics
from .dprime_balance import update_dprime_balance, within_race_predicted_time
from .config import STANDARD_AIR_DENSITY, estimate_frontal_area


def simulate_race(
    event_distance_m: float,
    segment_length_m: float,
    cs_ms: float,
    d_prime_m: float,
    race_script: pd.DataFrame,
    height_m: float,
    mass_kg: float,
    frontal_area_m2: float | None = None,
    air_density: float = STANDARD_AIR_DENSITY,
    shoe_re_delta_pct: float = 0.0,
    drafting_scale: float = 1.0,
    max_speed_ms: float | None = None,
    dprime_method: Literal["bellenger", "skiba"] = "bellenger",
    curve_offset_penalty: float = 0.85,
) -> Dict[str, Any]:
    """Simulate a track race segment-by-segment.

    Parameters
    ----------
    event_distance_m : float
        Race distance (e.g. 1609.34 for the mile).
    segment_length_m : float
        Default segment length (100 m).
    cs_ms : float
        Critical Speed (m/s).
    d_prime_m : float
        D′ (metres).
    race_script : DataFrame
        Must contain per-segment columns: segment_speed_ms, pos_of_n, gap_m,
        is_curve_offset.  Number of rows must match the number of segments.
    height_m, mass_kg : float
        Athlete anthropometrics.
    frontal_area_m2 : float or None
        Projected frontal area.  Estimated from height/mass if None.
    air_density : float
        Ambient air density (kg/m³).
    shoe_re_delta_pct : float
        Shoe RE improvement (positive = benefit, %).
    drafting_scale : float
        Global multiplier on drafting effect (for sensitivity).
    max_speed_ms : float or None
        Maximum achievable speed for speed-capped predictions.
    dprime_method : 'bellenger' or 'skiba'
    curve_offset_penalty : float
        Multiplier on drafting benefit for curve-offset segments (default 0.85).

    Returns
    -------
    dict with keys:
        segments_df    – DataFrame with full per-segment breakdown
        finish_time_s  – Predicted finish time (actual elapsed, from race script)
        predicted_time_s – Model-predicted finish time at race end
        summary        – dict of summary statistics
    """
    # ── 1. Prepare segments ───────────────────────────────────────────
    seg_df = generate_segments(event_distance_m, segment_length_m)
    n_seg = len(seg_df)

    # Validate race_script length
    if len(race_script) != n_seg:
        raise ValueError(
            f"race_script has {len(race_script)} rows but event produces "
            f"{n_seg} segments.  They must match."
        )

    if frontal_area_m2 is None:
        frontal_area_m2 = estimate_frontal_area(height_m, mass_kg)

    # ── 2. Merge segment geometry with race script ────────────────────
    script = race_script.reset_index(drop=True)
    seg_df = seg_df.reset_index(drop=True)

    # ── 3. Run simulation ─────────────────────────────────────────────
    results = []
    d_bal = d_prime_m        # start with full D′
    d_exp_accum = 0.0        # Skiba accumulated expenditure
    elapsed = 0.0            # cumulative time (s)

    for i in range(n_seg):
        seg_len  = seg_df.loc[i, "length_m"]
        is_bend  = seg_df.loc[i, "is_bend"]
        pct_bend = seg_df.loc[i, "pct_bend"]

        v_scripted   = float(script.loc[i, "segment_speed_ms"])
        pos_of_n_str = str(script.loc[i, "pos_of_n"])
        gap_m        = float(script.loc[i, "gap_m"]) if pd.notna(script.loc[i, "gap_m"]) else 1.3
        is_offset    = int(script.loc[i, "is_curve_offset"]) if pd.notna(script.loc[i, "is_curve_offset"]) else 0

        # ── D′ depletion enforcement ──────────────────────────────────
        # If anaerobic capacity is exhausted, the athlete cannot sustain
        # pace above Critical Speed.  Clamp actual speed to CS.
        if d_bal <= 0.01 and v_scripted > cs_ms:
            v_actual = cs_ms
        else:
            v_actual = v_scripted

        # Duration of this segment at actual speed
        seg_dur = seg_len / v_actual

        # ── 3a. Drafting drag reduction ───────────────────────────────
        r_drag = drag_reduction_prior(pos_of_n_str, gap_m, drafting_scale)

        # Apply curve-offset penalty if the runner is offset on a bend
        if is_offset and is_bend:
            r_drag *= curve_offset_multiplier(1, curve_offset_penalty)

        # ── 3b. Full metabolic calculation ────────────────────────────
        metab = compute_segment_metabolics(
            v_actual=v_actual,
            r_drag=r_drag,
            frontal_area_m2=frontal_area_m2,
            mass_kg=mass_kg,
            air_density=air_density,
            shoe_re_delta_pct=shoe_re_delta_pct,
        )
        v_eq = metab["v_eq"]

        # ── 3c. D′ balance update ─────────────────────────────────────
        d_bal, d_exp_accum = update_dprime_balance(
            d_prime_balance=d_bal,
            d_prime_0=d_prime_m,
            v_eq=v_eq,
            cs=cs_ms,
            segment_duration_s=seg_dur,
            method=dprime_method,
            d_prime_exp_accum=d_exp_accum,
        )

        # ── 3d. Elapsed time and remaining distance ───────────────────
        elapsed += seg_dur
        remaining_dist = event_distance_m - seg_df.loc[i, "end_m"]

        # ── 3e. Within-race predicted finish time ─────────────────────
        pred_time = within_race_predicted_time(
            elapsed_time_s=elapsed,
            remaining_distance_m=remaining_dist,
            d_prime_balance=d_bal,
            cs=cs_ms,
            max_speed=max_speed_ms,
        )

        # ── 3f. Collect results ───────────────────────────────────────
        row = {
            "segment_idx":       i + 1,
            "start_m":           seg_df.loc[i, "start_m"],
            "end_m":             seg_df.loc[i, "end_m"],
            "length_m":          seg_len,
            "is_bend":           is_bend,
            "pct_bend":          pct_bend,
            "v_scripted_ms":     v_scripted,
            "v_actual_ms":       v_actual,
            "v_eq_ms":           v_eq,
            "segment_time_s":    seg_dur,
            "elapsed_time_s":    elapsed,
            "remaining_m":       remaining_dist,
            "d_prime_balance_m": d_bal,
            "predicted_finish_s": pred_time,
            "pos_of_n":          pos_of_n_str,
            "gap_m":             gap_m,
            "is_curve_offset":   is_offset,
            **metab,  # includes r_drag, vo2 components, RE improvement, etc.
        }
        results.append(row)

    # ── 4. Build output ───────────────────────────────────────────────
    results_df = pd.DataFrame(results)
    actual_finish_time = results_df["elapsed_time_s"].iloc[-1]

    # If the runner's D' is depleted mid-race, the actual finish time would
    # be longer (they'd slow to CS).  Flag this.
    d_prime_depleted = results_df["d_prime_balance_m"].min() <= 0.01
    n_clamped = int((results_df["v_actual_ms"] < results_df["v_scripted_ms"]).sum())

    # Max segment speed in the race → used as MS_event (Bellenger)
    if max_speed_ms is None:
        max_speed_ms = results_df["v_actual_ms"].max()

    summary = {
        "event_distance_m":    event_distance_m,
        "cs_ms":               cs_ms,
        "d_prime_m":           d_prime_m,
        "finish_time_s":       actual_finish_time,
        "finish_time_str":     _format_time(actual_finish_time),
        "predicted_finish_s":  results_df["predicted_finish_s"].iloc[-1],
        "d_prime_final_m":     results_df["d_prime_balance_m"].iloc[-1],
        "d_prime_depleted":    d_prime_depleted,
        "avg_speed_ms":        event_distance_m / actual_finish_time,
        "avg_re_improvement":  results_df["re_improvement_pct"].mean(),
        "max_speed_ms":        max_speed_ms,
        "n_segments_clamped":  n_clamped,
    }

    return {
        "segments_df":       results_df,
        "finish_time_s":     actual_finish_time,
        "predicted_time_s":  results_df["predicted_finish_s"].iloc[-1],
        "summary":           summary,
    }


def _format_time(seconds: float) -> str:
    """Format seconds as M:SS.ss or H:MM:SS.ss."""
    if seconds < 0:
        return "N/A"
    mins = int(seconds // 60)
    secs = seconds - mins * 60
    if mins >= 60:
        hrs = mins // 60
        mins = mins % 60
        return f"{hrs}:{mins:02d}:{secs:05.2f}"
    return f"{mins}:{secs:05.2f}"
