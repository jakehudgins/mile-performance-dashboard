"""
optimizer.py — Pacing & drafting strategy optimization engine.

Core philosophy: For a given athlete (CS, D′) and scenario configuration
(drafting, shoes, air density), find the FASTEST achievable race time
that fully depletes D′ at the finish line.

Each pacing "strategy" is a shape constraint on the speed profile:
  - even:          all segments at same speed → 1-D scalar search
  - fast_start:    first N segments boosted, rest compensated → 1-D search
  - negative_ramp: linearly accelerating → 1-D search
  - negative_kick: steady then kick final N m → 1-D search
  - optimized:     unconstrained per-segment speeds → N-D global optimization

In every case, the solver finds the fastest time where D′ → 0 at finish.
"""

from __future__ import annotations
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Literal
from scipy.optimize import minimize_scalar, differential_evolution

from .segments import generate_segments
from .simulation import simulate_race
from .monte_carlo import run_monte_carlo


# ══════════════════════════════════════════════════════════════════════
# 1. Drafting configuration builder
# ══════════════════════════════════════════════════════════════════════

def build_drafting_config(
    n_segments: int,
    formation: str = "1of1",
    gap_m: float = 1.3,
    no_draft_final_m: float = 400.0,
    event_distance_m: float = 1609.34,
    segment_length_m: float = 100.0,
) -> pd.DataFrame:
    """Build a drafting configuration DataFrame for all segments.

    Segments starting at or beyond (event_distance - no_draft_final_m)
    are forced to solo ('1of1').
    """
    seg_df = generate_segments(event_distance_m, segment_length_m)
    draft_cutoff = event_distance_m - no_draft_final_m

    rows = []
    for i in range(n_segments):
        start_m = seg_df.loc[i, "start_m"]
        if start_m >= draft_cutoff:
            rows.append({"pos_of_n": "1of1", "gap_m": 0.0, "is_curve_offset": 0})
        else:
            rows.append({"pos_of_n": formation, "gap_m": gap_m, "is_curve_offset": 0})

    return pd.DataFrame(rows)


# ══════════════════════════════════════════════════════════════════════
# 2. Speed profile shape generators (parameterized by a scalar "level")
# ══════════════════════════════════════════════════════════════════════

def _even_profile(level: float, n_seg: int, seg_lengths: np.ndarray,
                  **kwargs) -> np.ndarray:
    """Constant speed = level for all segments."""
    return np.full(n_seg, level)


def _fast_start_profile(level: float, n_seg: int, seg_lengths: np.ndarray,
                        aggression_pct: float = 3.0,
                        n_fast_segments: int = 4, **kwargs) -> np.ndarray:
    """Front-loaded: first N segments at level*(1+agg%), rest compensated."""
    n_fast = min(n_fast_segments, n_seg)
    boost = level * (aggression_pct / 100.0)

    speeds = np.full(n_seg, level)
    speeds[:n_fast] += boost

    # Compensate later segments to maintain same overall time budget
    fast_time = (seg_lengths[:n_fast] / speeds[:n_fast]).sum()
    target_total_time = seg_lengths.sum() / level
    slow_time = target_total_time - fast_time
    if slow_time > 0 and n_seg > n_fast:
        speeds[n_fast:] = seg_lengths[n_fast:].sum() / slow_time

    return speeds


def _negative_ramp_profile(level: float, n_seg: int,
                           seg_lengths: np.ndarray, **kwargs) -> np.ndarray:
    """Linearly accelerating from ~2% below level to ~2% above."""
    ramp = np.linspace(-0.02, 0.02, n_seg)
    speeds = level * (1.0 + ramp)
    # Normalize to preserve target elapsed time at 'level' avg speed
    total_time = (seg_lengths / speeds).sum()
    target_time = seg_lengths.sum() / level
    speeds *= total_time / target_time
    return speeds


def _negative_kick_profile(level: float, n_seg: int,
                           seg_lengths: np.ndarray,
                           kick_distance_m: float = 300.0,
                           kick_boost_pct: float = 5.0, **kwargs) -> np.ndarray:
    """Steady pace then kick for final N metres."""
    cum_dist = np.cumsum(seg_lengths)
    total_dist = cum_dist[-1]
    kick_start = total_dist - kick_distance_m
    kick_mask = np.array([
        (cum_dist[i] - seg_lengths[i]) >= kick_start for i in range(n_seg)
    ])

    speeds = np.full(n_seg, level)
    kick_boost = level * (kick_boost_pct / 100.0)
    speeds[kick_mask] += kick_boost

    # Compensate steady segments
    steady_mask = ~kick_mask
    if steady_mask.any():
        target_total_time = seg_lengths.sum() / level
        kick_time = (seg_lengths[kick_mask] / speeds[kick_mask]).sum()
        steady_time = target_total_time - kick_time
        speeds[steady_mask] = seg_lengths[steady_mask].sum() / steady_time

    return speeds


# ══════════════════════════════════════════════════════════════════════
# 3. Find maximum sustainable speed for a shaped profile
# ══════════════════════════════════════════════════════════════════════

def find_max_sustainable_speed(
    profile_fn,
    cs_ms: float,
    d_prime_m: float,
    event_distance_m: float,
    segment_length_m: float,
    drafting_config: pd.DataFrame,
    height_m: float,
    mass_kg: float,
    air_density: float = 1.225,
    shoe_re_delta_pct: float = 0.0,
    dprime_method: str = "bellenger",
    speed_range: tuple = (5.0, 9.0),
    **profile_kwargs,
) -> Dict[str, Any]:
    """Find the speed level that minimizes finish time (fully spending D′).

    Uses bounded scalar optimization. The D′ depletion enforcement in
    simulate_race creates a U-shaped time-vs-speed curve: as speed level
    increases, time decreases until D′ runs out mid-race and pace gets
    clamped to CS, at which point time increases again. The minimum of
    this curve is the optimal performance point.
    """
    seg_df = generate_segments(event_distance_m, segment_length_m)
    n_seg = len(seg_df)
    seg_lengths = seg_df["length_m"].values
    drafting = drafting_config.reset_index(drop=True)

    def _objective(level: float) -> float:
        speeds = profile_fn(level, n_seg, seg_lengths, **profile_kwargs)
        script = drafting.copy()
        script["segment_speed_ms"] = speeds
        script = script[["segment_speed_ms", "pos_of_n", "gap_m", "is_curve_offset"]]

        try:
            result = simulate_race(
                event_distance_m=event_distance_m,
                segment_length_m=segment_length_m,
                cs_ms=cs_ms,
                d_prime_m=d_prime_m,
                race_script=script,
                height_m=height_m,
                mass_kg=mass_kg,
                air_density=air_density,
                shoe_re_delta_pct=shoe_re_delta_pct,
                dprime_method=dprime_method,
            )
            return result["finish_time_s"]
        except Exception:
            return 999.0

    lo, hi = speed_range
    opt = minimize_scalar(_objective, bounds=(lo, hi), method="bounded",
                          options={"xatol": 1e-5, "maxiter": 200})

    best_level = opt.x
    # Run final simulation at optimal level
    speeds = profile_fn(best_level, n_seg, seg_lengths, **profile_kwargs)
    final_script = drafting.copy()
    final_script["segment_speed_ms"] = speeds
    final_script = final_script[["segment_speed_ms", "pos_of_n", "gap_m", "is_curve_offset"]]

    best_result = simulate_race(
        event_distance_m=event_distance_m,
        segment_length_m=segment_length_m,
        cs_ms=cs_ms,
        d_prime_m=d_prime_m,
        race_script=final_script,
        height_m=height_m,
        mass_kg=mass_kg,
        air_density=air_density,
        shoe_re_delta_pct=shoe_re_delta_pct,
        dprime_method=dprime_method,
    )

    return {
        "optimal_level": best_level,
        "finish_time_s": best_result["finish_time_s"],
        "finish_time_str": best_result["summary"]["finish_time_str"],
        "race_script": final_script,
        "simulation_result": best_result,
        "d_prime_final": best_result["summary"]["d_prime_final_m"],
    }


# ══════════════════════════════════════════════════════════════════════
# 4. Free-form optimizer (differential evolution)
# ══════════════════════════════════════════════════════════════════════

def optimize_pacing(
    cs_ms: float,
    d_prime_m: float,
    event_distance_m: float,
    segment_length_m: float,
    drafting_config: pd.DataFrame,
    height_m: float,
    mass_kg: float,
    air_density: float = 1.225,
    shoe_re_delta_pct: float = 0.0,
    speed_bounds: tuple = (5.5, 8.0),
    max_delta_ms: float = 0.4,
    dprime_method: str = "bellenger",
    seed: int = 42,
    maxiter: int = 300,
    tol: float = 1e-4,
    polish: bool = True,
) -> Dict[str, Any]:
    """Find the per-segment speed profile that minimizes finish time.

    Uses differential_evolution with a smoothness constraint
    (max segment-to-segment speed change) to produce realistic profiles.
    D′ depletion enforcement naturally caps unsustainable pacing.
    """
    seg_df = generate_segments(event_distance_m, segment_length_m)
    n_seg = len(seg_df)
    drafting = drafting_config.reset_index(drop=True)

    bounds = [speed_bounds] * n_seg

    def objective(speeds: np.ndarray) -> float:
        # Smoothness constraint: reject implausible speed spikes
        if np.any(np.abs(np.diff(speeds)) > max_delta_ms):
            return 999.0

        script = drafting.copy()
        script["segment_speed_ms"] = speeds
        script = script[["segment_speed_ms", "pos_of_n", "gap_m", "is_curve_offset"]]

        try:
            result = simulate_race(
                event_distance_m=event_distance_m,
                segment_length_m=segment_length_m,
                cs_ms=cs_ms,
                d_prime_m=d_prime_m,
                race_script=script,
                height_m=height_m,
                mass_kg=mass_kg,
                air_density=air_density,
                shoe_re_delta_pct=shoe_re_delta_pct,
                dprime_method=dprime_method,
            )
            return result["finish_time_s"]
        except Exception:
            return 999.0

    opt_result = differential_evolution(
        objective,
        bounds=bounds,
        seed=seed,
        maxiter=maxiter,
        tol=tol,
        polish=polish,
        disp=False,
        workers=1,
    )

    optimal_speeds = opt_result.x
    final_script = drafting.copy()
    final_script["segment_speed_ms"] = optimal_speeds
    final_script = final_script[["segment_speed_ms", "pos_of_n", "gap_m", "is_curve_offset"]]

    final_result = simulate_race(
        event_distance_m=event_distance_m,
        segment_length_m=segment_length_m,
        cs_ms=cs_ms,
        d_prime_m=d_prime_m,
        race_script=final_script,
        height_m=height_m,
        mass_kg=mass_kg,
        air_density=air_density,
        shoe_re_delta_pct=shoe_re_delta_pct,
        dprime_method=dprime_method,
    )

    return {
        "optimal_speeds": optimal_speeds,
        "finish_time_s": final_result["finish_time_s"],
        "finish_time_str": final_result["summary"]["finish_time_str"],
        "race_script": final_script,
        "simulation_result": final_result,
        "d_prime_final": final_result["summary"]["d_prime_final_m"],
        "optimizer_result": opt_result,
    }


# ══════════════════════════════════════════════════════════════════════
# 5. Scenario matrix — each scenario finds its own optimal performance
# ══════════════════════════════════════════════════════════════════════

def run_scenario_matrix(
    scenarios: Dict[str, Dict[str, Any]],
    cs_ms: float,
    d_prime_m: float,
    event_distance_m: float = 1609.34,
    segment_length_m: float = 100.0,
    height_m: float = 1.84,
    mass_kg: float = 72.6,
    air_density: float = 1.225,
    dprime_method: str = "bellenger",
) -> Dict[str, Dict[str, Any]]:
    """Run multiple scenarios, each finding its fastest achievable time.

    Every scenario fully depletes D′. The comparison shows how much
    faster the athlete can go under different conditions.

    Parameters
    ----------
    scenarios : dict
        Keys are scenario labels, values are dicts with:
            strategy : str ('even', 'fast_start', 'negative_ramp',
                           'negative_kick', 'optimized')
            formation : str (e.g. '2of5', '1of1')
            gap_m : float
            shoe_pct : float (RE delta %)
            no_draft_final_m : float (default 400)
            aggression_pct : float (for fast_start, default 3.0)
            n_fast_segments : int (for fast_start, default 4)
            kick_distance_m : float (for negative_kick, default 300)
            kick_boost_pct : float (for negative_kick, default 5.0)
            speed_bounds : tuple (for optimized, default (5.5, 8.0))
            max_delta_ms : float (for optimized, default 0.4)
            air_density : float (optional override)
            curve_offset_pct : float (% of bend segments with offset, 0-100)
    """
    seg_df = generate_segments(event_distance_m, segment_length_m)
    n_seg = len(seg_df)

    PROFILE_MAP = {
        "even": _even_profile,
        "fast_start": _fast_start_profile,
        "negative_ramp": _negative_ramp_profile,
        "negative_kick": _negative_kick_profile,
    }

    results = {}

    for label, cfg in scenarios.items():
        strategy = cfg.get("strategy", "even")
        formation = cfg.get("formation", "1of1")
        gap_m = cfg.get("gap_m", 1.3)
        shoe_pct = cfg.get("shoe_pct", 0.0)
        no_draft_final = cfg.get("no_draft_final_m", 400.0)
        local_air = cfg.get("air_density", air_density)
        curve_offset_pct = cfg.get("curve_offset_pct", 0.0)

        # Build drafting config
        drafting = build_drafting_config(
            n_segments=n_seg,
            formation=formation,
            gap_m=gap_m,
            no_draft_final_m=no_draft_final,
            event_distance_m=event_distance_m,
            segment_length_m=segment_length_m,
        )

        # Apply curve offset to bend segments
        if curve_offset_pct > 0:
            bend_indices = seg_df.index[seg_df["is_bend"]].tolist()
            n_offset = max(1, int(round(len(bend_indices) * curve_offset_pct / 100.0)))
            for idx in bend_indices[:n_offset]:
                drafting.loc[idx, "is_curve_offset"] = 1

        common_kwargs = dict(
            cs_ms=cs_ms,
            d_prime_m=d_prime_m,
            event_distance_m=event_distance_m,
            segment_length_m=segment_length_m,
            drafting_config=drafting,
            height_m=height_m,
            mass_kg=mass_kg,
            air_density=local_air,
            shoe_re_delta_pct=shoe_pct,
            dprime_method=dprime_method,
        )

        if strategy == "optimized":
            speed_bounds = cfg.get("speed_bounds", (5.5, 8.0))
            max_delta = cfg.get("max_delta_ms", 0.4)
            res = optimize_pacing(
                **common_kwargs,
                speed_bounds=speed_bounds,
                max_delta_ms=max_delta,
            )
        else:
            profile_fn = PROFILE_MAP[strategy]
            profile_kwargs = {}
            if strategy == "fast_start":
                profile_kwargs["aggression_pct"] = cfg.get("aggression_pct", 3.0)
                profile_kwargs["n_fast_segments"] = cfg.get("n_fast_segments", 4)
            elif strategy == "negative_kick":
                profile_kwargs["kick_distance_m"] = cfg.get("kick_distance_m", 300.0)
                profile_kwargs["kick_boost_pct"] = cfg.get("kick_boost_pct", 5.0)

            res = find_max_sustainable_speed(
                profile_fn=profile_fn,
                **common_kwargs,
                **profile_kwargs,
            )

        results[label] = {
            "simulation_result": res["simulation_result"],
            "race_script": res["race_script"],
            "finish_time_s": res["finish_time_s"],
            "finish_time_str": res["finish_time_str"],
            "d_prime_final": res["d_prime_final"],
            "n_clamped": res["simulation_result"]["summary"]["n_segments_clamped"],
            "avg_re_improvement": res["simulation_result"]["summary"]["avg_re_improvement"],
        }

    return results


# ══════════════════════════════════════════════════════════════════════
# 6. Robustness check (MC on optimized/chosen script)
# ══════════════════════════════════════════════════════════════════════

def robustness_check(
    race_script: pd.DataFrame,
    cs_ms: float,
    d_prime_m: float,
    event_distance_m: float,
    segment_length_m: float,
    height_m: float,
    mass_kg: float,
    priors: pd.DataFrame,
    air_density: float = 1.225,
    shoe_re_delta_pct: float = 0.0,
    n_samples: int = 500,
    target_time_s: float | None = None,
    dprime_method: str = "bellenger",
    seed: int = 42,
) -> Dict[str, Any]:
    """Run Monte Carlo on a specific race script to quantify robustness."""
    base_params = {
        "CS": cs_ms,
        "D_prime": d_prime_m,
        "drafting_scale": 1.0,
        "shoe_re_delta_pct": shoe_re_delta_pct,
        "air_density_kgm3": air_density,
    }

    return run_monte_carlo(
        n_samples=n_samples,
        event_distance_m=event_distance_m,
        segment_length_m=segment_length_m,
        race_script=race_script,
        height_m=height_m,
        mass_kg=mass_kg,
        priors=priors,
        base_params=base_params,
        target_time_s=target_time_s,
        dprime_method=dprime_method,
        seed=seed,
    )


# ══════════════════════════════════════════════════════════════════════
# 7. Excel output writer
# ══════════════════════════════════════════════════════════════════════

def write_optimal_strategy(
    workbook_path: str,
    race_script: pd.DataFrame,
    scenario_label: str = "optimal_strategy",
    sheet_name: str = "optimal_strategy",
) -> None:
    """Append the optimal race script as a new sheet in the workbook."""
    from openpyxl import load_workbook as _load_wb

    output = race_script.copy()
    output.insert(0, "segment_idx", range(1, len(output) + 1))
    output["scenario"] = scenario_label

    wb = _load_wb(workbook_path)
    if sheet_name in wb.sheetnames:
        del wb[sheet_name]
    wb.save(workbook_path)

    with pd.ExcelWriter(workbook_path, engine="openpyxl", mode="a",
                        if_sheet_exists="replace") as writer:
        output.to_excel(writer, sheet_name=sheet_name, index=False)
