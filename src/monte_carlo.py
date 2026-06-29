"""
monte_carlo.py — Monte Carlo uncertainty propagation for race simulation.

Why Monte Carlo?
----------------
The D′-balance race model chains several sub-models, each with uncertainty:

1. **CS & D′** — Derived from a small number of historical races whose
   conditions (drafting, shoes, altitude, fitness timing) are imperfectly
   known.  Bellenger reports SEE of ±0.04 m/s for CS and ±12 m for D′.

2. **Drafting factor** — CFD studies (Beaumont, Schickhofer) estimate drag
   reduction under idealised conditions.  Real-race drafting is dynamic:
   formations shift, gaps vary, lateral offsets occur.  Bellenger notes the
   range is 1.1 %–2.9 % equivalent speed; using a fixed 2.18 % is an
   approximation.

3. **Metabolic chain nonlinearity** — Kipp et al. show that the VO₂–velocity
   curve is curvilinear.  Small errors in drag reduction compound through
   the cubic air-resistance term and the nonlinear inverse solve.

4. **Shoe RE delta** — Measured group means (e.g. 4 % for Vaporfly) have
   individual SD ≈ 1–2 pp; the benefit for a specific athlete is uncertain.

5. **Environment** — Air density varies with altitude, temperature, humidity.
   A ±0.02 kg/m³ change alters the aero VO₂ component.

Monte Carlo samples from these priors, runs the full simulation for each
draw, and returns a distribution of finish times with sensitivity metrics.

References
----------
- Bellenger et al. (2026): Acknowledgment of Limitations section
- Kipp et al. (2019): curvilinear VO₂–velocity and compounding uncertainty
- Beaumont et al. (2021): range of drafting benefits across formations
"""

from __future__ import annotations
import numpy as np
import pandas as pd
from typing import Dict, Any, List
from scipy import stats as sp_stats

from .simulation import simulate_race


def run_monte_carlo(
    n_samples: int,
    event_distance_m: float,
    segment_length_m: float,
    race_script: pd.DataFrame,
    height_m: float,
    mass_kg: float,
    priors: pd.DataFrame,
    base_params: Dict[str, float] | None = None,
    target_time_s: float | None = None,
    dprime_method: str = "bellenger",
    seed: int | None = 42,
) -> Dict[str, Any]:
    """Run Monte Carlo simulation over uncertain parameters.

    Parameters
    ----------
    n_samples : int
        Number of Monte Carlo draws.
    event_distance_m, segment_length_m : float
        Event configuration.
    race_script : DataFrame
        Per-segment speed plan (same format as ``simulate_race``).
    height_m, mass_kg : float
        Athlete anthropometrics (held constant across samples).
    priors : DataFrame
        Columns: parameter, mean, std, distribution [, lower_bound, upper_bound].
        Supported distributions: 'normal', 'uniform', 'truncnorm'.
        Recognised parameter names:
            CS, D_prime, drafting_scale, shoe_re_delta_pct, air_density_kgm3
    base_params : dict, optional
        Default values for parameters NOT in the priors sheet.  Keys match
        the parameter names above.
    target_time_s : float, optional
        If provided, the output includes the probability of beating this time.
    dprime_method : str
        'bellenger' or 'skiba'.
    seed : int or None
        Random seed for reproducibility.

    Returns
    -------
    dict with keys:
        finish_times   – array of simulated finish times (s)
        samples_df     – DataFrame of all sampled parameter values
        stats          – dict of summary statistics (mean, median, std, CI)
        prob_target    – probability of beating target_time_s (if provided)
        sensitivity    – DataFrame of Spearman correlations (input → finish time)
    """
    rng = np.random.default_rng(seed)

    # ── 1. Set up defaults ────────────────────────────────────────────
    defaults = {
        "CS":                  6.0,
        "D_prime":             200.0,
        "drafting_scale":      1.0,
        "shoe_re_delta_pct":   0.0,
        "air_density_kgm3":    1.225,
    }
    if base_params:
        defaults.update(base_params)

    # ── 2. Sample from priors ─────────────────────────────────────────
    sampled = {name: np.full(n_samples, defaults[name]) for name in defaults}

    for _, row in priors.iterrows():
        name = str(row["parameter"]).strip()
        if name not in defaults:
            continue  # skip unknown parameters

        mean = float(row["mean"])
        std  = float(row["std"])
        dist = str(row.get("distribution", "normal")).strip().lower()
        lb   = float(row["lower_bound"]) if "lower_bound" in row and pd.notna(row.get("lower_bound")) else -np.inf
        ub   = float(row["upper_bound"]) if "upper_bound" in row and pd.notna(row.get("upper_bound")) else  np.inf

        if dist == "normal":
            draws = rng.normal(mean, std, n_samples)
            draws = np.clip(draws, lb, ub)
        elif dist == "uniform":
            draws = rng.uniform(lb, ub, n_samples)
        elif dist == "truncnorm":
            a = (lb - mean) / std if std > 0 else -10
            b = (ub - mean) / std if std > 0 else  10
            draws = sp_stats.truncnorm.rvs(a, b, loc=mean, scale=std,
                                            size=n_samples, random_state=rng)
        else:
            draws = rng.normal(mean, std, n_samples)
            draws = np.clip(draws, lb, ub)

        sampled[name] = draws

    samples_df = pd.DataFrame(sampled)

    # ── 3. Run simulations ────────────────────────────────────────────
    finish_times = np.zeros(n_samples)

    for k in range(n_samples):
        try:
            result = simulate_race(
                event_distance_m=event_distance_m,
                segment_length_m=segment_length_m,
                cs_ms=sampled["CS"][k],
                d_prime_m=sampled["D_prime"][k],
                race_script=race_script,
                height_m=height_m,
                mass_kg=mass_kg,
                air_density=sampled["air_density_kgm3"][k],
                shoe_re_delta_pct=sampled["shoe_re_delta_pct"][k],
                drafting_scale=sampled["drafting_scale"][k],
                dprime_method=dprime_method,
            )
            finish_times[k] = result["finish_time_s"]
        except Exception:
            finish_times[k] = np.nan

    # ── 4. Statistics ─────────────────────────────────────────────────
    valid = finish_times[~np.isnan(finish_times)]
    stats = {
        "mean_s":     float(np.mean(valid)),
        "median_s":   float(np.median(valid)),
        "std_s":      float(np.std(valid, ddof=1)) if len(valid) > 1 else 0.0,
        "ci_95_lo_s": float(np.percentile(valid, 2.5)),
        "ci_95_hi_s": float(np.percentile(valid, 97.5)),
        "min_s":      float(np.min(valid)),
        "max_s":      float(np.max(valid)),
        "n_valid":    len(valid),
        "n_failed":   int(np.isnan(finish_times).sum()),
    }

    # Probability of hitting target
    prob = None
    if target_time_s is not None and len(valid) > 0:
        prob = float(np.mean(valid <= target_time_s))

    # ── 5. Sensitivity (Spearman rank correlations) ───────────────────
    sens_rows = []
    for param in defaults:
        vals = sampled[param]
        mask = ~np.isnan(finish_times)
        if np.std(vals[mask]) < 1e-12 or np.std(finish_times[mask]) < 1e-12:
            rho = 0.0
        else:
            import warnings as _w
            with _w.catch_warnings():
                _w.simplefilter("ignore")
                rho, _ = sp_stats.spearmanr(vals[mask], finish_times[mask])
        sens_rows.append({"parameter": param, "spearman_rho": rho,
                          "abs_rho": abs(rho)})

    sensitivity_df = (pd.DataFrame(sens_rows)
                        .sort_values("abs_rho", ascending=False)
                        .reset_index(drop=True))

    return {
        "finish_times":  finish_times,
        "samples_df":    samples_df,
        "stats":         stats,
        "prob_target":   prob,
        "sensitivity":   sensitivity_df,
    }
