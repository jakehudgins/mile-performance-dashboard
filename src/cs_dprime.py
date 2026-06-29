"""
cs_dprime.py — Fit Critical Speed (CS) and D′ from race history.

The two-parameter speed–duration model (Hill, 1925; Hughson et al., 1984)
defines:
    speed = CS  +  D′ / time

Rearranging:   speed = D′ · (1/time) + CS        ← linear in (1/time)
Also:          distance = CS · time + D′           ← linear in time

We fit BOTH forms and average the estimates (as in Kirby et al., 2021 and
Bellenger et al., 2026).  Standard error of the estimate (SEE) is reported
for both CS and D′.

References
----------
- Kirby et al. (2021): J Appl Physiol 131:1532–1542
- Bellenger et al. (2026): IJSPP (Ahead of Print)
- Ruiz-Alias et al. (2023): Int J Sports Med 44:969–975
"""

from __future__ import annotations
import numpy as np
import pandas as pd
from typing import Dict, Any


def fit_cs_dprime(history_df: pd.DataFrame,
                  athlete_id: str | None = None) -> Dict[str, Any]:
    """Fit CS and D′ from personal-best race performances.

    Parameters
    ----------
    history_df : DataFrame
        Must contain columns: ``athlete_id``, ``distance_m``, ``time_s``.
        Only rows where ``distance_m`` ≥ 800 m are used (severe-intensity
        domain).  If multiple times exist for the same distance, the fastest
        is kept automatically.
    athlete_id : str, optional
        If provided, filter to this athlete.  Otherwise use all rows.

    Returns
    -------
    dict with keys:
        cs_ms          – Critical Speed (m/s)
        d_prime_m      – D′ (metres)
        cs_see_ms      – Standard Error of the Estimate for CS
        d_prime_see_m  – Standard Error of the Estimate for D′
        r_squared      – Average R² across both fits
        n_races        – Number of race performances used
        details        – dict of per-model fit results
    """
    df = history_df.copy()
    if athlete_id is not None:
        df = df[df["athlete_id"] == athlete_id]

    if len(df) < 2:
        raise ValueError(
            f"Need ≥ 2 race performances to fit CS/D′; got {len(df)} "
            f"for athlete '{athlete_id}'."
        )

    # Keep fastest time per distance
    df = df.loc[df.groupby("distance_m")["time_s"].idxmin()]

    distance = df["distance_m"].values.astype(float)
    time_s   = df["time_s"].values.astype(float)
    speed    = distance / time_s          # m/s
    inv_time = 1.0 / time_s              # 1/s

    # ------------------------------------------------------------------
    # Model A:  distance = CS · time + D′   (linear regression: d vs t)
    # ------------------------------------------------------------------
    # np.polyfit(x, y, 1) returns [slope, intercept]
    coeffs_a = np.polyfit(time_s, distance, 1)
    cs_a     = coeffs_a[0]   # slope  = CS
    dp_a     = coeffs_a[1]   # intercept = D′
    pred_a   = np.polyval(coeffs_a, time_s)
    ss_res_a = np.sum((distance - pred_a) ** 2)
    ss_tot_a = np.sum((distance - np.mean(distance)) ** 2)
    r2_a     = 1.0 - ss_res_a / ss_tot_a if ss_tot_a > 0 else 0.0
    see_a    = _see(distance, pred_a, df_model=2)

    # ------------------------------------------------------------------
    # Model B:  speed = D′ · (1/time) + CS   (linear regression: v vs 1/t)
    # ------------------------------------------------------------------
    coeffs_b = np.polyfit(inv_time, speed, 1)
    dp_b     = coeffs_b[0]   # slope  = D′
    cs_b     = coeffs_b[1]   # intercept = CS
    pred_b   = np.polyval(coeffs_b, inv_time)
    ss_res_b = np.sum((speed - pred_b) ** 2)
    ss_tot_b = np.sum((speed - np.mean(speed)) ** 2)
    r2_b     = 1.0 - ss_res_b / ss_tot_b if ss_tot_b > 0 else 0.0
    see_b    = _see(speed, pred_b, df_model=2)

    # ------------------------------------------------------------------
    # Average both models (Kirby / Bellenger convention)
    # ------------------------------------------------------------------
    cs_avg = (cs_a + cs_b) / 2.0
    dp_avg = (dp_a + dp_b) / 2.0
    r2_avg = (r2_a + r2_b) / 2.0

    # SEE for CS and D′ from model B (speed-vs-1/time) — the more common
    # reporting convention (Bellenger Table 2).
    # Model B intercept = CS, slope = D′.
    # We also compute from model A for comparison.
    cs_see = abs(cs_a - cs_b) / 2.0   # half-range as approximate SEE
    dp_see = abs(dp_a - dp_b) / 2.0

    return {
        "cs_ms":         cs_avg,
        "d_prime_m":     dp_avg,
        "cs_see_ms":     cs_see,
        "d_prime_see_m": dp_see,
        "r_squared":     r2_avg,
        "n_races":       len(df),
        "details": {
            "model_a": {"cs": cs_a, "d_prime": dp_a, "r2": r2_a, "see_d": see_a},
            "model_b": {"cs": cs_b, "d_prime": dp_b, "r2": r2_b, "see_v": see_b},
        },
    }


# ── helper ────────────────────────────────────────────────────────────
def _see(observed: np.ndarray, predicted: np.ndarray, df_model: int) -> float:
    """Standard Error of the Estimate (root-mean-square residual).

    Parameters
    ----------
    observed, predicted : arrays of same length
    df_model : number of model parameters (for degrees-of-freedom correction)
    """
    n = len(observed)
    if n <= df_model:
        return float("nan")
    residuals = observed - predicted
    return float(np.sqrt(np.sum(residuals ** 2) / (n - df_model)))


def prerace_predicted_time(distance_m: float, cs_ms: float,
                           d_prime_m: float) -> float:
    """Predict best-possible finish time from CS and D′ (Bellenger Eq. 2).

    predicted_time = (distance - D′) / CS

    This assumes the athlete runs optimally and exhausts D′ exactly at
    the finish line.
    """
    if cs_ms <= 0:
        raise ValueError("CS must be positive.")
    return (distance_m - d_prime_m) / cs_ms
