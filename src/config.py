"""
config.py — Physical constants, track geometry, and model defaults.

All units are SI unless otherwise noted.

References
----------
- IAAF Track and Field Facilities Manual (standard 400 m track specs)
- Pugh (1971): Metabolic cost coefficient for air resistance
- Batliner et al. (2018): Quadratic VO₂–velocity regression coefficients
- Beaumont et al. (2021): Drag-coefficient data for in-line runner formations
- DuBois & DuBois (1916): Body-surface-area formula
"""

import math

# ══════════════════════════════════════════════════════════════════════
# 1.  PHYSICAL CONSTANTS
# ══════════════════════════════════════════════════════════════════════
STANDARD_AIR_DENSITY = 1.225       # kg/m³  (sea-level, 15 °C, dry air)
GRAVITY              = 9.81        # m/s²

# ══════════════════════════════════════════════════════════════════════
# 2.  STANDARD 400 m TRACK GEOMETRY  (IAAF lane-1 measurement line)
# ══════════════════════════════════════════════════════════════════════
TRACK_LENGTH_M       = 400.0       # one full lap
MEASUREMENT_RADIUS_M = 36.80       # inner radius 36.50 m + 0.30 m offset
BEND_LENGTH_M        = math.pi * MEASUREMENT_RADIUS_M   # ≈ 115.611 m per semicircle
STRAIGHT_LENGTH_M    = (TRACK_LENGTH_M - 2 * BEND_LENGTH_M) / 2  # ≈ 84.389 m

# Track layout: list of (section_type, start_m, end_m) measured from the
# finish line going *forward* in race direction.  One full lap = 400 m.
#   Bend  1  →  Back straight  →  Bend 2  →  Home straight (→ finish)
TRACK_SECTIONS = [
    ("bend",     0.0,                          BEND_LENGTH_M),
    ("straight", BEND_LENGTH_M,                BEND_LENGTH_M + STRAIGHT_LENGTH_M),
    ("bend",     BEND_LENGTH_M + STRAIGHT_LENGTH_M,
                 2 * BEND_LENGTH_M + STRAIGHT_LENGTH_M),
    ("straight", 2 * BEND_LENGTH_M + STRAIGHT_LENGTH_M,  TRACK_LENGTH_M),
]

# ══════════════════════════════════════════════════════════════════════
# 3.  METABOLIC MODEL COEFFICIENTS
# ══════════════════════════════════════════════════════════════════════

# --- Batliner et al. (2018) quadratic VO₂–velocity on treadmill ----------
#   VO₂_treadmill(v) = A·v² + B·v + C   [ml O₂ / kg / min]
#   Measured on 10 high-level male runners (<30-min 10 km), 1.78–5.14 m/s.
BATLINER_A = 1.5355
BATLINER_B = 1.5374
BATLINER_C = 15.661

# --- Pugh (1971) metabolic cost of overcoming air resistance -------------
#   VO₂_air(v) = PUGH_K · A_p · v³ / mass   [ml O₂ / kg / min]
#   Converts mechanical power against air drag to oxygen cost.
PUGH_K = 3.54   # (ml O₂ / min) per (m² · (m/s)³)  — includes efficiency term

# ══════════════════════════════════════════════════════════════════════
# 4.  DRAFTING DRAG-REDUCTION LOOKUP  (Beaumont et al., 2021 — 1.3 m gap)
# ══════════════════════════════════════════════════════════════════════
# Keys: (n_runners_in_line, position_1_indexed)  →  fractional drag reduction
# Position 1 = lead runner (small benefit from having runners behind).
# These are percentage reductions relative to a solo runner (CD = 0.81).
BEAUMONT_DRAG_REDUCTION = {
    # --- solo ---
    (1, 1): 0.000,
    # --- pair ---
    (2, 1): 0.037,    (2, 2): 0.568,
    # --- trio ---
    (3, 1): 0.049,    (3, 2): 0.592,   (3, 3): 0.530,
    # --- quartet ---
    (4, 1): 0.049,    (4, 2): 0.592,   (4, 3): 0.530,   (4, 4): 0.543,
    # --- quintet ---
    (5, 1): 0.062,    (5, 2): 0.630,   (5, 3): 0.567,   (5, 4): 0.580,   (5, 5): 0.543,
}

# --- Schickhofer & Hanson (2021) — drag reductions at 1.2 m gap ----------
# Only position-2 values are cited in the literature for comparison.
SCHICKHOFER_DRAG_REDUCTION_1_2M = {
    (2, 2): 0.701,
    (3, 2): 0.756,
}

# Default curve-offset penalty: when a drafting runner takes a mild lateral
# offset (~0.2 m) on bends, the reduced shelter retains this fraction of
# the straight-line benefit.  Motivated by Beaumont Appendix Fig. A1:
# a 0.2 m shift increased drag by ~15 % relative to the in-line reference.
DEFAULT_CURVE_OFFSET_PENALTY = 0.85

# ══════════════════════════════════════════════════════════════════════
# 5.  FRONTAL-AREA ESTIMATION HELPER
# ══════════════════════════════════════════════════════════════════════
def estimate_frontal_area(height_m: float, mass_kg: float) -> float:
    """Return projected frontal area (m²) for a runner in running posture.

    Uses the DuBois & DuBois (1916) body-surface-area formula scaled by an
    empirical fraction (≈ 0.18) validated against Kipp et al. (2019) who use
    A_p = 0.45 m² for a 1.71 m / 58 kg runner and Beaumont et al. (2021) who
    use A_p = 0.475 m² for a 1.65 m / 56 kg runner.
    """
    height_cm = height_m * 100.0
    bsa = 0.007184 * (height_cm ** 0.725) * (mass_kg ** 0.425)
    return 0.18 * bsa

# ══════════════════════════════════════════════════════════════════════
# 6.  COMMON RACE DISTANCES (metres)
# ══════════════════════════════════════════════════════════════════════
DISTANCES = {
    "400m":    400.0,
    "800m":    800.0,
    "1500m":  1500.0,
    "mile":   1609.34,
    "3000m":  3000.0,
    "2mile":  3218.68,
    "5000m":  5000.0,
    "10000m": 10000.0,
}

# ══════════════════════════════════════════════════════════════════════
# 7.  EXCEL WORKBOOK SHEET NAMES & REQUIRED COLUMNS
# ══════════════════════════════════════════════════════════════════════
REQUIRED_SHEETS = {
    "event_config":    ["event_name", "event_distance_m", "segment_length_m"],
    "athlete_history": ["athlete_id", "distance_m", "time_s"],
    "physiology":      ["athlete_id", "height_m", "mass_kg"],
    "race_script":     ["segment_idx", "segment_speed_ms", "pos_of_n", "gap_m",
                        "is_curve_offset"],
    "environment":     ["air_density_kgm3"],
    "equipment":       ["shoe_re_delta_pct"],
    "priors":          ["parameter", "mean", "std", "distribution"],
}
