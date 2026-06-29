"""
aerodynamics.py — Drafting drag-reduction model.

Maps per-segment user inputs (position in line, gap distance, curve offset)
to a fractional drag reduction (0 = no benefit, 1 = all drag removed).

The model blends two CFD data sources:
    • Beaumont et al. (2021) — 1.3 m gap, groups of 2–5 in-line runners
    • Schickhofer & Hanson (2021) — 1.2 m gap, groups of 2–3

Gap sensitivity is modelled as a linear scaling factor derived from the two
reference gaps.  Curve-offset penalty (default 0.85) is based on Beaumont's
Appendix showing ~15 % increase in drag for a 0.2 m lateral shift.

References
----------
- Beaumont et al. (2021): Sports Biomechanics 23:11, 2180–2195
- Schickhofer & Hanson (2021): J Biomechanics 122, 110457
- Bellenger et al. (2026): IJSPP (Ahead of Print) — amalgamated 2.18 % factor
"""

from __future__ import annotations
import re
from typing import Tuple

from .config import (
    BEAUMONT_DRAG_REDUCTION,
    SCHICKHOFER_DRAG_REDUCTION_1_2M,
    DEFAULT_CURVE_OFFSET_PENALTY,
)


# ══════════════════════════════════════════════════════════════════════
# Public API
# ══════════════════════════════════════════════════════════════════════

def parse_pos_of_n(pos_of_n: str) -> Tuple[int, int]:
    """Parse a position string like '2of5' → (position=2, n_runners=5).

    Also accepts '1of1' (solo, no drafting) and formats like '3/5', '3_of_5'.
    """
    s = str(pos_of_n).strip().lower().replace(" ", "")
    # Try patterns: "2of5", "2/5", "2_of_5", "2-of-5"
    m = re.match(r"(\d+)\s*(?:of|/|_of_|-of-)\s*(\d+)", s)
    if m:
        pos = int(m.group(1))
        n   = int(m.group(2))
    else:
        raise ValueError(
            f"Cannot parse pos_of_n='{pos_of_n}'. "
            "Expected format like '2of5', '2/5', or '1of1'."
        )
    if pos < 1 or pos > n:
        raise ValueError(f"Position {pos} out of range for {n} runners.")
    return pos, n


def drag_reduction_prior(
    pos_of_n: str,
    gap_m: float = 1.3,
    drafting_scale: float = 1.0,
) -> float:
    """Compute the fractional drag reduction for a runner in formation.

    Parameters
    ----------
    pos_of_n : str
        Position and formation size, e.g. '2of5'.  '1of1' = solo (no benefit).
    gap_m : float
        Axial gap to the runner directly ahead (metres).  Beaumont reference
        gap = 1.3 m.  The model adjusts for different gaps using Schickhofer
        data at 1.2 m.  Ignored for the lead runner (position 1).
    drafting_scale : float
        Optional global multiplier on the drafting benefit (useful for
        Monte Carlo sensitivity analysis).  Default = 1.0.

    Returns
    -------
    r_drag : float in [0, 1)
        Fractional drag reduction.  0 = no benefit; 0.63 = 63 % less drag.
    """
    pos, n = parse_pos_of_n(pos_of_n)

    if n == 1 or (pos == 1 and n == 1):
        return 0.0  # solo runner

    # ── Step 1: look up Beaumont base value at 1.3 m gap ──────────────
    r_base = _beaumont_lookup(pos, n)

    # ── Step 2: adjust for gap distance ───────────────────────────────
    if pos > 1:
        # Only non-leaders are affected by the gap to the runner ahead
        gap_factor = _gap_adjustment_factor(n, gap_m)
        r_adjusted = r_base * gap_factor
    else:
        # Lead runner benefit is small and not gap-dependent
        r_adjusted = r_base

    # ── Step 3: apply global scale and clamp ──────────────────────────
    r_drag = r_adjusted * drafting_scale
    return max(0.0, min(r_drag, 0.95))   # hard cap at 95 %


def curve_offset_multiplier(is_curve_offset: int | bool,
                            penalty: float = DEFAULT_CURVE_OFFSET_PENALTY
                            ) -> float:
    """Return multiplier on drag reduction when runner is laterally offset on a curve.

    Parameters
    ----------
    is_curve_offset : 0 or 1
        1 = runner takes a mild (~0.2 m) lateral offset on this bend segment.
    penalty : float
        Fraction of straight-line benefit retained (default 0.85).

    Returns
    -------
    float : 1.0 (full benefit) or ``penalty`` (reduced benefit).
    """
    if int(is_curve_offset):
        return penalty
    return 1.0


# ══════════════════════════════════════════════════════════════════════
# Internal helpers
# ══════════════════════════════════════════════════════════════════════

def _beaumont_lookup(pos: int, n: int) -> float:
    """Look up drag reduction from Beaumont et al. data.

    For formations larger than 5 runners, we approximate using the 5-runner
    data (positions beyond 5 get the average of positions 3–5 in a quintet).
    """
    key = (n, pos)
    if key in BEAUMONT_DRAG_REDUCTION:
        return BEAUMONT_DRAG_REDUCTION[key]

    # Extrapolation for n > 5  (use 5-runner pattern)
    if n > 5:
        # Lead runner: progressively better with more runners behind
        if pos == 1:
            return BEAUMONT_DRAG_REDUCTION[(5, 1)]
        # Position 2: best sheltered position, use 5-runner value
        if pos == 2:
            return BEAUMONT_DRAG_REDUCTION[(5, 2)]
        # Positions 3+: average of positions 3–5 in the 5-runner group
        avg_mid = (
            BEAUMONT_DRAG_REDUCTION[(5, 3)]
            + BEAUMONT_DRAG_REDUCTION[(5, 4)]
            + BEAUMONT_DRAG_REDUCTION[(5, 5)]
        ) / 3.0
        return avg_mid

    # Should not reach here for valid inputs
    return 0.0


def _gap_adjustment_factor(n: int, gap_m: float) -> float:
    """Compute a multiplier that scales the Beaumont (1.3 m) drag reduction
    for a different axial gap distance.

    At gap = 1.3 m the factor is 1.0 (Beaumont reference).
    At gap = 1.2 m the factor is derived from Schickhofer data (> 1.0).
    At gap ≥ 4.0 m the factor decays to zero (no practical benefit).

    The ratio between Schickhofer (1.2 m) and Beaumont (1.3 m) for position-2
    runners is used to calibrate:
        2 runners: 0.701 / 0.568 ≈ 1.234
        3 runners: 0.756 / 0.592 ≈ 1.277
    Average ≈ 1.26.  We use this as the factor at 1.2 m and linearly
    interpolate/extrapolate.
    """
    # Reference ratios at 1.2 m (Schickhofer / Beaumont)
    # Use the 2-runner ratio as default; use n-specific if available.
    if n >= 3:
        ratio_1_2 = 1.277  # from 3-runner comparison
    else:
        ratio_1_2 = 1.234  # from 2-runner comparison

    # Piecewise linear model
    REF_GAP = 1.3
    if gap_m <= 1.0:
        # Extrapolate linearly from 1.2 → 1.0 m
        slope = (ratio_1_2 - 1.0) / (REF_GAP - 1.2)  # per metre closer
        factor = ratio_1_2 + slope * (1.2 - gap_m)
    elif gap_m <= 1.2:
        # Interpolate between 1.0-extrapolated and 1.2
        slope = (ratio_1_2 - 1.0) / (REF_GAP - 1.2)
        factor_at_1_0 = ratio_1_2 + slope * 0.2
        factor = _lerp(gap_m, 1.0, 1.2, factor_at_1_0, ratio_1_2)
    elif gap_m <= REF_GAP:
        # Between 1.2 m and 1.3 m: interpolate between ratio and 1.0
        factor = _lerp(gap_m, 1.2, REF_GAP, ratio_1_2, 1.0)
    elif gap_m <= 4.0:
        # Decay from 1.0 at 1.3 m to 0.0 at 4.0 m (linear)
        factor = _lerp(gap_m, REF_GAP, 4.0, 1.0, 0.0)
    else:
        factor = 0.0

    return max(0.0, factor)


def _lerp(x: float, x0: float, x1: float, y0: float, y1: float) -> float:
    """Simple linear interpolation."""
    if abs(x1 - x0) < 1e-12:
        return (y0 + y1) / 2.0
    return y0 + (y1 - y0) * (x - x0) / (x1 - x0)
