"""
segments.py — Generate per-segment race tables with bend/straight metadata.

Rules
-----
1.  ``segment_length`` is the default length for every segment (e.g. 100 m).
2.  If ``event_distance`` is NOT an exact multiple of ``segment_length``, the
    EXTRA distance is absorbed into the FIRST segment (not the last).
        segment_1_length = segment_length + (event_distance % segment_length)
        remaining segments = segment_length each
    Example — mile (1609.34 m) at 100 m segmentation:
        segment 1 = 109.34 m  →  segments 2–16 = 100 m each  →  total 1609.34 m
3.  Each segment receives a ``pct_bend`` (0–1) calculated from where it falls
    on a repeating 400 m oval, and a boolean ``is_bend`` flag (True if
    pct_bend > 0.50).

References
----------
- IAAF Track & Field Facilities Manual  (400 m oval geometry)
- Bellenger et al. (2026): 100 m segmentation for 1500 m analysis
"""

from __future__ import annotations
import numpy as np
import pandas as pd
from .config import (
    TRACK_LENGTH_M,
    BEND_LENGTH_M,
    STRAIGHT_LENGTH_M,
    TRACK_SECTIONS,
)


def generate_segments(
    event_distance_m: float,
    segment_length_m: float = 100.0,
    track_length_m: float = TRACK_LENGTH_M,
    start_offset_m: float | None = None,
) -> pd.DataFrame:
    """Create a segment table for a track race.

    Parameters
    ----------
    event_distance_m : float
        Total race distance (e.g., 1609.34 for the mile).
    segment_length_m : float
        Default segment length (metres).  Default 100.
    track_length_m : float
        Lap length.  Default 400.
    start_offset_m : float or None
        Position on the track (from the finish line, going forward in race
        direction) where the race starts.  If ``None``, computed automatically
        as ``(track_length - event_distance % track_length) % track_length``.

    Returns
    -------
    DataFrame with columns:
        segment_idx   – 1-indexed segment number
        start_m       – cumulative start distance from race start
        end_m         – cumulative end distance
        length_m      – segment length (first segment may be longer)
        pct_bend      – fraction of segment on a bend (0–1)
        is_bend       – True if pct_bend > 0.50
    """
    # ── 1. Compute segment lengths ────────────────────────────────────
    remainder = event_distance_m % segment_length_m
    if remainder < 1e-9:
        # Clean multiple — all segments equal
        n_segments = int(round(event_distance_m / segment_length_m))
        lengths = [segment_length_m] * n_segments
    else:
        # Extra distance goes into the FIRST segment
        first_seg = segment_length_m + remainder
        n_remaining = int(round((event_distance_m - first_seg) / segment_length_m))
        lengths = [first_seg] + [segment_length_m] * n_remaining

    # Sanity check
    total = sum(lengths)
    assert abs(total - event_distance_m) < 0.01, (
        f"Segment lengths sum to {total:.2f} m but event is {event_distance_m:.2f} m"
    )

    # ── 2. Compute cumulative start/end positions ─────────────────────
    starts = np.concatenate([[0.0], np.cumsum(lengths[:-1])])
    ends   = np.cumsum(lengths)

    # ── 3. Determine track start offset ───────────────────────────────
    if start_offset_m is None:
        # Standard convention: the finish line is at track position 0 (≡ 400).
        # Running forward through the lap, the start is at:
        start_offset_m = (track_length_m - (event_distance_m % track_length_m)) % track_length_m

    # ── 4. Compute bend percentage for each segment ───────────────────
    pct_bends = np.array([
        _bend_fraction(start_offset_m + s, start_offset_m + e, track_length_m)
        for s, e in zip(starts, ends)
    ])

    # ── 5. Build DataFrame ────────────────────────────────────────────
    return pd.DataFrame({
        "segment_idx": np.arange(1, len(lengths) + 1),
        "start_m":     np.round(starts, 4),
        "end_m":       np.round(ends, 4),
        "length_m":    np.round(lengths, 4),
        "pct_bend":    np.round(pct_bends, 4),
        "is_bend":     pct_bends > 0.50,
    })


# ══════════════════════════════════════════════════════════════════════
# Internal helpers
# ══════════════════════════════════════════════════════════════════════

def _bend_fraction(
    abs_start: float,
    abs_end: float,
    track_length: float = TRACK_LENGTH_M,
) -> float:
    """Return the fraction of [abs_start, abs_end) that lies on a bend.

    Positions are in absolute race metres (may exceed track_length for
    multiple laps).  They are reduced modulo ``track_length`` to find the
    track position, then overlaid onto the standard section layout.
    """
    total_length = abs_end - abs_start
    if total_length <= 0:
        return 0.0

    bend_dist = 0.0
    pos = abs_start
    remaining = total_length
    max_iter = int(total_length / 10) + 200  # safety limit

    for _ in range(max_iter):
        if remaining <= 1e-9:
            break

        # Current track position (mod lap), clamped to [0, track_length)
        track_pos = pos % track_length
        if track_pos >= track_length - 1e-9:
            track_pos = 0.0  # wrap exactly

        # Find which section we're in
        matched = False
        for sec_type, sec_start, sec_end in TRACK_SECTIONS:
            if sec_start - 1e-9 <= track_pos < sec_end - 1e-9:
                # Distance to end of this section
                dist_to_sec_end = sec_end - track_pos
                step = min(dist_to_sec_end, remaining)
                if step < 1e-12:
                    step = min(1e-6, remaining)  # nudge forward
                if sec_type == "bend":
                    bend_dist += step
                pos += step
                remaining -= step
                matched = True
                break
        if not matched:
            # Edge case: nudge forward
            pos += 1e-6
            remaining -= 1e-6

    return bend_dist / total_length if total_length > 0 else 0.0
