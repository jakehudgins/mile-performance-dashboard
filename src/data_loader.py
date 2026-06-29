"""
data_loader.py — Load and validate the single-workbook input file.

The workbook must contain these sheets:
    event_config, athlete_history, physiology,
    race_script, environment, equipment, priors

Each sheet has a defined schema (see config.REQUIRED_SHEETS).
This module also provides ``create_demo_workbook()`` to generate a
template Excel file pre-populated with demo data for a mile race.
"""

from __future__ import annotations
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Any

from .config import REQUIRED_SHEETS, DISTANCES


def load_workbook(path: str | Path) -> Dict[str, pd.DataFrame]:
    """Load and validate every sheet from the input workbook.

    Parameters
    ----------
    path : str or Path
        Path to the .xlsx workbook.

    Returns
    -------
    dict : {sheet_name: DataFrame}

    Raises
    ------
    FileNotFoundError, ValueError on missing sheets / columns.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Workbook not found: {path}")

    xls = pd.ExcelFile(path)
    sheets: Dict[str, pd.DataFrame] = {}

    for sheet_name, required_cols in REQUIRED_SHEETS.items():
        if sheet_name not in xls.sheet_names:
            raise ValueError(
                f"Missing required sheet '{sheet_name}' in {path.name}. "
                f"Found sheets: {xls.sheet_names}"
            )
        df = pd.read_excel(xls, sheet_name=sheet_name)
        # Normalise column names
        df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
        missing = [c for c in required_cols if c not in df.columns]
        if missing:
            raise ValueError(
                f"Sheet '{sheet_name}' is missing columns: {missing}. "
                f"Found: {list(df.columns)}"
            )
        sheets[sheet_name] = df

    return sheets


def create_demo_workbook(path: str | Path) -> Path:
    """Write a demo .xlsx workbook for a mile race with placeholder data.

    The demo models a hypothetical elite miler:
        CS ≈ 6.10 m/s, D′ ≈ 210 m, target ~ 3:52

    Returns the written Path.
    """
    path = Path(path)

    # ── event_config ──────────────────────────────────────────────────
    event_config = pd.DataFrame([{
        "event_name":       "Mile",
        "event_distance_m": 1609.34,
        "segment_length_m": 100.0,
        "track_length_m":   400.0,
    }])

    # ── athlete_history (personal bests for CS/D′ fitting) ────────────
    athlete_history = pd.DataFrame([
        {"athlete_id": "demo_miler", "distance_m": 1500, "time_s": 214.0,
         "date": "2024-07-01", "event_name": "Diamond League"},
        {"athlete_id": "demo_miler", "distance_m": 1609.34, "time_s": 232.0,
         "date": "2024-06-15", "event_name": "Prefontaine Classic"},
        {"athlete_id": "demo_miler", "distance_m": 3000, "time_s": 454.0,
         "date": "2024-05-20", "event_name": "Oslo DL"},
        {"athlete_id": "demo_miler", "distance_m": 5000, "time_s": 780.0,
         "date": "2023-09-10", "event_name": "Brussels DL"},
    ])

    # ── physiology ────────────────────────────────────────────────────
    physiology = pd.DataFrame([{
        "athlete_id":      "demo_miler",
        "height_m":        1.80,
        "mass_kg":         66.0,
        "frontal_area_m2": None,   # will be estimated
        "max_speed_ms":    8.5,
    }])

    # ── race_script (16 segments for mile: 109.34 + 15×100) ──────────
    # Slightly positive-split strategy with a kick
    n_seg = 16
    # First segment is 109.34 m, rest 100 m
    base_speed = 6.93  # m/s ≈ 3:52 mile pace
    speeds = [7.10] + [6.90] * 7 + [6.85] * 4 + [6.95, 7.05, 7.20, 7.50]
    # Drafting: sit in 3rd of 5 for most of race, move up at end
    pos_list = ["3of5"] * 12 + ["2of4", "2of3", "1of2", "1of1"]
    gaps     = [1.3] * 12 + [1.2, 1.2, 1.3, 0.0]
    offsets  = [1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0]

    race_script = pd.DataFrame({
        "segment_idx":      list(range(1, n_seg + 1)),
        "segment_speed_ms": speeds,
        "pos_of_n":         pos_list,
        "gap_m":            gaps,
        "is_curve_offset":  offsets,
    })

    # ── environment ───────────────────────────────────────────────────
    environment = pd.DataFrame([{
        "air_density_kgm3": 1.225,
        "altitude_m":       0,
        "temperature_c":    20,
        "wind_speed_ms":    0,
        "wind_direction":   "none",
    }])

    # ── equipment ─────────────────────────────────────────────────────
    equipment = pd.DataFrame([{
        "shoe_re_delta_pct": 2.0,   # modest shoe benefit
    }])

    # ── priors (for Monte Carlo) ──────────────────────────────────────
    priors = pd.DataFrame([
        {"parameter": "CS",                "mean": 6.10, "std": 0.05,
         "distribution": "normal",  "lower_bound": 5.90, "upper_bound": 6.30},
        {"parameter": "D_prime",           "mean": 210,  "std": 15,
         "distribution": "normal",  "lower_bound": 170,  "upper_bound": 260},
        {"parameter": "drafting_scale",    "mean": 1.0,  "std": 0.12,
         "distribution": "truncnorm", "lower_bound": 0.6,  "upper_bound": 1.4},
        {"parameter": "shoe_re_delta_pct", "mean": 2.0,  "std": 1.0,
         "distribution": "normal",  "lower_bound": 0.0,  "upper_bound": 5.0},
        {"parameter": "air_density_kgm3",  "mean": 1.225, "std": 0.02,
         "distribution": "normal",  "lower_bound": 1.1,  "upper_bound": 1.35},
    ])

    # ── Write workbook ────────────────────────────────────────────────
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        event_config.to_excel(writer, sheet_name="event_config", index=False)
        athlete_history.to_excel(writer, sheet_name="athlete_history", index=False)
        physiology.to_excel(writer, sheet_name="physiology", index=False)
        race_script.to_excel(writer, sheet_name="race_script", index=False)
        environment.to_excel(writer, sheet_name="environment", index=False)
        equipment.to_excel(writer, sheet_name="equipment", index=False)
        priors.to_excel(writer, sheet_name="priors", index=False)

    return path
