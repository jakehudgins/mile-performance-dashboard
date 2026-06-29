"""
Track Race Performance Model
=============================
A modular package for predicting track race performance (400 m – 10,000 m)
using critical speed / D′ bioenergetics, aerodynamic drafting effects,
and curvilinear VO₂–velocity metabolic modeling.

Key references
--------------
- Kirby et al. (2021) — D′ balance model for race finishing-position prediction
- Bellenger et al. (2026) — Drafting impact on D′ balance (Paris 1500 m)
- Beaumont et al. (2021) — CFD drag-reduction data for in-line runners
- Schickhofer & Hanson (2021) — Gap-distance sensitivity of drafting drag
- Kipp et al. (2019) — Curvilinear VO₂–velocity and performance prediction
- Batliner et al. (2018) — Quadratic VO₂–velocity from wide speed range
- Pugh (1971) — Metabolic cost of air resistance in running
"""

from .config import *                    # noqa: F401, F403
from .cs_dprime import fit_cs_dprime
from .segments import generate_segments
from .aerodynamics import drag_reduction_prior, curve_offset_multiplier
from .metabolism import (
    vo2_treadmill,
    vo2_air_resistance,
    vo2_total_overground,
    drafted_vo2,
    solve_equivalent_speed,
)
from .dprime_balance import update_dprime_balance
from .simulation import simulate_race
from .monte_carlo import run_monte_carlo
from .data_loader import load_workbook
from .optimizer import (
    optimize_pacing,
    build_drafting_config,
    find_max_sustainable_speed,
    run_scenario_matrix,
    robustness_check,
    write_optimal_strategy,
)
from .plotting import (
    plot_speed_plan,
    plot_dprime_balance,
    plot_drag_and_vo2,
    plot_scenario_comparison,
    plot_monte_carlo_results,
    plot_scenario_summary_table,
    plot_sensitivity_waterfall,
    plot_speed_profile_overlay,
    plot_dashboard_savings,
)
from .dashboard import (
    compute_segment_savings,
    build_drafting_plan,
    format_time,
)
