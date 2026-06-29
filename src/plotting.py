"""
plotting.py — Visualisation helpers for race simulation results.

All functions return a matplotlib Figure so they can be displayed inline
in Jupyter or saved to file.
"""

from __future__ import annotations
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from typing import Dict, Any, List, Optional


# ══════════════════════════════════════════════════════════════════════
# 1. Speed plan with bend shading
# ══════════════════════════════════════════════════════════════════════

def plot_speed_plan(seg_df: pd.DataFrame, cs_ms: float | None = None,
                    title: str = "Segment Speed Plan") -> plt.Figure:
    """Bar chart of per-segment actual & equivalent speeds, bends shaded.

    Parameters
    ----------
    seg_df : DataFrame from simulate_race()['segments_df']
    cs_ms  : Critical Speed line (optional)
    """
    fig, ax = plt.subplots(figsize=(12, 4))
    x = seg_df["end_m"]
    width = seg_df["length_m"] * 0.9

    # Shade bend regions
    for _, row in seg_df.iterrows():
        if row["is_bend"]:
            ax.axvspan(row["start_m"], row["end_m"],
                       color="lightblue", alpha=0.25, zorder=0)

    ax.bar(x - width / 2, seg_df["v_actual_ms"], width=width,
           color="steelblue", label="v_actual", zorder=2)
    ax.bar(x - width / 2, seg_df["v_eq_ms"], width=width * 0.5,
           color="coral", alpha=0.7, label="v_eq (metabolic)", zorder=3)

    if cs_ms is not None:
        ax.axhline(cs_ms, color="green", ls="--", lw=1.5, label=f"CS = {cs_ms:.2f} m/s")

    ax.set_xlabel("Distance (m)")
    ax.set_ylabel("Speed (m/s)")
    ax.set_title(title)
    ax.legend(loc="upper right", fontsize=8)
    plt.tight_layout()
    return fig


# ══════════════════════════════════════════════════════════════════════
# 2. D′ balance vs distance
# ══════════════════════════════════════════════════════════════════════

def plot_dprime_balance(seg_df: pd.DataFrame, d_prime_0: float,
                        title: str = "D′ Balance vs Distance") -> plt.Figure:
    fig, ax = plt.subplots(figsize=(12, 4))

    xvals = np.concatenate([[0], seg_df["end_m"].values])
    yvals = np.concatenate([[d_prime_0], seg_df["d_prime_balance_m"].values])

    ax.plot(xvals, yvals, "o-", color="darkred", lw=2, markersize=4)
    ax.axhline(0, color="black", ls=":", lw=0.8)
    ax.fill_between(xvals, yvals, alpha=0.15, color="darkred")

    # Shade bends
    for _, row in seg_df.iterrows():
        if row["is_bend"]:
            ax.axvspan(row["start_m"], row["end_m"],
                       color="lightblue", alpha=0.2, zorder=0)

    ax.set_xlabel("Distance (m)")
    ax.set_ylabel("D′ balance (m)")
    ax.set_title(title)
    plt.tight_layout()
    return fig


# ══════════════════════════════════════════════════════════════════════
# 3. Per-segment drag reduction & VO₂ savings
# ══════════════════════════════════════════════════════════════════════

def plot_drag_and_vo2(seg_df: pd.DataFrame,
                      title: str = "Drafting & Metabolic Effects") -> plt.Figure:
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 6), sharex=True)

    x = seg_df["end_m"]

    # Drag reduction
    ax1.bar(x, seg_df["r_drag"] * 100, width=seg_df["length_m"] * 0.8,
            color="teal", alpha=0.7)
    ax1.set_ylabel("Drag reduction (%)")
    ax1.set_title(title)

    # RE improvement
    ax2.bar(x, seg_df["re_improvement_pct"], width=seg_df["length_m"] * 0.8,
            color="darkorange", alpha=0.7)
    ax2.set_ylabel("RE improvement (%)")
    ax2.set_xlabel("Distance (m)")

    for ax in (ax1, ax2):
        for _, row in seg_df.iterrows():
            if row["is_bend"]:
                ax.axvspan(row["start_m"], row["end_m"],
                           color="lightblue", alpha=0.2, zorder=0)

    plt.tight_layout()
    return fig


# ══════════════════════════════════════════════════════════════════════
# 4. Scenario comparison (overlay multiple simulations)
# ══════════════════════════════════════════════════════════════════════

def plot_scenario_comparison(
    scenarios: Dict[str, Dict[str, Any]],
    metric: str = "d_prime_balance_m",
    ylabel: str = "D′ balance (m)",
    title: str = "Scenario Comparison",
) -> plt.Figure:
    """Overlay a metric from multiple simulation results.

    Parameters
    ----------
    scenarios : dict  {label: simulate_race() result}
    metric    : column name in segments_df to plot
    """
    fig, ax = plt.subplots(figsize=(12, 5))
    colors = plt.cm.tab10.colors

    for idx, (label, res) in enumerate(scenarios.items()):
        df = res["segments_df"]
        c = colors[idx % len(colors)]
        xvals = np.concatenate([[0], df["end_m"].values])

        if metric == "d_prime_balance_m":
            yvals = np.concatenate([[res["summary"]["d_prime_m"]], df[metric].values])
        else:
            yvals = np.concatenate([[df[metric].iloc[0]], df[metric].values])

        ft = res["summary"]["finish_time_str"]
        ax.plot(xvals, yvals, "o-", color=c, lw=2, markersize=3,
                label=f"{label} ({ft})")

    ax.set_xlabel("Distance (m)")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(fontsize=8)
    plt.tight_layout()
    return fig


# ══════════════════════════════════════════════════════════════════════
# 5. Monte Carlo results
# ══════════════════════════════════════════════════════════════════════

def plot_monte_carlo_results(
    mc_result: Dict[str, Any],
    target_time_s: float | None = None,
    title: str = "Monte Carlo Finish-Time Distribution",
) -> plt.Figure:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ft = mc_result["finish_times"]
    valid = ft[~np.isnan(ft)]
    stats = mc_result["stats"]

    # ── Histogram ─────────────────────────────────────────────────────
    ax1.hist(valid, bins=min(50, max(10, len(valid) // 20)),
             color="steelblue", edgecolor="white", alpha=0.8)
    ax1.axvline(stats["mean_s"], color="red", ls="--", lw=2,
                label=f'Mean = {stats["mean_s"]:.2f} s')
    ax1.axvline(stats["ci_95_lo_s"], color="grey", ls=":", lw=1,
                label=f'95% CI = [{stats["ci_95_lo_s"]:.1f}, {stats["ci_95_hi_s"]:.1f}]')
    ax1.axvline(stats["ci_95_hi_s"], color="grey", ls=":", lw=1)

    if target_time_s is not None:
        ax1.axvline(target_time_s, color="green", ls="-", lw=2,
                    label=f"Target = {target_time_s:.1f} s")
        if mc_result["prob_target"] is not None:
            ax1.set_title(f'P(finish ≤ target) = {mc_result["prob_target"]:.1%}')

    ax1.set_xlabel("Finish time (s)")
    ax1.set_ylabel("Count")
    ax1.legend(fontsize=8)
    ax1.set_title(title)

    # ── Sensitivity bar chart ─────────────────────────────────────────
    sens = mc_result["sensitivity"]
    colors = ["coral" if r < 0 else "steelblue" for r in sens["spearman_rho"]]
    ax2.barh(sens["parameter"], sens["spearman_rho"], color=colors, alpha=0.8)
    ax2.set_xlabel("Spearman ρ  (correlation with finish time)")
    ax2.set_title("Sensitivity Analysis")
    ax2.axvline(0, color="black", lw=0.5)

    plt.tight_layout()
    return fig


# ══════════════════════════════════════════════════════════════════════
# 6. Scenario summary table (bar chart of finish times + deltas)
# ══════════════════════════════════════════════════════════════════════

def plot_scenario_summary_table(
    scenario_results: Dict[str, Dict[str, Any]],
    baseline_label: str | None = None,
    title: str = "Scenario Comparison — Finish Times",
) -> plt.Figure:
    """Horizontal bar chart of finish times with time deltas from baseline.

    Parameters
    ----------
    scenario_results : dict
        Output of run_scenario_matrix().
    baseline_label : str or None
        Which scenario is baseline for delta computation.
        If None, uses the slowest scenario.
    """
    # Sort by finish time (slowest first for visual)
    sorted_items = sorted(scenario_results.items(),
                          key=lambda x: x[1]["finish_time_s"], reverse=True)
    labels = [item[0] for item in sorted_items]
    times = [item[1]["finish_time_s"] for item in sorted_items]

    if baseline_label is None:
        baseline_time = max(times)
    else:
        baseline_time = scenario_results[baseline_label]["finish_time_s"]

    deltas = [t - baseline_time for t in times]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, max(4, len(labels) * 0.6)),
                                    gridspec_kw={"width_ratios": [3, 2]})

    # Finish times bar chart
    colors_bar = plt.cm.RdYlGn_r(np.linspace(0.2, 0.8, len(labels)))
    bars = ax1.barh(range(len(labels)), times, color=colors_bar, edgecolor="white")
    ax1.set_yticks(range(len(labels)))
    ax1.set_yticklabels(labels, fontsize=9)
    ax1.set_xlabel("Finish time (s)")
    ax1.set_title(title)

    # Add time annotations
    for i, (t, bar) in enumerate(zip(times, bars)):
        mins = int(t // 60)
        secs = t % 60
        ax1.text(bar.get_width() + 0.3, bar.get_y() + bar.get_height() / 2,
                 f"{mins}:{secs:05.2f}", va="center", fontsize=8)

    # Delta chart (seconds saved vs baseline)
    delta_colors = ["green" if d < 0 else "grey" if d == 0 else "red" for d in deltas]
    ax2.barh(range(len(labels)), deltas, color=delta_colors, edgecolor="white")
    ax2.set_yticks(range(len(labels)))
    ax2.set_yticklabels([""] * len(labels))
    ax2.set_xlabel("Δ from baseline (s)")
    ax2.set_title("Time Saved")
    ax2.axvline(0, color="black", lw=0.8)

    for i, d in enumerate(deltas):
        if d != 0:
            ax2.text(d + (0.1 if d > 0 else -0.1), i,
                     f"{d:+.2f}s", va="center", fontsize=8,
                     ha="left" if d > 0 else "right")

    plt.tight_layout()
    return fig


# ══════════════════════════════════════════════════════════════════════
# 7. Sensitivity waterfall (factor contribution chart)
# ══════════════════════════════════════════════════════════════════════

def plot_sensitivity_waterfall(
    scenario_results: Dict[str, Dict[str, Any]],
    factor_pairs: List[tuple],
    title: str = "Factor Contribution — Seconds Saved",
) -> plt.Figure:
    """Waterfall chart showing how many seconds each factor contributes.

    Parameters
    ----------
    scenario_results : dict
        Output of run_scenario_matrix().
    factor_pairs : list of (label, scenario_a, scenario_b) tuples
        Each tuple defines a factor:
        - label: display name (e.g. "Drafting benefit")
        - scenario_a: baseline scenario name
        - scenario_b: scenario with the factor applied
        The contribution = finish_time_a - finish_time_b (positive = saving).
    """
    factors = []
    contributions = []

    for label, scene_a, scene_b in factor_pairs:
        time_a = scenario_results[scene_a]["finish_time_s"]
        time_b = scenario_results[scene_b]["finish_time_s"]
        factors.append(label)
        contributions.append(time_a - time_b)

    fig, ax = plt.subplots(figsize=(10, max(4, len(factors) * 0.7)))

    bar_colors = ["#2ecc71" if c > 0 else "#e74c3c" for c in contributions]
    bars = ax.barh(range(len(factors)), contributions, color=bar_colors,
                   edgecolor="white", height=0.6)
    ax.set_yticks(range(len(factors)))
    ax.set_yticklabels(factors, fontsize=10)
    ax.set_xlabel("Seconds saved")
    ax.set_title(title)
    ax.axvline(0, color="black", lw=0.8)

    for i, (c, bar) in enumerate(zip(contributions, bars)):
        ax.text(c + (0.05 if c >= 0 else -0.05), i,
                f"{c:+.2f}s", va="center", fontsize=9,
                ha="left" if c >= 0 else "right")

    plt.tight_layout()
    return fig


# ══════════════════════════════════════════════════════════════════════
# 8. Speed profile overlay (compare strategies)
# ══════════════════════════════════════════════════════════════════════

def plot_speed_profile_overlay(
    scenario_results: Dict[str, Dict[str, Any]],
    cs_ms: float | None = None,
    title: str = "Speed Profile Comparison",
) -> plt.Figure:
    """Overlay per-segment speed profiles from multiple scenarios.

    Parameters
    ----------
    scenario_results : dict
        Output of run_scenario_matrix().
    cs_ms : float or None
        Critical Speed line.
    """
    fig, ax = plt.subplots(figsize=(12, 5))
    line_colors = plt.cm.tab10.colors

    for idx, (label, res) in enumerate(scenario_results.items()):
        df = res["simulation_result"]["segments_df"]
        x = df["end_m"]
        c = line_colors[idx % len(line_colors)]
        ft = res["finish_time_str"]
        ax.plot(x, df["v_actual_ms"], "o-", color=c, lw=2, markersize=4,
                label=f"{label} ({ft})")

    if cs_ms is not None:
        ax.axhline(cs_ms, color="black", ls="--", lw=1.5, alpha=0.6,
                   label=f"CS = {cs_ms:.3f} m/s")

    # Shade final 400m
    if len(scenario_results) > 0:
        first_res = next(iter(scenario_results.values()))
        df = first_res["simulation_result"]["segments_df"]
        total_dist = df["end_m"].iloc[-1]
        ax.axvspan(total_dist - 400, total_dist,
                   color="lightyellow", alpha=0.3, label="Final 400m (solo)")

    ax.set_xlabel("Distance (m)")
    ax.set_ylabel("Speed (m/s)")
    ax.set_title(title)
    ax.legend(fontsize=8, loc="upper right")
    plt.tight_layout()
    return fig


# ══════════════════════════════════════════════════════════════════════
# 9. Dashboard: per-segment savings visualization
# ══════════════════════════════════════════════════════════════════════

def plot_dashboard_savings(
    result: Dict[str, Any],
    title: str = "Performance Advantage Dashboard",
) -> plt.Figure:
    """Visualize per-segment time savings from drafting and shoes.

    Parameters
    ----------
    result : dict
        Output of compute_segment_savings().
    """
    df = result["segments_df"]
    n_seg = len(df)

    fig, axes = plt.subplots(3, 1, figsize=(13, 9), sharex=True,
                             gridspec_kw={"height_ratios": [2, 1.2, 1]})

    x = df["end_m"]
    width = df["length_m"] * 0.8

    # ── Panel 1: Time saved per segment (stacked: drafting + shoes) ───
    ax = axes[0]
    draft_ms = df["drafting_saved_s"] * 1000  # convert to milliseconds
    shoe_ms = df["shoe_saved_s"] * 1000

    ax.bar(x, draft_ms, width=width, color="#2ecc71", alpha=0.85,
           label="Drafting savings")
    ax.bar(x, shoe_ms, width=width, bottom=draft_ms, color="#3498db",
           alpha=0.85, label="Shoe savings")

    # Shade solo segments
    for _, row in df.iterrows():
        if row["pos_of_n"] == "1of1":
            ax.axvspan(row["start_m"], row["end_m"],
                       color="lightgrey", alpha=0.2, zorder=0)

    ax.set_ylabel("Time saved (ms)")
    ax.set_title(title)
    ax.legend(loc="upper right", fontsize=9)
    ax.axhline(0, color="black", lw=0.5)

    # ── Panel 2: Cumulative time saved ────────────────────────────────
    ax = axes[1]
    cum_saved = df["time_saved_s"].cumsum()
    ax.plot(x, cum_saved, "o-", color="#e74c3c", lw=2, markersize=4)
    ax.fill_between(x, cum_saved, alpha=0.15, color="#e74c3c")
    ax.set_ylabel("Cumulative saved (s)")
    ax.axhline(0, color="black", lw=0.5)

    # Annotate final value
    final_saved = cum_saved.iloc[-1]
    ax.annotate(f"{final_saved:.2f}s saved",
                xy=(x.iloc[-1], final_saved),
                xytext=(-60, 10), textcoords="offset points",
                fontsize=10, fontweight="bold", color="#e74c3c",
                arrowprops=dict(arrowstyle="->", color="#e74c3c"))

    # ── Panel 3: Equivalent speed vs baseline ─────────────────────────
    ax = axes[2]
    ax.bar(x, df["equivalent_speed_ms"], width=width, color="#9b59b6",
           alpha=0.7, label="Equivalent speed")
    ax.axhline(df["baseline_speed_ms"].iloc[0], color="black", ls="--",
               lw=1.5, label=f"Baseline = {df['baseline_speed_ms'].iloc[0]:.3f} m/s")

    # Color-code drafted vs solo
    for _, row in df.iterrows():
        if row["pos_of_n"] == "1of1":
            ax.axvspan(row["start_m"], row["end_m"],
                       color="lightgrey", alpha=0.2, zorder=0)

    ax.set_xlabel("Distance (m)")
    ax.set_ylabel("Speed (m/s)")
    ax.legend(loc="upper right", fontsize=9)

    plt.tight_layout()
    return fig
