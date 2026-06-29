"""
Mile Performance Advantage Dashboard — Streamlit App

Computes how much faster an athlete could run for the same metabolic energy
under different drafting and shoe configurations.

Based on: Beaumont et al. (2021), Schickhofer & Hanson (2021), Kipp et al. (2019),
Batliner et al. (2018), Pugh (1971).
"""

import sys
import os
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

# Add parent directory so we can import src modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.dashboard import compute_segment_savings, build_drafting_plan, format_time
from src.plotting import plot_dashboard_savings
from src.segments import generate_segments
from src.config import estimate_frontal_area

# ══════════════════════════════════════════════════════════════════════
# Page config
# ══════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="Mile Performance Advantage",
    page_icon="🏃",
    layout="wide",
)

st.title("🏃 Mile Performance Advantage Dashboard")
st.markdown("""
Given a **baseline even pace**, this tool computes how much faster the athlete
could run for the **same metabolic energy** under different drafting and shoe
configurations.
""")

# ══════════════════════════════════════════════════════════════════════
# Sidebar: Global settings
# ══════════════════════════════════════════════════════════════════════
st.sidebar.header("⚙️ Settings")

st.sidebar.subheader("Baseline Pace")
col_min, col_sec = st.sidebar.columns(2)
with col_min:
    pace_min = st.number_input("Minutes", min_value=2, max_value=6, value=3, step=1)
with col_sec:
    pace_sec = st.number_input("Seconds", min_value=0.00, max_value=59.99,
                                value=45.00, step=0.01, format="%.2f")

st.sidebar.subheader("Athlete")
height_m = st.sidebar.number_input("Height (m)", min_value=1.50, max_value=2.10,
                                    value=1.84, step=0.01, format="%.2f")
mass_kg = st.sidebar.number_input("Mass (kg)", min_value=45.0, max_value=100.0,
                                   value=72.6, step=0.1, format="%.1f")
air_density = st.sidebar.number_input("Air density (kg/m³)", min_value=0.90, max_value=1.35,
                                       value=1.225, step=0.005, format="%.3f")

st.sidebar.subheader("Shoe / Product")
shoe_re_pct = st.sidebar.slider("Shoe RE improvement (%)", min_value=0.0,
                                 max_value=6.0, value=0.0, step=0.1, format="%.1f")

st.sidebar.subheader("Drafting — Zone Default")
formation_options = [
    "1of1", "1of2", "2of2",
    "1of3", "2of3", "3of3",
    "1of4", "2of4", "3of4", "4of4",
    "1of5", "2of5", "3of5", "4of5", "5of5",
]
formation_labels = {
    "1of1": "Solo (1of1)",
    "1of2": "Front of pair (1of2)",
    "2of2": "2nd of 2 (2of2)",
    "1of3": "Front of 3 (1of3)",
    "2of3": "2nd of 3 (2of3)",
    "3of3": "3rd of 3 (3of3)",
    "1of4": "Front of 4 (1of4)",
    "2of4": "2nd of 4 (2of4)",
    "3of4": "3rd of 4 (3of4)",
    "4of4": "4th of 4 (4of4)",
    "1of5": "Front of 5 (1of5)",
    "2of5": "2nd of 5 (2of5)",
    "3of5": "3rd of 5 (3of5)",
    "4of5": "4th of 5 (4of5)",
    "5of5": "5th of 5 (5of5)",
}

default_formation = st.sidebar.selectbox(
    "Default formation",
    options=formation_options,
    index=formation_options.index("2of5"),
    format_func=lambda x: formation_labels[x],
)
default_gap = st.sidebar.slider("Default gap (m)", min_value=0.5, max_value=3.0,
                                 value=1.0, step=0.1, format="%.1f")
no_draft_final = st.sidebar.select_slider("Solo final distance (m)",
                                           options=[200.0, 300.0, 400.0], value=400.0)

# ══════════════════════════════════════════════════════════════════════
# Race geometry
# ══════════════════════════════════════════════════════════════════════
EVENT_DIST = 1609.34  # mile
SEG_LEN = 100.0
seg_df = generate_segments(EVENT_DIST, SEG_LEN)
N_SEG = len(seg_df)

# ══════════════════════════════════════════════════════════════════════
# Per-segment configuration
# ══════════════════════════════════════════════════════════════════════
st.subheader("📋 Per-Segment Drafting Configuration")
st.markdown("Customize position and gap for each segment. Leave as 'Zone default' to use sidebar settings.")

with st.expander("Click to expand per-segment controls", expanded=False):
    seg_data = []
    cols = st.columns(4)
    for i in range(N_SEG):
        col = cols[i % 4]
        with col:
            st.markdown(f"**Seg {i+1}** ({seg_df.loc[i, 'start_m']:.0f}–{seg_df.loc[i, 'end_m']:.0f} m)")
            pos = st.selectbox(
                f"Position##seg{i}",
                options=["default"] + formation_options,
                index=0,
                format_func=lambda x: "Zone default" if x == "default" else formation_labels.get(x, x),
                key=f"pos_{i}",
                label_visibility="collapsed",
            )
            gap = st.number_input(
                f"Gap##seg{i}",
                min_value=0.0, max_value=4.0, value=0.0, step=0.1,
                format="%.1f", key=f"gap_{i}",
                label_visibility="collapsed",
            )
            seg_data.append({"pos": pos, "gap": gap})

# ══════════════════════════════════════════════════════════════════════
# Compute
# ══════════════════════════════════════════════════════════════════════
pace_seconds = pace_min * 60 + pace_sec
baseline_speed = EVENT_DIST / pace_seconds

# Build overrides from per-segment controls
overrides = {}
for i, sd in enumerate(seg_data):
    override = {}
    if sd["pos"] != "default":
        override["pos_of_n"] = sd["pos"]
    if sd["gap"] > 0:
        override["gap_m"] = sd["gap"]
    if override:
        overrides[i] = override

plan = build_drafting_plan(
    n_segments=N_SEG,
    zone_config={"formation": default_formation, "gap_m": default_gap},
    per_segment_overrides=overrides,
    event_distance_m=EVENT_DIST,
    segment_length_m=SEG_LEN,
    no_draft_final_m=no_draft_final,
)

result = compute_segment_savings(
    baseline_speed_ms=baseline_speed,
    event_distance_m=EVENT_DIST,
    segment_length_m=SEG_LEN,
    drafting_plan=plan,
    shoe_re_pct=shoe_re_pct,
    air_density=air_density,
    height_m=height_m,
    mass_kg=mass_kg,
)

# ══════════════════════════════════════════════════════════════════════
# Results
# ══════════════════════════════════════════════════════════════════════
st.markdown("---")

eq_time = result["equivalent_finish_s"]
saved = result["time_saved_s"]
pct = result["pct_improvement"]
draft_s = result["drafting_contribution_s"]
shoe_s = result["shoe_contribution_s"]

# Big metrics
col1, col2, col3, col4 = st.columns(4)
col1.metric("Baseline", format_time(pace_seconds))
col2.metric("Equivalent Time", format_time(eq_time))
col3.metric("Time Saved", f"{saved:.2f} s")
col4.metric("Improvement", f"{pct:.2f} %")

col5, col6 = st.columns(2)
col5.metric("↳ Drafting", f"{draft_s:.2f} s")
col6.metric("↳ Shoes/Product", f"{shoe_s:.2f} s")

# ══════════════════════════════════════════════════════════════════════
# Visualization
# ══════════════════════════════════════════════════════════════════════
st.markdown("---")
st.subheader("📊 Per-Segment Breakdown")

fig = plot_dashboard_savings(
    result,
    title=f"Performance Advantage: {format_time(pace_seconds)} → {format_time(eq_time)} ({saved:+.2f}s)",
)
st.pyplot(fig)
plt.close(fig)

# ══════════════════════════════════════════════════════════════════════
# Segment table
# ══════════════════════════════════════════════════════════════════════
with st.expander("View per-segment data table"):
    display_df = result["segments_df"][[
        "segment_idx", "start_m", "end_m", "pos_of_n", "gap_m",
        "r_drag", "baseline_speed_ms", "equivalent_speed_ms",
        "time_saved_s", "drafting_saved_s", "shoe_saved_s", "re_improvement_pct",
    ]].round(4)
    st.dataframe(display_df, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════
# Footer
# ══════════════════════════════════════════════════════════════════════
st.markdown("---")
st.caption("""
**Model references:** Beaumont et al. (2021) — CFD drag reduction;
Schickhofer & Hanson (2021) — gap sensitivity;
Kipp et al. (2019) & Batliner et al. (2018) — VO₂–velocity;
Pugh (1971) — metabolic cost of air resistance.
""")
