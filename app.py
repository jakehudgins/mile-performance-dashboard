"""
1-Mile Performance Prediction Dashboard — Streamlit App

Computes how much faster an athlete could run for the same metabolic energy
under different drafting and product configurations.
"""

import sys
import os
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

# Add current directory so we can import src modules
sys.path.insert(0, os.path.dirname(__file__))

from src.dashboard import compute_segment_savings, build_drafting_plan, format_time, calculate_air_density
from src.segments import generate_segments
from src.config import estimate_frontal_area

# ══════════════════════════════════════════════════════════════════════
# Page config
# ══════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="1-Mile Performance Prediction",
    page_icon="🏃",
    layout="wide",
)

st.title("🏃 1-Mile Performance Prediction Dashboard")
st.markdown("""
**How to use this tool:**
1. Set your **baseline even pace** and **athlete parameters** in the sidebar
2. Choose whether to use a **default drafting formation** or configure **each segment individually**
3. Adjust **Product RE %** for shoe/apparel aerodynamic benefits
4. The model instantly computes the equivalent faster finish time at the same metabolic cost

The results show how many seconds are saved per 100m segment from drafting and product interventions.
""")

with st.expander("📖 Model Details & Assumptions"):
    st.markdown("""
**Research basis:**
- **Aerodynamic drag reduction:** Beaumont et al. (2021) — CFD simulations of in-line running formations. Drag reductions range from ~4% (leader) to ~63% (2nd of 5).
- **Gap sensitivity:** Schickhofer & Hanson (2021) — benefit decays linearly to near-zero at 4.0 m gap.
- **Intrinsic metabolic cost:** Batliner et al. (2018) — quadratic VO₂–velocity relationship from treadmill data.
- **Air resistance cost:** Pugh (1971) — cubic relationship between speed and oxygen cost of overcoming drag.
- **Speed conversion:** Kipp et al. (2019) — curvilinear VO₂–velocity model to invert savings into equivalent faster speed.

**Key assumptions:**
- Drafting benefits are from steady-state CFD in still air — crosswinds/gusts will alter actual benefit.
- Gap between runners is assumed constant within each 100m segment.
- Product RE improvement applies to the intrinsic (musculoskeletal) cost only, not air resistance.
- Frontal area is estimated from height and mass (DuBois BSA × 0.18).
- The interaction between drafting and product benefits is nonlinear and correctly modeled via root-finding.
- Group sizes are limited to 1–5 runners per the validated CFD data. Positions must be ≤ group size.
    """)

# ══════════════════════════════════════════════════════════════════════
# Sidebar: Settings
# ══════════════════════════════════════════════════════════════════════
st.sidebar.header("⚙️ Settings")

st.sidebar.subheader("Baseline Pace")
col_min, col_sec = st.sidebar.columns(2)
with col_min:
    pace_min = st.number_input("Minutes", min_value=2, max_value=6, value=3, step=1)
with col_sec:
    pace_sec = st.number_input("Seconds", min_value=0.00, max_value=59.99,
                                value=49.00, step=0.01, format="%.2f")

st.sidebar.subheader("Athlete")
height_m = st.sidebar.number_input("Height (m)", min_value=1.50, max_value=2.10,
                                    value=1.84, step=0.01, format="%.2f")
mass_kg = st.sidebar.number_input("Mass (kg)", min_value=45.0, max_value=100.0,
                                   value=72.6, step=0.1, format="%.1f")

st.sidebar.subheader("Environment")
temp_f = st.sidebar.number_input("Temperature (°F)", min_value=0.0, max_value=120.0,
                                  value=75.0, step=1.0, format="%.0f")
temp_c = (temp_f - 32.0) * 5.0 / 9.0
rh_pct = st.sidebar.number_input("Relative Humidity (%)", min_value=0.0, max_value=100.0,
                                  value=52.0, step=1.0, format="%.0f")
pressure_hpa = st.sidebar.number_input("Barometric Pressure (hPa)", min_value=800.0,
                                        max_value=1100.0, value=1015.0, step=1.0, format="%.1f")

air_density = calculate_air_density(temp_c, rh_pct, pressure_hpa)
st.sidebar.metric("Calculated Air Density", f"{air_density:.4f} kg/m³")

st.sidebar.subheader("Product")
product_re_pct = st.sidebar.slider("Product RE Improvement (%)", min_value=0.0,
                                    max_value=6.0, value=0.0, step=0.1, format="%.1f")

# ══════════════════════════════════════════════════════════════════════
# Race geometry
# ══════════════════════════════════════════════════════════════════════
EVENT_DIST = 1609.34
SEG_LEN = 100.0
seg_df = generate_segments(EVENT_DIST, SEG_LEN)
N_SEG = len(seg_df)

# ══════════════════════════════════════════════════════════════════════
# Compute (runs before display so results appear above config)
# ══════════════════════════════════════════════════════════════════════
# We need to read the configuration first, then show results above it.
# Use session state for per-segment data.

# ── Drafting toggle ───────────────────────────────────────────────
st.subheader("Drafting Configuration")
use_defaults = st.toggle("Use default formation for all drafted segments", value=True)

# ── Default formation controls ────────────────────────────────────
if use_defaults:
    def_col1, def_col2, def_col3 = st.columns(3)
    formation_options = [
        "1of1", "1of2", "2of2",
        "1of3", "2of3", "3of3",
        "1of4", "2of4", "3of4", "4of4",
        "1of5", "2of5", "3of5", "4of5", "5of5",
    ]
    formation_labels = {
        "1of1": "Solo (1of1)", "1of2": "Front of pair (1of2)",
        "2of2": "2nd of 2 (2of2)", "1of3": "Front of 3 (1of3)",
        "2of3": "2nd of 3 (2of3)", "3of3": "3rd of 3 (3of3)",
        "1of4": "Front of 4 (1of4)", "2of4": "2nd of 4 (2of4)",
        "3of4": "3rd of 4 (3of4)", "4of4": "4th of 4 (4of4)",
        "1of5": "Front of 5 (1of5)", "2of5": "2nd of 5 (2of5)",
        "3of5": "3rd of 5 (3of5)", "4of5": "4th of 5 (4of5)",
        "5of5": "5th of 5 (5of5)",
    }
    with def_col1:
        default_formation = st.selectbox(
            "Formation", options=formation_options,
            index=formation_options.index("2of5"),
            format_func=lambda x: formation_labels[x],
        )
    with def_col2:
        default_gap = st.slider("Gap (m)", min_value=0.5, max_value=3.0,
                                 value=1.0, step=0.1, format="%.1f")
    with def_col3:
        no_draft_final = st.select_slider("Solo final distance (m)",
                                           options=[200.0, 300.0, 400.0], value=400.0)

    # Build plan from defaults
    plan = build_drafting_plan(
        n_segments=N_SEG,
        zone_config={"formation": default_formation, "gap_m": default_gap},
        event_distance_m=EVENT_DIST,
        segment_length_m=SEG_LEN,
        no_draft_final_m=no_draft_final,
    )
else:
    default_formation = "2of5"
    default_gap = 1.0
    no_draft_final = 400.0

    # ── Per-segment configuration in scrollable container ─────────
    st.markdown("**Per-Segment Configuration**  *(Set position, group size, and gap for each segment)*")

    # Reset to defaults button
    if st.button("🔄 Reset all segments to default (Position 2 of 5 at 1.0m)"):
        for i in range(N_SEG):
            st.session_state[f"pos_{i}"] = 2
            st.session_state[f"grp_{i}"] = 5
            st.session_state[f"gap_{i}"] = 1.0
        st.rerun()

    seg_config_data = []
    has_error = False

    # Scrollable container for per-segment inputs
    with st.container(height=400):
        cols_header = st.columns([0.6, 1.5, 1, 1, 1, 0.8])
        cols_header[0].markdown("**Seg**")
        cols_header[1].markdown("**Distance**")
        cols_header[2].markdown("**Position**")
        cols_header[3].markdown("**Group Size**")
        cols_header[4].markdown("**Gap (m)**")
        cols_header[5].markdown("**Type**")

        for i in range(N_SEG):
            is_bend = seg_df.loc[i, "is_bend"]
            seg_type = "🟦 Curve" if is_bend else "⬜ Straight"
            bg_color = "#e3f2fd" if is_bend else "#ffffff"
            start_m = int(seg_df.loc[i, "start_m"])
            end_m = int(seg_df.loc[i, "end_m"])

            cols = st.columns([0.6, 1.5, 1, 1, 1, 0.8])
            cols[0].markdown(f"<div style='background:{bg_color};padding:4px;border-radius:4px;text-align:center;'>"
                           f"<b>{i+1}</b></div>", unsafe_allow_html=True)
            cols[1].markdown(f"<div style='background:{bg_color};padding:4px;border-radius:4px;'>"
                           f"{start_m}–{end_m} m</div>", unsafe_allow_html=True)

            pos = cols[2].number_input(f"Pos##{i}", min_value=0, max_value=5,
                                        value=0, step=1, key=f"pos_{i}",
                                        label_visibility="collapsed")
            group = cols[3].number_input(f"Grp##{i}", min_value=0, max_value=5,
                                          value=0, step=1, key=f"grp_{i}",
                                          label_visibility="collapsed")
            gap = cols[4].number_input(f"Gap##{i}", min_value=0.0, max_value=4.0,
                                       value=0.0, step=0.1, format="%.1f",
                                       key=f"gap_{i}", label_visibility="collapsed")
            cols[5].markdown(f"<div style='background:{bg_color};padding:4px;border-radius:4px;'>"
                           f"{seg_type}</div>", unsafe_allow_html=True)

            # Validate
            if pos > 0 and group > 0:
                if pos > group:
                    st.error(f"Segment {i+1}: Position ({pos}) cannot exceed group size ({group}).")
                    has_error = True
                elif group < 1 or group > 5 or pos < 1 or pos > 5:
                    st.error(f"Segment {i+1}: Position and group size must be whole numbers 1–5.")
                    has_error = True
                else:
                    seg_config_data.append({
                        "idx": i,
                        "pos_of_n": f"{pos}of{group}",
                        "gap_m": gap if gap > 0 else 1.0,
                    })
            else:
                # Unfilled = solo
                seg_config_data.append({
                    "idx": i,
                    "pos_of_n": "1of1",
                    "gap_m": 0.0,
                })

    if has_error:
        st.error("Please fix the errors above before results can be computed.")
        st.stop()

    # Build plan from per-segment data
    plan_rows = []
    for sd in seg_config_data:
        plan_rows.append({
            "pos_of_n": sd["pos_of_n"],
            "gap_m": sd["gap_m"],
            "is_curve_offset": 0,
        })
    plan = pd.DataFrame(plan_rows)

# ══════════════════════════════════════════════════════════════════════
# Compute results
# ══════════════════════════════════════════════════════════════════════
pace_seconds = pace_min * 60 + pace_sec
baseline_speed = EVENT_DIST / pace_seconds

result = compute_segment_savings(
    baseline_speed_ms=baseline_speed,
    event_distance_m=EVENT_DIST,
    segment_length_m=SEG_LEN,
    drafting_plan=plan,
    shoe_re_pct=product_re_pct,
    air_density=air_density,
    height_m=height_m,
    mass_kg=mass_kg,
)

# ══════════════════════════════════════════════════════════════════════
# Results (ABOVE the plots)
# ══════════════════════════════════════════════════════════════════════
st.markdown("---")
st.subheader("📊 Performance Prediction")

eq_time = result["equivalent_finish_s"]
saved = result["time_saved_s"]
pct = result["pct_improvement"]
draft_s = result["drafting_contribution_s"]
shoe_s = result["shoe_contribution_s"]

col1, col2, col3, col4 = st.columns(4)
col1.metric("Baseline", format_time(pace_seconds))
col2.metric("Equivalent Time", format_time(eq_time))
col3.metric("Time Saved", f"{saved:.2f} s")
col4.metric("Improvement", f"{pct:.2f} %")

col5, col6 = st.columns(2)
col5.metric("↳ Drafting", f"{draft_s:.2f} s")
col6.metric("↳ Product RE", f"{shoe_s:.2f} s")

# ══════════════════════════════════════════════════════════════════════
# Plots (side by side)
# ══════════════════════════════════════════════════════════════════════
st.markdown("---")

df = result["segments_df"]
plot_col1, plot_col2 = st.columns(2)

# ── Left plot: Time saved per segment (bars pointing DOWN) ────────
with plot_col1:
    fig1, ax1 = plt.subplots(figsize=(7, 4.5))

    x = df["end_m"]
    width = df["length_m"] * 0.75

    draft_neg = -df["drafting_saved_s"]
    shoe_neg = -df["shoe_saved_s"]

    ax1.bar(x, draft_neg, width=width, color="#2ecc71", alpha=0.85, label="Drafting")
    ax1.bar(x, shoe_neg, width=width, bottom=draft_neg, color="#3498db", alpha=0.85, label="Product RE")

    # Shade solo segments
    for _, row in df.iterrows():
        if row["pos_of_n"] == "1of1":
            ax1.axvspan(row["start_m"], row["end_m"], color="lightgrey", alpha=0.15, zorder=0)

    ax1.axhline(0, color="black", lw=0.8)
    ax1.set_xlabel("Distance (m)")
    ax1.set_ylabel("Time Saved (s)")
    ax1.set_title("Time Saved Per Segment (vs Baseline)")
    ax1.legend(loc="lower right", fontsize=9)

    # Dynamic y-axis
    y_min = (draft_neg + shoe_neg).min()
    if y_min < 0:
        ax1.set_ylim(y_min * 1.25, max(0.02, -y_min * 0.08))
    else:
        ax1.set_ylim(-0.05, 0.02)
    ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.2f}"))

    plt.tight_layout()
    st.pyplot(fig1)
    plt.close(fig1)

# ── Right plot: Speed comparison line plot ────────────────────────
with plot_col2:
    fig2, ax2 = plt.subplots(figsize=(7, 4.5))

    x = df["end_m"]
    baseline_v = df["baseline_speed_ms"].iloc[0]

    ax2.axhline(baseline_v, color="black", ls="--", lw=2, label=f"Baseline ({baseline_v:.3f} m/s)")
    ax2.plot(x, df["equivalent_speed_ms"], "o-", color="#9b59b6", lw=2,
             markersize=5, label="Equivalent speed")

    # Shade curve segments
    for i_seg in range(N_SEG):
        if seg_df.loc[i_seg, "is_bend"]:
            ax2.axvspan(seg_df.loc[i_seg, "start_m"], seg_df.loc[i_seg, "end_m"],
                       color="lightblue", alpha=0.15, zorder=0)

    ax2.set_xlabel("Distance (m)")
    ax2.set_ylabel("Speed (m/s)")
    ax2.set_title("Equivalent Speed at Same Metabolic Cost")
    ax2.legend(loc="upper right", fontsize=9)

    # Dynamic y-axis
    v_max = df["equivalent_speed_ms"].max()
    padding = max(0.02, (v_max - baseline_v) * 0.3)
    ax2.set_ylim(baseline_v - padding, v_max + padding)

    plt.tight_layout()
    st.pyplot(fig2)
    plt.close(fig2)

# ══════════════════════════════════════════════════════════════════════
# Data table (expandable, at bottom)
# ══════════════════════════════════════════════════════════════════════
with st.expander("View per-segment data table"):
    display_df = result["segments_df"][[
        "segment_idx", "start_m", "end_m", "is_bend", "pos_of_n", "gap_m",
        "r_drag", "baseline_speed_ms", "equivalent_speed_ms",
        "time_saved_s", "drafting_saved_s", "shoe_saved_s", "re_improvement_pct",
    ]].round(4)
    st.dataframe(display_df, width="stretch")

# ══════════════════════════════════════════════════════════════════════
# Footer
# ══════════════════════════════════════════════════════════════════════
st.markdown("---")
st.caption("""
**References:** Beaumont et al. (2021) — CFD drag reduction;
Schickhofer & Hanson (2021) — gap sensitivity;
Kipp et al. (2019) & Batliner et al. (2018) — VO₂–velocity;
Pugh (1971) — metabolic cost of air resistance.
""")
