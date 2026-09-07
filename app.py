import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import time
from typing import Dict, List, Any, Optional

# Core imports
from src.core.threat_engine import PeriodicThreat, AgileThreat, IntermittentThreat, UnknownThreat
from src.core.digital_twin import DigitalTwinEngine
from src.core.virtual_receiver import VirtualReceiver
from src.core.sage_engine import SAGEScanEngine
from src.core.comparison_engine import ComparisonSimulator
from src.core.demo_controller import JudgeDemoController

# Component renderers
from src.components.spectrum_view import render_spectrum_heatmap, render_channel_matrix_figure
from src.components.threat_panel import render_threat_panel
from src.components.prediction_panel import render_prediction_panel
from src.components.decision_explanation import render_decision_explanation
from src.components.comparison_panel import render_comparison_panel
from src.components.event_timeline import render_event_timeline
from src.components.digital_twin_inspector import render_digital_twin_inspector

# Page Config
st.set_page_config(
    page_title="SAGE-SCAN | Predictive RF Threat Detection",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------------------------------------------------
# SESSION STATE INITIALIZATION
# ---------------------------------------------------------
if "sim_engine" not in st.session_state:
    num_bands = 50
    st.session_state.num_bands = num_bands
    st.session_state.seed = 42
    st.session_state.w_hist = 0.3
    st.session_state.w_temp = 0.5
    st.session_state.w_trans = 0.4
    st.session_state.c_explore = 0.3
    
    st.session_state.digital_twin = DigitalTwinEngine(num_bands=num_bands, max_time=1000, seed=42)
    st.session_state.sage_engine = SAGEScanEngine(num_bands=num_bands, w_hist=0.3, w_temp=0.5, w_trans=0.4, c_explore=0.3)
    st.session_state.sim_engine = ComparisonSimulator(st.session_state.digital_twin, st.session_state.sage_engine, seed=42)
    st.session_state.demo_controller = JudgeDemoController(seed=42)
    
    st.session_state.is_running = False
    st.session_state.sim_speed = 0.15
    st.session_state.technical_view = False
    st.session_state.god_mode = False
    st.session_state.selected_channel_inspect = 15
    st.session_state.confidence_history = []

# ---------------------------------------------------------
# SIDEBAR CONTROLS
# ---------------------------------------------------------
st.sidebar.title("📡 SAGE-SCAN")
st.sidebar.caption("**Predictive RF Threat Detection & Intelligent Spectrum Scheduler**")

nav_page = st.sidebar.radio(
    "NAVIGATION",
    [
        "📡 Live Command Center",
        "🎬 Judge Demonstration",
        "⚡ Comparative Benchmark",
        "🛠️ Scenario & Threat Injector",
        "📖 Architecture & Insights"
    ]
)

st.sidebar.markdown("---")
st.sidebar.subheader("🕹️ Simulation Engine Controls")

c1, c2 = st.sidebar.columns(2)
with c1:
    if st.button("▶ PLAY", use_container_width=True, type="primary"):
        st.session_state.is_running = True
with c2:
    if st.button("⏸ PAUSE", use_container_width=True):
        st.session_state.is_running = False

c3, c4 = st.sidebar.columns(2)
with c3:
    if st.button("⏭ STEP", use_container_width=True):
        if nav_page == "🎬 Judge Demonstration":
            st.session_state.demo_controller.step()
        else:
            st.session_state.sim_engine.step()
with c4:
    if st.button("↻ RESET", use_container_width=True):
        st.session_state.sim_engine.reset()
        st.session_state.demo_controller.reset()
        st.session_state.is_running = False
        st.session_state.confidence_history = []

if "speed_select_slider" not in st.session_state:
    st.session_state.speed_select_slider = "1.0x"

speed_label = st.sidebar.select_slider(
    "Simulation Speed",
    options=["0.5x", "1.0x", "2.0x", "5.0x", "10.0x"],
    key="speed_select_slider"
)
speed_map = {
    "0.5x": {"delay": 0.40, "steps": 1},
    "1.0x": {"delay": 0.15, "steps": 1},
    "2.0x": {"delay": 0.08, "steps": 2},
    "5.0x": {"delay": 0.04, "steps": 5},
    "10.0x": {"delay": 0.01, "steps": 10},
}
speed_cfg = speed_map.get(speed_label, speed_map["1.0x"])
st.session_state.sim_speed = speed_cfg["delay"]
st.session_state.sim_delay = speed_cfg["delay"]
st.session_state.sim_steps = speed_cfg["steps"]

st.sidebar.markdown("---")
st.sidebar.subheader("⚙️ Quick Threat Injections")
if st.sidebar.button("🛸 SPAWN UNKNOWN THREAT (F2)", use_container_width=True):
    st.session_state.digital_twin.spawn_unknown_threat(band=2)
    st.sidebar.success("Covert signal spawned on F2! SAGE exploration will seek it.")

if st.sidebar.button("⚡ INJECT FAST AGILE HOPPER", use_container_width=True):
    fast_hopper = AgileThreat(threat_id="AGILE-FAST", name="FAST HOPPER", bands=[5, 18, 32, 44], hop_interval=8, prob=1.0)
    st.session_state.digital_twin.inject_threat(fast_hopper)
    st.sidebar.success("Injected fast agile radar (F5 → F18 → F32 → F44)")

st.sidebar.markdown("---")
st.session_state.god_mode = st.sidebar.toggle("👁️ Digital Twin Inspector (God Mode)", value=st.session_state.god_mode)
st.session_state.technical_view = st.sidebar.toggle("🔬 Technical View (Formulas & Scores)", value=st.session_state.technical_view)

# ---------------------------------------------------------
# AUTO-STEPPING WHEN PLAY IS ACTIVE
# ---------------------------------------------------------
if st.session_state.is_running:
    steps_to_run = st.session_state.get("sim_steps", 1)
    for _ in range(steps_to_run):
        if nav_page == "🎬 Judge Demonstration":
            if st.session_state.demo_controller.is_finished or st.session_state.demo_controller.digital_twin.current_time >= st.session_state.demo_controller.max_demo_time:
                st.session_state.is_running = False
                break
            else:
                st.session_state.demo_controller.step()
                if st.session_state.demo_controller.is_finished:
                    st.session_state.is_running = False
                    break
        else:
            if st.session_state.digital_twin.current_time >= st.session_state.digital_twin.max_time - 1:
                st.session_state.is_running = False
                break
            else:
                st.session_state.sim_engine.step()
                if st.session_state.digital_twin.current_time >= st.session_state.digital_twin.max_time - 1:
                    st.session_state.is_running = False
                    break
        
        # Record confidence history strictly from live model decision
        curr_t = st.session_state.digital_twin.current_time
        last_dec = st.session_state.sage_engine.last_decision
        if last_dec and last_dec.get("best_prediction"):
            conf = float(last_dec["best_prediction"].get("confidence", 0.0))
        elif last_dec and last_dec.get("is_prediction_active"):
            conf = float(last_dec.get("prediction_confidence", 0.0))
        else:
            conf = 0.0
        st.session_state.confidence_history.append({"time": curr_t, "confidence": conf * 100})
        if len(st.session_state.confidence_history) > 100:
            st.session_state.confidence_history.pop(0)

# ---------------------------------------------------------
# PAGE 1: LIVE COMMAND CENTER
# ---------------------------------------------------------
if nav_page == "📡 Live Command Center":
    # Header Banner
    with st.container(border=True):
        h1, h2 = st.columns([3, 1])
        with h1:
            st.title("📡 SAGE-SCAN // Live Spectrum Command Center")
            st.caption('**"Don\'t scan for threats. Predict them."** — Closed-Loop Adaptive Spectrum Scheduling')
        with h2:
            st.metric(label="SYSTEM STATUS", value="ONLINE ⚡", delta="Sensing Spectrum")

    # Conceptual loop visualizer
    t_clock = st.session_state.digital_twin.current_time
    st.info("🔄 **THE SAGE-SCAN CLOSED LOOP:** `1. OBSERVE` ➔ `2. LEARN` ➔ `3. PREDICT` ➔ `4. PRE-POSITION` ➔ `5. INTERCEPT` ➔ `6. LEARN AGAIN`")

    # Extract state variables
    sage_engine = st.session_state.sage_engine
    digital_twin = st.session_state.digital_twin
    sim_engine = st.session_state.sim_engine
    
    rx_band = sim_engine.sage_receiver.current_band
    last_decision = sage_engine.last_decision
    last_alert = sage_engine.last_prediction_alert
    
    # Upcoming periodic pulse preview
    upcoming_pulse = sage_engine.temporal_model.get_upcoming_pulse(15, t_clock)

    # 1. Prediction & Status Panel
    render_prediction_panel(
        current_time=t_clock,
        receiver_band=rx_band,
        decision_info=last_decision,
        last_alert=last_alert,
        upcoming_pulse=upcoming_pulse
    )

    st.markdown("---")

    # 2. Main Spectrum Visualizer Grid & Waterfall
    left_col, right_col = st.columns([1, 1])
    
    with left_col:
        st.markdown("### 🎛️ 50-Channel Interactive Spectrum Matrix")
        st.caption("F0 → F49 Channels. Real-time receiver lock (📡), predicted target (🔮), threat emission (🚨), and intercepts (🎯).")
        
        gt_snapshot = digital_twin.get_ground_truth_snapshot()
        active_bands = gt_snapshot.get("active_bands", [])
        pred_target = last_decision.get("selected_band", None) if last_decision else None

        fig_matrix = render_channel_matrix_figure(
            num_bands=50,
            current_receiver_band=rx_band,
            predicted_band=pred_target,
            active_threat_bands=active_bands,
            detection_counts=sage_engine.history_model.num_hits,
            observation_counts=sage_engine.exploration_model.observation_counts,
            selected_inspect_band=st.session_state.selected_channel_inspect
        )
        st.plotly_chart(fig_matrix, use_container_width=True)

        # Interactive channel selector
        c_sel, c_info = st.columns([1, 2])
        with c_sel:
            st.session_state.selected_channel_inspect = st.selectbox(
                "Inspect Channel Profile",
                options=list(range(50)),
                index=st.session_state.selected_channel_inspect,
                format_func=lambda x: f"Channel F{x}"
            )
        with c_info:
            insp_b = st.session_state.selected_channel_inspect
            b_prof = sage_engine.history_model.get_band_profile(insp_b)
            diag = sage_engine.temporal_model.get_learning_diagnostics(insp_b, t_clock)
            exp_prof = sage_engine.exploration_model.get_exploration_profile(insp_b, t_clock)
            
            with st.container(border=True):
                countdown_str = f" (in {diag['slots_until_next']} slot{'s' if diag['slots_until_next'] > 1 else ''} ⏳)" if (diag['slots_until_next'] is not None and diag['slots_until_next'] > 0) else (" (ACTIVE NOW 🎯)" if diag['slots_until_next'] == 0 else "")
                st.markdown(
                    f"**📡 Channel F{insp_b} — Live Pattern Learning Progression:**\n\n"
                    f"• **Learning Stage:** `{diag['learning_stage']}` &nbsp;|&nbsp; **Monitoring:** `{exp_prof['status']}`\n"
                    f"• **Receiver Detections ({len(diag['detection_timestamps'])} hits):** `{diag['timestamps_display']}`\n"
                    f"• **Learned Intervals (Δt):** `{diag['intervals_display']}` &nbsp;|&nbsp; **Consistency:** `{diag['period_consistency']}`\n"
                    f"• **Estimated Period:** `{diag['estimated_period']}` slots &nbsp;|&nbsp; **Prediction Confidence:** `{diag['confidence_pct']}`\n"
                    f"• **Next Predicted Pulse:** `{diag['next_predicted_pulse']}`{countdown_str}"
                )

    with right_col:
        st.markdown("### 🌊 Real-Time Spectrum Waterfall Heatmap")
        st.caption("Time-Frequency 2D Spectrogram showing past threat emissions, scans, and verified intercepts.")
        
        fig_waterfall = render_spectrum_heatmap(
            ground_truth_matrix=digital_twin.ground_truth,
            receiver_history=sim_engine.sage_receiver.observation_history,
            current_time=t_clock,
            num_bands=50,
            window_slots=45,
            predicted_band=pred_target
        )
        st.plotly_chart(fig_waterfall, use_container_width=True)

    st.markdown("---")

    # 3. Decision Explainability Panel
    render_decision_explanation(last_decision, technical_view=st.session_state.technical_view)

    st.markdown("---")

    # 4. Threats Panel & Live Event Timeline
    t_col, e_col = st.columns([1, 1])
    with t_col:
        render_threat_panel(gt_snapshot.get("threat_states", []), t_clock)
    with e_col:
        render_event_timeline(sim_engine.history, max_items=12)

    # 5. Optional Digital Twin Inspector (God Mode)
    if st.session_state.god_mode:
        st.markdown("---")
        p15, off15, conf15 = sage_engine.temporal_model.get_periodicity(15)
        sage_summary = {
            "periodic_insight": {
                "band": 15,
                "period": p15,
                "confidence_pct": f"{conf15*100:.0f}%"
            },
            "agile_chains": sage_engine.transition_model.get_learned_chains(),
            "unexplored_count": len(sage_engine.exploration_model.get_unexplored_bands())
        }
        render_digital_twin_inspector(gt_snapshot, sage_summary)

# ---------------------------------------------------------
# PAGE 2: JUDGE DEMONSTRATION MODE
# ---------------------------------------------------------
elif nav_page == "🎬 Judge Demonstration":
    controller = st.session_state.demo_controller
    current_phase = controller.current_phase
    t_clock = controller.digital_twin.current_time

    with st.container(border=True):
        st.title("🎬 JUDGE DEMONSTRATION // Deterministic 7-Phase Showcase")
        st.caption("A scripted 60–90 second reproducible demonstration proving the complete 14-step intelligence lifecycle.")

    # Phase progress bar
    progress_val = min(1.0, current_phase / 7.0)
    st.progress(progress_val)
    
    phase_title = controller.phase_descriptions.get(current_phase, "")
    st.warning(f"**{phase_title}** (Simulation Time: **t = {t_clock} / 120**)")

    # Demo Controls
    d_c1, d_c2, d_c3, d_c4 = st.columns(4)
    with d_c1:
        if st.button("▶ START / RESUME DEMO", use_container_width=True, type="primary"):
            st.session_state.is_running = True
    with d_c2:
        if st.button("⏸ PAUSE DEMO", use_container_width=True):
            st.session_state.is_running = False
    with d_c3:
        if st.button("⏭ SINGLE STEP", use_container_width=True):
            controller.step()
    with d_c4:
        if st.button("↻ RESTART DEMO", use_container_width=True):
            controller.reset()
            st.session_state.is_running = False

    st.markdown("---")

    # Display live demo elements
    sage_eng = controller.sage_engine
    sim = controller.comparison_sim
    dt = controller.digital_twin
    
    last_dec = sage_eng.last_decision
    last_alert = sage_eng.last_prediction_alert
    rx_band = sim.sage_receiver.current_band

    # Prediction & alert panel
    render_prediction_panel(
        current_time=t_clock,
        receiver_band=rx_band,
        decision_info=last_dec,
        last_alert=last_alert,
        upcoming_pulse=sage_eng.temporal_model.get_upcoming_pulse(15, t_clock)
    )

    st.markdown("---")

    # Side-by-side Spectrum & Performance
    d_col1, d_col2 = st.columns([1, 1])
    with d_col1:
        st.markdown("#### Live Spectrum Status")
        gt_snap = dt.get_ground_truth_snapshot()
        fig_demo_matrix = render_channel_matrix_figure(
            num_bands=50,
            current_receiver_band=rx_band,
            predicted_band=last_dec.get("selected_band", None) if last_dec else None,
            active_threat_bands=gt_snap.get("active_bands", []),
            detection_counts=sage_eng.history_model.num_hits,
            observation_counts=sage_eng.exploration_model.observation_counts
        )
        st.plotly_chart(fig_demo_matrix, use_container_width=True)

    with d_col2:
        st.markdown("#### SAGE-SCAN vs. Sequential Sweep")
        comp_summary = sim.get_comparison_summary()
        render_comparison_panel(comp_summary)

    if current_phase >= 7:
        st.markdown("---")
        st.success(
            "### 🎉 DEMONSTRATION COMPLETE\n\n"
            "**The Digital Twin shows the world. SAGE-SCAN learns the world.**\n\n"
            "• Traditional Scanner: `SCAN ➔ MISS ➔ SCAN ➔ MISS`\n"
            "• SAGE-SCAN: `OBSERVE ➔ LEARN ➔ PREDICT ➔ PRE-POSITION ➔ INTERCEPT`"
        )

# ---------------------------------------------------------
# PAGE 3: COMPARATIVE BENCHMARK
# ---------------------------------------------------------
elif nav_page == "⚡ Comparative Benchmark":
    with st.container(border=True):
        st.title("⚡ COMPARATIVE BENCHMARK ENGINE // Head-to-Head Testing")
        st.caption("Simultaneously evaluates Sequential Sweep against SAGE-SCAN across identical simulated RF timelines.")

    sim_engine = st.session_state.sim_engine
    comp_summary = sim_engine.get_comparison_summary()
    
    render_comparison_panel(comp_summary)

    st.markdown("---")
    st.markdown("### 📊 Real-Time Detection Timeline Comparison")
    
    hist = sim_engine.history
    if hist:
        df_hist = []
        for r in hist:
            t = r["time"]
            df_hist.append({
                "Time": t,
                "Sequential Band": r["seq_band"],
                "SAGE-SCAN Band": r["sage_band"],
                "Sequential Detected": 1 if r["seq_obs"]["detected"] else 0,
                "SAGE Detected": 1 if r["sage_obs"]["detected"] else 0
            })
        df_plot = pd.DataFrame(df_hist)
        
        fig_cum = go.Figure()
        fig_cum.add_trace(go.Scatter(x=df_plot["Time"], y=np.cumsum(df_plot["SAGE Detected"]), mode="lines+markers", name="SAGE-SCAN Detections", line=dict(color="#10b981", width=3)))
        fig_cum.add_trace(go.Scatter(x=df_plot["Time"], y=np.cumsum(df_plot["Sequential Detected"]), mode="lines+markers", name="Sequential Sweep Detections", line=dict(color="#ef4444", width=2, dash="dash")))
        
        fig_cum.update_layout(
            title="Cumulative Target Signals Intercepted Over Time",
            xaxis=dict(title="Simulation Time Slot (t)", gridcolor="#1f2937"),
            yaxis=dict(title="Cumulative Interceptions", gridcolor="#1f2937"),
            template="plotly_dark",
            height=300,
            paper_bgcolor="#0e1117",
            plot_bgcolor="#0b0f19"
        )
        st.plotly_chart(fig_cum, use_container_width=True)
    else:
        st.info("Advance simulation to build cumulative detection comparison curves.")

# ---------------------------------------------------------
# PAGE 4: SCENARIO EDITOR & THREAT INJECTOR
# ---------------------------------------------------------
elif nav_page == "🛠️ Scenario & Threat Injector":
    with st.container(border=True):
        st.title("🛠️ DIGITAL TWIN SCENARIO EDITOR // Sandbox")
        st.caption("Configure custom RF emitter profiles, hop intervals, burst probabilities, and inject live scenarios into the Digital Twin.")

    c_ed1, c_ed2 = st.columns(2)
    with c_ed1:
        st.markdown("#### Custom Periodic Radar Emitter")
        p_name = st.text_input("Threat Name", value="CUSTOM-PERIODIC-01")
        p_band = st.slider("Target Channel (F0 - F49)", 0, 49, 22)
        p_period = st.slider("Pulse Interval (slots)", 2, 25, 9)
        p_offset = st.slider("Time Offset", 0, 10, 0)
        
        if st.button("➕ Inject Periodic Threat", use_container_width=True):
            new_p = PeriodicThreat(threat_id=f"THREAT-P{p_band}", name=p_name, band=p_band, period=p_period, offset=p_offset)
            st.session_state.digital_twin.inject_threat(new_p)
            st.success(f"Injected {p_name} on F{p_band} with period={p_period} slots!")

    with c_ed2:
        st.markdown("#### Custom Agile Hopping Radar")
        a_name = st.text_input("Agile Threat Name", value="CUSTOM-AGILE-01")
        a_bands_str = st.text_input("Hopping Channels (comma-separated)", value="8, 16, 24, 32")
        a_hop_int = st.slider("Dwell per Hop (slots)", 5, 30, 12)
        
        if st.button("➕ Inject Agile Threat", use_container_width=True):
            try:
                bands_list = [int(b.strip()) for b in a_bands_str.split(",") if b.strip().isdigit()]
                if bands_list:
                    new_a = AgileThreat(threat_id=f"THREAT-A{bands_list[0]}", name=a_name, bands=bands_list, hop_interval=a_hop_int)
                    st.session_state.digital_twin.inject_threat(new_a)
                    st.success(f"Injected agile threat hopping across {bands_list}!")
            except Exception as e:
                st.error(f"Invalid input: {e}")

# ---------------------------------------------------------
# PAGE 5: ARCHITECTURE & INSIGHTS
# ---------------------------------------------------------
else:
    with st.container(border=True):
        st.title("📖 SYSTEM ARCHITECTURE & PHILOSOPHY // Blueprint")
        st.caption("The fundamental mathematical and structural difference between Blind Scanning and Predictive Sensing.")

    st.markdown("""
    ### 🏛️ The SAGE-SCAN Closed-Loop Architecture
    ```
    ┌──────────────────────────────────────────────────────────────┐
    │                      DIGITAL TWIN                            │
    │  Ground Truth RF Environment: Physical Signals, Time, World  │
    └──────────────────────────────┬───────────────────────────────┘
                                   │
                                   ▼
    ┌──────────────────────────────────────────────────────────────┐
    │                    VIRTUAL RECEIVER                          │
    │  Bandwidth-Limited Sensor (Instantaneous Bandwidth W = 1)    │
    └──────────────────────────────┬───────────────────────────────┘
                                   │ Partial Observation (HIT / MISS)
                                   ▼
    ┌──────────────────────────────────────────────────────────────┐
    │                       SAGE-SCAN                              │
    │  1. History Model (Hit rate empirical tracking)              │
    │  2. Temporal Model (Periodicity & Autocorrelation)           │
    │  3. Transition Model (Agility Markov Hopping Paths)          │
    │  4. Exploration Model (UCB Cold-Band Uncertainty)            │
    └──────────────────────────────┬───────────────────────────────┘
                                   │
                                   ▼ Next Optimal Frequency Action
    ┌──────────────────────────────────────────────────────────────┐
    │                       SCHEDULER                              │
    │  Pre-positions receiver before signal arrival                │
    └──────────────────────────────┬───────────────────────────────┘
                                   │
                                   └────────────► Back to Receiver
    ```
    
    ### 🎯 The Four Intelligent Signals:
    1. **History Model ($P_{\\text{hist}}$):** Empirical hit-to-observation ratio for bursty and dense channels.
    2. **Temporal Model ($P_{\\text{temp}}$):** Modulo-based periodicity engine discovering cadence ($P$) and offset, generating predictions with pre-positioning lead time.
    3. **Transition Model ($P_{\\text{trans}}$):** Markov chain estimating next frequency jump probabilities $P(F_{\\text{next}} \\mid F_{\\text{prev}})$.
    4. **Exploration Model (UCB):** Uncertainty growth term $c \\cdot \\sqrt{\\frac{\\ln T}{N_f + 1}}$ guaranteeing zero blind spots.
    
    ---
    
    ### 💡 Core Product Message
    > **"Traditional scanning asks 'Where should I scan next?'**  
    > **SAGE-SCAN asks 'Where is a signal most likely to appear next?'"**
    """)

# Auto-rerun trigger when simulation is playing
if st.session_state.is_running:
    time.sleep(st.session_state.get("sim_delay", 0.15))
    st.rerun()
