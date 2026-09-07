import streamlit as st
import pandas as pd
import plotly.express as px
from typing import Dict, Any

def render_decision_explanation(decision_info: Dict[str, Any], technical_view: bool = False):
    """Renders the transparent 'Why Did I Choose This?' decision breakdown panel."""
    st.markdown("### 🧠 Why Did I Choose This? — Decision Explainability")
    
    if not decision_info:
        st.info("Simulation initializing. Awaiting first scheduling decision...")
        return

    selected_b = decision_info.get("selected_band", 0)
    decision_type = decision_info.get("decision_type", "EXPLORE")
    dominant = decision_info.get("dominant_component", "Exploration")
    reason = decision_info.get("reason", "Exploring unobserved channels")
    is_pred = decision_info.get("is_prediction_active", False)
    
    action_label = decision_info.get("action_label", decision_info.get("decision_type", "EXPLORE"))
    best_pred = decision_info.get("best_prediction")
    
    if best_pred:
        conf_pct = best_pred.get("confidence_pct", "0%")
        delta_conf = f"Target: {best_pred.get('channel_label', f'F{selected_b}')} ({best_pred.get('time_to_event', 0)}s ⏳)"
        d_color = "normal"
    elif is_pred:
        conf_pct = decision_info.get("prediction_confidence_pct", "0%")
        delta_conf = f"Driver: {dominant}"
        d_color = "normal"
    else:
        conf_pct = "N/A"
        uncert = decision_info.get("exploration_uncertainty", 0.0)
        delta_conf = f"Uncertainty: {uncert:.2f}"
        d_color = "off"
        
    sel_score = decision_info.get("selection_score", 0.0)
    strategy = decision_info.get("strategy", "Information Gain")
    target_time = decision_info.get("target_time", 0)

    # 1. Summary Cards (Clear separation of Selection Score and Prediction Confidence)
    with st.container(border=True):
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.metric("SELECTED CHANNEL", f"F{selected_b}", delta=f"Action: {action_label}")
        with c2:
            st.metric("SELECTION SCORE", f"{sel_score:.3f}", delta="Scheduler Preference", delta_color="off")
        with c3:
            st.metric("PREDICTION CONFIDENCE", conf_pct, delta=delta_conf, delta_color=d_color)
        with c4:
            st.metric("STRATEGY", strategy, delta=f"Target Slot: t = {target_time}", delta_color="off")
        st.caption(f"**Action Rationale:** {reason}")

    # 2. Four Component Scores Breakdown Table & Plotly Horizontal Chart
    breakdowns = decision_info.get("scores_breakdown", {})
    if breakdowns:
        rows = []
        chart_data = []
        for name, data in breakdowns.items():
            w = data.get("weight", 0.0)
            raw = data.get("raw", 0.0)
            score = data.get("score", 0.0)
            pct = (score / sel_score * 100) if sel_score > 0 else 0.0
            
            rows.append({
                "Decision Signal": name,
                "Model Weight (w)": f"{w:.2f}",
                "Raw Value": f"{raw:.3f}",
                "Weighted Score": f"{score:.3f}",
                "Contribution": f"{pct:.1f}%"
            })
            chart_data.append({"Signal": name, "Weighted Score": score})

        df_table = pd.DataFrame(rows)
        df_chart = pd.DataFrame(chart_data)

        col_t, col_c = st.columns([1, 1])
        with col_t:
            st.dataframe(df_table, use_container_width=True, hide_index=True)
            
        with col_c:
            fig = px.bar(
                df_chart,
                x="Weighted Score",
                y="Signal",
                orientation="h",
                template="plotly_dark",
                color="Signal",
                color_discrete_map={
                    "History": "#38bdf8",
                    "Temporal / Timing": "#10b981",
                    "Transition / Agile": "#f59e0b",
                    "Exploration (UCB)": "#a855f7"
                }
            )
            fig.update_layout(
                height=160,
                margin=dict(l=10, r=10, t=10, b=10),
                paper_bgcolor="#0e1117",
                plot_bgcolor="#0b0f19",
                showlegend=False
            )
            st.plotly_chart(fig, use_container_width=True)

    if technical_view:
        st.markdown("---")
        st.markdown("#### 🔬 Mathematical Formulation")
        st.latex(r"\text{SelectionScore}(f, t) = w_{\text{hist}} \cdot P_{\text{hist}}(f) + w_{\text{temp}} \cdot P_{\text{temp}}(f, t) + w_{\text{trans}} \cdot P_{\text{trans}}(f) + c \cdot \sqrt{\frac{\ln T}{N_f + 1}}")
        st.caption("Selection score dictates receiver scheduling; Prediction confidence represents estimated physical emission likelihood.")
