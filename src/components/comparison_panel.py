import streamlit as st
import pandas as pd
import plotly.express as px
from typing import Dict, Any

def render_comparison_panel(comparison_summary: Dict[str, Any]):
    """Renders the comprehensive performance comparison between Sequential Scanner and SAGE-SCAN."""
    st.markdown("### ⚡ Live Head-to-Head Benchmark: Sequential vs. SAGE-SCAN")
    st.caption("Running simultaneously against the exact same Digital Twin RF environment and threat timeline (W = 1).")

    sage_m = comparison_summary.get("sage", {})
    seq_m = comparison_summary.get("sequential", {})

    comp_rows = [
        {
            "Metric": "🎯 Signals Intercepted (Hits)", 
            "Sequential Scanner": f"{seq_m.get('signals_intercepted', 0)} ({seq_m.get('detection_rate_pct', '0.0%')})", 
            "SAGE-SCAN": f"{sage_m.get('signals_intercepted', 0)} ({sage_m.get('detection_rate_pct', '0.0%')})"
        },
        {
            "Metric": "⚪ Signals Missed", 
            "Sequential Scanner": f"{seq_m.get('signals_missed', 0)} ({seq_m.get('miss_rate_pct', '0.0%')})", 
            "SAGE-SCAN": f"{sage_m.get('signals_missed', 0)} ({sage_m.get('miss_rate_pct', '0.0%')})"
        },
        {
            "Metric": "⏱ Average Detection Delay", 
            "Sequential Scanner": seq_m.get("avg_delay_label", "0.0 slots"), 
            "SAGE-SCAN": sage_m.get("avg_delay_label", "0.0 slots")
        },
        {
            "Metric": "🔮 Prediction Accuracy", 
            "Sequential Scanner": "N/A (Blind Sweep)", 
            "SAGE-SCAN": f"{sage_m.get('prediction_hits', 0)}/{sage_m.get('prediction_attempts', 0)} ({sage_m.get('prediction_accuracy_pct', '0.0%')})"
        },
        {
            "Metric": "⚡ Preemptive Intercepts", 
            "Sequential Scanner": "0", 
            "SAGE-SCAN": f"{sage_m.get('preemptive_intercepts', 0)} ({sage_m.get('preemptive_intercept_rate_pct', '0.0%')})"
        },
        {
            "Metric": "🌐 Spectrum Channel Coverage", 
            "Sequential Scanner": seq_m.get("channel_coverage_pct", "0.0%"), 
            "SAGE-SCAN": sage_m.get("channel_coverage_pct", "0.0%")
        },
        {
            "Metric": "🛸 Unknown Threats Discovered", 
            "Sequential Scanner": seq_m.get("unknown_threats_discovered", 0), 
            "SAGE-SCAN": sage_m.get("unknown_threats_discovered", 0)
        }
    ]
    df_comp = pd.DataFrame(comp_rows)

    col1, col2 = st.columns([1, 1])
    with col1:
        st.dataframe(df_comp, use_container_width=True, hide_index=True)

    with col2:
        chart_data = [
            {"Scanner": "Sequential", "Metric": "Intercepted", "Value": seq_m.get("signals_intercepted", 0)},
            {"Scanner": "SAGE-SCAN", "Metric": "Intercepted", "Value": sage_m.get("signals_intercepted", 0)},
            {"Scanner": "Sequential", "Metric": "Missed", "Value": seq_m.get("signals_missed", 0)},
            {"Scanner": "SAGE-SCAN", "Metric": "Missed", "Value": sage_m.get("signals_missed", 0)},
            {"Scanner": "Sequential", "Metric": "Preemptive", "Value": 0},
            {"Scanner": "SAGE-SCAN", "Metric": "Preemptive", "Value": sage_m.get("preemptive_intercepts", 0)}
        ]
        df_plot = pd.DataFrame(chart_data)
        
        fig = px.bar(
            df_plot,
            x="Metric",
            y="Value",
            color="Scanner",
            barmode="group",
            template="plotly_dark",
            color_discrete_map={"Sequential": "#ef4444", "SAGE-SCAN": "#10b981"}
        )
        fig.update_layout(
            height=240,
            margin=dict(l=10, r=10, t=20, b=10),
            paper_bgcolor="#0e1117",
            plot_bgcolor="#0b0f19",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        st.plotly_chart(fig, use_container_width=True)
