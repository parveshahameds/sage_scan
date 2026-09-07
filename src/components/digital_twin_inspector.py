import streamlit as st
import pandas as pd
from typing import Dict, Any, List

def render_digital_twin_inspector(
    ground_truth_snapshot: Dict[str, Any],
    sage_knowledge_summary: Dict[str, Any]
):
    """Renders the Digital Twin Inspector (God Mode) comparing True Reality vs Learned AI Knowledge."""
    st.markdown("### 👁️ Digital Twin Inspector: Ground Truth vs. Learned Knowledge")
    st.info("💡 **God Mode Active:** This view reveals what the physical world is doing versus what SAGE-SCAN has learned under partial observability.")

    col1, col2 = st.columns(2)
    with col1:
        with st.container(border=True):
            st.error("🌍 **DIGITAL TWIN GROUND TRUTH (The World)**\n\n*Complete physical reality known only to the simulator.*")

            threat_states = ground_truth_snapshot.get("threat_states", [])
            gt_rows = []
            for t in threat_states:
                gt_rows.append({
                    "Threat": t["name"],
                    "Channel": t["frequency_label"],
                    "True Behavior": t["behavior"],
                    "Current Status": t["status"]
                })
            st.dataframe(pd.DataFrame(gt_rows), use_container_width=True, hide_index=True)
            active_list = ground_truth_snapshot.get('active_channels', [])
            active_str = ', '.join([f'F{b}' for b in active_list]) if active_list else 'None (Silent)'
            st.caption(f"Active Channels Right Now: **{active_str}**")

    with col2:
        with st.container(border=True):
            st.success("🧠 **SAGE-SCAN INFERRED STATE (The Brain)**\n\n*Inferred strictly from 1-channel receiver observations over time.*")

            sage_rows = []
            # Periodic insight
            p_info = sage_knowledge_summary.get("periodic_insight", {})
            if p_info.get("period", 0) > 0:
                sage_rows.append({
                    "Inferred Pattern": f"Periodic Pulse on F{p_info.get('band', 15)}",
                    "Learned Value": f"Period = {p_info.get('period')} slots",
                    "Confidence": p_info.get("confidence_pct", "0%")
                })
            else:
                sage_rows.append({
                    "Inferred Pattern": "Periodic Pulse",
                    "Learned Value": "Gathering observations...",
                    "Confidence": "0%"
                })

            # Agile insight
            agile_info = sage_knowledge_summary.get("agile_chains", [])
            if agile_info:
                top_chain = agile_info[0]
                sage_rows.append({
                    "Inferred Pattern": f"Agile Hop F{top_chain['from_band']} → F{top_chain['to_band']}",
                    "Learned Value": f"Transitions: {top_chain['count']}",
                    "Confidence": top_chain.get("confidence_pct", "0%")
                })
            else:
                sage_rows.append({
                    "Inferred Pattern": "Agile Hopping Chain",
                    "Learned Value": "Tracking transitions...",
                    "Confidence": "0%"
                })

            # Exploration status
            unexplored_count = sage_knowledge_summary.get("unexplored_count", 0)
            sage_rows.append({
                "Inferred Pattern": "Spectrum Awareness",
                "Learned Value": f"{unexplored_count} cold channels left",
                "Confidence": "UCB Exploration Active"
            })

            st.dataframe(pd.DataFrame(sage_rows), use_container_width=True, hide_index=True)
            st.caption("SAGE-SCAN never reads ground-truth variables directly.")
