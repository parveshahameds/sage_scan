import streamlit as st
from typing import List, Dict, Any

def render_threat_panel(threat_states: List[Dict[str, Any]], current_time: int):
    """Renders Digital Twin simulated threat objects and their real-time states using native containers."""
    st.markdown("### 🎯 Active Simulated Threat Objects")
    
    if not threat_states:
        st.info("No active threat objects in environment.")
        return

    cols = st.columns(len(threat_states))
    for idx, threat in enumerate(threat_states):
        with cols[idx]:
            with st.container(border=True):
                is_active = threat.get("active", False)
                status_label = "EMITTING 🚨" if is_active else "SILENT ⏳"
                
                st.markdown(f"**{threat['id']}** &nbsp; `{status_label}`")
                st.subheader(f"{threat['name']}")
                st.markdown(f"**Channel:** `F{threat.get('current_band', 0)}`")
                st.caption(threat.get("behavior", ""))
                
                # Contextual extra metrics
                t_type = threat.get("type", "")
                if t_type == "Periodic Radar":
                    st.markdown(f"• **Period:** `{threat.get('period', 7)}` slots\n• **Next Pulse:** `t={threat.get('next_emission_time', 0)}`")
                elif t_type == "Frequency Agile":
                    st.markdown(f"• **Next Hop:** `{threat.get('next_hop_label', 'N/A')}`\n• **Hop In:** `{threat.get('slots_until_hop', 0)}` slots")
                elif t_type == "Intermittent / Burst":
                    burst_p = threat.get("burst_probability", 0.18) * 100
                    st.markdown(f"• **Burst Rate:** `{burst_p:.0f}%`\n• **Profile:** `Empirical`")
                else:
                    st.markdown("• **Covert Signal**\n• **Discovery:** `Exploration`")
