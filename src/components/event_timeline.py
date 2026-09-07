import streamlit as st
import pandas as pd
from typing import List, Dict, Any

def render_event_timeline(history_records: List[Dict[str, Any]], max_items: int = 15):
    """Renders a live event timeline of simulation actions and predictions using native Streamlit UI."""
    st.markdown("### 📜 Live Event & Prediction Timeline")
    
    if not history_records:
        st.caption("No events logged yet. Advance simulation to generate timeline.")
        return

    # Take most recent records in reverse chronological order
    recent = list(reversed(history_records[-max_items:]))
    
    events_data = []
    for rec in recent:
        t = rec["time"]
        sage_b = rec["sage_band"]
        sage_obs = rec["sage_obs"]
        alert = rec.get("prediction_alert")
        detected = sage_obs.get("detected", False)
        
        event_tag = "Scanned Channel"
        icon = "📡"
        
        if alert and alert.get("type") == "PREDICTION_SUCCESS":
            icon = "🎯"
            event_tag = "PREDICTION SUCCESS"
        elif alert and alert.get("type") == "PREEMPTIVE_INTERCEPT":
            icon = "⚡"
            event_tag = "PREEMPTIVE INTERCEPT"
        elif alert and alert.get("type") == "UNKNOWN_DISCOVERED":
            icon = "🛸"
            event_tag = "UNKNOWN SIGNAL DISCOVERED"
        elif alert and alert.get("type") == "PREDICTION_MISS":
            icon = "❌"
            event_tag = "Prediction Miss"
        elif detected:
            icon = "🟢"
            event_tag = "Signal Intercepted"
        else:
            icon = "⚪"
            event_tag = "Idle Scan"

        events_data.append({
            "Time": f"t = {t}",
            "Status": f"{icon} {event_tag}",
            "Channel": f"F{sage_b}",
            "Observation": "HIT 🎯" if detected else "MISS ⚪"
        })

    df_events = pd.DataFrame(events_data)
    st.dataframe(df_events, use_container_width=True, hide_index=True)
