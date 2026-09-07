import streamlit as st
from typing import Dict, Any, Optional

def render_prediction_panel(
    current_time: int,
    receiver_band: int,
    decision_info: Dict[str, Any],
    last_alert: Optional[Dict[str, Any]],
    upcoming_pulse: Optional[Dict[str, Any]]
):
    """Renders the prediction command center, countdowns, and real-time prediction alerts using native Streamlit UI."""
    
    # 1. Prediction alert banner using native Streamlit notification containers
    if last_alert and (current_time - last_alert.get("time", 0) <= 2):
        alert_type = last_alert.get("type", "")
        title = last_alert.get("title", "")
        msg = last_alert.get("message", "")
        band = last_alert.get("band", receiver_band)
        
        if alert_type == "PREDICTION_SUCCESS":
            st.success(f"### {title} (Channel F{band})\n{msg}", icon="🎯")
        elif alert_type == "PREEMPTIVE_INTERCEPT":
            st.info(f"### {title} (Channel F{band})\n{msg}", icon="⚡")
        elif alert_type == "UNKNOWN_DISCOVERED":
            st.warning(f"### {title} (Channel F{band})\n{msg}", icon="🛸")
        elif alert_type == "PREDICTION_MISS":
            st.error(f"### {title} (Channel F{band})\n{msg}", icon="❌")
        else:
            st.info(f"### {title} (Channel F{band})\n{msg}", icon="📡")

    # 2. Extract prediction and action metadata
    best_pred = decision_info.get("best_prediction") if decision_info else None
    pred_band = decision_info.get("selected_band", receiver_band) if decision_info else receiver_band
    sel_score = decision_info.get("selection_score", 0.0) if decision_info else 0.0
    action_label = decision_info.get("action_label", decision_info.get("decision_type", "EXPLORE")) if decision_info else "EXPLORE"
    strategy_label = decision_info.get("strategy", "Information Gain") if decision_info else "Information Gain"

    # 3. Virtual Receiver Status Cards via native Streamlit metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        with st.container(border=True):
            st.caption("📡 TUNED RECEIVER")
            st.metric(label="Receiver Channel", value=f"F{receiver_band}", delta="Instantaneous W=1", delta_color="off")

    with col2:
        with st.container(border=True):
            st.caption("🔮 SAGE TARGET")
            st.metric(label="Selected Channel", value=f"F{pred_band}", delta=f"Action: {action_label} ({sel_score:.3f})", delta_color="normal")

    with col3:
        if best_pred:
            conf_val = best_pred.get("confidence_pct", "0%")
            target_lbl = best_pred.get("channel_label", f"F{pred_band}")
            t_left = best_pred.get("time_to_event", 0)
            if t_left == 0:
                delta_txt = f"Target: {target_lbl} (Active NOW 🎯)"
            else:
                delta_txt = f"Target: {target_lbl} (in {t_left}s ⏳)"
            d_color = "normal"
        else:
            conf_val = "N/A"
            uncert = decision_info.get("exploration_uncertainty", 0.0) if decision_info else 0.0
            delta_txt = f"Uncertainty: {uncert:.2f}"
            d_color = "off"
            
        with st.container(border=True):
            st.caption("🎯 PREDICTION CONFIDENCE")
            st.metric(label="Confidence", value=conf_val, delta=delta_txt, delta_color=d_color)

    with col4:
        with st.container(border=True):
            st.caption("⚡ SCHEDULING STRATEGY")
            st.metric(label="Strategy", value=strategy_label, delta=f"Action: {action_label}", delta_color="off")

    # 4. Prominent Predictive Pre-Positioning Banner
    if best_pred and best_pred.get("confidence", 0) >= 0.70:
        t_event = best_pred.get("time_to_event", 0)
        p_ch = best_pred.get("channel_label", "F15")
        p_time = best_pred.get("predicted_time", current_time)
        p_conf = best_pred.get("confidence_pct", "98%")
        evidence = best_pred.get("evidence", "Periodic pattern discovered")
        
        if t_event == 0:
            st.success(f"🎯 **TARGET ACTIVE NOW ({p_ch})** (Confidence: {p_conf}) — Receiver Pre-Positioned & Sensed! `{evidence}`", icon="⚡")
        elif t_event <= 3:
            st.warning(f"📡 **PREDICTIVE TARGET: {p_ch} will emit at t={p_time} (in {t_event} slot{'s' if t_event > 1 else ''})** — Confidence: {p_conf} | Action: **PRE-POSITION RECEIVER**", icon="🎯")
