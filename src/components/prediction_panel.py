import streamlit as st
from typing import Dict, Any, Optional

def render_prediction_panel(
    current_time: int,
    receiver_band: int,
    decision_info: Dict[str, Any],
    last_alert: Optional[Dict[str, Any]],
    upcoming_pulse: Optional[Dict[str, Any]]
):
    """Renders the prediction command center, static upper status bar, and metrics cards with zero layout shift."""
    
    best_pred = decision_info.get("best_prediction") if decision_info else None
    pred_band = decision_info.get("selected_band", receiver_band) if decision_info else receiver_band
    sel_score = decision_info.get("selection_score", 0.0) if decision_info else 0.0
    action_label = decision_info.get("action_label", decision_info.get("decision_type", "EXPLORE")) if decision_info else "EXPLORE"
    strategy_label = decision_info.get("strategy", "Information Gain") if decision_info else "Information Gain"

    # 1. Permanent Static Status Banner in Upper Header (Consistent 48px height, 0px vertical layout shift)
    status_icon = "📡"
    status_border = "rgba(71, 85, 105, 0.4)"
    status_bg = "rgba(15, 23, 42, 0.75)"
    status_content = f"""<span style="color:#94a3b8; font-weight:600;">SPECTRUM SURVEILLANCE ACTIVE</span> 
    <span style="color:#64748b; margin:0 8px;">|</span> 
    <span style="color:#cbd5e1;">50-Band Receiver sensing slot <b>t={current_time}</b></span>
    <span style="color:#64748b; margin:0 8px;">|</span> 
    <span style="color:#94a3b8;">Strategy: <b>{strategy_label}</b></span>"""

    # Check alert priority
    if last_alert and (current_time - last_alert.get("time", 0) <= 2):
        alert_type = last_alert.get("type", "")
        title = last_alert.get("title", "")
        msg = last_alert.get("message", "")
        band = last_alert.get("band", receiver_band)
        
        if alert_type == "PREDICTION_SUCCESS":
            status_icon = "🎯"
            status_bg = "rgba(6, 78, 59, 0.35)"
            status_border = "rgba(16, 185, 129, 0.7)"
            status_content = f"""<span style="color:#6ee7b7; font-weight:700;">{title} (Channel F{band})</span> 
            <span style="color:#10b981; margin:0 8px;">—</span> 
            <span style="color:#d1fae5;">{msg}</span>"""
        elif alert_type == "PREEMPTIVE_INTERCEPT":
            status_icon = "⚡"
            status_bg = "rgba(12, 74, 110, 0.35)"
            status_border = "rgba(14, 165, 233, 0.7)"
            status_content = f"""<span style="color:#7dd3fc; font-weight:700;">{title} (Channel F{band})</span> 
            <span style="color:#0ea5e9; margin:0 8px;">—</span> 
            <span style="color:#e0f2fe;">{msg}</span>"""
        elif alert_type == "UNKNOWN_DISCOVERED":
            status_icon = "🛸"
            status_bg = "rgba(120, 53, 15, 0.35)"
            status_border = "rgba(245, 158, 11, 0.7)"
            status_content = f"""<span style="color:#fcd34d; font-weight:700;">{title} (Channel F{band})</span> 
            <span style="color:#f59e0b; margin:0 8px;">—</span> 
            <span style="color:#fef3c7;">{msg}</span>"""
        else:
            status_icon = "📡"
            status_bg = "rgba(30, 58, 138, 0.35)"
            status_border = "rgba(59, 130, 246, 0.7)"
            status_content = f"""<span style="color:#93c5fd; font-weight:700;">{title} (Channel F{band})</span> 
            <span style="color:#3b82f6; margin:0 8px;">—</span> 
            <span style="color:#eff6ff;">{msg}</span>"""

    elif best_pred and best_pred.get("confidence", 0) >= 0.70:
        t_event = best_pred.get("time_to_event", 0)
        p_ch = best_pred.get("channel_label", f"F{pred_band}")
        p_time = best_pred.get("predicted_time", current_time)
        p_conf = best_pred.get("confidence_pct", "98%")
        evidence = best_pred.get("evidence", "Periodic pattern discovered")

        if t_event == 0:
            status_icon = "⚡"
            status_bg = "rgba(6, 78, 59, 0.4)"
            status_border = "rgba(34, 197, 94, 0.8)"
            status_content = f"""<span style="color:#4ade80; font-weight:700;">TARGET ACTIVE NOW ({p_ch})</span> 
            <span style="color:#22c55e; margin:0 8px;">—</span> 
            <span style="color:#f0fdf4;">Confidence: <b>{p_conf}</b> | Receiver Pre-Positioned & Sensed! (<i>{evidence}</i>)</span>"""
        elif t_event <= 3:
            status_icon = "🎯"
            status_bg = "rgba(113, 63, 18, 0.35)"
            status_border = "rgba(234, 179, 8, 0.8)"
            slots_str = f"in {t_event} slot{'s' if t_event > 1 else ''}"
            status_content = f"""<span style="color:#fde047; font-weight:700;">📡 PREDICTIVE TARGET: {p_ch} will emit at t={p_time} ({slots_str})</span> 
            <span style="color:#eab308; margin:0 8px;">—</span> 
            <span style="color:#fef9c3;">Confidence: <b>{p_conf}</b> | Action: <b style="color:#fde047; text-decoration: underline;">PRE-POSITION RECEIVER</b></span>"""

    # Static upper header status box
    st.markdown(
        f"""
        <div style="
            background: {status_bg};
            border: 1px solid {status_border};
            border-radius: 8px;
            padding: 9px 14px;
            margin-bottom: 12px;
            display: flex;
            align-items: center;
            min-height: 48px;
            box-sizing: border-box;
            font-size: 0.92rem;
            line-height: 1.35;
        ">
            <span style="font-size: 1.25rem; margin-right: 10px; display: inline-flex; align-items: center; flex-shrink: 0;">{status_icon}</span>
            <div style="flex-grow: 1; overflow: hidden; text-overflow: ellipsis; white-space: nowrap;">{status_content}</div>
        </div>
        """,
        unsafe_allow_html=True
    )

    # 2. Virtual Receiver Status Cards via native Streamlit metrics
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
