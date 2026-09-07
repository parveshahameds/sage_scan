import plotly.graph_objects as go
import numpy as np
from typing import Dict, List, Any, Optional

def render_spectrum_heatmap(
    ground_truth_matrix: np.ndarray,
    receiver_history: List[Dict[str, Any]],
    current_time: int,
    num_bands: int = 50,
    window_slots: int = 60,
    predicted_band: Optional[int] = None
) -> go.Figure:
    """Renders a 2D time-frequency waterfall spectrogram showing threat emissions and receiver intercepts."""
    start_t = max(0, current_time - window_slots + 1)
    end_t = current_time + 1
    t_len = end_t - start_t
    
    # 0 = Inactive, 1 = Ground Truth Emission, 2 = Scanned Miss, 3 = Predicted, 4 = Successful Intercept
    matrix = np.zeros((t_len, num_bands))
    
    # Fill ground truth
    gt_slice = ground_truth_matrix[start_t:end_t]
    for rel_t in range(t_len):
        abs_t = start_t + rel_t
        for b in range(num_bands):
            if abs_t < len(ground_truth_matrix) and ground_truth_matrix[abs_t, b] == 1:
                matrix[rel_t, b] = 1 # Signal active

    # Overlay receiver observations
    for obs in receiver_history:
        t = obs["time"]
        if start_t <= t < end_t:
            rel_t = t - start_t
            b = obs["band"]
            if obs["detected"]:
                matrix[rel_t, b] = 4 # Intercept
            else:
                if matrix[rel_t, b] == 0:
                    matrix[rel_t, b] = 2 # Scanned empty
    
    # Highlight prediction for current time
    if predicted_band is not None and t_len > 0:
        if matrix[t_len - 1, predicted_band] == 0:
            matrix[t_len - 1, predicted_band] = 3

    colorscale = [
        [0.0, "#0b0f19"],
        [0.2, "#ef4444"],
        [0.4, "#2563eb"],
        [0.6, "#8b5cf6"],
        [1.0, "#10b981"]
    ]

    time_labels = np.arange(start_t, end_t)

    fig = go.Figure(data=go.Heatmap(
        z=matrix.T,
        x=time_labels,
        y=np.arange(num_bands),
        colorscale=colorscale,
        zmin=0,
        zmax=4,
        showscale=False,
        hoverongaps=False,
        hovertemplate="Time: t=%{x}<br>Channel: F%{y}<extra></extra>"
    ))

    fig.update_layout(
        title=f"Live Spectrum Waterfall (t={start_t} → t={current_time})",
        xaxis=dict(title="Simulation Time Slot (t)", dtick=max(1, t_len // 10), showgrid=True, gridcolor="#1f2937"),
        yaxis=dict(title="Frequency Channels (F0 - F49)", dtick=5, showgrid=True, gridcolor="#1f2937"),
        height=320,
        margin=dict(l=40, r=20, t=40, b=30),
        template="plotly_dark",
        paper_bgcolor="#0e1117",
        plot_bgcolor="#0b0f19"
    )
    return fig


def render_channel_matrix_figure(
    num_bands: int,
    current_receiver_band: int,
    predicted_band: Optional[int],
    active_threat_bands: List[int],
    detection_counts: np.ndarray,
    observation_counts: np.ndarray,
    selected_inspect_band: Optional[int] = None
) -> go.Figure:
    """Renders the 50-channel visual spectrum as a 5x10 interactive Plotly matrix without any raw HTML strings."""
    cols = 10
    rows = (num_bands + cols - 1) // cols
    
    # Grid values:
    # 0: Inactive/Observed, 1: Cold/Unexplored, 2: Threat Emitting, 3: Tuned RX, 4: Predicted Target, 5: Hit/Intercepted
    grid_z = np.zeros((rows, cols))
    annotations = []
    hover_texts = []

    for r in range(rows):
        hover_row = []
        for c in range(cols):
            b = r * cols + c
            if b >= num_bands:
                hover_row.append("")
                continue

            is_rx = (b == current_receiver_band)
            is_pred = (b == predicted_band)
            is_active = (b in active_threat_bands)
            is_unexplored = (observation_counts[b] == 0)
            is_selected = (b == selected_inspect_band)
            hits = int(detection_counts[b])
            obs = int(observation_counts[b])

            status_str = "Idle"
            val = 0
            if is_active and is_rx:
                val = 5
                status_str = "INTERCEPT 🎯"
            elif is_active:
                val = 2
                status_str = "THREAT EMITTING 🚨"
            elif is_rx:
                val = 3
                status_str = "RECEIVER TUNED 📡"
            elif is_pred:
                val = 4
                status_str = "PREDICTED TARGET 🔮"
            elif is_unexplored:
                val = 1
                status_str = "COLD / UNEXPLORED"
            elif hits > 0:
                val = 0.5
                status_str = f"{hits} Detections"

            grid_z[r, c] = val
            
            tag = ""
            if is_active and is_rx:
                tag = "🎯"
            elif is_active:
                tag = "🚨"
            elif is_rx:
                tag = "📡"
            elif is_pred:
                tag = "🔮"
            elif is_unexplored:
                tag = "❄️"
            elif hits > 0:
                tag = f"✓{hits}"

            ann_text = f"<b>F{b}</b><br>{tag}" if tag else f"<b>F{b}</b>"
            annotations.append(dict(
                x=c,
                y=r,
                text=ann_text,
                showarrow=False,
                font=dict(color="#ffffff" if val > 0 else "#94a3b8", size=11)
            ))
            
            hover_row.append(f"<b>Channel F{b}</b><br>Status: {status_str}<br>Total Observations: {obs}<br>Detections: {hits}")
        hover_texts.append(hover_row)

    # Colorscale: 0 Dark, 1 Cold Slate, 2 Red Threat, 3 Blue RX, 4 Purple Pred, 5 Green Hit
    colorscale = [
        [0.0, "#111827"],
        [0.1, "#1e293b"],
        [0.2, "#334155"],
        [0.4, "#dc2626"],
        [0.6, "#2563eb"],
        [0.8, "#7c3aed"],
        [1.0, "#059669"]
    ]

    fig = go.Figure(data=go.Heatmap(
        z=grid_z,
        colorscale=colorscale,
        zmin=0,
        zmax=5,
        showscale=False,
        hovertext=hover_texts,
        hoverinfo="text"
    ))

    fig.update_layout(
        annotations=annotations,
        xaxis=dict(showgrid=False, showticklabels=False, zeroline=False),
        yaxis=dict(showgrid=False, showticklabels=False, zeroline=False, autorange="reversed"),
        height=280,
        margin=dict(l=10, r=10, t=10, b=10),
        template="plotly_dark",
        paper_bgcolor="#0e1117",
        plot_bgcolor="#0b0f19"
    )
    return fig
