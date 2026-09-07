import pytest
from src.core.digital_twin import DigitalTwinEngine
from src.core.sage_engine import SAGEScanEngine
from src.core.comparison_engine import ComparisonSimulator
from src.core.demo_controller import JudgeDemoController
from src.components.spectrum_view import render_spectrum_heatmap, render_channel_matrix_figure
from src.components.prediction_panel import render_prediction_panel
from src.components.threat_panel import render_threat_panel
from src.components.decision_explanation import render_decision_explanation
from src.components.comparison_panel import render_comparison_panel
from src.components.event_timeline import render_event_timeline
from src.components.digital_twin_inspector import render_digital_twin_inspector

def test_full_simulation_and_component_lifecycle():
    dt = DigitalTwinEngine(num_bands=50, max_time=150, seed=42)
    sage = SAGEScanEngine(num_bands=50)
    sim = ComparisonSimulator(dt, sage, seed=42)
    demo = JudgeDemoController(seed=42)

    # 1. Run simulation steps & trigger prediction events
    for step_i in range(50):
        step_data = sim.step()
        assert "sage_band" in step_data
        assert "seq_band" in step_data

    # 2. Inject unknown threat on F2
    unknown = dt.spawn_unknown_threat(band=2)
    assert unknown.band == 2

    # Step more to let exploration discover F2
    for step_i in range(30):
        sim.step()

    # 3. Verify component visual models produce valid figures without errors
    gt_matrix = dt.ground_truth
    rx_history = sim.sage_receiver.observation_history
    curr_t = dt.current_time
    
    # Waterfall figure
    fig_waterfall = render_spectrum_heatmap(gt_matrix, rx_history, curr_t, num_bands=50)
    assert fig_waterfall is not None
    assert len(fig_waterfall.data) > 0

    # 50-Channel Matrix figure
    gt_snap = dt.get_ground_truth_snapshot()
    fig_matrix = render_channel_matrix_figure(
        num_bands=50,
        current_receiver_band=sim.sage_receiver.current_band,
        predicted_band=15,
        active_threat_bands=gt_snap.get("active_bands", []),
        detection_counts=sage.history_model.num_hits,
        observation_counts=sage.exploration_model.observation_counts
    )
    assert fig_matrix is not None
    assert len(fig_matrix.data) > 0

    # Comparison summary
    summary = sim.get_comparison_summary()
    assert summary["sage"]["total_scans"] == 80
    assert summary["sequential"]["total_scans"] == 80

    # 4. Judge Demo lifecycle
    for _ in range(80):
        d_step = demo.step()
        assert "phase_info" in d_step

    assert demo.current_phase >= 5
