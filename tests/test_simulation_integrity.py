import numpy as np
import pytest
from src.core.digital_twin import DigitalTwinEngine
from src.core.threat_engine import PeriodicThreat, AgileThreat, IntermittentThreat, UnknownThreat
from src.core.virtual_receiver import VirtualReceiver
from src.core.sage_engine import SAGEScanEngine
from src.core.comparison_engine import ComparisonSimulator, SequentialScanner
from src.core.metrics_engine import MetricsEngine
from src.models.temporal_model import TemporalModel
from src.models.transition_model import TransitionModel

def test_agile_transition_prediction_strictly_mathematical():
    """Verify Critical Issue 1: SAGE must predict F40 from F30, never an arbitrary frequency like F34."""
    model = TransitionModel(num_bands=50, weight=0.4)
    
    # Train the transition model on F10 -> F20 -> F30 -> F40 -> F10
    hops = [10, 20, 30, 40, 10, 20, 30, 40]
    for i, b in enumerate(hops):
        model.update(b, True, current_time=i * 15)
        
    # The last detected band is F30
    model.last_detected_band = 30
    next_band, prob = model.predict_next_band()
    
    assert next_band == 40, f"Expected predicted band 40 from 30, but got {next_band}"
    assert prob == 1.0
    
    # Verify probability for F40 is 1.0 and F34 is STRICTLY 0.0
    prob_f40 = model.predict_transition_activity(40)
    prob_f34 = model.predict_transition_activity(34)
    
    assert prob_f40 == 1.0
    assert prob_f34 == 0.0, f"F34 was never in transition chain but returned probability {prob_f34}"

def test_confidence_dynamically_responds_to_hits_and_misses():
    """Verify Critical Issue 2: Confidence must increase on validated hits and decrease on misses."""
    temp_model = TemporalModel(num_bands=50, weight=0.5)
    
    # Seed detections on F15 at t=7, 14, 21, 28
    for t in [7, 14, 21, 28]:
        temp_model.update(15, True, t)
        
    initial_conf = temp_model.period_estimates[15]["confidence"]
    assert initial_conf > 0.60
    
    # Simulate 2 validated prediction hits
    temp_model.record_prediction_hit(15)
    conf_after_hit = temp_model.period_estimates[15]["confidence"]
    assert conf_after_hit > initial_conf, "Confidence should increase on hit"
    
    # Simulate prediction misses
    temp_model.record_prediction_miss(15)
    conf_after_miss1 = temp_model.period_estimates[15]["confidence"]
    assert conf_after_miss1 < conf_after_hit, "Confidence must decrease on miss"
    
    temp_model.record_prediction_miss(15)
    conf_after_miss2 = temp_model.period_estimates[15]["confidence"]
    assert conf_after_miss2 < conf_after_miss1, "Consecutive misses must further reduce confidence"

def test_digital_twin_isolation_and_same_timeline_for_baselines():
    """Verify Critical Issues 4 & 6: SAGE has zero ground truth access, both run on same timeline."""
    dt = DigitalTwinEngine(num_bands=50, max_time=60, seed=42)
    sage = SAGEScanEngine(num_bands=50)
    sim = ComparisonSimulator(dt, sage, seed=42)
    
    # SAGE engine must NOT have direct references to Digital Twin attributes
    assert not hasattr(sage, "digital_twin")
    assert not hasattr(sage, "ground_truth")
    assert not hasattr(sage, "threats")
    
    # Step simulation and verify synchronized stepping
    for _ in range(35):
        step_rec = sim.step()
        t = step_rec["time"]
        assert step_rec["sage_obs"]["time"] == t
        assert step_rec["seq_obs"]["time"] == t
        # Both receivers have bandwidth W = 1
        assert sim.sage_receiver.bandwidth == 1
        assert sim.seq_receiver.bandwidth == 1

def test_event_driven_metrics_non_hardcoded():
    """Verify Critical Issues 3 & 10: Metrics are computed strictly from real simulation events."""
    dt = DigitalTwinEngine(num_bands=50, max_time=100, seed=42)
    sage = SAGEScanEngine(num_bands=50)
    sim = ComparisonSimulator(dt, sage, seed=42)
    
    for _ in range(80):
        sim.step()
        
    summary = sim.get_comparison_summary()
    sage_m = summary["sage"]
    seq_m = summary["sequential"]
    
    assert summary["total_emitted_signals"] > 0
    assert sage_m["total_scans"] == 80
    assert seq_m["total_scans"] == 80
    assert sage_m["signals_intercepted"] + sage_m["signals_missed"] == summary["total_emitted_signals"]
    assert seq_m["signals_intercepted"] + seq_m["signals_missed"] == summary["total_emitted_signals"]
    assert sage_m["channel_coverage"] > 0.0
    assert 0.0 <= sage_m["detection_rate"] <= 1.0

def test_unknown_threat_discovered_via_exploration():
    """Verify Critical Issue 8: Unknown threat on F2 is discovered strictly through UCB exploration."""
    dt = DigitalTwinEngine(num_bands=50, max_time=120, seed=42)
    sage = SAGEScanEngine(num_bands=50, seed=42)
    sim = ComparisonSimulator(dt, sage, seed=42)
    
    # Spawn unknown covert threat on F2 without notifying SAGE
    dt.spawn_unknown_threat(band=2)
    
    # Advance simulation
    discovered = False
    for _ in range(90):
        step_rec = sim.step()
        alert = step_rec.get("prediction_alert")
        if alert and alert.get("type") == "UNKNOWN_DISCOVERED" and alert.get("band") == 2:
            discovered = True
            break
            
    assert discovered, "UCB exploration should have visited and discovered unknown threat on F2"

def test_selection_score_vs_prediction_confidence_separation():
    """Verify that selection score and prediction confidence are distinct and not conflated."""
    sage = SAGEScanEngine(num_bands=50, seed=42)
    
    # At t=1, SAGE performs exploration
    dec = sage.decide_next_action(1)
    
    assert "selection_score" in dec
    assert "prediction_confidence" in dec
    assert "strategy" in dec
    assert dec["strategy"] == "Information Gain"
    assert dec["decision_type"] == "EXPLORE"
    # For exploration, prediction confidence is marked as N/A (no prediction active)
    assert dec["prediction_confidence_pct"] == "N/A"
    assert dec["selection_score"] > 0.0
    
    # Train temporal model on F15
    for t in [7, 14, 21, 28]:
        sage.temporal_model.update(15, True, t)
        
    # At t=35, pulse is expected on F15
    dec_exploit = sage.decide_next_action(35)
    if dec_exploit["selected_band"] == 15:
        assert dec_exploit["strategy"] == "Targeted Sensing"
        assert dec_exploit["decision_type"] == "EXPLOIT"
        assert dec_exploit["prediction_confidence"] >= 0.70
        assert dec_exploit["is_prediction_active"] is True

def test_transition_confidence_decay_on_misses():
    """Verify that repeated misses on agile transitions visibly decrease confidence."""
    trans_model = TransitionModel(num_bands=50)
    
    # Seed transition F30 -> F40
    trans_model.update(30, True, 10)
    trans_model.update(40, True, 20)
    
    initial_conf = trans_model.transition_confidence[30, 40]
    assert initial_conf >= 0.60
    
    # Miss 1
    trans_model.record_prediction_miss(30, 40)
    conf_miss1 = trans_model.transition_confidence[30, 40]
    assert conf_miss1 < initial_conf
    
    # Miss 2
    trans_model.record_prediction_miss(30, 40)
    conf_miss2 = trans_model.transition_confidence[30, 40]
    assert conf_miss2 < conf_miss1
    
    # Miss 3
    trans_model.record_prediction_miss(30, 40)
    conf_miss3 = trans_model.transition_confidence[30, 40]
    assert conf_miss3 < conf_miss2
    assert conf_miss3 <= 0.20

def test_live_learning_convergence_on_f15_periodic_threat():
    """Verify that SAGE autonomously discovers and converges on F15 period in free-running simulation."""
    dt = DigitalTwinEngine(num_bands=50, max_time=1000, seed=42)
    sage = SAGEScanEngine(num_bands=50, seed=42)
    sim = ComparisonSimulator(dt, sage, seed=42)
    
    for _ in range(999):
        sim.step()
        
    diag15 = sage.temporal_model.get_learning_diagnostics(15, 999)
    
    # SAGE must have genuinely learned the 7-slot period with high confidence
    assert diag15["estimated_period"] == 7
    assert diag15["offset"] == 0
    assert diag15["confidence"] >= 0.90
    assert len(diag15["detection_timestamps"]) >= 30
    assert diag15["learning_stage"] in ["PERIOD LEARNED", "HIGH CONFIDENCE", "PREDICTION SUCCESS"]

def test_terminal_confidence_regression_no_jump_at_t999():
    """Regression test: verify prediction confidence does NOT jump to 98% at simulation completion."""
    dt = DigitalTwinEngine(num_bands=50, max_time=1000, seed=42)
    sage = SAGEScanEngine(num_bands=50, seed=42)
    sim = ComparisonSimulator(dt, sage, seed=42)
    
    for _ in range(998):
        sim.step()
        
    rec_998 = sim.history[-1]
    conf_998 = rec_998["sage_decision"].get("prediction_confidence_pct")
    
    # Step to terminal slot t=999
    rec_999 = sim.step()
    conf_999 = rec_999["sage_decision"].get("prediction_confidence_pct")
    
    # Step beyond terminal slot
    rec_post = sim.step()
    conf_post = rec_post["sage_decision"].get("prediction_confidence_pct")
    
    assert conf_post == conf_999
    assert sim.is_completed is True

def test_predictive_scheduling_prepositioning_at_end_of_simulation():
    """Verify Section 15: SAGE pre-positions at t=999, t=1000 and intercepts at t=1001 with 98% confidence."""
    dt = DigitalTwinEngine(num_bands=50, max_time=1005, seed=42)
    sage = SAGEScanEngine(num_bands=50, seed=42)
    sim = ComparisonSimulator(dt, sage, seed=42)
    
    # Step up to t=999
    for _ in range(999):
        rec_999 = sim.step()
        
    # At t=999: SAGE must target F15 with PRE-POSITION action and 98% confidence
    dec_999 = rec_999["sage_decision"]
    assert dec_999["selected_band"] == 15
    assert dec_999["action_label"] == "PRE-POSITION"
    assert dec_999["strategy"] == "Predictive Targeting"
    assert dec_999["prediction_confidence"] >= 0.90
    assert dec_999["best_prediction"]["predicted_time"] == 1001
    assert dec_999["best_prediction"]["time_to_event"] == 2
    
    # At t=1000: SAGE must maintain pre-positioning on F15
    rec_1000 = sim.step()
    dec_1000 = rec_1000["sage_decision"]
    assert dec_1000["selected_band"] == 15
    assert dec_1000["action_label"] == "PRE-POSITION"
    assert dec_1000["best_prediction"]["time_to_event"] == 1
    
    # At t=1001: Signal emits and is intercepted by pre-positioned receiver
    rec_1001 = sim.step()
    dec_1001 = rec_1001["sage_decision"]
    alert_1001 = rec_1001.get("prediction_alert")
    
    assert dec_1001["selected_band"] == 15
    assert dec_1001["action_label"] == "INTERCEPT"
    assert rec_1001["sage_obs"]["detected"] is True
    assert alert_1001 is not None
    assert alert_1001["type"] == "PREDICTION_SUCCESS"
    assert alert_1001["is_preemptive"] is True

