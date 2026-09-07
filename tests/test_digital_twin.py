import numpy as np
import pytest
from src.core.digital_twin import DigitalTwinEngine
from src.core.threat_engine import PeriodicThreat, AgileThreat, UnknownThreat
from src.core.virtual_receiver import VirtualReceiver
from src.core.sage_engine import SAGEScanEngine
from src.core.comparison_engine import ComparisonSimulator
from src.core.demo_controller import JudgeDemoController

def test_digital_twin_emissions():
    dt = DigitalTwinEngine(num_bands=50, max_time=50, seed=42)
    # Step through 25 steps
    for _ in range(25):
        dt.step()
    
    # Check periodic threat emissions on F15 at t=7, 14, 21
    assert dt.ground_truth[7, 15] == 1
    assert dt.ground_truth[14, 15] == 1
    assert dt.ground_truth[21, 15] == 1
    # Silent at t=8 on F15
    assert dt.ground_truth[8, 15] == 0

def test_virtual_receiver_bandwidth_constraint():
    dt = DigitalTwinEngine(num_bands=50, max_time=30, seed=42)
    rx = VirtualReceiver(num_bands=50, bandwidth=1, pd=1.0, pfa=0.0, seed=42)
    
    # Step DT to t=7 (F15 is active)
    for _ in range(7):
        dt.step()
        
    # Listen to F12 -> Should be a miss
    rx.tune_to(12)
    obs_miss = rx.observe(dt)
    assert obs_miss["detected"] is False
    
    # Listen to F15 -> Should detect
    rx.tune_to(15)
    obs_hit = rx.observe(dt)
    assert obs_hit["detected"] is True
    assert obs_hit["band"] == 15

def test_comparison_simulator():
    dt = DigitalTwinEngine(num_bands=50, max_time=50, seed=42)
    sage = SAGEScanEngine(num_bands=50)
    sim = ComparisonSimulator(dt, sage, seed=42)
    
    for _ in range(40):
        step_res = sim.step()
        assert "sage_band" in step_res
        assert "seq_band" in step_res
        
    summary = sim.get_comparison_summary()
    assert summary["time"] == 40
    assert summary["sage"]["total_scans"] == 40
    assert summary["sequential"]["total_scans"] == 40

def test_judge_demo_controller():
    controller = JudgeDemoController(seed=42)
    
    # Run through Phase 1 to Phase 6
    for _ in range(80):
        step_res = controller.step()
        
    assert controller.current_phase >= 5
    assert controller.unknown_threat_injected is True
