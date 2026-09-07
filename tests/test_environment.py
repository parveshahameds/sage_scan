import pytest
import numpy as np
import os
from src.emitters import PeriodicEmitter, IntermittentEmitter
from src.rf_environment import RFEnvironment

def test_periodic_emitter():
    # Emitter in band 5, period 4, pulse width 1
    emitter = PeriodicEmitter(name="E1", band=5, period=4, pulse_width=1, offset=0, value=2.0)
    rng = np.random.default_rng(42)
    truth = emitter.generate_truth(total_time=12, num_bands=10, rng=rng)
    
    # Expected activity at t=0, 4, 8 in band 5
    assert truth[0, 5] == 1
    assert truth[1, 5] == 0
    assert truth[4, 5] == 1
    assert truth[8, 5] == 1
    assert np.sum(truth) == 3

def test_rf_environment():
    emitter1 = PeriodicEmitter(name="E1", band=2, period=3, pulse_width=1, offset=0, value=1.0)
    emitter2 = IntermittentEmitter(name="E2", band=4, prob=1.0, value=3.0)
    
    env = RFEnvironment(num_bands=10, total_time=10, emitters=[emitter1, emitter2], seed=42)
    
    # Verify environment ground truth shapes
    assert env.get_ground_truth().shape == (10, 10)
    
    # At t=0:
    # E1 active at band 2 (since 0 % 3 == 0)
    # E2 active at band 4 (since prob=1.0)
    q2 = env.query(0, 2)
    assert q2["active"] is True
    assert "E1" in q2["emitters"]
    assert q2["max_value"] == 1.0
    
    q4 = env.query(0, 4)
    assert q4["active"] is True
    assert "E2" in q4["emitters"]
    assert q4["max_value"] == 3.0

    q9 = env.query(0, 9)
    assert q9["active"] is False
    assert q9["max_value"] == 0.0

def test_rf_environment_trace_loading():
    trace_path = "sage_scan/data/test_turing_trace.csv"
    if os.path.exists(trace_path):
        os.remove(trace_path)
        
    env = RFEnvironment(num_bands=50, total_time=100, emitters=None, seed=42, trace_path=trace_path)
    
    # Verify file is generated
    assert os.path.exists(trace_path)
    
    # Verify ground truth matches shape
    assert env.get_ground_truth().shape == (100, 50)
    
    # Verify that some slots are active
    assert np.sum(env.get_ground_truth()) > 0
    
    # Cleanup
    if os.path.exists(trace_path):
        os.remove(trace_path)
