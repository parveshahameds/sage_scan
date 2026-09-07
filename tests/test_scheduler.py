import pytest
import numpy as np
from src.features import FeatureModel
from src.scheduler import SmartScheduler

def test_periodicity_detector():
    fm = FeatureModel(num_bands=5)
    
    # Simulate hits at t=5, 10, 15, 20 on band 1
    # Period should be 5, offset 0
    obs = []
    for t in [5, 10, 15, 20]:
        obs.append({"time": t, "band": 1, "detected": True, "ground_truth_active": True, "is_false_alarm": False, "is_missed_detection": False, "emitters": ["E1"], "max_value": 1.0})
    
    fm.update(obs, current_time=20)
    
    p, offset, conf = fm.get_periodicity(1)
    assert p == 5
    assert offset == 0
    assert conf == 1.0
    
    # Predict activity at t=25 (should be active) and t=27 (should be inactive)
    assert fm.predict_temporal_activity(1, 25) > 0.8
    assert fm.predict_temporal_activity(1, 27) == 0.0

def test_smart_scheduler_decisions():
    # Setup scheduler for 5 bands, bandwidth 1
    sched = SmartScheduler(num_bands=5, bandwidth=1, w_hist=0.5, w_temp=0.5, w_trans=0.0, c_explore=0.1)
    
    # Initially all bands have 0 observations, so exploration dominates.
    # It will choose a band and perform update.
    band, dwell = sched.select_action(current_time=0)
    assert 0 <= band < 5
    assert dwell == 1
    
    # Update scheduler with a HIT on band 2 at t=0
    obs = [{"time": 0, "band": 2, "detected": True, "ground_truth_active": True, "is_false_alarm": False, "is_missed_detection": False, "emitters": ["E1"], "max_value": 1.0}]
    sched.update(obs, current_time=0)
    
    # Now band 2 has a historical hit rate of 1.0. If we query, band 2 should have higher score from history
    # Select next action
    next_band, next_dwell = sched.select_action(current_time=1)
    # Band 2 is preferred because it has a hit rate of 1.0, while others have 0.0, and T=1 is small so exploration term won't dominate yet.
    assert next_band == 2
