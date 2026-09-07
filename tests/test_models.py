import numpy as np
import pytest
from src.models.history_model import HistoryModel
from src.models.temporal_model import TemporalModel
from src.models.transition_model import TransitionModel
from src.models.exploration_model import ExplorationModel

def test_history_model():
    model = HistoryModel(num_bands=50, weight=0.3)
    # Band 10 observed 10 times, 7 hits
    for t in range(10):
        detected = (t < 7)
        model.update(10, detected, t)
    
    profile = model.get_band_profile(10)
    assert profile["observations"] == 10
    assert profile["hits"] == 7
    assert profile["hit_rate"] == 0.7
    assert model.compute_score(10) == pytest.approx(0.3 * 0.7)

def test_temporal_model_periodicity_discovery():
    model = TemporalModel(num_bands=50, weight=0.5)
    # Emit on F15 every 7 slots at t=7, 14, 21, 28, 35
    for t in [7, 14, 21, 28, 35]:
        model.update(15, True, t)
    
    p, offset, conf = model.get_periodicity(15)
    assert p == 7
    assert offset == 0
    assert conf > 0.8
    
    # Predict activity at future time t=42
    prob = model.predict_activity(15, 42)
    assert prob > 0.8
    # Predict activity at t=43 (should be 0 or small)
    assert model.predict_activity(15, 43) < 0.2

def test_transition_model_agility():
    model = TransitionModel(num_bands=50, weight=0.4)
    # Sequence: 10 -> 20 -> 30 -> 40 -> 10
    seq = [10, 20, 30, 40, 10, 20, 30, 40]
    for i, b in enumerate(seq):
        model.update(b, True, i * 15)
        
    # After detecting on 30, it should predict 40
    model.last_detected_band = 30
    next_band, prob = model.predict_next_band()
    assert next_band == 40
    assert prob == 1.0
    assert model.compute_score(40) == pytest.approx(0.4)

def test_exploration_model():
    model = ExplorationModel(num_bands=50, coefficient=0.3)
    # Band 2 is never observed
    model.update(15, 1)
    model.update(15, 2)
    model.update(15, 3)
    
    score_f2 = model.compute_score(2)
    score_f15 = model.compute_score(15)
    # Unobserved F2 should have higher exploration score than frequently visited F15
    assert score_f2 > score_f15
