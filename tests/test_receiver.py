import pytest
import numpy as np
from src.emitters import IntermittentEmitter
from src.rf_environment import RFEnvironment
from src.receiver import Receiver

def test_receiver_observations():
    # Setup environment: Emitter always active on band 2
    emitter = IntermittentEmitter(name="E1", band=2, prob=1.0, value=1.0)
    env = RFEnvironment(num_bands=5, total_time=5, emitters=[emitter], seed=42)
    
    # Receiver: bandwidth 2, pd=1.0, pfa=0.0
    rec = Receiver(bandwidth=2, pd=1.0, pfa=0.0, seed=42)
    
    # Observe from t=0, selected band 1, dwell 2
    # Observed bands: [1, 2] (since bandwidth is 2, selected band is 1)
    obs = rec.observe(env, start_time=0, selected_band=1, dwell_time=2)
    
    # Expected observations length: 2 (dwell) * 2 (bandwidth) = 4
    assert len(obs) == 4
    
    # Check that band 2 observations are detected as True
    band_2_obs = [o for o in obs if o["band"] == 2]
    assert len(band_2_obs) == 2
    for o in band_2_obs:
        assert o["detected"] is True
        assert o["ground_truth_active"] is True
        assert "E1" in o["emitters"]
        
    # Check that band 1 observations are detected as False (since no active emitter and Pfa=0)
    band_1_obs = [o for o in obs if o["band"] == 1]
    assert len(band_1_obs) == 2
    for o in band_1_obs:
        assert o["detected"] is False
        assert o["ground_truth_active"] is False

def test_receiver_wrap_around():
    emitter = IntermittentEmitter(name="E1", band=0, prob=1.0, value=1.0)
    env = RFEnvironment(num_bands=5, total_time=5, emitters=[emitter], seed=42)
    
    rec = Receiver(bandwidth=2, pd=1.0, pfa=0.0, seed=42)
    
    # Start at band 4. Observed bands: [4, (4+1)%5] -> [4, 0]
    obs = rec.observe(env, start_time=0, selected_band=4, dwell_time=1)
    
    assert len(obs) == 2
    bands = [o["band"] for o in obs]
    assert 4 in bands
    assert 0 in bands
    
    obs_0 = [o for o in obs if o["band"] == 0][0]
    assert obs_0["detected"] is True
