import numpy as np
from typing import Dict, Any, List
from src.core.digital_twin import DigitalTwinEngine

class VirtualReceiver:
    """The Virtual Receiver is the hardware-constrained sensor.
    
    It can listen to ONLY ONE frequency channel at a time (bandwidth = 1).
    It observes the Digital Twin environment and outputs noisy HIT / MISS detections.
    """
    def __init__(self, num_bands: int = 50, bandwidth: int = 1, pd: float = 0.95, pfa: float = 0.02, seed: int = 42):
        self.num_bands = num_bands
        self.bandwidth = bandwidth
        self.pd = pd
        self.pfa = pfa
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        
        self.current_band = 0
        self.last_observation: Dict[str, Any] = {}
        self.observation_history: List[Dict[str, Any]] = []

    def tune_to(self, band: int):
        """Sets the receiver's tuned frequency channel."""
        self.current_band = int(band % self.num_bands)

    def observe(self, digital_twin: DigitalTwinEngine) -> Dict[str, Any]:
        """Samples the RF spectrum at current_band through the receiver front-end."""
        phys_state = digital_twin.query_physical_spectrum(self.current_band)
        is_actually_active = phys_state["is_active"]
        t = phys_state["time"]
        
        detected = False
        is_false_alarm = False
        is_miss = False

        if is_actually_active:
            if self.rng.random() < self.pd:
                detected = True
            else:
                is_miss = True
        else:
            if self.rng.random() < self.pfa:
                detected = True
                is_false_alarm = True

        obs = {
            "time": t,
            "band": self.current_band,
            "detected": detected,
            "is_false_alarm": is_false_alarm,
            "is_miss": is_miss,
            "ground_truth_active": is_actually_active # Kept internally for metrics engine
        }
        
        self.last_observation = obs
        self.observation_history.append(obs)
        return obs

    def reset(self, seed: int = None):
        if seed is not None:
            self.seed = seed
        self.rng = np.random.default_rng(self.seed)
        self.current_band = 0
        self.last_observation = {}
        self.observation_history = []
