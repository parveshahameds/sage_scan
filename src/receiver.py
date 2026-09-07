import numpy as np
from typing import List, Dict, Any
from src.rf_environment import RFEnvironment

class Receiver:
    """Simulates a bandwidth-limited RF receiver with configurable Pd and Pfa."""
    def __init__(self, bandwidth: int = 5, pd: float = 0.9, pfa: float = 0.05, seed: int = 42):
        self.bandwidth = bandwidth
        self.pd = pd
        self.pfa = pfa
        self.rng = np.random.default_rng(seed)

    def set_seed(self, seed: int):
        """Sets the random seed for reproducibility."""
        self.rng = np.random.default_rng(seed)

    def observe(self, env: RFEnvironment, start_time: int, selected_band: int, dwell_time: int) -> List[Dict[str, Any]]:
        """Simulates observations over the dwell time and instantaneous bandwidth window.
        
        The window is of size `bandwidth` starting at `selected_band` (wrapping modulo num_bands).
        Returns a list of observation dicts for each time step and band in the window.
        """
        observations = []
        num_bands = env.num_bands
        total_time = env.total_time

        # Calculate the observed bands window
        observed_bands = [(selected_band + i) % num_bands for i in range(self.bandwidth)]

        for d in range(dwell_time):
            t = start_time + d
            if t >= total_time:
                break

            for band in observed_bands:
                truth = env.query(t, band)
                actual_active = truth["active"]
                emitters = truth["emitters"]
                max_value = truth["max_value"]

                # Apply probability of detection (Pd) and probability of false alarm (Pfa)
                detected = False
                is_false_alarm = False
                is_missed_detection = False

                if actual_active:
                    if self.rng.random() < self.pd:
                        detected = True
                    else:
                        is_missed_detection = True
                else:
                    if self.rng.random() < self.pfa:
                        detected = True
                        is_false_alarm = True

                observations.append({
                    "time": t,
                    "band": band,
                    "detected": detected,
                    "ground_truth_active": actual_active,
                    "is_false_alarm": is_false_alarm,
                    "is_missed_detection": is_missed_detection,
                    "emitters": emitters,
                    "max_value": max_value
                })

        return observations
