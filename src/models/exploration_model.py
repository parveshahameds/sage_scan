import numpy as np
from typing import Dict, List, Any

class ExplorationModel:
    """Calculates Upper Confidence Bound (UCB) exploration values with Phase Dispersion to ensure complete spectrum coverage without harmonic phase-locking."""
    def __init__(self, num_bands: int = 50, coefficient: float = 0.3):
        self.num_bands = num_bands
        self.coefficient = coefficient
        self.reset()

    def reset(self):
        self.observation_counts = np.zeros(self.num_bands, dtype=int)
        self.last_observation_time = -np.ones(self.num_bands, dtype=int)
        self.total_actions = 0

    def update(self, band: int, current_time: int):
        self.observation_counts[band] += 1
        self.last_observation_time[band] = current_time
        self.total_actions += 1

    def compute_exploration_value(self, band: int, current_time: int = 0) -> float:
        """Computes raw UCB uncertainty value with recency and phase dispersion."""
        T = max(1, self.total_actions)
        n_obs = self.observation_counts[band]
        base_ucb = np.sqrt(np.log(T + 1) / (n_obs + 1))
        
        last_t = self.last_observation_time[band]
        time_since = (current_time - last_t) if last_t != -1 else (current_time + 5)
        
        # Recency term guarantees cold channels are naturally re-visited
        recency = min(1.0, time_since / 20.0)
        
        # Prime-phase dispersion during cold-start exploration (t <= 50)
        # Prevents phase-locking where a 50-channel sweep misses modulo 7 pulses
        cold_dispersion = 0.0
        if current_time <= 50 and n_obs == 0:
            scheduled_band = (current_time * 11 + 3) % self.num_bands
            if band == scheduled_band:
                cold_dispersion = 0.50
                
        return float(base_ucb + 0.35 * recency + cold_dispersion)

    def compute_score(self, band: int, current_time: int = 0) -> float:
        """Returns weighted exploration score."""
        val = self.compute_exploration_value(band, current_time)
        return float(self.coefficient * val)

    def get_unexplored_bands(self) -> List[int]:
        return [i for i in range(self.num_bands) if self.observation_counts[i] == 0]

    def get_rarely_observed_bands(self, threshold: int = 2) -> List[int]:
        return [i for i in range(self.num_bands) if 0 < self.observation_counts[i] <= threshold]

    def get_exploration_profile(self, band: int, current_time: int) -> Dict[str, Any]:
        obs = int(self.observation_counts[band])
        last_t = int(self.last_observation_time[band])
        time_since = (current_time - last_t) if last_t != -1 else current_time
        val = self.compute_exploration_value(band, current_time)
        score = self.compute_score(band, current_time)
        
        status = "Unexplored" if obs == 0 else ("Rarely Observed" if obs <= 2 else "Active Monitoring")
        
        return {
            "band": band,
            "observations": obs,
            "time_since_last_observation": time_since,
            "uncertainty_value": val,
            "score": score,
            "status": status
        }
