import numpy as np
from typing import Dict, List, Any, Optional

class HistoryModel:
    """Tracks historical observation and hit statistics for all frequency channels."""
    def __init__(self, num_bands: int = 50, weight: float = 0.3):
        self.num_bands = num_bands
        self.weight = weight
        self.reset()

    def reset(self):
        self.num_observations = np.zeros(self.num_bands, dtype=int)
        self.num_hits = np.zeros(self.num_bands, dtype=int)
        self.last_detection_time = -np.ones(self.num_bands, dtype=int)
        self.last_observation_time = -np.ones(self.num_bands, dtype=int)
        self.hit_history: Dict[int, List[int]] = {i: [] for i in range(self.num_bands)}

    def update(self, band: int, detected: bool, current_time: int):
        self.num_observations[band] += 1
        self.last_observation_time[band] = current_time
        if detected:
            self.num_hits[band] += 1
            self.last_detection_time[band] = current_time
            self.hit_history[band].append(current_time)

    def get_hit_rate(self, band: int) -> float:
        obs = self.num_observations[band]
        if obs > 0:
            return float(self.num_hits[band] / obs)
        return 0.0

    def compute_score(self, band: int) -> float:
        """Returns the weighted historical contribution score."""
        return float(self.weight * self.get_hit_rate(band))

    def get_band_profile(self, band: int) -> Dict[str, Any]:
        obs = int(self.num_observations[band])
        hits = int(self.num_hits[band])
        rate = self.get_hit_rate(band)
        last_t = int(self.last_detection_time[band])
        return {
            "band": band,
            "observations": obs,
            "hits": hits,
            "hit_rate": rate,
            "hit_rate_pct": f"{rate * 100:.1f}%",
            "last_detection_time": last_t,
            "score": self.compute_score(band)
        }
