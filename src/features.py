import numpy as np
from typing import List, Dict, Tuple, Any

class FeatureModel:
    """Extracts features and maintains history from receiver observations to support scheduling decisions."""
    def __init__(self, num_bands: int):
        self.num_bands = num_bands
        self.reset()

    def reset(self):
        # Per-band statistics
        self.num_observations = np.zeros(self.num_bands, dtype=int)
        self.num_hits = np.zeros(self.num_bands, dtype=int)
        self.last_observation_time = -np.ones(self.num_bands, dtype=int)
        self.last_hit_time = -np.ones(self.num_bands, dtype=int)
        self.hit_history: Dict[int, List[int]] = {i: [] for i in range(self.num_bands)}

        # Periodicity estimates: (period, offset, confidence)
        self.periodicity_estimates: Dict[int, Tuple[int, int, float]] = {i: (0, 0, 0.0) for i in range(self.num_bands)}

        # Frequency transition matrix for tracking agility
        self.transition_counts = np.zeros((self.num_bands, self.num_bands), dtype=int)
        self.last_detected_band = -1
        self.last_detected_time = -1

    def update(self, observations: List[Dict[str, Any]], current_time: int):
        """Updates internal statistics with a new batch of observations from a dwell period."""
        # Group observations by time step to handle temporal sequences
        sorted_obs = sorted(observations, key=lambda x: (x["time"], x["band"]))
        
        for obs in sorted_obs:
            t = obs["time"]
            b = obs["band"]
            detected = obs["detected"]

            # Update observation counts
            self.num_observations[b] += 1
            self.last_observation_time[b] = t

            if detected:
                self.num_hits[b] += 1
                self.last_hit_time[b] = t
                self.hit_history[b].append(t)

                # Track transitions for frequency agile emitters
                if self.last_detected_band != -1 and t - self.last_detected_time <= 15:
                    # Increment transition count if it's a consecutive active step within a window
                    if self.last_detected_band != b:
                        self.transition_counts[self.last_detected_band, b] += 1

                self.last_detected_band = b
                self.last_detected_time = t

        # Update periodicity model for bands that received new hits
        unique_bands = set(obs["band"] for obs in observations if obs["detected"])
        for b in unique_bands:
            self._update_periodicity(b)

    def _update_periodicity(self, band: int):
        """Detects if hits in a band follow a periodic pattern."""
        hit_times = self.hit_history[band]
        if len(hit_times) < 3:
            self.periodicity_estimates[band] = (0, 0, 0.0)
            return

        best_p = 0
        best_offset = 0
        best_confidence = 0.0

        # Search for periods from 2 up to half the observation span, capped at 100
        span = hit_times[-1] - hit_times[0]
        max_p = min(100, span // 2)
        if max_p < 2:
            self.periodicity_estimates[band] = (0, 0, 0.0)
            return

        for p in range(2, max_p + 1):
            modulos = [t % p for t in hit_times]
            counts = np.bincount(modulos, minlength=p)
            offset = int(np.argmax(counts))
            hits_matching = counts[offset]
            confidence = hits_matching / len(hit_times)

            # We prefer smaller periods (simplest explanation)
            # If we find a period with high confidence (>0.9), we prefer to keep it
            # rather than replacing it with a larger multiple of itself.
            if confidence > best_confidence:
                if confidence > 0.9 and best_confidence > 0.9:
                    # Already found a smaller period that works well
                    continue
                best_p = p
                best_offset = offset
                best_confidence = confidence

        self.periodicity_estimates[band] = (best_p, best_offset, best_confidence)

    def get_periodicity(self, band: int) -> Tuple[int, int, float]:
        return self.periodicity_estimates.get(band, (0, 0, 0.0))

    def predict_temporal_activity(self, band: int, t: int) -> float:
        """Estimates probability that band is active at future time t based on periodicity."""
        p, offset, conf = self.get_periodicity(band)
        if p > 0 and conf > 0.5:
            # Check how close time t is to the predicted periodic cycle
            # Use a gaussian-like decay around the expected modulo
            dist = (t - offset) % p
            # Dist can be represented as distance to 0 or p
            dist_to_cycle = min(dist, p - dist)
            if dist_to_cycle <= 1:
                # Return confidence weighted by distance
                return conf * (1.0 - 0.3 * dist_to_cycle)
        return 0.0

    def predict_transition_activity(self, band: int) -> float:
        """Estimates transition probability to this band from the last detected band."""
        if self.last_detected_band == -1:
            return 0.0

        total_transitions = np.sum(self.transition_counts[self.last_detected_band])
        if total_transitions > 0:
            return self.transition_counts[self.last_detected_band, band] / total_transitions
        return 0.0

    def get_historical_rate(self, band: int) -> float:
        """Returns the hit-to-observation ratio for a band."""
        obs = self.num_observations[band]
        if obs > 0:
            return self.num_hits[band] / obs
        return 0.0
