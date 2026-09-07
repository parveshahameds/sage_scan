import numpy as np
from typing import Dict, List, Tuple, Any, Optional

class TransitionModel:
    """Tracks frequency-hopping transitions to predict the next frequency in agile threats.
    
    Guarantees that transitions are strictly derived from observed empirical hops.
    Never predicts arbitrary channels.
    """
    def __init__(self, num_bands: int = 50, weight: float = 0.4, max_hop_window: int = 25):
        self.num_bands = num_bands
        self.weight = weight
        self.max_hop_window = max_hop_window
        self.reset()

    def reset(self):
        # transition_counts[src, dst]
        self.transition_counts = np.zeros((self.num_bands, self.num_bands), dtype=int)
        # transition_confidence[src, dst]
        self.transition_confidence = np.zeros((self.num_bands, self.num_bands), dtype=float)
        
        self.last_detected_band: int = -1
        self.last_detected_time: int = -1
        self.recent_transitions: List[Tuple[int, int, int]] = [] # (src, dst, time)

    def update(self, band: int, detected: bool, current_time: int):
        """Processes a detection and updates transition counts if a jump occurred within max_hop_window."""
        if detected:
            if self.last_detected_band != -1 and (current_time - self.last_detected_time) <= self.max_hop_window:
                if self.last_detected_band != band:
                    src = self.last_detected_band
                    dst = band
                    self.transition_counts[src, dst] += 1
                    
                    # Update transition confidence based on frequency of occurrence
                    total_from_src = np.sum(self.transition_counts[src])
                    prob = self.transition_counts[src, dst] / total_from_src
                    # Base confidence scales with repeated observations
                    self.transition_confidence[src, dst] = float(min(0.95, prob * (0.6 + 0.15 * min(3, self.transition_counts[src, dst]))))
                    
                    self.recent_transitions.append((src, dst, current_time))
                    if len(self.recent_transitions) > 50:
                        self.recent_transitions.pop(0)

            self.last_detected_band = band
            self.last_detected_time = current_time

    def record_prediction_hit(self, from_band: int, to_band: int):
        """Increases confidence when an agile hop prediction is validated."""
        if 0 <= from_band < self.num_bands and 0 <= to_band < self.num_bands:
            self.transition_confidence[from_band, to_band] = float(min(0.98, self.transition_confidence[from_band, to_band] + 0.08))

    def record_prediction_miss(self, from_band: int, to_band: int):
        """Decreases confidence when an agile hop prediction misses."""
        if 0 <= from_band < self.num_bands and 0 <= to_band < self.num_bands:
            self.transition_confidence[from_band, to_band] = float(max(0.05, self.transition_confidence[from_band, to_band] - 0.25))

    def get_transition_probability(self, from_band: int, to_band: int) -> float:
        if from_band < 0 or from_band >= self.num_bands or to_band < 0 or to_band >= self.num_bands:
            return 0.0
        total_out = np.sum(self.transition_counts[from_band])
        if total_out > 0:
            return float(self.transition_counts[from_band, to_band] / total_out)
        return 0.0

    def predict_next_band(self) -> Optional[Tuple[int, float]]:
        """Returns (predicted_target_band, transition_probability) from the last detected band."""
        if self.last_detected_band == -1:
            return None
        src = self.last_detected_band
        row = self.transition_counts[src]
        total = np.sum(row)
        if total == 0:
            return None
            
        best_dst = int(np.argmax(row))
        prob = float(row[best_dst] / total)
        return best_dst, prob

    def predict_transition_activity(self, band: int) -> float:
        """Returns the transition probability from last_detected_band to `band`.
        
        Returns 0.0 if no transition has been observed between last_detected_band and `band`.
        """
        if self.last_detected_band == -1 or band < 0 or band >= self.num_bands:
            return 0.0
            
        src = self.last_detected_band
        dst = band
        count = self.transition_counts[src, dst]
        
        if count >= 1:
            total_from_src = np.sum(self.transition_counts[src])
            prob = float(count / total_from_src)
            return prob
        return 0.0

    def compute_score(self, band: int) -> float:
        """Returns weighted transition score contribution."""
        prob = self.predict_transition_activity(band)
        return float(self.weight * prob)

    def get_learned_chains(self) -> List[Dict[str, Any]]:
        chains = []
        for src in range(self.num_bands):
            total = np.sum(self.transition_counts[src])
            if total >= 1:
                for dst in range(self.num_bands):
                    count = self.transition_counts[src, dst]
                    if count > 0:
                        prob = float(count / total)
                        conf = float(self.transition_confidence[src, dst])
                        chains.append({
                            "from_band": src,
                            "to_band": dst,
                            "count": int(count),
                            "probability": prob,
                            "confidence": conf,
                            "confidence_pct": f"{conf * 100:.0f}%"
                        })
        return sorted(chains, key=lambda x: x["count"], reverse=True)
