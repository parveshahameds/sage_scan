import numpy as np
from typing import Dict, List, Tuple, Any, Optional

class TemporalModel:
    """Discovers periodicity in signal detections and dynamically manages cycle confidence based on empirical prediction hits and misses.
    
    Tracks the full 9-stage learning progression:
    UNKNOWN -> OBSERVING -> FIRST DETECTION -> PATTERN HYPOTHESIS -> PERIOD LEARNED ->
    PREDICTION GENERATED -> HIGH CONFIDENCE -> PRE-POSITIONED -> PREDICTION SUCCESS
    """
    def __init__(self, num_bands: int = 50, weight: float = 0.5):
        self.num_bands = num_bands
        self.weight = weight
        self.reset()

    def reset(self):
        # Maps band -> [period, offset, confidence, consecutive_misses, successful_hits, is_active]
        self.period_estimates: Dict[int, Dict[str, Any]] = {
            i: {
                "period": 0,
                "offset": 0,
                "confidence": 0.0,
                "consecutive_misses": 0,
                "successful_hits": 0,
                "is_active": False
            } for i in range(self.num_bands)
        }
        self.detection_timestamps: Dict[int, List[int]] = {i: [] for i in range(self.num_bands)}
        self.learning_stages: Dict[int, str] = {i: "UNKNOWN" for i in range(self.num_bands)}
        self.observation_counts: Dict[int, int] = {i: 0 for i in range(self.num_bands)}

    def get_periodicity(self, band: int) -> Tuple[int, int, float]:
        """Returns (period, offset, confidence) tuple for backward compatibility and tests."""
        state = self.period_estimates.get(band, {})
        return (state.get("period", 0), state.get("offset", 0), state.get("confidence", 0.0))

    def update(self, band: int, detected: bool, current_time: int):
        """Updates timestamps when a signal is detected and re-evaluates candidate periods."""
        self.observation_counts[band] += 1
        
        if detected:
            self.detection_timestamps[band].append(current_time)
            
            if len(self.detection_timestamps[band]) == 1:
                self.learning_stages[band] = "FIRST DETECTION"
            elif len(self.detection_timestamps[band]) >= 2:
                self._analyze_periodicity(band)
        else:
            if len(self.detection_timestamps[band]) == 0:
                self.learning_stages[band] = "OBSERVING"

    def _analyze_periodicity(self, band: int):
        timestamps = self.detection_timestamps[band]
        if len(timestamps) < 2:
            return

        span = timestamps[-1] - timestamps[0]
        max_p = min(30, max(2, span))
        best_p = 0
        best_offset = 0
        best_conf = 0.0

        for p in range(2, max_p + 1):
            modulos = [t % p for t in timestamps]
            counts = np.bincount(modulos, minlength=p)
            offset = int(np.argmax(counts))
            matches = counts[offset]
            conf = float(matches / len(timestamps))

            # Prefer higher periods if consistency is equal (avoids sub-harmonic locking)
            if conf > best_conf or (conf == best_conf and conf >= 0.80 and p > best_p):
                best_p = p
                best_offset = offset
                best_conf = conf

        # Threshold for locked periodicity: >= 2 detections matching candidate period
        if len(timestamps) >= 2 and best_p > 0 and best_conf >= 0.60:
            current_conf = self.period_estimates[band]["confidence"]
            # Base confidence grows with observation count and consistency
            new_conf = max(current_conf, min(0.95, best_conf * (0.60 + 0.08 * min(5, len(timestamps)))))
            
            self.period_estimates[band]["period"] = best_p
            self.period_estimates[band]["offset"] = best_offset
            self.period_estimates[band]["confidence"] = float(new_conf)
            self.period_estimates[band]["is_active"] = True
            self.period_estimates[band]["consecutive_misses"] = 0
            
            if new_conf >= 0.80:
                self.learning_stages[band] = "HIGH CONFIDENCE"
            else:
                self.learning_stages[band] = "PERIOD LEARNED"

    def record_prediction_hit(self, band: int):
        """Called when a predicted periodic pulse is successfully intercepted: INCREASES confidence."""
        state = self.period_estimates[band]
        if state["is_active"]:
            state["successful_hits"] += 1
            state["consecutive_misses"] = 0
            # Confidence approaches 0.98 asymptotically on validated hits
            state["confidence"] = float(min(0.98, state["confidence"] + 0.08))
            self.learning_stages[band] = "PREDICTION SUCCESS"
            if state["confidence"] >= 0.80:
                self.learning_stages[band] = "HIGH CONFIDENCE"

    def record_prediction_miss(self, band: int):
        """Called when a predicted periodic pulse is missed: DECREASES confidence dynamically."""
        state = self.period_estimates[band]
        if state["is_active"]:
            state["consecutive_misses"] += 1
            # Substantial penalty on miss: -0.22 per miss
            state["confidence"] = float(max(0.05, state["confidence"] - 0.22))
            
            # If misses accumulate (confidence drops below 0.35), deactivate periodic exploitation
            if state["confidence"] < 0.35 or state["consecutive_misses"] >= 2:
                state["is_active"] = False
                self.learning_stages[band] = "OBSERVING"

    def predict_activity(self, band: int, target_time: int) -> float:
        """Returns the probability [0, 1] that band will be active at target_time based on periodic cycle."""
        state = self.period_estimates.get(band)
        if not state or not state["is_active"] or state["confidence"] < 0.35:
            return 0.0
            
        p = state["period"]
        offset = state["offset"]
        conf = state["confidence"]
        
        if p > 0:
            dist = (target_time - offset) % p
            if dist == 0:
                # Exact predicted pulse slot
                self.learning_stages[band] = "PREDICTION GENERATED"
                return float(conf)
            elif dist == p - 1:
                # 1 slot before pulse (lead time for pre-positioning)
                self.learning_stages[band] = "PRE-POSITIONED"
                return float(conf * 0.70)
        return 0.0

    def compute_score(self, band: int, current_time: int) -> float:
        """Returns weighted temporal contribution, including active hypothesis probing."""
        state = self.period_estimates.get(band, {})
        if state.get("is_active") and state.get("confidence", 0) >= 0.35:
            prob = self.predict_activity(band, current_time)
            return float(self.weight * prob)
            
        # Active Hypothesis Probing: if 1 detection has been observed, probe candidate harmonic intervals
        timestamps = self.detection_timestamps[band]
        if len(timestamps) == 1:
            t1 = timestamps[0]
            dt_elapsed = current_time - t1
            if dt_elapsed in [5, 7, 9, 11, 14]:
                self.learning_stages[band] = "PATTERN HYPOTHESIS"
                return float(self.weight * 0.85) # High priority follow-up probe
        return 0.0

    def get_upcoming_pulse(self, band: int, current_time: int) -> Optional[Dict[str, Any]]:
        state = self.period_estimates.get(band)
        if state and state["is_active"] and state["confidence"] >= 0.35:
            p = state["period"]
            offset = state["offset"]
            conf = state["confidence"]
            slots_until = (offset - (current_time % p)) % p
            return {
                "band": band,
                "period": p,
                "offset": offset,
                "confidence": conf,
                "confidence_pct": f"{conf * 100:.0f}%",
                "slots_until_next": int(slots_until),
                "next_expected_time": int(current_time + slots_until)
            }
        return None

    def get_learning_diagnostics(self, band: int, current_time: int) -> Dict[str, Any]:
        """Provides complete explainability diagnostics for learning progression and debug metrics."""
        timestamps = self.detection_timestamps[band]
        intervals = [timestamps[i+1] - timestamps[i] for i in range(len(timestamps) - 1)]
        state = self.period_estimates.get(band, {})
        
        p = state.get("period", 0)
        offset = state.get("offset", 0)
        conf = state.get("confidence", 0.0)
        is_active = state.get("is_active", False)
        
        consistency_pct = "0%"
        if p > 0 and len(timestamps) >= 2:
            mods = [t % p for t in timestamps]
            matches = sum(1 for m in mods if m == offset)
            consistency_pct = f"{(matches / len(timestamps)) * 100:.0f}%"
            
        slots_until = None
        next_pulse = None
        if p > 0 and is_active:
            slots_until = int((offset - (current_time % p)) % p)
            next_pulse = int(current_time + slots_until)

        stage = self.learning_stages.get(band, "UNKNOWN")
        if conf >= 0.80 and is_active:
            stage = "HIGH CONFIDENCE"
        elif is_active:
            stage = "PERIOD LEARNED"

        return {
            "band": band,
            "learning_stage": stage,
            "observation_count": self.observation_counts.get(band, 0),
            "detection_timestamps": timestamps,
            "timestamps_display": f"[{', '.join(str(t) for t in timestamps[-8:])}]" if timestamps else "[]",
            "learned_intervals": intervals,
            "intervals_display": f"[{', '.join(str(d) for d in intervals[-8:])}]" if intervals else "None",
            "estimated_period": p if p > 0 else "None",
            "offset": offset,
            "period_consistency": consistency_pct,
            "confidence": conf,
            "confidence_pct": f"{conf * 100:.0f}%",
            "next_predicted_pulse": f"t = {next_pulse}" if next_pulse is not None else "Unknown",
            "slots_until_next": slots_until,
            "is_active": is_active
        }
