import numpy as np
from typing import Dict, List, Any, Optional

class SimulatedThreat:
    """Base class for all simulated threat objects in the Digital Twin environment."""
    def __init__(self, threat_id: str, name: str, threat_type: str, priority: float = 1.0):
        self.threat_id = threat_id
        self.name = name
        self.threat_type = threat_type
        self.priority = priority
        self.active_now = False
        self.current_frequency = -1
        self.history_active_times: List[int] = []

    def update_state(self, current_time: int, rng: np.random.Generator) -> bool:
        """Updates the internal state for current_time and returns True if emitting a signal."""
        raise NotImplementedError

    def get_state(self, current_time: int) -> Dict[str, Any]:
        """Returns the Digital Twin Ground-Truth state for inspector display."""
        raise NotImplementedError


class PeriodicThreat(SimulatedThreat):
    """THREAT-01 / PERIODIC: Emits periodic pulses at fixed intervals (e.g. F15 every 7 slots)."""
    def __init__(self, threat_id: str = "THREAT-01", name: str = "PERIODIC RADAR", 
                 band: int = 15, period: int = 7, pulse_width: int = 1, offset: int = 0, priority: float = 1.5):
        super().__init__(threat_id, name, "Periodic Radar", priority)
        self.band = band
        self.period = period
        self.pulse_width = pulse_width
        self.offset = offset
        self.current_frequency = band

    def update_state(self, current_time: int, rng: np.random.Generator) -> bool:
        self.current_frequency = self.band
        is_pulse = ((current_time - self.offset) % self.period) < self.pulse_width
        self.active_now = bool(is_pulse)
        if self.active_now:
            self.history_active_times.append(current_time)
        return self.active_now

    def get_state(self, current_time: int) -> Dict[str, Any]:
        slots_to_next = (self.offset - (current_time % self.period)) % self.period
        next_t = current_time + slots_to_next if slots_to_next > 0 else current_time + self.period
        return {
            "id": self.threat_id,
            "name": self.name,
            "type": self.threat_type,
            "current_band": self.band,
            "frequency_label": f"F{self.band}",
            "behavior": f"Pulse every {self.period} slots",
            "period": self.period,
            "offset": self.offset,
            "status": "EMITTING 🚨" if self.active_now else "SILENT ⏳",
            "active": self.active_now,
            "next_emission_time": next_t,
            "slots_until_next": int(slots_to_next if slots_to_next > 0 else self.period),
            "priority": self.priority
        }


class AgileThreat(SimulatedThreat):
    """THREAT-02 / AGILE: Hops across frequency bands in sequence (e.g. F10 -> F20 -> F30 -> F40)."""
    def __init__(self, threat_id: str = "THREAT-02", name: str = "FREQUENCY AGILE RADAR",
                 bands: List[int] = None, hop_interval: int = 15, prob: float = 1.0, priority: float = 2.0):
        super().__init__(threat_id, name, "Frequency Agile", priority)
        self.bands = bands if bands is not None else [10, 20, 30, 40]
        self.hop_interval = hop_interval
        self.prob = prob
        self.current_band_idx = 0
        self.current_frequency = self.bands[0]

    def update_state(self, current_time: int, rng: np.random.Generator) -> bool:
        self.current_band_idx = (current_time // self.hop_interval) % len(self.bands)
        self.current_frequency = self.bands[self.current_band_idx]
        self.active_now = bool(rng.random() < self.prob)
        if self.active_now:
            self.history_active_times.append(current_time)
        return self.active_now

    def get_state(self, current_time: int) -> Dict[str, Any]:
        next_idx = (self.current_band_idx + 1) % len(self.bands)
        next_band = self.bands[next_idx]
        slots_to_hop = self.hop_interval - (current_time % self.hop_interval)
        return {
            "id": self.threat_id,
            "name": self.name,
            "type": self.threat_type,
            "current_band": self.current_frequency,
            "frequency_label": f"F{self.current_frequency}",
            "behavior": f"Hops sequence {' → '.join([f'F{b}' for b in self.bands])}",
            "hop_sequence": self.bands,
            "next_predicted_hop": next_band,
            "next_hop_label": f"F{next_band}",
            "slots_until_hop": int(slots_to_hop),
            "status": "EMITTING 🚨" if self.active_now else "HOPPING / QUIET",
            "active": self.active_now,
            "priority": self.priority
        }


class IntermittentThreat(SimulatedThreat):
    """THREAT-03 / INTERMITTENT: Emits unpredictable bursts on a specific frequency (e.g. F35)."""
    def __init__(self, threat_id: str = "THREAT-03", name: str = "BURST / INTERMITTENT JAMMER",
                 band: int = 35, burst_prob: float = 0.18, priority: float = 1.2):
        super().__init__(threat_id, name, "Intermittent / Burst", priority)
        self.band = band
        self.burst_prob = burst_prob
        self.current_frequency = band

    def update_state(self, current_time: int, rng: np.random.Generator) -> bool:
        self.current_frequency = self.band
        self.active_now = bool(rng.random() < self.burst_prob)
        if self.active_now:
            self.history_active_times.append(current_time)
        return self.active_now

    def get_state(self, current_time: int) -> Dict[str, Any]:
        return {
            "id": self.threat_id,
            "name": self.name,
            "type": self.threat_type,
            "current_band": self.band,
            "frequency_label": f"F{self.band}",
            "behavior": f"Unpredictable bursts (p={self.burst_prob:.2f})",
            "status": "BURSTING 🚨" if self.active_now else "SILENT ⏳",
            "active": self.active_now,
            "burst_probability": self.burst_prob,
            "priority": self.priority
        }


class UnknownThreat(SimulatedThreat):
    """Spawnable unknown threat on an unobserved channel (e.g. F2)."""
    def __init__(self, threat_id: str = "THREAT-UNKNOWN", name: str = "UNKNOWN COVERT TRANSMITTER",
                 band: int = 2, prob: float = 1.0, priority: float = 2.5):
        super().__init__(threat_id, name, "Covert / Unknown", priority)
        self.band = band
        self.prob = prob
        self.current_frequency = band

    def update_state(self, current_time: int, rng: np.random.Generator) -> bool:
        self.current_frequency = self.band
        self.active_now = bool(rng.random() < self.prob)
        if self.active_now:
            self.history_active_times.append(current_time)
        return self.active_now

    def get_state(self, current_time: int) -> Dict[str, Any]:
        return {
            "id": self.threat_id,
            "name": self.name,
            "type": self.threat_type,
            "current_band": self.band,
            "frequency_label": f"F{self.band}",
            "behavior": f"Active covert signal on F{self.band}",
            "status": "TRANSMITTING 🚨" if self.active_now else "SILENT",
            "active": self.active_now,
            "priority": self.priority
        }
