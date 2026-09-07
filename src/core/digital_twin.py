import numpy as np
from typing import Dict, List, Any, Optional
from src.core.threat_engine import (
    SimulatedThreat, PeriodicThreat, AgileThreat, IntermittentThreat, UnknownThreat
)

class DigitalTwinEngine:
    """The Digital Twin is the virtual RF world.
    
    It maintains ground truth physics, simulation time, simulated threat objects,
    and RF spectrum emissions. SAGE-SCAN algorithm NEVER reads this directly!
    """
    def __init__(self, num_bands: int = 50, max_time: int = 1005, seed: int = 42, custom_threats: List[SimulatedThreat] = None):
        self.num_bands = num_bands
        self.max_time = max_time
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        
        self.custom_threats = custom_threats
        self.threats: List[SimulatedThreat] = []
        self.current_time = 0
        
        # Ground truth matrix: (max_time, num_bands)
        self.ground_truth = np.zeros((self.max_time, self.num_bands), dtype=int)
        self.event_log: List[Dict[str, Any]] = []
        self.snapshots: Dict[int, Dict[str, Any]] = {}
        
        self.reset()

    def reset(self):
        self.rng = np.random.default_rng(self.seed)
        self.current_time = 0
        self.ground_truth = np.zeros((self.max_time, self.num_bands), dtype=int)
        self.event_log = []
        self.snapshots = {}
        
        if self.custom_threats is not None:
            self.threats = [t for t in self.custom_threats]
        else:
            self.threats = [
                PeriodicThreat(threat_id="THREAT-01", name="PERIODIC RADAR", band=15, period=7, pulse_width=1, offset=0),
                AgileThreat(threat_id="THREAT-02", name="FREQUENCY AGILE", bands=[10, 20, 30, 40], hop_interval=15, prob=1.0),
                IntermittentThreat(threat_id="THREAT-03", name="INTERMITTENT BURST", band=35, burst_prob=0.18)
            ]
        
        # Record initial snapshot
        self._record_snapshot()

    def step(self) -> int:
        """Advances the Digital Twin world by one time step."""
        if self.current_time >= self.max_time - 1:
            return self.current_time

        self.current_time += 1
        t = self.current_time

        # Update all threat states
        active_bands = []
        for threat in self.threats:
            is_active = threat.update_state(t, self.rng)
            if is_active:
                b = threat.current_frequency % self.num_bands
                self.ground_truth[t, b] = 1
                active_bands.append((threat.threat_id, b))
                self.event_log.append({
                    "time": t,
                    "event": "EMISSION",
                    "threat_id": threat.threat_id,
                    "threat_name": threat.name,
                    "band": b,
                    "message": f"🚨 {threat.name} emitted signal on F{b}"
                })

        self._record_snapshot()
        return self.current_time

    def _record_snapshot(self):
        t = self.current_time
        threat_states = [threat.get_state(t) for threat in self.threats]
        active_channels = [b for b in range(self.num_bands) if self.ground_truth[t, b] == 1]
        self.snapshots[t] = {
            "time": t,
            "threat_states": threat_states,
            "active_channels": active_channels,
            "ground_truth_row": self.ground_truth[t].copy()
        }

    def inject_threat(self, threat: SimulatedThreat):
        """Injects a new threat dynamically into the Digital Twin."""
        self.threats.append(threat)
        self.event_log.append({
            "time": self.current_time,
            "event": "INJECT_THREAT",
            "threat_id": threat.threat_id,
            "threat_name": threat.name,
            "band": threat.current_frequency,
            "message": f"⚡ Injected {threat.name} on F{threat.current_frequency}"
        })

    def spawn_unknown_threat(self, band: int = 2) -> UnknownThreat:
        """Spawns an unknown threat on an unobserved channel (e.g. F2)."""
        unknown = UnknownThreat(threat_id=f"UNKNOWN-F{band}", name=f"COVERT TRANSMITTER (F{band})", band=band, prob=1.0)
        self.inject_threat(unknown)
        return unknown

    def query_physical_spectrum(self, band: int) -> Dict[str, Any]:
        """Returns physical state for a specific band at current_time."""
        t = self.current_time
        is_active = bool(self.ground_truth[t, band] == 1)
        emitting_threats = [threat.threat_id for threat in self.threats if threat.current_frequency == band and threat.active_now]
        return {
            "time": t,
            "band": band,
            "is_active": is_active,
            "emitting_threats": emitting_threats
        }

    def get_ground_truth_snapshot(self) -> Dict[str, Any]:
        """Provides full world state for Digital Twin Inspector (God Mode)."""
        t = self.current_time
        return {
            "current_time": t,
            "threat_count": len(self.threats),
            "threat_states": [threat.get_state(t) for threat in self.threats],
            "active_bands": [b for b in range(self.num_bands) if self.ground_truth[t, b] == 1],
            "total_emissions_so_far": int(np.sum(self.ground_truth[:t + 1]))
        }
