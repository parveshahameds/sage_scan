import numpy as np
import logging
from typing import Dict, Any, List
from src.core.digital_twin import DigitalTwinEngine
from src.core.virtual_receiver import VirtualReceiver
from src.core.sage_engine import SAGEScanEngine
from src.core.metrics_engine import MetricsEngine

logger = logging.getLogger("ComparisonSimulator")

class SequentialScanner:
    """A traditional open-loop receiver that sweeps frequencies sequentially: F0 -> F1 -> ... -> F49 -> F0."""
    def __init__(self, num_bands: int = 50):
        self.num_bands = num_bands
        self.current_band = 0

    def reset(self):
        self.current_band = 0

    def get_next_band(self) -> int:
        band = self.current_band
        self.current_band = (self.current_band + 1) % self.num_bands
        return band


class ComparisonSimulator:
    """Executes SAGE-SCAN and Sequential Scanner simultaneously against the identical Digital Twin environment.
    
    Both scanners receive identical receiver constraints (Instantaneous Bandwidth W = 1)
    and observe the same underlying physical RF events.
    """
    def __init__(self, digital_twin: DigitalTwinEngine, sage_engine: SAGEScanEngine, seed: int = 42):
        self.digital_twin = digital_twin
        self.sage_engine = sage_engine
        self.sequential_scanner = SequentialScanner(digital_twin.num_bands)
        
        # Virtual receivers (strictly W=1)
        self.sage_receiver = VirtualReceiver(digital_twin.num_bands, bandwidth=1, seed=seed)
        self.seq_receiver = VirtualReceiver(digital_twin.num_bands, bandwidth=1, seed=seed + 1)
        
        # Performance Metrics Engines
        self.sage_metrics = MetricsEngine(digital_twin.num_bands)
        self.seq_metrics = MetricsEngine(digital_twin.num_bands)
        
        # Track initial active timestamps for delay calculation
        self.first_emission_times: Dict[int, int] = {}
        self.history: List[Dict[str, Any]] = []
        self.step_logs: List[Dict[str, Any]] = []
        self.is_completed = False

    def reset(self):
        self.digital_twin.reset()
        self.sage_engine.reset()
        self.sequential_scanner.reset()
        self.sage_receiver.reset()
        self.seq_receiver.reset()
        self.sage_metrics.reset()
        self.seq_metrics.reset()
        self.first_emission_times = {}
        self.history = []
        self.step_logs = []
        self.is_completed = False

    def step(self) -> Dict[str, Any]:
        """Advances both scanners and the Digital Twin by one simulation time slot.
        
        When reaching the simulation endpoint, freezes the final live model state.
        """
        # If simulation has already reached terminal slot, return frozen final record
        if self.digital_twin.current_time >= self.digital_twin.max_time - 1:
            self.is_completed = True
            if self.history:
                return self.history[-1]
            return {
                "time": self.digital_twin.current_time,
                "sage_band": self.sage_receiver.current_band,
                "seq_band": self.seq_receiver.current_band,
                "sage_obs": self.sage_receiver.last_observation,
                "seq_obs": self.seq_receiver.last_observation,
                "sage_decision": self.sage_engine.last_decision,
                "prediction_alert": self.sage_engine.last_prediction_alert
            }

        # 1. Digital Twin advances world clock and physical signals emit
        t = self.digital_twin.step()
        if t >= self.digital_twin.max_time - 1:
            self.is_completed = True
        
        # Count total active physical signals across all channels in the world at step t
        num_active_in_world = int(np.sum(self.digital_twin.ground_truth[t]))
        
        # Record ground truth active channels for latency tracking
        for b in range(self.digital_twin.num_bands):
            if self.digital_twin.ground_truth[t, b] == 1:
                if b not in self.first_emission_times:
                    self.first_emission_times[b] = t
        
        # 2. SAGE-SCAN decides its next channel based on its internal models
        sage_decision = self.sage_engine.decide_next_action(t)
        sage_band = sage_decision["selected_band"]
        
        # Record prediction attempt if SAGE generated an active predictive target
        if sage_decision.get("is_prediction_active"):
            self.sage_metrics.record_prediction_attempt()
        
        # 3. Sequential Scanner chooses its next sequential channel
        seq_band = self.sequential_scanner.get_next_band()
        
        # 4. Tune physical receiver sensors
        self.sage_receiver.tune_to(sage_band)
        self.seq_receiver.tune_to(seq_band)
        
        # 5. Virtual receivers observe physical spectrum (Bandwidth W=1)
        sage_obs = self.sage_receiver.observe(self.digital_twin)
        seq_obs = self.seq_receiver.observe(self.digital_twin)
        
        # 6. Calculate detection delays
        sage_delay = 0
        if sage_obs["detected"] and sage_band in self.first_emission_times:
            sage_delay = t - self.first_emission_times[sage_band]
            del self.first_emission_times[sage_band]
            
        seq_delay = 0
        if seq_obs["detected"] and seq_band in self.first_emission_times:
            seq_delay = t - self.first_emission_times[seq_band]
        
        # 7. Record observation outcomes and exact world emission accounting
        self.sage_metrics.record_step_emissions(
            band=sage_band,
            detected=sage_obs["detected"],
            is_actually_active=sage_obs["ground_truth_active"],
            is_false_alarm=sage_obs["is_false_alarm"],
            num_active_in_world=num_active_in_world,
            delay=sage_delay
        )
        self.seq_metrics.record_step_emissions(
            band=seq_band,
            detected=seq_obs["detected"],
            is_actually_active=seq_obs["ground_truth_active"],
            is_false_alarm=seq_obs["is_false_alarm"],
            num_active_in_world=num_active_in_world,
            delay=seq_delay
        )
        
        # 8. SAGE-SCAN learns from its observation and validates predictions
        prediction_alert = self.sage_engine.update_from_observation(sage_obs, t)
        
        # Record prediction hit/miss events strictly from validated prediction outcome
        if prediction_alert:
            a_type = prediction_alert.get("type")
            if a_type in ["PREDICTION_SUCCESS", "PREEMPTIVE_INTERCEPT"]:
                self.sage_metrics.record_prediction_hit(is_preemptive=True)
            elif a_type == "PREDICTION_MISS":
                self.sage_metrics.record_prediction_miss()
            elif a_type == "UNKNOWN_DISCOVERED":
                self.sage_metrics.record_unknown_threat_discovered()

        # 9. Structured Audit Logging
        pred_acc = self.sage_metrics.prediction_accuracy
        log_entry = {
            "t": t,
            "selected_channel": f"F{sage_band}",
            "prediction_target": f"F{sage_band}" if sage_decision.get("is_prediction_active") else "None",
            "prediction_confidence": sage_decision.get("prediction_confidence_pct", "N/A"),
            "prediction_accuracy": f"{pred_acc * 100:.1f}%",
            "selection_score": f"{sage_decision.get('selection_score', 0.0):.3f}",
            "simulation_running": not self.is_completed,
            "simulation_complete": self.is_completed,
            "confidence_source": sage_decision.get("confidence_source", "Exploration / Information Gain")
        }
        self.step_logs.append(log_entry)

        step_record = {
            "time": t,
            "sage_band": sage_band,
            "seq_band": seq_band,
            "sage_obs": sage_obs,
            "seq_obs": seq_obs,
            "sage_decision": sage_decision,
            "prediction_alert": prediction_alert,
            "audit_log": log_entry
        }
        self.history.append(step_record)
        return step_record

    def get_comparison_summary(self) -> Dict[str, Any]:
        """Calculates benchmark metrics strictly from real event logs."""
        t = self.digital_twin.current_time
        total_emissions = int(np.sum(self.digital_twin.ground_truth[:t + 1]))
        return {
            "time": t,
            "total_emitted_signals": total_emissions,
            "sage": self.sage_metrics.get_summary(total_emissions),
            "sequential": self.seq_metrics.get_summary(total_emissions)
        }
