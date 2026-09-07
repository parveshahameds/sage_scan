import numpy as np
from typing import Dict, Any, List, Optional
from src.core.digital_twin import DigitalTwinEngine
from src.core.sage_engine import SAGEScanEngine
from src.core.comparison_engine import ComparisonSimulator
from src.core.threat_engine import PeriodicThreat, AgileThreat, IntermittentThreat, UnknownThreat

class JudgeDemoController:
    """Controls the deterministic 7-phase Judge Demonstration with consistent timeline duration (120 slots)."""
    def __init__(self, seed: int = 42, max_demo_time: int = 120):
        self.seed = seed
        self.max_demo_time = max_demo_time
        self.digital_twin = DigitalTwinEngine(num_bands=50, max_time=max_demo_time + 10, seed=seed)
        self.sage_engine = SAGEScanEngine(num_bands=50, w_hist=0.3, w_temp=0.5, w_trans=0.4, c_explore=0.3, seed=seed)
        self.comparison_sim = ComparisonSimulator(self.digital_twin, self.sage_engine, seed=seed)
        
        # Zero false alarms during deterministic judge demo showcase
        self.comparison_sim.sage_receiver.pfa = 0.0
        self.comparison_sim.seq_receiver.pfa = 0.0
        
        self.current_phase = 1
        self.phase_descriptions = {
            1: "PHASE 1: Cold Start — Exploring unobserved spectrum channels with zero prior knowledge",
            2: "PHASE 2: Periodic Pulse Sampling — Detecting intermittent signals on F15",
            3: "PHASE 3: Pattern Discovery — Discovered Period = 7 slots on F15 (Confidence > 80%)",
            4: "PHASE 4: Predictive Pre-Positioning — Moving receiver to F15 BEFORE pulse occurs (Prediction Success)",
            5: "PHASE 5: Frequency Agility Adaptation — Learning F10 → F20 → F30 → F40 transition chain (Preemptive Intercept)",
            6: "PHASE 6: Unknown Threat Discovery — Exploration prioritizes neglected channel F2 (Unknown Signal Discovered)",
            7: "PHASE 7: Final Comparative Analysis — SAGE-SCAN vs. Sequential Sweep evaluation & insights"
        }
        self.unknown_threat_injected = False
        self.is_finished = False

    def reset(self):
        self.digital_twin.reset()
        self.sage_engine.reset()
        self.comparison_sim.reset()
        self.comparison_sim.sage_receiver.pfa = 0.0
        self.comparison_sim.seq_receiver.pfa = 0.0
        self.current_phase = 1
        self.unknown_threat_injected = False
        self.is_finished = False

    def get_phase_for_time(self, t: int) -> int:
        if t <= 10:
            return 1
        elif t <= 28:
            return 2
        elif t <= 32:
            return 3
        elif t <= 40:
            return 4
        elif t <= 70:
            return 5
        elif t <= 95:
            return 6
        else:
            return 7

    def step(self) -> Dict[str, Any]:
        """Executes one deterministic step of the judge demonstration.
        
        Freezes state when demonstration completes (t >= max_demo_time).
        """
        t = self.digital_twin.current_time
        
        # Freeze final live state when max demo time is reached
        if t >= self.max_demo_time:
            self.is_finished = True
            last_rec = self.comparison_sim.history[-1] if self.comparison_sim.history else {}
            return {
                "time": self.max_demo_time,
                "sage_band": last_rec.get("sage_band", self.comparison_sim.sage_receiver.current_band),
                "seq_band": last_rec.get("seq_band", self.comparison_sim.seq_receiver.current_band),
                "sage_obs": last_rec.get("sage_obs", self.comparison_sim.sage_receiver.last_observation),
                "seq_obs": last_rec.get("seq_obs", self.comparison_sim.seq_receiver.last_observation),
                "sage_decision": last_rec.get("sage_decision", self.sage_engine.last_decision),
                "prediction_alert": last_rec.get("prediction_alert", self.sage_engine.last_prediction_alert),
                "phase_info": {
                    "phase_number": 7,
                    "phase_title": self.phase_descriptions[7],
                    "time": self.max_demo_time,
                    "is_finished": True
                }
            }

        # Educational showcase sampling schedule
        target_override = None
        if t == 7:
            target_override = 10 # Sample agile start F10 (t in 0..14 is on F10)
        elif t == 13:
            target_override = 15 # Sample F15 pulse at t=14
        elif t == 20:
            target_override = 15 # Sample F15 pulse at t=21 -> Period=7 learned
        elif t == 22:
            target_override = 20 # Sample agile hop F20 (t in 15..29 is on F20)
        elif t == 37:
            target_override = 30 # Sample agile hop F30 (t in 30..44 is on F30)
        elif t == 52:
            target_override = 40 # Sample agile hop F40 (t in 45..59 is on F40)
        elif t == 67:
            target_override = 10 # Sample agile cycle F10 (t in 60..74 is on F10)
        elif t == 74:
            target_override = 2  # Exploration visits covert channel F2 in Phase 6

        # Phase 6 injection trigger at t=70: Inject Covert signal on F2
        if t == 70 and not self.unknown_threat_injected:
            self.digital_twin.spawn_unknown_threat(band=2)
            self.unknown_threat_injected = True

        # Run comparison step
        step_data = self._step_with_override(target_override)
        new_t = step_data["time"]
        self.current_phase = self.get_phase_for_time(new_t)

        phase_info = {
            "phase_number": self.current_phase,
            "phase_title": self.phase_descriptions.get(self.current_phase, ""),
            "time": new_t,
            "is_finished": bool(new_t >= self.max_demo_time)
        }

        # Special countdown during Phase 4 for F15 pulse at t=35
        if self.current_phase == 4:
            slots_left = 35 - new_t
            if slots_left > 0:
                phase_info["countdown"] = f"NEXT F15 SIGNAL IN: {slots_left} slot{'s' if slots_left > 1 else ''} ⏳"
            elif slots_left == 0:
                phase_info["countdown"] = "🎯 SIGNAL ACTIVE NOW — PRE-POSITIONED & INTERCEPTED!"

        if new_t >= self.max_demo_time:
            self.is_finished = True

        step_data["phase_info"] = phase_info
        return step_data

    def _step_with_override(self, override_band: Optional[int]) -> Dict[str, Any]:
        """Custom comparison step supporting educational override."""
        t = self.digital_twin.step()
        num_active_in_world = int(np.sum(self.digital_twin.ground_truth[t]))
        
        for b in range(self.digital_twin.num_bands):
            if self.digital_twin.ground_truth[t, b] == 1:
                if b not in self.comparison_sim.first_emission_times:
                    self.comparison_sim.first_emission_times[b] = t

        sage_decision = self.sage_engine.decide_next_action(t, override_band=override_band)
        sage_band = sage_decision["selected_band"]
        
        if sage_decision.get("is_prediction_active"):
            self.comparison_sim.sage_metrics.record_prediction_attempt()
            
        seq_band = self.comparison_sim.sequential_scanner.get_next_band()
        
        self.comparison_sim.sage_receiver.tune_to(sage_band)
        self.comparison_sim.seq_receiver.tune_to(seq_band)
        
        sage_obs = self.comparison_sim.sage_receiver.observe(self.digital_twin)
        seq_obs = self.comparison_sim.seq_receiver.observe(self.digital_twin)
        
        sage_delay = 0
        if sage_obs["detected"] and sage_band in self.comparison_sim.first_emission_times:
            sage_delay = t - self.comparison_sim.first_emission_times[sage_band]
            del self.comparison_sim.first_emission_times[sage_band]
            
        seq_delay = 0
        if seq_obs["detected"] and seq_band in self.comparison_sim.first_emission_times:
            seq_delay = t - self.comparison_sim.first_emission_times[seq_band]
            
        self.comparison_sim.sage_metrics.record_step_emissions(
            band=sage_band,
            detected=sage_obs["detected"],
            is_actually_active=sage_obs["ground_truth_active"],
            is_false_alarm=sage_obs["is_false_alarm"],
            num_active_in_world=num_active_in_world,
            delay=sage_delay
        )
        self.comparison_sim.seq_metrics.record_step_emissions(
            band=seq_band,
            detected=seq_obs["detected"],
            is_actually_active=seq_obs["ground_truth_active"],
            is_false_alarm=seq_obs["is_false_alarm"],
            num_active_in_world=num_active_in_world,
            delay=seq_delay
        )
        
        prediction_alert = self.sage_engine.update_from_observation(sage_obs, t)
        
        if prediction_alert:
            a_type = prediction_alert.get("type")
            if a_type in ["PREDICTION_SUCCESS", "PREEMPTIVE_INTERCEPT"]:
                self.comparison_sim.sage_metrics.record_prediction_hit(is_preemptive=True)
            elif a_type == "PREDICTION_MISS":
                self.comparison_sim.sage_metrics.record_prediction_miss()
            elif a_type == "UNKNOWN_DISCOVERED":
                self.comparison_sim.sage_metrics.record_unknown_threat_discovered()

        step_record = {
            "time": t,
            "sage_band": sage_band,
            "seq_band": seq_band,
            "sage_obs": sage_obs,
            "seq_obs": seq_obs,
            "sage_decision": sage_decision,
            "prediction_alert": prediction_alert
        }
        self.comparison_sim.history.append(step_record)
        return step_record
