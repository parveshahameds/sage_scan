import numpy as np
from typing import Tuple, List, Dict, Any
from src.features import FeatureModel

class SmartScheduler:
    """Smart Adaptive Electronic Support Receiver Scheduler.
    
    Balances exploration and exploitation using a prioritized UCB algorithm.
    """
    def __init__(self, num_bands: int, bandwidth: int, 
                 w_hist: float = 0.3, w_temp: float = 0.5, w_trans: float = 0.4, 
                 c_explore: float = 0.2, max_dwell: int = 3):
        self.num_bands = num_bands
        self.bandwidth = bandwidth
        self.w_hist = w_hist
        self.w_temp = w_temp
        self.w_trans = w_trans
        self.c_explore = c_explore
        self.max_dwell = max_dwell
        
        self.feature_model = FeatureModel(num_bands)
        self.total_actions = 0
        self.last_action_scores: Dict[str, Any] = {}
        self.all_scores_history: List[Dict[str, np.ndarray]] = []

    def reset(self):
        self.feature_model.reset()
        self.total_actions = 0
        self.last_action_scores = {}
        self.all_scores_history = []

    def select_action(self, current_time: int) -> Tuple[int, int]:
        """Calculates expected reward scores for all bands and selects the optimal band and dwell time."""
        scores_hist = np.zeros(self.num_bands)
        scores_temp = np.zeros(self.num_bands)
        scores_trans = np.zeros(self.num_bands)
        scores_explore = np.zeros(self.num_bands)
        scores_total = np.zeros(self.num_bands)

        T = self.total_actions

        for f in range(self.num_bands):
            # 1. Historical Hit Rate
            p_hist = self.feature_model.get_historical_rate(f)
            scores_hist[f] = self.w_hist * p_hist

            # 2. Temporal Prediction (Periodicity)
            p_temp = self.feature_model.predict_temporal_activity(f, current_time)
            scores_temp[f] = self.w_temp * p_temp

            # 3. Transition Prediction (Frequency Agility)
            p_trans = self.feature_model.predict_transition_activity(f)
            scores_trans[f] = self.w_trans * p_trans

            # 4. Exploration Term (UCB Confidence Bound)
            n_obs = self.feature_model.num_observations[f]
            explore_val = np.sqrt(np.log(max(1, T)) / (n_obs + 1))
            scores_explore[f] = self.c_explore * explore_val

            # Total Score
            scores_total[f] = scores_hist[f] + scores_temp[f] + scores_trans[f] + scores_explore[f]

        # Select band with maximum UCB score
        selected_band = int(np.argmax(scores_total))

        # Determine dwell time adaptively
        # If temporal or transition predictions are high, dwell longer to capture bursts/pulses.
        # Otherwise, dwell for 1 step to scan quickly.
        max_prior = max(
            self.feature_model.predict_temporal_activity(selected_band, current_time),
            self.feature_model.predict_transition_activity(selected_band)
        )
        
        dwell_time = 1
        if max_prior > 0.4:
            dwell_time = self.max_dwell
        
        # Determine if this was an EXPLOIT or EXPLORE decision
        exploit_score = scores_hist[selected_band] + scores_temp[selected_band] + scores_trans[selected_band]
        explore_score = scores_explore[selected_band]
        decision = "EXPLOIT" if exploit_score >= explore_score else "EXPLORE"

        # Save score explanation for UI
        self.last_action_scores = {
            "selected_band": selected_band,
            "dwell_time": dwell_time,
            "decision": decision,
            "historical_rate": self.feature_model.get_historical_rate(selected_band),
            "historical_score": scores_hist[selected_band],
            "temporal_score": scores_temp[selected_band],
            "temporal_prediction": self.feature_model.predict_temporal_activity(selected_band, current_time),
            "transition_score": scores_trans[selected_band],
            "transition_prediction": self.feature_model.predict_transition_activity(selected_band),
            "exploration_score": scores_explore[selected_band],
            "exploration_value": explore_val,
            "total_score": scores_total[selected_band],
            "time_since_last_observation": (current_time - self.feature_model.last_observation_time[selected_band]) if self.feature_model.last_observation_time[selected_band] != -1 else current_time,
            "time_since_last_hit": (current_time - self.feature_model.last_hit_time[selected_band]) if self.feature_model.last_hit_time[selected_band] != -1 else -1,
            "period": self.feature_model.get_periodicity(selected_band)[0],
            "period_confidence": self.feature_model.get_periodicity(selected_band)[2]
        }

        # Keep a history of all scores at each selection step
        self.all_scores_history.append({
            "time": current_time,
            "scores_hist": scores_hist,
            "scores_temp": scores_temp,
            "scores_trans": scores_trans,
            "scores_explore": scores_explore,
            "scores_total": scores_total,
            "selected_band": selected_band,
            "decision": decision
        })

        self.total_actions += 1
        return selected_band, dwell_time

    def update(self, observations: List[Dict[str, Any]], current_time: int) -> None:
        """Passes observations to the Feature Model to update statistics."""
        self.feature_model.update(observations, current_time)
