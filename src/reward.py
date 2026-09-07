from typing import List, Dict, Any, Tuple

class RewardCalculator:
    """Calculates rewards for receiver actions to guide scheduling updates."""
    def __init__(self, 
                 hit_reward: float = 10.0, 
                 value_weight: float = 5.0, 
                 miss_penalty: float = -1.0, 
                 false_alarm_penalty: float = -3.0,
                 dwell_cost: float = -0.5, 
                 repeat_inactive_penalty: float = -1.0):
        self.hit_reward = hit_reward
        self.value_weight = value_weight
        self.miss_penalty = miss_penalty
        self.false_alarm_penalty = false_alarm_penalty
        self.dwell_cost = dwell_cost
        self.repeat_inactive_penalty = repeat_inactive_penalty

    def calculate_reward(self, observations: List[Dict[str, Any]], dwell_time: int, consecutive_inactive_scans: Dict[int, int]) -> Tuple[float, Dict[str, float]]:
        """Calculates total reward for a list of observations.
        
        Returns the total reward and a breakdown of components.
        """
        total_reward = 0.0
        breakdown = {
            "hit_reward": 0.0,
            "miss_penalty": 0.0,
            "false_alarm_penalty": 0.0,
            "idle_cost": 0.0,
            "dwell_cost": 0.0,
            "repeat_inactive_penalty": 0.0
        }

        # Apply base dwell time cost
        dwell_penalty = dwell_time * self.dwell_cost
        total_reward += dwell_penalty
        breakdown["dwell_cost"] = dwell_penalty

        unique_bands = set(obs["band"] for obs in observations)

        for obs in observations:
            detected = obs["detected"]
            actual_active = obs["ground_truth_active"]
            max_val = obs["max_value"]
            band = obs["band"]

            if detected:
                if actual_active:
                    # True Hit: Base reward + bonus for emitter value
                    r = self.hit_reward + (self.value_weight * max_val)
                    total_reward += r
                    breakdown["hit_reward"] += r
                else:
                    # False Alarm: Significant penalty
                    r = self.false_alarm_penalty
                    total_reward += r
                    breakdown["false_alarm_penalty"] += r
            else:
                if actual_active:
                    # Missed Detection: Penalty for missing a target
                    r = self.miss_penalty
                    total_reward += r
                    breakdown["miss_penalty"] += r
                else:
                    # Normal empty spectrum scan: small idle cost
                    r = -0.1
                    total_reward += r
                    breakdown["idle_cost"] += r

        # Repeated inactive scan penalty
        # If the band has been scanned multiple times consecutively with no detections, penalize
        for band in unique_bands:
            inactive_streak = consecutive_inactive_scans.get(band, 0)
            if inactive_streak > 2:
                # Apply penalty for staying on a dead band
                penalty = self.repeat_inactive_penalty * (inactive_streak - 2)
                total_reward += penalty
                breakdown["repeat_inactive_penalty"] += penalty

        return total_reward, breakdown
