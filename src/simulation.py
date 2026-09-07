import numpy as np
from typing import List, Dict, Any, Tuple
from src.rf_environment import RFEnvironment
from src.receiver import Receiver
from src.reward import RewardCalculator
from src.metrics import compute_simulation_metrics

class Simulation:
    """Manages a single execution run of the RF receiver simulation."""
    def __init__(self, env: RFEnvironment, receiver: Receiver, scheduler: Any, reward_calculator: RewardCalculator = None):
        self.env = env
        self.receiver = receiver
        self.scheduler = scheduler
        self.reward_calculator = reward_calculator if reward_calculator is not None else RewardCalculator()
        
        self.reset()

    def reset(self):
        self.current_time = 0
        self.observations: List[Dict[str, Any]] = []
        self.rewards: List[float] = []
        self.reward_breakdowns: List[Dict[str, float]] = []
        self.decisions: List[Dict[str, Any]] = []
        
        # Track consecutive inactive scans for each band
        self.consecutive_inactive_scans = {i: 0 for i in range(self.env.num_bands)}
        self.scheduler.reset()
        if hasattr(self.receiver, 'set_seed'):
            # Re-seed receiver to align runs if needed, using env seed or fixed seed
            self.receiver.set_seed(self.env.seed)

    def step(self) -> bool:
        """Executes one scheduling action (which covers dwell_time steps).
        
        Returns True if simulation is ongoing, False if finished.
        """
        if self.current_time >= self.env.total_time:
            return False

        # 1. Scheduler selects band and dwell time
        selected_band, dwell_time = self.scheduler.select_action(self.current_time)
        
        # Clamp dwell time if it exceeds remaining simulation time
        if self.current_time + dwell_time > self.env.total_time:
            dwell_time = self.env.total_time - self.current_time

        if dwell_time <= 0:
            return False

        # 2. Receiver observes spectrum window
        obs_batch = self.receiver.observe(self.env, self.current_time, selected_band, dwell_time)
        self.observations.extend(obs_batch)

        # Update consecutive inactive scans
        # Group observations by band to see which ones had hits
        band_hits = {b: False for b in range(self.env.num_bands)}
        observed_bands = set()
        for obs in obs_batch:
            b = obs["band"]
            observed_bands.add(b)
            if obs["detected"]:
                band_hits[b] = True

        for b in observed_bands:
            if band_hits[b]:
                self.consecutive_inactive_scans[b] = 0
            else:
                self.consecutive_inactive_scans[b] += 1

        # 3. Calculate Reward
        reward, breakdown = self.reward_calculator.calculate_reward(obs_batch, dwell_time, self.consecutive_inactive_scans)
        self.rewards.append(reward)
        self.reward_breakdowns.append(breakdown)

        # 4. Update Scheduler
        self.scheduler.update(obs_batch, self.current_time)

        # Save decision metadata
        decision_info = {
            "time": self.current_time,
            "selected_band": selected_band,
            "dwell_time": dwell_time,
            "reward": reward,
            "decision": getattr(self.scheduler, 'last_action_scores', {}).get('decision', 'EXPLOIT')
        }
        self.decisions.append(decision_info)

        # 5. Advance simulation clock
        self.current_time += dwell_time
        return True

    def run(self) -> Dict[str, Any]:
        """Runs the simulation to completion."""
        self.reset()
        while self.step():
            pass
        return self.get_results()

    def get_results(self) -> Dict[str, Any]:
        """Computes performance metrics and formats historical data for visualization."""
        metrics = compute_simulation_metrics(
            observations=self.observations,
            ground_truth=self.env.get_ground_truth(),
            emitter_truths=self.env.emitter_truths,
            emitter_map=self.env.emitter_map,
            rewards=self.rewards,
            decisions=self.decisions
        )
        return {
            "metrics": metrics,
            "observations": self.observations,
            "rewards": self.rewards,
            "reward_breakdowns": self.reward_breakdowns,
            "decisions": self.decisions
        }
