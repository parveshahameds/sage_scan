import numpy as np
from typing import List, Dict, Any, Callable
from src.rf_environment import RFEnvironment
from src.receiver import Receiver
from src.scheduler import SmartScheduler
from src.reward import RewardCalculator
from src.simulation import Simulation

def run_training(
    env_factory: Callable[[int], RFEnvironment],
    receiver: Receiver,
    scheduler: Any,
    reward_calculator: RewardCalculator,
    num_episodes: int = 10,
    base_seed: int = 42
) -> List[Dict[str, Any]]:
    """Runs a series of simulation episodes where the scheduler accumulates knowledge.
    
    Returns a list of result dicts, one for each episode, showing learning progress.
    """
    # Reset scheduler to initial state before training starts
    scheduler.reset()
    
    episode_results = []

    for ep in range(num_episodes):
        # Generate environment for this episode with a varying but deterministic seed
        ep_seed = base_seed + ep
        env = env_factory(ep_seed)
        
        # Instantiate simulation
        sim = Simulation(env, receiver, scheduler, reward_calculator)
        
        # Reset simulation state (this clears clock, but we manually preserve scheduler's feature model)
        sim.current_time = 0
        sim.observations = []
        sim.rewards = []
        sim.reward_breakdowns = []
        sim.decisions = []
        sim.consecutive_inactive_scans = {i: 0 for i in range(env.num_bands)}
        
        # Note: We do NOT call scheduler.reset() here to allow knowledge transfer!
        # Run simulation to completion
        while sim.step():
            pass
            
        results = sim.get_results()
        
        episode_results.append({
            "episode": ep + 1,
            "seed": ep_seed,
            "metrics": results["metrics"]
        })

    return episode_results
