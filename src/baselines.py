import numpy as np
from typing import Tuple, List, Dict, Any

class SequentialSweepScheduler:
    """A traditional open-loop scanning strategy that sweeps frequencies sequentially."""
    def __init__(self, num_bands: int, bandwidth: int, default_dwell: int = 1):
        self.num_bands = num_bands
        self.bandwidth = bandwidth
        self.default_dwell = default_dwell
        self.current_band = 0

    def reset(self):
        self.current_band = 0

    def select_action(self, current_time: int) -> Tuple[int, int]:
        """Selects the next frequency band in order, using the default dwell time."""
        action = (self.current_band, self.default_dwell)
        # Advance the search window by its bandwidth
        # Stepping by bandwidth models a contiguous tiled scan of the spectrum.
        # Ensure we wrap around the spectrum properly.
        self.current_band = (self.current_band + self.bandwidth) % self.num_bands
        return action

    def update(self, observations: List[Dict[str, Any]], current_time: int) -> None:
        """No-op for baseline open-loop sweep."""
        pass


class RandomSearchScheduler:
    """A baseline strategy that randomly selects frequency bands at each step."""
    def __init__(self, num_bands: int, bandwidth: int, default_dwell: int = 1, seed: int = 42):
        self.num_bands = num_bands
        self.bandwidth = bandwidth
        self.default_dwell = default_dwell
        self.rng = np.random.default_rng(seed)

    def reset(self):
        pass

    def select_action(self, current_time: int) -> Tuple[int, int]:
        """Selects a random frequency band to observe."""
        selected_band = int(self.rng.integers(0, self.num_bands))
        return (selected_band, self.default_dwell)

    def update(self, observations: List[Dict[str, Any]], current_time: int) -> None:
        """No-op for baseline random search."""
        pass
