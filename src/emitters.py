import numpy as np
from typing import List, Dict, Any

class Emitter:
    """Base class for all emitter types."""
    def __init__(self, name: str, value: float = 1.0, start_time: int = 0, end_time: int = 1000):
        self.name = name
        self.value = value
        self.start_time = start_time
        self.end_time = end_time

    def generate_truth(self, total_time: int, num_bands: int, rng: np.random.Generator) -> np.ndarray:
        """Generates a binary activity matrix of shape (total_time, num_bands)."""
        raise NotImplementedError

    def get_config(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "type": self.__class__.__name__,
            "value": self.value,
            "start_time": self.start_time,
            "end_time": self.end_time
        }


class PeriodicEmitter(Emitter):
    """Emits at fixed time intervals in a specific frequency band."""
    def __init__(self, name: str, band: int, period: int, pulse_width: int = 1, offset: int = 0, value: float = 1.0, start_time: int = 0, end_time: int = 1000):
        super().__init__(name, value, start_time, end_time)
        self.band = band
        self.period = period
        self.pulse_width = pulse_width
        self.offset = offset

    def generate_truth(self, total_time: int, num_bands: int, rng: np.random.Generator) -> np.ndarray:
        truth = np.zeros((total_time, num_bands), dtype=int)
        actual_end = min(self.end_time, total_time)
        for t in range(self.start_time, actual_end):
            if (t - self.offset) % self.period < self.pulse_width:
                truth[t, self.band % num_bands] = 1
        return truth

    def get_config(self) -> Dict[str, Any]:
        config = super().get_config()
        config.update({
            "band": self.band,
            "period": self.period,
            "pulse_width": self.pulse_width,
            "offset": self.offset
        })
        return config


class IntermittentEmitter(Emitter):
    """Emits randomly with a given probability in a specific frequency band."""
    def __init__(self, name: str, band: int, prob: float, value: float = 1.0, start_time: int = 0, end_time: int = 1000):
        super().__init__(name, value, start_time, end_time)
        self.band = band
        self.prob = prob

    def generate_truth(self, total_time: int, num_bands: int, rng: np.random.Generator) -> np.ndarray:
        truth = np.zeros((total_time, num_bands), dtype=int)
        actual_end = min(self.end_time, total_time)
        b = self.band % num_bands
        for t in range(self.start_time, actual_end):
            if rng.random() < self.prob:
                truth[t, b] = 1
        return truth

    def get_config(self) -> Dict[str, Any]:
        config = super().get_config()
        config.update({
            "band": self.band,
            "prob": self.prob
        })
        return config


class FrequencyAgileEmitter(Emitter):
    """Hops between multiple frequency bands over time."""
    def __init__(self, name: str, bands: List[int], hop_interval: int, prob: float = 1.0, value: float = 1.5, start_time: int = 0, end_time: int = 1000):
        super().__init__(name, value, start_time, end_time)
        self.bands = bands
        self.hop_interval = hop_interval
        self.prob = prob

    def generate_truth(self, total_time: int, num_bands: int, rng: np.random.Generator) -> np.ndarray:
        truth = np.zeros((total_time, num_bands), dtype=int)
        actual_end = min(self.end_time, total_time)
        for t in range(self.start_time, actual_end):
            band_idx = (t // self.hop_interval) % len(self.bands)
            band = self.bands[band_idx] % num_bands
            if rng.random() < self.prob:
                truth[t, band] = 1
        return truth

    def get_config(self) -> Dict[str, Any]:
        config = super().get_config()
        config.update({
            "bands": self.bands,
            "hop_interval": self.hop_interval,
            "prob": self.prob
        })
        return config


class BurstEmitter(Emitter):
    """Emits in bursts of a given duration with a start probability."""
    def __init__(self, name: str, band: int, burst_duration: int, burst_start_prob: float, value: float = 1.2, start_time: int = 0, end_time: int = 1000):
        super().__init__(name, value, start_time, end_time)
        self.band = band
        self.burst_duration = burst_duration
        self.burst_start_prob = burst_start_prob

    def generate_truth(self, total_time: int, num_bands: int, rng: np.random.Generator) -> np.ndarray:
        truth = np.zeros((total_time, num_bands), dtype=int)
        actual_end = min(self.end_time, total_time)
        in_burst = False
        burst_remaining = 0
        b = self.band % num_bands
        for t in range(self.start_time, actual_end):
            if in_burst:
                truth[t, b] = 1
                burst_remaining -= 1
                if burst_remaining <= 0:
                    in_burst = False
            else:
                if rng.random() < self.burst_start_prob:
                    in_burst = True
                    burst_remaining = self.burst_duration
                    truth[t, b] = 1
                    burst_remaining -= 1
        return truth

    def get_config(self) -> Dict[str, Any]:
        config = super().get_config()
        config.update({
            "band": self.band,
            "burst_duration": self.burst_duration,
            "burst_start_prob": self.burst_start_prob
        })
        return config


class ScanningEmitter(Emitter):
    """Sweeps sequentially through frequency bands."""
    def __init__(self, name: str, start_band: int, dwell_at_band: int, prob: float = 1.0, value: float = 1.0, start_time: int = 0, end_time: int = 1000):
        super().__init__(name, value, start_time, end_time)
        self.start_band = start_band
        self.dwell_at_band = dwell_at_band
        self.prob = prob

    def generate_truth(self, total_time: int, num_bands: int, rng: np.random.Generator) -> np.ndarray:
        truth = np.zeros((total_time, num_bands), dtype=int)
        actual_end = min(self.end_time, total_time)
        for t in range(self.start_time, actual_end):
            band = (self.start_band + (t // self.dwell_at_band)) % num_bands
            if rng.random() < self.prob:
                truth[t, band] = 1
        return truth

    def get_config(self) -> Dict[str, Any]:
        config = super().get_config()
        config.update({
            "start_band": self.start_band,
            "dwell_at_band": self.dwell_at_band,
            "prob": self.prob
        })
        return config


class MultiBandEmitter(Emitter):
    """Emits in multiple frequency bands simultaneously."""
    def __init__(self, name: str, bands: List[int], prob: float, value: float = 1.3, start_time: int = 0, end_time: int = 1000):
        super().__init__(name, value, start_time, end_time)
        self.bands = bands
        self.prob = prob

    def generate_truth(self, total_time: int, num_bands: int, rng: np.random.Generator) -> np.ndarray:
        truth = np.zeros((total_time, num_bands), dtype=int)
        actual_end = min(self.end_time, total_time)
        for t in range(self.start_time, actual_end):
            if rng.random() < self.prob:
                for band in self.bands:
                    truth[t, band % num_bands] = 1
        return truth

    def get_config(self) -> Dict[str, Any]:
        config = super().get_config()
        config.update({
            "bands": self.bands,
            "prob": self.prob
        })
        return config
