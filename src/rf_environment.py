import numpy as np
import pandas as pd
import os
from typing import List, Dict, Any
from src.emitters import Emitter

def generate_mock_turing_dataset(filepath: str):
    """Generates a realistic mock-up of the Turing Synthetic Radar Dataset (TSRD) schema."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    # We will generate mock pulses over a timeframe of 1,000,000 microseconds (1 second)
    total_duration = 1000000.0
    rng = np.random.default_rng(42)
    pulses = []
    
    # Emitter 0: Periodic Radar (band 3000 MHz, period 20,000 us, 50 Hz pulse rate)
    for toa in np.arange(1000, total_duration, 20000):
        pulses.append({
            "toa_us": float(toa + rng.uniform(-50, 50)),
            "frequency_mhz": 3000.0,
            "pulse_width_us": 2.5,
            "aoa_deg": 120.0,
            "amplitude_db": -12.0,
            "emitter_id": 0
        })
        
    # Emitter 1: Periodic Radar (band 5500 MHz, period 35,000 us, ~28 Hz rate)
    for toa in np.arange(500, total_duration, 35000):
        pulses.append({
            "toa_us": float(toa + rng.uniform(-30, 30)),
            "frequency_mhz": 5500.0,
            "pulse_width_us": 1.5,
            "aoa_deg": 45.0,
            "amplitude_db": -15.0,
            "emitter_id": 1
        })
        
    # Emitter 2: Frequency Agile Radar (hops between 8000, 10000, 12000 MHz every 15,000 us)
    bands = [8000.0, 10000.0, 12000.0]
    current_hop_idx = 0
    for toa in np.arange(2000, total_duration, 15000):
        if int(toa // 15000) % 4 == 0:
            current_hop_idx = (current_hop_idx + 1) % len(bands)
        pulses.append({
            "toa_us": float(toa),
            "frequency_mhz": bands[current_hop_idx],
            "pulse_width_us": 1.0,
            "aoa_deg": 280.0,
            "amplitude_db": -18.0,
            "emitter_id": 2
        })
        
    # Emitter 3: Intermittent threat (15000 MHz, active with 10% prob every 10,000 us)
    for toa in np.arange(3000, total_duration, 10000):
        if rng.random() < 0.12:
            pulses.append({
                "toa_us": float(toa),
                "frequency_mhz": 15000.0,
                "pulse_width_us": 0.5,
                "aoa_deg": 90.0,
                "amplitude_db": -22.0,
                "emitter_id": 3
            })

    # Sort pulses chronologically by Time of Arrival (ToA)
    pulses = sorted(pulses, key=lambda x: x["toa_us"])
    df = pd.DataFrame(pulses)
    df.to_csv(filepath, index=False)


class RFEnvironment:
    """Simulates the RF Spectrum Ground Truth, supporting synthetic generators or PDW trace datasets."""
    def __init__(self, num_bands: int = 50, total_time: int = 1000, emitters: List[Emitter] = None, seed: int = 42, trace_path: str = None):
        self.num_bands = num_bands
        self.total_time = total_time
        self.emitters = emitters if emitters is not None else []
        self.seed = seed
        self.trace_path = trace_path
        self.rng = np.random.default_rng(seed)
        
        self.emitter_map: Dict[str, Emitter] = {e.name: e for e in self.emitters}
        self.emitter_truths: Dict[str, np.ndarray] = {}
        self.ground_truth = np.zeros((self.total_time, self.num_bands), dtype=int)
        
        if self.trace_path:
            self._load_from_trace()
        else:
            self._generate_environment()

    def _generate_environment(self):
        """Generates ground truth using standard synthetic emitters."""
        if not self.emitters:
            return

        for emitter in self.emitters:
            truth = emitter.generate_truth(self.total_time, self.num_bands, self.rng)
            self.emitter_truths[emitter.name] = truth

        truths_list = list(self.emitter_truths.values())
        if truths_list:
            self.ground_truth = np.clip(np.sum(truths_list, axis=0), 0, 1)

    def _load_from_trace(self):
        """Loads and converts a Pulse Descriptor Word (PDW) trace dataset into the discrete simulation grid."""
        # Ensure directory and mock trace are generated if it doesn't exist
        if not os.path.exists(self.trace_path):
            generate_mock_turing_dataset(self.trace_path)
            
        df = pd.read_csv(self.trace_path)
        
        # Check that we have data
        if df.empty:
            return

        # Determine frequency/time bounds from the trace to scale to our simulation grid
        min_toa = df["toa_us"].min()
        max_toa = df["toa_us"].max()
        
        min_freq = df["frequency_mhz"].min()
        max_freq = df["frequency_mhz"].max()

        # Handle edge cases where bounds are identical
        toa_range = max_toa - min_toa if max_toa != min_toa else 1e-6
        freq_range = max_freq - min_freq if max_freq != min_freq else 1e-6

        # Map each emitter_id to a concrete Emitter object and register in self.emitter_map
        unique_emitters = df["emitter_id"].unique()
        self.emitter_map = {}
        self.emitter_truths = {}
        
        for em_id in unique_emitters:
            emitter_name = f"Turing_Emitter_{int(em_id)}"
            # Associate values based on emitter index
            self.emitter_map[emitter_name] = Emitter(
                name=emitter_name, 
                value=1.0 + float(em_id) * 0.2, 
                start_time=0, 
                end_time=self.total_time
            )
            self.emitter_truths[emitter_name] = np.zeros((self.total_time, self.num_bands), dtype=int)

        # Map each PDW pulse to (time_slot, band_index) in the ground truth grid
        for _, row in df.iterrows():
            toa = row["toa_us"]
            freq = row["frequency_mhz"]
            em_id = int(row["emitter_id"])
            emitter_name = f"Turing_Emitter_{em_id}"

            # Calculate discrete time slot and band index
            t_slot = int(((toa - min_toa) / toa_range) * (self.total_time - 1))
            b_idx = int(((freq - min_freq) / freq_range) * (self.num_bands - 1))

            # Bounds clamping
            t_slot = max(0, min(self.total_time - 1, t_slot))
            b_idx = max(0, min(self.num_bands - 1, b_idx))

            # Record activity
            self.emitter_truths[emitter_name][t_slot, b_idx] = 1

        # Sum and clip all truths into the composite ground truth matrix
        truths_list = list(self.emitter_truths.values())
        if truths_list:
            self.ground_truth = np.clip(np.sum(truths_list, axis=0), 0, 1)

    def get_ground_truth(self) -> np.ndarray:
        """Returns the full ground truth binary matrix."""
        return self.ground_truth

    def query(self, t: int, band: int) -> Dict[str, Any]:
        """Queries the ground truth at a specific time and band."""
        if t < 0 or t >= self.total_time or band < 0 or band >= self.num_bands:
            return {
                "active": False,
                "emitters": [],
                "max_value": 0.0
            }

        active_emitters = []
        max_value = 0.0
        for name, truth in self.emitter_truths.items():
            if truth[t, band] == 1:
                active_emitters.append(name)
                max_value = max(max_value, self.emitter_map[name].value)

        return {
            "active": len(active_emitters) > 0,
            "emitters": active_emitters,
            "max_value": max_value
        }
