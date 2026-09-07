import numpy as np
from typing import List, Dict, Any

def compute_simulation_metrics(
    observations: List[Dict[str, Any]], 
    ground_truth: np.ndarray, 
    emitter_truths: Dict[str, np.ndarray], 
    emitter_map: Dict[str, Any],
    rewards: List[float],
    decisions: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """Computes all 8 core performance metrics from simulation history."""
    
    total_steps = ground_truth.shape[0]
    num_bands = ground_truth.shape[1]

    # Initialize counters
    total_observations = len(observations)
    successful_detections = 0
    false_alarms = 0
    missed_detections = 0
    non_transmission_obs = 0
    active_transmission_obs = 0

    # Group observations for easy calculation
    for obs in observations:
        detected = obs["detected"]
        actual_active = obs["ground_truth_active"]
        is_fa = obs["is_false_alarm"]
        is_miss = obs["is_missed_detection"]

        if actual_active:
            active_transmission_obs += 1
            if detected:
                successful_detections += 1
            if is_miss:
                missed_detections += 1
        else:
            non_transmission_obs += 1
            if detected:
                false_alarms += 1

    # 1. Probability of Detection (Pd)
    # Pd = successful detections / available transmission opportunities in observed bands
    pd_val = (successful_detections / active_transmission_obs) if active_transmission_obs > 0 else 0.0

    # 2. Probability of False Alarm (Pfa)
    # Pfa = false detections / non-transmission observations
    pfa_val = (false_alarms / non_transmission_obs) if non_transmission_obs > 0 else 0.0

    # 3. Average Intercept Rate
    # successful interceptions / unit time
    # Here, a successful interception can be defined as any step where we got a HIT (successful detection)
    avg_intercept_rate = successful_detections / total_steps if total_steps > 0 else 0.0

    # 4. Average Reward
    avg_reward = np.mean(rewards) if rewards else 0.0

    # 5. Percentage of Correct Predictions
    # A prediction is defined by the scheduler's decision to EXPLOIT a band.
    # Prediction is correct if the band was actually active at that step.
    exploit_decisions = [d for d in decisions if d.get("decision") == "EXPLOIT"]
    correct_predictions = 0
    for d in exploit_decisions:
        t = d["time"]
        band = d["selected_band"]
        # Check if ground truth was active at t, band
        if ground_truth[t, band] == 1:
            correct_predictions += 1
    pct_correct_predictions = (correct_predictions / len(exploit_decisions)) if exploit_decisions else 0.0

    # 6 & 7. Average Intercept Time and Average Intercept Time Error
    # Parse activation segments for each emitter
    intercept_delays = []
    
    for name, truth in emitter_truths.items():
        emitter = emitter_map[name]
        band = emitter.get_config().get("band", -1)
        
        # If it is frequency agile or multi-band, it could be in multiple bands.
        # Let's find segments of activity for this emitter.
        # Find all t where emitter was active
        active_indices = np.where(truth == 1)
        if len(active_indices[0]) == 0:
            continue
            
        # Group active time steps into contiguous segments (bursts/pulses)
        # We can look at active (t, band) pairs.
        active_points = sorted(list(zip(active_indices[0], active_indices[1])))
        
        # Segment points by emitter-specific contiguous blocks
        segments = []
        if active_points:
            current_seg = [active_points[0]]
            for pt in active_points[1:]:
                # If same band and consecutive time, or agile and consecutive time
                # Let's group by consecutive time steps for this emitter
                if pt[0] == current_seg[-1][0] + 1:
                    current_seg.append(pt)
                else:
                    segments.append(current_seg)
                    current_seg = [pt]
            segments.append(current_seg)

        for seg in segments:
            seg_start_time = seg[0][0]
            seg_end_time = seg[-1][0]
            # Check if receiver intercepted this segment
            # Receiver intercept occurs if there is an observation at t in observed bands
            # and that observation registered a successful detection of this emitter
            intercepted = False
            intercept_time = -1
            
            for pt in seg:
                t_active = pt[0]
                b_active = pt[1]
                
                # Was this observed and detected?
                for obs in observations:
                    if obs["time"] == t_active and obs["band"] == b_active and obs["detected"]:
                        intercepted = True
                        intercept_time = t_active
                        break
                if intercepted:
                    break
            
            if intercepted:
                delay = intercept_time - seg_start_time
                intercept_delays.append(delay)

    avg_intercept_time = np.mean(intercept_delays) if intercept_delays else 0.0
    # Average Intercept Time Error:
    # absolute difference between actual emitter transmission time (any active slot) and receiver interception time
    # Let's represent this as the average delay error (which is identical to the intercept delay itself, or 0 if instantaneous)
    avg_intercept_time_error = np.mean([abs(d) for d in intercept_delays]) if intercept_delays else 0.0

    # 8. Interception Ratio
    # intercepted transmissions / total transmissions
    # Total active slots in the ground truth
    total_transmissions = np.sum(ground_truth)
    # Total active slots that were detected by the receiver
    detected_slots = set()
    for obs in observations:
        if obs["ground_truth_active"] and obs["detected"]:
            detected_slots.add((obs["time"], obs["band"]))
    
    interception_ratio = (len(detected_slots) / total_transmissions) if total_transmissions > 0 else 0.0

    return {
        "pd": pd_val,
        "pfa": pfa_val,
        "avg_intercept_rate": avg_intercept_rate,
        "avg_reward": avg_reward,
        "pct_correct_predictions": pct_correct_predictions,
        "avg_intercept_time": avg_intercept_time,
        "avg_intercept_time_error": avg_intercept_time_error,
        "interception_ratio": interception_ratio,
        "total_observations": total_observations,
        "successful_detections": successful_detections,
        "false_alarms": false_alarms,
        "missed_detections": missed_detections
    }
