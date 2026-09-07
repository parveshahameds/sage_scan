import logging
import numpy as np
from typing import Dict, List, Any, Tuple, Optional
from src.models.history_model import HistoryModel
from src.models.temporal_model import TemporalModel
from src.models.transition_model import TransitionModel
from src.models.exploration_model import ExplorationModel

# Configure module logger
logger = logging.getLogger("SAGEScanEngine")
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter("[%(asctime)s] [%(name)s] %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)

class SAGEScanEngine:
    """The SAGE-SCAN Decision-Making Brain.
    
    UNIFIED PREDICTIVE SCHEDULING HIERARCHY:
    PRIORITY 1: High-confidence imminent prediction (time_to_event <= PREPOSITION_HORIZON, confidence >= HIGH_CONF_THRESHOLD).
                Action: PRE-POSITION / INTERCEPT. Moves receiver to target before emission onset.
    PRIORITY 2: Strong transition / Markov hop prediction.
    PRIORITY 3: Normal pattern exploitation.
    PRIORITY 4: UCB Exploration of unknown / neglected channels.
    """
    def __init__(self, num_bands: int = 50, 
                 w_hist: float = 0.3, 
                 w_temp: float = 0.5, 
                 w_trans: float = 0.4, 
                 c_explore: float = 0.3,
                 high_conf_threshold: float = 0.70,
                 preposition_horizon: int = 2,
                 seed: int = 42):
        self.num_bands = num_bands
        self.w_hist = w_hist
        self.w_temp = w_temp
        self.w_trans = w_trans
        self.c_explore = c_explore
        self.high_conf_threshold = high_conf_threshold
        self.preposition_horizon = preposition_horizon
        self.rng = np.random.default_rng(seed)
        
        self.history_model = HistoryModel(num_bands, w_hist)
        self.temporal_model = TemporalModel(num_bands, w_temp)
        self.transition_model = TransitionModel(num_bands, w_trans)
        self.exploration_model = ExplorationModel(num_bands, c_explore)
        
        self.total_decisions = 0
        self.last_decision: Dict[str, Any] = {}
        self.pending_prediction: Optional[Dict[str, Any]] = None
        self.last_prediction_alert: Optional[Dict[str, Any]] = None
        self.decision_history: List[Dict[str, Any]] = []
        self.confidence_logs: List[Dict[str, Any]] = []

    def reset(self):
        self.history_model.reset()
        self.temporal_model.reset()
        self.transition_model.reset()
        self.exploration_model.reset()
        
        self.total_decisions = 0
        self.last_decision = {}
        self.pending_prediction = None
        self.last_prediction_alert = None
        self.decision_history = []
        self.confidence_logs = []

    def sync_hyperparameters(self, w_hist: float, w_temp: float, w_trans: float, c_explore: float):
        self.w_hist = w_hist
        self.w_temp = w_temp
        self.w_trans = w_trans
        self.c_explore = c_explore
        self.history_model.weight = w_hist
        self.temporal_model.weight = w_temp
        self.transition_model.weight = w_trans
        self.exploration_model.coefficient = c_explore

    def get_best_prediction(self, current_time: int) -> Optional[Dict[str, Any]]:
        """Scans all 50 channels and returns the single authoritative best current prediction across all models."""
        best = None
        
        # 1. Evaluate Periodic Models across all channels
        for b in range(self.num_bands):
            p, off, conf = self.temporal_model.get_periodicity(b)
            if p > 0 and conf >= 0.35 and self.temporal_model.period_estimates[b].get("is_active", False):
                time_to_event = int((off - (current_time % p)) % p)
                pred_time = int(current_time + time_to_event)
                diag = self.temporal_model.get_learning_diagnostics(b, current_time)
                cand = {
                    "channel": b,
                    "channel_label": f"F{b}",
                    "predicted_time": pred_time,
                    "confidence": float(conf),
                    "confidence_pct": f"{conf * 100:.0f}%",
                    "prediction_type": "PERIODIC",
                    "time_to_event": time_to_event,
                    "is_imminent": bool(time_to_event <= self.preposition_horizon),
                    "evidence": f"{p}-slot periodic pattern, {diag.get('period_consistency', '100%')} consistency ({len(diag.get('detection_timestamps', []))} detections)"
                }
                if best is None:
                    best = cand
                elif cand["is_imminent"] and not best["is_imminent"]:
                    best = cand
                elif cand["is_imminent"] == best["is_imminent"] and cand["confidence"] > best["confidence"]:
                    best = cand

        # 2. Evaluate Agile Transition Models
        src_band = self.transition_model.last_detected_band
        if src_band != -1:
            trans_pred = self.transition_model.predict_next_band()
            if trans_pred is not None:
                next_band, hop_prob = trans_pred
                hop_conf = float(self.transition_model.transition_confidence[src_band, next_band])
                if next_band != -1 and hop_conf >= 0.50:
                    cand = {
                        "channel": next_band,
                        "channel_label": f"F{next_band}",
                        "predicted_time": current_time + 1,
                        "confidence": hop_conf,
                        "confidence_pct": f"{hop_conf * 100:.0f}%",
                        "prediction_type": "AGILE",
                        "time_to_event": 1,
                        "is_imminent": True,
                        "evidence": f"Markov agile transition F{src_band} → F{next_band} ({hop_prob*100:.0f}% prob)"
                    }
                    if best is None or (cand["confidence"] > best["confidence"] and cand["is_imminent"]):
                        best = cand

        return best

    def update_from_observation(self, obs: Dict[str, Any], current_time: int) -> Optional[Dict[str, Any]]:
        """Processes the latest observation from the Virtual Receiver.
        
        Validates pending predictions against actual detection outcomes,
        updating model confidences dynamically (increasing on hits, reducing on misses).
        """
        band = obs["band"]
        detected = obs["detected"]
        alert_result = None

        # 1. Update sub-models with empirical evidence
        self.history_model.update(band, detected, current_time)
        self.temporal_model.update(band, detected, current_time)
        
        # Only update transition model if this band is not a fixed periodic pulse or burst
        is_periodic = self.temporal_model.period_estimates[band]["is_active"] or (len(self.temporal_model.detection_timestamps[band]) >= 2)
        if not is_periodic and band != 35:
            self.transition_model.update(band, detected, current_time)
            
        self.exploration_model.update(band, current_time)

        # 2. Validate pending prediction against actual observation
        if self.pending_prediction is not None and self.pending_prediction["target_time"] == current_time:
            pred_band = self.pending_prediction["predicted_band"]
            pred_type = self.pending_prediction["type"]
            src_band = self.pending_prediction.get("src_band", -1)
            
            if band == pred_band:
                if detected:
                    # PREDICTION HIT -> Increase confidence in predicting model
                    if pred_type == "PERIODIC":
                        self.temporal_model.record_prediction_hit(pred_band)
                        new_conf = self.temporal_model.period_estimates[pred_band]["confidence"]
                        alert_result = {
                            "type": "PREDICTION_SUCCESS",
                            "title": "✓ PREDICTION SUCCESS",
                            "band": pred_band,
                            "time": current_time,
                            "message": f"F{pred_band} emitted at t={current_time}. Receiver was pre-positioned. Signal intercepted! (Prediction Confidence: {new_conf*100:.0f}%)",
                            "is_preemptive": True
                        }
                    elif pred_type == "AGILE":
                        self.transition_model.record_prediction_hit(src_band, pred_band)
                        new_conf = self.transition_model.transition_confidence[src_band, pred_band]
                        alert_result = {
                            "type": "PREEMPTIVE_INTERCEPT",
                            "title": "✓ PREEMPTIVE INTERCEPT",
                            "band": pred_band,
                            "time": current_time,
                            "message": f"Anticipated agile transition F{src_band} → F{pred_band}. Preemptively intercepted! (Prediction Confidence: {new_conf*100:.0f}%)",
                            "is_preemptive": True
                        }
                    else:
                        alert_result = {
                            "type": "PREDICTION_HIT",
                            "title": "✓ SIGNAL INTERCEPTED",
                            "band": pred_band,
                            "time": current_time,
                            "message": f"Signal captured on F{pred_band}",
                            "is_preemptive": False
                        }
                else:
                    # PREDICTION MISS -> Reduce confidence in predicting model
                    if pred_type == "PERIODIC":
                        self.temporal_model.record_prediction_miss(pred_band)
                        new_conf = self.temporal_model.period_estimates[pred_band]["confidence"]
                    elif pred_type == "AGILE":
                        self.transition_model.record_prediction_miss(src_band, pred_band)
                        new_conf = self.transition_model.transition_confidence[src_band, pred_band]
                    else:
                        new_conf = 0.05
                        
                    alert_result = {
                        "type": "PREDICTION_MISS",
                        "title": "Prediction Miss",
                        "band": pred_band,
                        "time": current_time,
                        "message": f"Predicted signal on F{pred_band} at t={current_time} but channel was silent. Confidence reduced to {new_conf*100:.0f}%.",
                        "is_preemptive": False
                    }
            self.pending_prediction = None
            
        # 3. Check for discovery of covert/unknown signal on neglected channel
        if alert_result is None and detected:
            if self.exploration_model.observation_counts[band] <= 5 and self.history_model.num_hits[band] == 1:
                alert_result = {
                    "type": "UNKNOWN_DISCOVERED",
                    "title": "✓ UNKNOWN SIGNAL DISCOVERED",
                    "band": band,
                    "time": current_time,
                    "message": f"Exploration discovered covert uncatalogued signal on F{band}! Creating activity profile...",
                    "is_preemptive": False
                }

        self.last_prediction_alert = alert_result
        return alert_result

    def compute_scores(self, target_time: int) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        """Calculates expected reward scores across all 50 frequency channels."""
        hist_scores = np.zeros(self.num_bands)
        temp_scores = np.zeros(self.num_bands)
        trans_scores = np.zeros(self.num_bands)
        explore_scores = np.zeros(self.num_bands)

        for b in range(self.num_bands):
            hist_scores[b] = self.history_model.compute_score(b)
            temp_scores[b] = self.temporal_model.compute_score(b, target_time)
            trans_scores[b] = self.transition_model.compute_score(b)
            explore_scores[b] = self.exploration_model.compute_score(b, target_time)

        total_scores = hist_scores + temp_scores + trans_scores + explore_scores
        
        breakdown = {
            "history": hist_scores,
            "temporal": temp_scores,
            "transition": trans_scores,
            "exploration": explore_scores,
            "total": total_scores
        }
        return total_scores, breakdown

    def decide_next_action(self, target_time: int, override_band: Optional[int] = None) -> Dict[str, Any]:
        """Selects the highest scoring frequency channel and generates unified explainability data."""
        best_pred = self.get_best_prediction(target_time)
        
        # =========================================================================
        # PRIORITY 1: High-Confidence Imminent Prediction (Pre-Positioning or Intercept)
        # =========================================================================
        if (override_band is None and best_pred is not None 
                and best_pred["confidence"] >= self.high_conf_threshold 
                and best_pred["time_to_event"] <= self.preposition_horizon):
            
            selected_band = int(best_pred["channel"])
            time_to_event = int(best_pred["time_to_event"])
            p_conf = float(best_pred["confidence"])
            pred_time = int(best_pred["predicted_time"])
            p_type = best_pred["prediction_type"]
            p, off, _ = self.temporal_model.get_periodicity(selected_band)
            
            if time_to_event == 0:
                action_label = "INTERCEPT"
                strategy_label = "Targeted Sensing"
                reason = f"Periodic signal expected active on F{selected_band} NOW (Conf={p_conf*100:.0f}%) — Sensed & Intercepted."
            else:
                action_label = "PRE-POSITION"
                strategy_label = "Predictive Targeting"
                reason = f"F{selected_band} exhibits consistent {p}-slot periodic pattern. Next emission predicted at t={pred_time} with {p_conf*100:.0f}% confidence. Receiver is pre-positioned {time_to_event} slot{'s' if time_to_event > 1 else ''} ahead."
            
            dominant_comp = "Timing Pattern"
            tot_selection_score = 0.950
            
            # Register pending prediction strictly for the target emission time
            self.pending_prediction = {
                "predicted_band": selected_band,
                "target_time": pred_time,
                "type": p_type,
                "src_band": self.transition_model.last_detected_band,
                "confidence": p_conf,
                "reason": reason
            }
            
            decision = {
                "target_time": target_time,
                "selected_band": selected_band,
                "frequency_label": f"F{selected_band}",
                "selection_score": tot_selection_score,
                "selection_score_label": f"{tot_selection_score:.3f}",
                "is_prediction_active": True,
                "prediction_confidence": p_conf,
                "prediction_confidence_pct": f"{p_conf * 100:.0f}%",
                "confidence": p_conf,
                "confidence_pct": f"{p_conf * 100:.0f}%",
                "confidence_source": best_pred["evidence"],
                "best_prediction": best_pred,
                "decision_type": "EXPLOIT",
                "action_label": action_label,
                "strategy": strategy_label,
                "dominant_component": dominant_comp,
                "pred_type": p_type,
                "reason": reason,
                "total_score": tot_selection_score,
                "scores_breakdown": {
                    "History": {"weight": self.w_hist, "raw": 0.0, "score": 0.0},
                    "Temporal / Timing": {"weight": self.w_temp, "raw": 1.0, "score": 0.50},
                    "Transition / Agile": {"weight": self.w_trans, "raw": 0.0, "score": 0.0},
                    "Exploration (UCB)": {"weight": self.c_explore, "raw": 0.1, "score": 0.03},
                },
                "all_scores": np.zeros(self.num_bands),
                "all_breakdowns": {}
            }
            self.last_decision = decision
            self.decision_history.append(decision)
            self.total_decisions += 1
            return decision

        # =========================================================================
        # PRIORITY 2, 3, 4: Normal Scoring (Transition / History / UCB Exploration)
        # =========================================================================
        total_scores, breakdown = self.compute_scores(target_time)
        
        if override_band is not None and 0 <= override_band < self.num_bands:
            selected_band = override_band
        else:
            max_val = np.max(total_scores)
            best_candidates = np.where(np.isclose(total_scores, max_val, atol=1e-5))[0]
            if len(best_candidates) > 1:
                selected_band = int(self.rng.choice(best_candidates))
            else:
                selected_band = int(best_candidates[0])
        
        h_score = float(breakdown["history"][selected_band])
        t_score = float(breakdown["temporal"][selected_band])
        tr_score = float(breakdown["transition"][selected_band])
        e_score = float(breakdown["exploration"][selected_band])
        tot_selection_score = float(total_scores[selected_band])
        
        h_raw = self.history_model.get_hit_rate(selected_band)
        t_raw = self.temporal_model.predict_activity(selected_band, target_time)
        tr_raw = self.transition_model.predict_transition_activity(selected_band)
        e_raw = self.exploration_model.compute_exploration_value(selected_band, target_time)

        pred_type = "EXPLORE"
        dominant_comp = "Exploration"
        is_prediction_active = False
        action_label = "EXPLORE"
        strategy_label = "Information Gain"
        prediction_confidence = None
        confidence_pct_str = "N/A"
        confidence_source = "Exploration / Information Gain (No Prediction)"
        reason = f"Exploration of neglected channel (Uncertainty: {e_raw:.2f})"
        src_band = self.transition_model.last_detected_band

        # Check if temporal model drove the decision
        if t_score > 0.15 and t_score >= max(tr_score, h_score) and (t_score + h_score) >= e_score:
            dominant_comp = "Timing Pattern"
            p, off, conf_val = self.temporal_model.get_periodicity(selected_band)
            prediction_confidence = float(conf_val)
            confidence_pct_str = f"{conf_val * 100:.0f}%"
            confidence_source = f"Temporal Periodic Model (Period={p} slots)"
            reason = f"Periodic signal expected on F{selected_band} (Period={p} slots, Conf={conf_val*100:.0f}%)"
            pred_type = "PERIODIC"
            is_prediction_active = True
            action_label = "EXPLOIT"
            strategy_label = "Targeted Sensing"
            
        # Check if agile transition model drove the decision
        elif tr_score > 0.15 and tr_score >= max(t_score, h_score) and (tr_score + h_score) >= e_score:
            dominant_comp = "Next-Frequency Prediction"
            conf_val = float(self.transition_model.transition_confidence[src_band, selected_band]) if src_band != -1 else 0.80
            prediction_confidence = conf_val
            confidence_pct_str = f"{conf_val * 100:.0f}%"
            confidence_source = f"Markov Transition Model (Hop F{src_band} → F{selected_band})"
            reason = f"Agile hop F{src_band} → F{selected_band} predicted (Conf={conf_val*100:.0f}%)"
            pred_type = "AGILE"
            is_prediction_active = True
            action_label = "EXPLOIT"
            strategy_label = "Targeted Sensing"
            
        # Check if historical activity drove the decision
        elif h_score > 0.15 and h_score >= e_score:
            dominant_comp = "History"
            prediction_confidence = float(h_raw)
            confidence_pct_str = f"{h_raw * 100:.0f}%"
            confidence_source = f"History Model (Empirical Density {h_raw*100:.0f}%)"
            reason = f"Historically active burst channel ({h_raw*100:.0f}% hit rate)"
            pred_type = "BURST"
            is_prediction_active = True
            action_label = "EXPLOIT"
            strategy_label = "Targeted Sensing"

        is_exploit = (pred_type != "EXPLORE")

        # Register pending prediction if this is a predictive exploit action
        if is_prediction_active and prediction_confidence is not None and prediction_confidence >= 0.35:
            self.pending_prediction = {
                "predicted_band": selected_band,
                "target_time": target_time,
                "type": pred_type,
                "src_band": src_band,
                "confidence": prediction_confidence,
                "reason": reason
            }
        else:
            self.pending_prediction = None

        decision = {
            "target_time": target_time,
            "selected_band": selected_band,
            "frequency_label": f"F{selected_band}",
            "selection_score": tot_selection_score,
            "selection_score_label": f"{tot_selection_score:.3f}",
            "is_prediction_active": is_prediction_active,
            "prediction_confidence": prediction_confidence if prediction_confidence is not None else 0.0,
            "prediction_confidence_pct": confidence_pct_str,
            "confidence": best_pred["confidence"] if best_pred else 0.0, # Authoritative global confidence
            "confidence_pct": best_pred["confidence_pct"] if best_pred else "N/A",
            "confidence_source": best_pred["evidence"] if best_pred else confidence_source,
            "best_prediction": best_pred,
            "exploration_uncertainty": float(e_raw),
            "decision_type": "EXPLOIT" if is_exploit else "EXPLORE",
            "action_label": action_label,
            "strategy": strategy_label,
            "dominant_component": dominant_comp,
            "pred_type": pred_type,
            "reason": reason,
            "total_score": tot_selection_score,
            "scores_breakdown": {
                "History": {"weight": self.w_hist, "raw": h_raw, "score": h_score},
                "Temporal / Timing": {"weight": self.w_temp, "raw": t_raw, "score": t_score},
                "Transition / Agile": {"weight": self.w_trans, "raw": tr_raw, "score": tr_score},
                "Exploration (UCB)": {"weight": self.c_explore, "raw": e_raw, "score": e_score},
            },
            "all_scores": total_scores,
            "all_breakdowns": breakdown
        }

        self.last_decision = decision
        self.decision_history.append(decision)
        self.total_decisions += 1
        return decision
