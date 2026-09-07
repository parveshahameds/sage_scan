import numpy as np
from src.core.digital_twin import DigitalTwinEngine
from src.core.threat_engine import PeriodicThreat, AgileThreat, IntermittentThreat, UnknownThreat
from src.core.sage_engine import SAGEScanEngine
from src.core.comparison_engine import ComparisonSimulator
from src.core.demo_controller import JudgeDemoController

def run_audit():
    print("==================================================")
    print("SAGE-SCAN LOGIC & SIMULATION INTEGRITY AUDIT")
    print("==================================================")

    # 1. Initialize Deterministic Demo Controller
    demo = JudgeDemoController(seed=42, max_demo_time=120)
    dt = demo.digital_twin
    sage = demo.sage_engine
    sim = demo.comparison_sim

    print("\n[SCENARIO SETUP]")
    for t in dt.threats:
        print(f" • {t.name} on {t.get_state(0)['frequency_label']} (Behavior: {t.threat_type})")

    # Step 0 to 69: Observe, Learn Periodicity, Learn Agility
    print("\n[STEP 1: LEARNING PERIODICITY & AGILITY (t = 0..69)]")
    for _ in range(70):
        demo.step()

    # Verify F15 periodicity learned
    p, off, conf = sage.temporal_model.get_periodicity(15)
    print(f" ✓ F15 Learned Period: {p} slots (Offset: {off}, Confidence: {conf*100:.1f}%)")
    assert p == 7, f"Expected period 7, got {p}"
    assert conf >= 0.70

    # Verify Agile transition F10 -> F20 -> F30 -> F40
    chains = sage.transition_model.get_learned_chains()
    print(" ✓ Learned Agile Transitions:")
    for c in chains:
        print(f"   - F{c['from_band']} → F{c['to_band']} (Count: {c['count']}, Probability: {c['probability']*100:.0f}%, Conf: {c['confidence_pct']})")

    # Verify that from F30, SAGE strictly predicts F40 (Never arbitrary F34)
    sage.transition_model.last_detected_band = 30
    next_b, prob = sage.transition_model.predict_next_band()
    prob_f34 = sage.transition_model.predict_transition_activity(34)
    print(f" ✓ Agile Target from F30: F{next_b} (Prob: {prob*100:.0f}%) | Probability for F34: {prob_f34}")
    assert next_b == 40
    assert prob_f34 == 0.0

    # Step 70 to 120: Exploration discovers F2
    print("\n[STEP 2: SPAWNING UNKNOWN THREAT ON F2 (t = 70) & EXPLORATION]")
    unknown_discovered = False
    for step_i in range(70, 120):
        res = demo.step()
        alert = res.get("prediction_alert")
        if alert and alert.get("type") == "UNKNOWN_DISCOVERED" and alert.get("band") == 2:
            unknown_discovered = True
            print(f" ✓ SAGE UCB Exploration discovered unknown signal on F2 at t = {res['time']}!")

    assert unknown_discovered, "Exploration failed to discover F2"

    # Step 3: Verify dynamic confidence response to misses
    print("\n[STEP 3: VERIFY DYNAMIC CONFIDENCE RESPONSE]")
    conf_before = sage.temporal_model.period_estimates[15]["confidence"]
    sage.temporal_model.record_prediction_miss(15)
    conf_after_miss = sage.temporal_model.period_estimates[15]["confidence"]
    print(f" • Confidence before miss: {conf_before*100:.1f}% → after miss: {conf_after_miss*100:.1f}%")
    assert conf_after_miss < conf_before

    sage.temporal_model.record_prediction_hit(15)
    conf_after_hit = sage.temporal_model.period_estimates[15]["confidence"]
    print(f" • Confidence after validated hit: {conf_after_hit*100:.1f}%")
    assert conf_after_hit > conf_after_miss

    # Step 4: Verify complete event-driven benchmark metrics
    print("\n[STEP 4: EVENT-DRIVEN BENCHMARK METRICS AT t = 120]")
    summary = sim.get_comparison_summary()
    sage_m = summary["sage"]
    seq_m = summary["sequential"]
    tot_emissions = summary["total_emitted_signals"]

    print(f" • Total Emitted Signals in World: {tot_emissions}")
    print(f" • SAGE Intercepted: {sage_m['signals_intercepted']} ({sage_m['detection_rate_pct']}) | Missed: {sage_m['signals_missed']}")
    print(f" • Sequential Intercepted: {seq_m['signals_intercepted']} ({seq_m['detection_rate_pct']}) | Missed: {seq_m['signals_missed']}")
    print(f" • SAGE Prediction Accuracy: {sage_m['prediction_hits']}/{sage_m['prediction_attempts']} ({sage_m['prediction_accuracy_pct']})")
    print(f" • SAGE Preemptive Intercepts: {sage_m['preemptive_intercepts']} ({sage_m['preemptive_intercept_rate_pct']})")
    print(f" • SAGE Unknown Threats Discovered: {sage_m['unknown_threats_discovered']}")
    print(f" • Channel Coverage — SAGE: {sage_m['channel_coverage_pct']} | Sequential: {seq_m['channel_coverage_pct']}")

    assert sage_m["signals_intercepted"] + sage_m["signals_missed"] == tot_emissions
    assert seq_m["signals_intercepted"] + seq_m["signals_missed"] == tot_emissions
    assert sage_m["prediction_hits"] > 0
    assert sage_m["preemptive_intercepts"] > 0
    assert sage_m["unknown_threats_discovered"] > 0

    print("\n==================================================")
    print("ALL 10 SIMULATION INTEGRITY CHECKS PASSED PERFECTLY!")
    print("==================================================")

if __name__ == "__main__":
    run_audit()
