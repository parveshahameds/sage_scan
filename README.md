# SAGE-SCAN — Predictive RF Threat Detection & Intelligent Spectrum Scanner
> ### *"Don't scan for threats. Predict them."*
> **"Traditional scanning asks *'Where should I scan next?'* — SAGE-SCAN asks *'Where is a signal most likely to appear next?'"***

---

### ⚠️ SAFETY & SIMULATION SCOPE DISCLAIMER
**SAGE-SCAN is a pure SOFTWARE SIMULATION and EDUCATIONAL DEMONSTRATOR.**  
It does not interface with real RF hardware, software-defined radios (SDRs), antennas, transmitters, military EW hardware, jammers, or targeting systems. All spectrum states, threat behaviors, and signals are synthetically simulated in the **Digital Twin** environment for academic and algorithmic research purposes.

---

## 1. Executive Summary & Core Philosophy

In Electronic Warfare (EW) and Electronic Support (ES), receivers must monitor wide frequency bands (e.g. 50 channels: $F_0 \rightarrow F_{49}$) to intercept hostile radars. However, hardware physics limits the receiver's **instantaneous bandwidth to only ONE frequency channel ($W=1$) at a time**.

* **Traditional Scanner:** Sequentially rotates channels ($F_0 \rightarrow F_1 \rightarrow F_2 \rightarrow \dots \rightarrow F_{49} \rightarrow F_0$). It wastes scan time on silent channels and continually misses transient, periodic, and frequency-agile threats.
* **SAGE-SCAN:** Replaces blind sweeping with closed-loop predictive scheduling. By observing scan outcomes online, it models threat behaviors and predicts where and when signals will appear next.

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                       The SAGE-SCAN Sensing Loop                            │
│                                                                             │
│   1. OBSERVE ➔ 2. LEARN ➔ 3. PREDICT ➔ 4. PRE-POSITION ➔ 5. INTERCEPT       │
│                               ▲                                     │       │
│                               └────────── 6. LEARN AGAIN ───────────┘       │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 2. The Digital Twin Architecture

The application is structured into two decoupled, closed-loop layers:
> **"The Digital Twin is the world. SAGE-SCAN is the brain."**

```
┌──────────────────────────────────────────────────────────────┐
│                      DIGITAL TWIN                            │
│  Ground Truth RF Environment: Physical Signals, Time, World  │
│  Simulated Threats: THREAT-01 (Periodic), THREAT-02 (Agile), │
│  THREAT-03 (Intermittent), Covert Emitters                   │
└──────────────────────────────┬───────────────────────────────┘
                               │
                               ▼
┌──────────────────────────────────────────────────────────────┐
│                    VIRTUAL RECEIVER                          │
│  Bandwidth-Limited Sensor (Instantaneous Bandwidth W = 1)    │
│  Detects signals with realistic noise floor, Pd, and Pfa     │
└──────────────────────────────┬───────────────────────────────┘
                               │ Partial Observation (HIT / MISS)
                               ▼
┌──────────────────────────────────────────────────────────────┐
│                       SAGE-SCAN                              │
│  1. History Model (Empirical hit rate tracking)              │
│  2. Temporal Model (Periodicity & Cycle Autocorrelation)     │
│  3. Transition Model (Markov Hopping Transition Table)       │
│  4. Exploration Model (UCB Uncertainty for Cold Channels)    │
└──────────────────────────────┬───────────────────────────────┘
                               │
                               ▼ Next Target Frequency & Dwell
┌──────────────────────────────────────────────────────────────┐
│                       SCHEDULER                              │
│  Pre-positions receiver before signal arrival                │
└──────────────────────────────┬───────────────────────────────┘
                               │
                               └────────────► Back to Receiver
```

---

## 3. The Four Core SAGE-SCAN Models

For every frequency channel $f \in [0, 49]$, SAGE-SCAN computes a composite expected reward score:

$$\text{Score}(f, t) = \underbrace{w_{\text{hist}} \cdot P_{\text{hist}}(f)}_{\text{History Model}} + \underbrace{w_{\text{temp}} \cdot P_{\text{temp}}(f, t)}_{\text{Temporal Model}} + \underbrace{w_{\text{trans}} \cdot P_{\text{trans}}(f)}_{\text{Transition Model}} + \underbrace{c \cdot \sqrt{\frac{\ln T}{N_f + 1}}}_{\text{Exploration Model}}$$

### 1. History Model (`src/models/history_model.py`)
* Tracks total looks $N_f$, successful hits $H_f$, and hit rate $H_f / N_f$.
* Gives high priority to historically active or high-burst channels (such as **THREAT-03** on $F_{35}$).

### 2. Temporal Model (`src/models/temporal_model.py`)
* Analyzes detection timestamps to identify pulse intervals (e.g. $t=7, 14, 21, 28 \rightarrow \text{Period } P = 7$).
* Computes cycle confidence and predicts pulse arrival with pre-positioning lead time before the signal occurs.

### 3. Transition Model (`src/models/transition_model.py`)
* Maintains a Markov transition table $\text{counts}[F_{\text{prev}}, F_{\text{next}}]$ for frequency-agile threats (e.g. $F_{10} \rightarrow F_{20} \rightarrow F_{30} \rightarrow F_{40}$).
* Predicts the next agile hop and triggers **PREEMPTIVE INTERCEPT** before the next pulse.

### 4. Exploration Model (`src/models/exploration_model.py`)
* Quantifies uncertainty on neglected channels using the Upper Confidence Bound (UCB) term.
* Prevents the receiver from getting locked onto known threats and guarantees the discovery of covert/unknown threats (e.g. **UNKNOWN THREAT** on $F_2$).

---

## 4. Simulated Threat Objects

| Threat ID | Type | Frequency / Path | Behavior | Learned Insight |
| :--- | :--- | :--- | :--- | :--- |
| **THREAT-01** | Periodic Radar | $F_{15}$ | Pulses every **7 time slots** | Period $P=7$ discovered, confidence $>90\%$, triggers **PREDICTION SUCCESS** |
| **THREAT-02** | Frequency Agile | $F_{10} \rightarrow F_{20} \rightarrow F_{30} \rightarrow F_{40}$ | Hops every 15 slots in sequence | Markov chain learned, triggers **PREEMPTIVE INTERCEPT** |
| **THREAT-03** | Intermittent Jammer | $F_{35}$ | Unpredictable bursts ($p=0.18$) | Empirical hit-rate profile created |
| **THREAT-UNKNOWN** | Covert Signal | $F_2$ | Injected uncatalogued transmitter | Discovered via UCB exploration, triggers **UNKNOWN SIGNAL DISCOVERED** |

---

## 5. Performance Metrics

All metrics are calculated live from real simulation events:
1. **Detections (Hits):** Total number of physical signal emissions intercepted.
2. **Missed Signals:** Physical emissions where the receiver was tuned elsewhere.
3. **Average Detection Delay (slots):** Time elapsed between threat emission start and receiver capture.
4. **Prediction Successes:** Pre-positioned intercepts on predicted channels.
5. **Unknown Threats Discovered:** Uncatalogued signals discovered strictly via exploration.
6. **System Interception Ratio:** $\frac{\text{Intercepted Transmissions}}{\text{Total Ground-Truth Transmissions}}$

---

## 6. Project Directory Structure

```
sage_scan/
│
├── app.py                         # Streamlit Command Center Dashboard
├── requirements.txt               # Dependencies
├── README.md                      # Comprehensive Documentation
│
├── src/
│   ├── models/                    # The 4 Core ML Sub-Models
│   │   ├── history_model.py       # Hit rate and observation tracking
│   │   ├── temporal_model.py      # Periodicity and cycle prediction
│   │   ├── transition_model.py    # Agility Markov transition tables
│   │   └── exploration_model.py   # UCB cold-band uncertainty engine
│   │
│   ├── core/                      # Simulation & Intelligence Engines
│   │   ├── threat_engine.py       # Simulated Threat Objects (Periodic, Agile, etc.)
│   │   ├── digital_twin.py        # Ground Truth Virtual RF Environment
│   │   ├── virtual_receiver.py    # Bandwidth-limited receiver front-end
│   │   ├── sage_engine.py         # SAGE-SCAN Decision Engine & Explainability
│   │   ├── metrics_engine.py      # Live comparative metrics calculator
│   │   ├── comparison_engine.py   # Sequential Scanner vs SAGE-SCAN Runner
│   │   └── demo_controller.py     # Deterministic 7-Phase Judge Demo
│   │
│   └── components/                # Visual UI Renderers
│       ├── spectrum_view.py       # 50-channel visual matrix & Waterfall heatmap
│       ├── threat_panel.py        # Threat state cards
│       ├── prediction_panel.py    # Prediction command center & alerts
│       ├── decision_explanation.py# "Why Did I Choose This?" breakdown
│       ├── comparison_panel.py    # Head-to-head live benchmark
│       ├── event_timeline.py      # Scrolling live event log
│       └── digital_twin_inspector.py # God Mode comparison
│
└── tests/                         # Pytest Unit Test Suite
    ├── test_models.py             # Unit tests for the 4 models
    ├── test_digital_twin.py       # Unit tests for Digital Twin physics
    ├── test_environment.py        # Synthetic RF test suite
    ├── test_receiver.py           # Virtual receiver unit tests
    └── test_scheduler.py          # Scheduler decision tests
```

---

## 7. How to Run the Application

### 1. Prerequisites
Ensure you have Python 3.8+ installed.

### 2. Set Up Virtual Environment & Dependencies
From the repository root:
```bash
# Create and activate virtual environment
python3 -m venv .venv
source .venv/bin/activate  # macOS / Linux
# .venv\Scripts\activate   # Windows

# Install required dependencies
pip install -r sage_scan/requirements.txt
```

### 3. Run Unit Tests
Run the 15-test verification suite:
```bash
PYTHONPATH=sage_scan pytest sage_scan/tests
```

### 4. Launch the Live Application
```bash
streamlit run sage_scan/app.py
```
Open your browser at `http://localhost:8501`.

---

## 8. How to Run the Judge Demonstration

1. In the application sidebar, navigate to **🎬 Judge Demonstration**.
2. Click **▶ START / RESUME DEMO**.
3. Watch the automated 7-phase sequence:
   * **Phase 1 (Cold Start):** SAGE explores cold channels with no prior data.
   * **Phase 2 (Sampling):** Intercepts pulses on $F_{15}$.
   * **Phase 3 (Discovery):** Learns $P=7$ on $F_{15}$ (Confidence $>90\%$).
   * **Phase 4 (Pre-positioning):** Countdown $3 \rightarrow 2 \rightarrow 1 \rightarrow$ receiver moves to $F_{15} \rightarrow$ **✓ PREDICTION SUCCESS!**
   * **Phase 5 (Agility):** Tracks $F_{10} \rightarrow F_{20} \rightarrow F_{30} \rightarrow F_{40} \rightarrow$ **✓ PREEMPTIVE INTERCEPT!**
   * **Phase 6 (Exploration):** Spawns covert signal on $F_2 \rightarrow$ discovered via UCB $\rightarrow$ **✓ UNKNOWN SIGNAL DISCOVERED!**
   * **Phase 7 (Comparison):** Displays live performance benchmark proving SAGE-SCAN vastly outperforms Sequential Sweep.
