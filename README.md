# Tail-Aware Phase 1 — source and reproducibility

Python 3.11+ dependencies: numpy, pandas, scipy, matplotlib. Install with `pip install numpy pandas scipy matplotlib`.

Run in the same directory containing the Python scripts:

1. `python adb_benchmark_experiment.py --help`
2. `python tailaware_phase1_v2_experiments.py --help`
3. `python tailaware_v2_quantum_oracle.py --help`
4. `python tailaware_v2_cost_audit.py`

NOTE: Scripts are the originals used for the experiments, and write outputs to `/mnt/data`. On a local computer, edit `OUT=Path('/mnt/data')` in each script to your desired output directory. Reproducing 200-replicate experiments may take substantial CPU time.

The experiments use the ADB synthetic benchmark. Tail-aware tensor-guided importance sampling uses small-state conditional lookahead and does NOT establish scalable or hardware quantum advantage. Quantum demonstrations are noiseless simulation and idealized estimation, not QPU executions.

Primary scripts: `adb_benchmark_experiment.py` (ADB model), `tailaware_phase1_v2_experiments.py` (200x classical baselines and tensor-guided IS), `tailaware_v2_quantum_oracle.py` (8-qubit coherent-oracle simulation and QAE), `tailaware_v2_cost_audit.py` (resource proxy). Supporting sampler: `adb_tailaware_new_pilot.py`.
