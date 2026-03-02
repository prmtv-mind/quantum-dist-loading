# README.md

```markdown
# Benchmarking Variational Quantum Circuits for Probability Distribution Loading

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![Qiskit 1.2.4](https://img.shields.io/badge/Qiskit-1.2.4-purple)](https://qiskit.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

This repository provides the complete code and data for the paper **"Benchmarking Variational Quantum Circuits for Probability Distribution Loading"** by Youssef Nassar. It implements a systematic, large‑scale benchmarking framework to evaluate how variational quantum circuits (VQCs) load classical probability distributions into quantum states—a critical subroutine for quantum Monte Carlo, quantum finance, and quantum machine learning.

The experiments encompass **2,880 individual runs** across six distribution shapes, three circuit architectures, four circuit depths, two qubit counts, two classical optimizers, and ten random initialization seeds. All results are fully reproducible using Qiskit 1.x simulators.

---

## Key Findings

- **Distribution shape is the primary determinant of loading difficulty.**  
  Uniform distributions achieve the highest fidelity (`F = 0.8617`), while Binomial distributions prove most challenging (`F = 0.6981`). A strong positive correlation (`r = +0.87`) between target entropy and achieved fidelity indicates that higher‑entropy distributions are closer to the natural output of Ry‑CZ circuits.

- **Under best practices (COBYLA optimizer, depth 4), fidelity improves markedly:**  
  Geometric: `0.9077`, Uniform: `0.9015`, Bimodal: `0.8577`.

- **Architecture differences are modest** (maximum gap `ΔF = 0.031`); the circular topology offers a slight advantage for asymmetric distributions.

- **COBYLA substantially outperforms SPSA** within a 500‑iteration budget  
  (mean fidelity `0.848` vs `0.699`, convergence rate `86%` vs `19%`).

- **Fidelity degrades significantly from `n = 3` to `n = 4` qubits** (drops of `–0.085` to `–0.206`), showing that a fixed‑depth parameter budget does not compensate for a doubled Hilbert space dimension.

- **Bimodal distributions** exhibit the steepest improvement with depth and do not saturate at `D = 4`, indicating a need for deeper circuits.

---

## Repository Structure

```
quantum-dist-loading/
├── src/
│   ├── algorithms.py          # Target generation, circuit construction, cost evaluation
│   ├── optimizers.py          # COBYLA and SPSA wrappers
│   ├── experiment_runner.py   # Full factorial runner
│   └── metrics.py             # Fidelity, KL, JS, TV distance calculations
├── config/                     # Factor level definitions
├── notebooks/
│   ├── 01_visualize_targets.ipynb
│   ├── 02_analyze_results.ipynb
│   └── 03_figures_paper.ipynb
├── results/                    # JSON records per run, aggregated CSVs
├── requirements.txt
├── LICENSE
└── README.md
```

---

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/prntv-mind/quantum-dist-loading.git
   cd quantum-dist-loading
   ```

2. Create a virtual environment (recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate   # Linux/macOS
   venv\Scripts\activate      # Windows
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

   Core dependencies:
   - `qiskit==1.2.4`
   - `qiskit-aer`
   - `numpy>=1.26`
   - `scipy>=1.12`
   - `matplotlib>=3.8`

---

## Usage

### Reproduce the full experiment (2,880 runs)

```bash
python src/experiment_runner.py
```

This command generates a `results/` folder containing one JSON file per experimental run and an aggregated summary CSV.  
**Note:** The full experiment may take several hours on a typical laptop; consider reducing the number of seeds for a quick test.

### Run a single configuration manually

Example Python snippet:

```python
from src.algorithms import generate_target, evaluate_cost
from src.optimizers import optimize_cobyla
import numpy as np

# Configuration
n_qubits = 3
depth = 2
architecture = 'linear'
distribution = 'bimodal'
seed = 42

# Generate target distribution
p_des = generate_target(distribution, n_qubits)

# Initialize parameters
np.random.seed(seed)
theta0 = np.random.uniform(0, 2*np.pi, size=n_qubits*(depth+1))

# Optimize
theta_opt, history = optimize_cobyla(theta0, p_des, n_qubits, depth, architecture)

# Evaluate final fidelity
p_gen = evaluate_cost(theta_opt, p_des, n_qubits, depth, architecture, return_probs=True)
fidelity = np.sum(np.sqrt(p_gen * p_des))**2
print(f"Final fidelity: {fidelity:.4f}")
```

### Explore results with Jupyter notebooks

Launch Jupyter and open `notebooks/02_analyze_results.ipynb` to reproduce the figures and tables from the paper.

---

## Experiment Configuration

The full factorial design comprises:

| Factor          | Levels                                                                           | Count |
|-----------------|----------------------------------------------------------------------------------|-------|
| Distribution    | Binomial, Uniform, Poisson(1.5), Poisson(2.5), Geometric, Bimodal               | 6     |
| Architecture    | Linear, Circular, Alternating                                                   | 3     |
| Depth (`D`)     | 1, 2, 3, 4                                                                       | 4     |
| Qubits (`n`)    | 3, 4                                                                             | 2     |
| Optimizer       | COBYLA, SPSA                                                                     | 2     |
| Random seed     | 0–9                                                                              | 10    |
| **Total runs**  |                                                                                  | 2880  |

Each run uses 1,000 shots and an L2 cost function. Convergence is defined as `L2 < 0.15`.

---

## Outputs

- **Raw JSON per run:** Contains all configuration parameters, final metrics (fidelity, KL divergence, Jensen‑Shannon divergence, total variation distance), and cost history.
- **Aggregated CSV:** Summary statistics grouped by experimental factors.
- **Figures:** Jupyter notebooks generate the plots shown in the paper (depth scaling, architecture heatmap, entropy correlation, etc.).

---


## License

This project is licensed under the MIT License – see the [LICENSE](LICENSE) file for details.

---

## Acknowledgements

This work was conducted at the African Leadership Academy. The author thanks the open‑source Qiskit community for providing an excellent framework for quantum simulation.
```

# About (GitHub repository description)

A systematic benchmarking framework for variational quantum circuits designed to load classical probability distributions into quantum states. The study compares circuit architectures, optimizers, and distribution types across 2,880 simulated experiments.
