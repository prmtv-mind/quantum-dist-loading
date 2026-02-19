# Quantum Distribution Loading Benchmarking

Systematic benchmarking framework for variational quantum circuits loading probability distributions.

## Requirements

- Python 3.12+
- Qiskit 1.2.4+
- Modern scientific Python stack

## Setup

````bash
# Create environment
conda create -n qdist-bench python=3.12 -y
conda activate qdist-bench

# Install dependencies
pip install -r requirements.txt

# Verify
python -c "import qiskit; print(f'Qiskit {qiskit.__version__}')"


# Project Structure

```bash
src/: Core implementation
    circuits/: Quantum circuit implementations
    distributions/: Target distribution generators
    optimization/: Optimizer implementations
    metrics/: Evaluation metrics
    utils/: Helper functions

experiments/: Experiment configurations and runners

analysis/: Analysis scripts and visualization

notebooks/: Jupyter notebooks for exploration

results/: Experimental results storage
````

# Git workflow:

```bash
# Initial commit
git add .
git commit -m "Initial project setup"

# Create GitHub repository and push
git remote add origin https://github.com/yourusername/quantum-dist-loading.git
git branch -M main
git push -u origin main

# Development workflow
git checkout -b feature/circuit-implementation
# Make changes
git add .
git commit -m "Implement basic circuit structure"
git push origin feature/circuit-implementation
# Create pull request on GitHub
```
