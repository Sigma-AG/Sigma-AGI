# Sigma-AGI: Dynamic Neurogenesis on TPUs

![JAX](https://img.shields.io/badge/Built_with-JAX-blue.svg)
![Status](https://img.shields.io/badge/Status-Research_Preview-orange.svg)
![License](https://img.shields.io/badge/License-Apache_2.0-green.svg)

**Sigma-AGI** is an experimental framework for evolving **Liquid Neural Networks** with **Dynamic Topology**.

Instead of training fixed weights on fixed architectures, Sigma-AGI agents "grow" their brains from scratch. Starting with a minimal set of neurons, they evolve complex, sparse structures through interaction with physics-based environments.

## Key Features

- **Dynamic Neurogenesis**: Agents dynamically add/prune neurons and connections during their lifetime
- **JAX-Native**: Fully compiled evolution strategy using `jax.jit`, `jax.vmap`, and masking techniques for static-graph compatibility
- **Energy-Based Survival**: Intelligence emerges from the strict constraint of metabolic efficiency
- **Self-Organized Criticality**: Phase transitions enable "quantum leaps" in capability
- **Scalable**: Designed to scale to 4,096+ agents on TPU pods

## Architecture

```
Sigma-AGI/
├── README.md
├── requirements.txt
├── LICENSE
└── src/
    └── sigma_core.py      # Core implementation
```

## Installation

```bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/Sigma-AGI.git
cd Sigma-AGI

# Install dependencies
pip install -r requirements.txt
```

## Usage

### Quick Test
```bash
python src/sigma_core.py
```

### Full Evolution
```bash
python -c "from src.sigma_core import evolve; evolve(n_generations=1000, population_size=256)"
```

### TPU Execution
```bash
# On Google Cloud TPU
python src/sigma_core.py --population 4096
```

## Why TPUs?

Our "Dynamic Masking" architecture is uniquely suited for TPUs:

1. **Memory-Bound Sparse Operations**: To enable dynamic growth within JAX's static graph, we pre-allocate maximal capacity (50k neurons) and use active masks. This requires significant HBM that exceeds consumer GPUs.

2. **Population Scaling**: We aim to run 4,096+ parallel agents to study phase transitions in evolutionary topology. TPU's 128GB+ memory is critical for holding massive population states.

3. **JAX Optimization**: The entire evolutionary loop (inference, selection, mutation) is compiled via `jax.jit` and `jax.lax.scan` to run entirely on the accelerator.

## Research Goal

To demonstrate that **structural evolution** (finding the right graph) is more sample-efficient than weight optimization for embodied AGI tasks.

### Core Hypothesis

Following the "Bitter Lesson" (Sutton, 2019):
- Let evolution discover topology
- Let evolution discover neuron types
- Let evolution discover connection patterns
- No human priors about "good" architectures

## Key Concepts

### Dynamic Neurogenesis
```python
# Start minimal
INITIAL_NEURONS = 1000
INITIAL_CONNECTIONS = 500

# Grow on demand (up to limits)
MAX_NEURONS = 50000
MAX_CONNECTIONS = 500000
```

### Resource Penalty
```python
# Complexity has cost (Bitter Lesson)
fitness = survival_reward - alpha * n_neurons - beta * n_connections
```

### Self-Organized Criticality
```python
# Target branching ratio = 1.0 (edge of chaos)
# Enables phase transitions in capability
```

## JAX Features Used

- `jax.jit` - XLA compilation for acceleration
- `jax.vmap` - Vectorized population evaluation
- `jax.lax.scan` - Efficient episode simulation
- Dynamic masking - Static-graph compatible growth

## License

Apache 2.0 - See [LICENSE](LICENSE)

## Citation

```bibtex
@software{sigma_agi_2026,
  title = {Sigma-AGI: Dynamic Neurogenesis on TPUs},
  author = {Sigma-AGI Project},
  year = {2026},
  url = {https://github.com/YOUR_USERNAME/Sigma-AGI}
}
```

## Acknowledgments

- JAX team at Google for the amazing framework
- Bitter Lesson (Sutton, 2019) for the guiding philosophy
