# WignerSim

A Python library for designing, simulating and visualizing Wigner quasiprobability distributions of quantum states.

## Overview

WignerSim enables users to engineer and analyze quantum states with arbitrary Wigner quasiprobability distributions, including:

- **Fock states** (number states)
- **Coherent states** (Gaussian states)
- **Squeezed states** (reduced noise in one quadrature)
- **Cat states** (superpositions of coherent states)
- **Thermal states** (mixed states)
- **Displaced Fock states** and custom superpositions

The Wigner function provides a phase-space representation of quantum states and can exhibit negative values, which are signatures of non-classical quantum behavior.

## Installation

Install the required dependencies:

```bash
pip install -r requirements.txt
```

Or install the package in development mode:

```bash
pip install -e .
```

## Quick Start

```python
import numpy as np
from wignersim import CoherentState, CatState, wigner_function, plot_wigner

# Create a cat state (superposition of coherent states)
cat = CatState(alpha=2.5, kind='even', n_max=40)

# Define phase space grid
xvec = np.linspace(-5, 5, 100)
yvec = np.linspace(-5, 5, 100)

# Calculate Wigner function
W = wigner_function(cat.density_matrix(), xvec, yvec)

# Visualize
plot_wigner(W, xvec, yvec, title='Cat State Wigner Function')
```

## Features

### Supported Quantum States

- **FockState(n)**: Number state |n⟩
- **CoherentState(alpha)**: Coherent state |α⟩
- **SqueezedState(r, phi)**: Squeezed vacuum state
- **CatState(alpha, kind)**: Schrödinger cat states ('even', 'odd', 'yurke')
- **ThermalState(n_mean)**: Thermal mixed state
- **DisplacedFockState(n, alpha)**: Displaced number state D(α)|n⟩

### Visualization Tools

- `plot_wigner()`: 2D contour plots
- `plot_wigner_3d()`: 3D surface plots
- `plot_comparison()`: Side-by-side comparisons
- `animate_wigner_evolution()`: Time evolution animations

## Examples

Run the example scripts to see various quantum states:

```bash
# Basic quantum states
cd examples
python basic_states.py

# Cat states with quantum interference
python cat_states.py

# Custom engineered states
python custom_states.py
```

## Testing

Run the test suite:

```bash
python -m pytest tests/
```

Or using unittest:

```bash
python -m unittest discover tests/
```

## Key Concepts

### Wigner Function

The Wigner function W(x,y) is a quasi-probability distribution in phase space that provides a complete description of a quantum state. Unlike classical probability distributions, it can take negative values, which indicate non-classical quantum behavior.

### Cat States

Schrödinger cat states are superpositions of macroscopically distinct coherent states:
- Even cat: (|α⟩ + |-α⟩) / √2
- Odd cat: (|α⟩ - |-α⟩) / √2

These states exhibit quantum interference fringes in their Wigner functions.

### Squeezed States

Squeezed states have reduced quantum noise in one quadrature below the standard quantum limit, at the expense of increased noise in the conjugate quadrature.

## Requirements

- Python >= 3.7
- NumPy >= 1.21.0
- SciPy >= 1.7.0
- Matplotlib >= 3.4.0

## License

MIT License - see LICENSE file for details.

## Author

Volkan Gurses

## Citation

If you use this software in your research, please cite it appropriately.
