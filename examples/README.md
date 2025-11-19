# WignerSim Examples

This directory contains example scripts demonstrating various capabilities of the WignerSim library.

## Quick Start

```bash
# Run any example script
python basic_states.py
python cat_states.py
python custom_states.py
python demo.py
```

## Examples Overview

### 1. basic_states.py
Demonstrates fundamental quantum states and their Wigner functions:
- Fock states (vacuum, |1⟩, |5⟩)
- Coherent states at various positions
- Squeezed vacuum states
- Thermal states

**Output:** 4 PNG files showing different state categories

### 2. cat_states.py
Focuses on Schrödinger cat states:
- Even cat states: (|α⟩ + |-α⟩) / √2
- Odd cat states: (|α⟩ - |-α⟩) / √2
- Yurke cat states: (|α⟩ + i|-α⟩) / √2
- Cat states with different amplitudes

**Key Feature:** Shows quantum interference fringes and negative Wigner values

**Output:** 2 PNG files demonstrating cat state types

### 3. custom_states.py
Advanced state engineering:
- Displaced Fock states D(α)|n⟩
- Superpositions of multiple coherent states
- Rotated squeezed states
- High-amplitude states
- 3D visualization

**Output:** 3 PNG files including a 3D surface plot

### 4. demo.py
Interactive demonstration with detailed explanations:
- Single cat state with annotations
- Comparison of 5 different quantum states
- Educational commentary on quantum features

**Output:** 2 PNG files with comprehensive comparison

## Creating Custom States

### Example: Your Own Superposition

```python
import numpy as np
from wignersim import CoherentState, wigner_function, plot_wigner

# Create a superposition of 4 coherent states in a diamond pattern
n_max = 40
rho = np.zeros((n_max, n_max), dtype=complex)

alphas = [2, 2j, -2, -2j]  # Four positions
for alpha in alphas:
    state = CoherentState(alpha, n_max=n_max)
    rho += state.density_matrix() / len(alphas)

# Calculate and plot Wigner function
xvec = np.linspace(-4, 4, 100)
yvec = np.linspace(-4, 4, 100)
W = wigner_function(rho, xvec, yvec)
plot_wigner(W, xvec, yvec, title='Diamond Pattern State')
```

### Example: High-Dimensional Entanglement (Mixed State)

```python
from wignersim import FockState, ThermalState

# Create a mixed state combining Fock and thermal components
fock = FockState(5, n_max=40)
thermal = ThermalState(n_mean=2, n_max=40)

# Mix them
rho_mixed = 0.5 * fock.density_matrix() + 0.5 * thermal.density_matrix()

# Visualize
W = wigner_function(rho_mixed, xvec, yvec)
plot_wigner(W, xvec, yvec, title='Mixed Fock-Thermal State')
```

### Example: Squeezed Cat State

```python
from wignersim import SqueezedState, CatState

# Create a cat state
cat = CatState(2.0, kind='even', n_max=40)

# Or manually build custom states
# This is more advanced - combine density matrices with proper normalization
```

## Understanding the Output

### Color Coding
- **Blue (positive)**: Classical-like probability distribution
- **Red (negative)**: Non-classical quantum behavior
- **White (near zero)**: Transition regions

### Key Observations

**Vacuum State:**
- Gaussian centered at origin
- Always positive (most classical single Fock state)

**Coherent States:**
- Gaussian displaced from origin
- Always positive (closest to classical light)
- Width = 1/√2 (minimum uncertainty)

**Fock States (n > 0):**
- Negative regions at center
- Non-classical features increase with n

**Squeezed States:**
- Elliptical Gaussian shape
- Demonstrates uncertainty redistribution
- Below-standard-quantum-limit noise in one quadrature

**Cat States:**
- Multiple peaks (at coherent state positions)
- Interference fringes between peaks
- Highly negative values (very non-classical)
- Signature of macroscopic quantum superposition

## Advanced Usage

### Animating State Evolution

```python
from wignersim import CoherentState, wigner_function
from wignersim.visualization import animate_wigner_evolution
import numpy as np

# Create sequence of coherent states moving in a circle
times = np.linspace(0, 2*np.pi, 50)
W_sequence = []

for t in times:
    alpha = 2 * np.exp(1j * t)
    state = CoherentState(alpha, n_max=30)
    W = wigner_function(state.density_matrix(), xvec, yvec)
    W_sequence.append(W)

# Create animation
anim = animate_wigner_evolution(W_sequence, xvec, yvec, times,
                                save_path='rotation.gif')
```

### 3D Visualization

```python
from wignersim import plot_wigner_3d

# Any state
W = wigner_function(state.density_matrix(), xvec, yvec)
fig, ax = plot_wigner_3d(W, xvec, yvec, 
                         title='3D Wigner Function',
                         elev=25, azim=60)
```

## Tips

1. **Grid Size**: Use finer grids (more points) for better accuracy
   - Trade-off with computation time
   - Typical: 80-150 points per dimension

2. **Grid Range**: Adjust based on state properties
   - Coherent states: center ± 3-4 units
   - Cat states: include both coherent components ± margins
   - High amplitude: larger range needed

3. **Truncation (n_max)**: Higher for states with large photon numbers
   - Coherent |α⟩: n_max ~ 10 + 3*|α|²
   - Cat states: n_max ~ 30-50
   - Squeezed: n_max ~ 30-50

4. **Negative Values**: Indicate non-classicality
   - More negative = more quantum
   - Vacuum has no negative values
   - Fock n=1 has slight negativity
   - Cat states have strong negativity

## Further Reading

- Wigner function: Phase-space quantum mechanics
- Cat states: Schrödinger's cat thought experiment
- Squeezed light: Quantum noise reduction
- Negative quasi-probabilities: Quantum weirdness indicator

## Troubleshooting

**Low trace values?**
- Increase n_max (Hilbert space truncation issue)

**Blurry features?**
- Use finer grid (more points in xvec, yvec)

**Slow computation?**
- Reduce n_max if possible
- Use coarser grid
- Consider wigner_clenshaw for large systems

**Memory errors?**
- Reduce n_max
- Process in smaller batches
