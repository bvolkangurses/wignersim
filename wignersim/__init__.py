"""
Wigner quasiprobability distribution simulator for quantum states.

This package provides tools to simulate and visualize the Wigner function
for various quantum states including coherent states, squeezed states,
cat states, and more.
"""

from .wigner import wigner_function
from .states import (
    QuantumState,
    FockState,
    CoherentState,
    SqueezedState,
    CatState,
    ThermalState,
    DisplacedFockState,
    SuperpositionState,
)
from .visualization import plot_wigner, plot_wigner_3d

__version__ = "0.1.0"

__all__ = [
    "wigner_function",
    "QuantumState",
    "FockState",
    "CoherentState",
    "SqueezedState",
    "CatState",
    "ThermalState",
    "DisplacedFockState",
    "SuperpositionState",
    "plot_wigner",
    "plot_wigner_3d",
]
