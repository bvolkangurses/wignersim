"""
Interactive demonstration of WignerSim capabilities.

This script provides a simple demo of creating and visualizing
quantum states with their Wigner functions.
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
import os

# Add parent directory to path to import wignersim
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from wignersim import (
    FockState, CoherentState, SqueezedState, CatState,
    wigner_function, plot_wigner
)


def demo_cat_state():
    """Demonstrate a cat state with quantum interference."""
    print("=" * 60)
    print("WIGNERSIM DEMONSTRATION: Schrödinger Cat State")
    print("=" * 60)
    print()
    print("Creating an 'even' cat state: |ψ⟩ = (|α⟩ + |-α⟩) / √2")
    print("where α = 2.5")
    print()
    print("Cat states are superpositions of macroscopically distinct")
    print("coherent states, showing quantum interference at a")
    print("macroscopic scale.")
    print()
    
    # Create cat state
    cat = CatState(alpha=2.5, kind='even', n_max=40)
    
    # Define phase space grid
    xvec = np.linspace(-5, 5, 100)
    yvec = np.linspace(-5, 5, 100)
    
    # Calculate Wigner function
    print("Calculating Wigner function...")
    W = wigner_function(cat.density_matrix(), xvec, yvec)
    
    # Create visualization
    print("Creating visualization...")
    fig, ax = plot_wigner(W, xvec, yvec, 
                         title='Schrödinger Cat State (α=2.5)')
    
    # Add annotations
    ax.plot([2.5], [0], 'wo', markersize=10, markeredgecolor='k', 
            markeredgewidth=2, label='|2.5⟩')
    ax.plot([-2.5], [0], 'wo', markersize=10, markeredgecolor='k',
            markeredgewidth=2, label='|-2.5⟩')
    ax.legend(loc='upper right')
    
    plt.savefig('demo_cat_state.png', dpi=150, bbox_inches='tight')
    print()
    print("Saved: demo_cat_state.png")
    print()
    print("Key observations:")
    print("- Two Gaussian peaks at α=±2.5 (the coherent state locations)")
    print("- Interference fringes in the center")
    print("- Blue regions show NEGATIVE Wigner function values")
    print("  (signature of non-classical quantum behavior)")
    print()
    print(f"State purity: {cat.purity():.6f}")
    print(f"Trace: {cat.trace():.6f}")
    print()


def demo_comparison():
    """Compare different quantum states."""
    print("=" * 60)
    print("COMPARING QUANTUM STATES")
    print("=" * 60)
    print()
    
    # Create different states
    states = {
        'vacuum': (FockState(0, n_max=20), 'Vacuum |0⟩'),
        'fock_3': (FockState(3, n_max=20), 'Fock |3⟩'),
        'coherent': (CoherentState(2, n_max=25), 'Coherent |2⟩'),
        'squeezed': (SqueezedState(1.0, n_max=30), 'Squeezed (r=1)'),
        'cat': (CatState(2, kind='even', n_max=30), 'Cat |2⟩+|-2⟩'),
    }
    
    # Define phase space grid
    xvec = np.linspace(-4, 4, 80)
    yvec = np.linspace(-4, 4, 80)
    
    # Calculate Wigner functions
    print("Calculating Wigner functions for 5 different states...")
    W_dict = {}
    for key, (state, title) in states.items():
        W = wigner_function(state.density_matrix(), xvec, yvec)
        W_dict[key] = (W, title)
    
    # Create comparison plot
    print("Creating comparison plot...")
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    # Find global color scale
    all_W = [W for W, _ in W_dict.values()]
    vmax = max(np.max(np.abs(W)) for W in all_W)
    vmin = -vmax
    
    for idx, (key, (W, title)) in enumerate(W_dict.items()):
        ax = axes[idx]
        cf = ax.contourf(xvec, yvec, W, levels=20, cmap='RdBu',
                        vmin=vmin, vmax=vmax)
        ax.contour(xvec, yvec, W, levels=20, colors='k',
                  alpha=0.3, linewidths=0.5)
        ax.set_xlabel('Re(α)')
        ax.set_ylabel('Im(α)')
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.set_aspect('equal')
    
    # Hide last subplot
    axes[-1].axis('off')
    
    # Add colorbar
    fig.colorbar(cf, ax=axes, label='W(x,y)', 
                orientation='horizontal', fraction=0.05, pad=0.1)
    
    plt.tight_layout()
    plt.savefig('demo_comparison.png', dpi=150, bbox_inches='tight')
    print()
    print("Saved: demo_comparison.png")
    print()
    print("Observations:")
    print("- Vacuum: Positive everywhere (classical)")
    print("- Fock |3⟩: Negative regions (non-classical)")
    print("- Coherent: Gaussian, positive (most classical)")
    print("- Squeezed: Elliptical, shows uncertainty redistribution")
    print("- Cat: Interference pattern (highly non-classical)")
    print()


def main():
    """Run all demonstrations."""
    print()
    print("╔════════════════════════════════════════════════════════╗")
    print("║          WIGNERSIM - Quantum State Simulator          ║")
    print("║   Wigner Quasiprobability Distribution Visualization  ║")
    print("╚════════════════════════════════════════════════════════╝")
    print()
    
    # Run demonstrations
    demo_cat_state()
    demo_comparison()
    
    print("=" * 60)
    print("DEMONSTRATION COMPLETE")
    print("=" * 60)
    print()
    print("WignerSim allows you to engineer quantum states with")
    print("arbitrary Wigner distributions, including:")
    print()
    print("  • Cat states (quantum superpositions)")
    print("  • Squeezed states (noise reduction)")
    print("  • Fock states (number states)")
    print("  • Coherent states (classical-like)")
    print("  • Thermal states (mixed states)")
    print("  • Custom superpositions")
    print()
    print("The Wigner function provides a complete phase-space")
    print("representation of quantum states. Negative values are")
    print("a signature of non-classical quantum behavior!")
    print()


if __name__ == "__main__":
    main()
