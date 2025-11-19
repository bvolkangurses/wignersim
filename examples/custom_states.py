"""
Example: Engineering custom quantum states.

This script demonstrates how to engineer arbitrary quantum states
by combining basic states and computing their Wigner functions.
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
import os

# Add parent directory to path to import wignersim
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from wignersim import (
    CoherentState, FockState, SqueezedState, DisplacedFockState,
    wigner_function
)


def create_custom_superposition(n_max=40):
    """
    Create a custom superposition state manually.
    
    Example: Superposition of multiple coherent states arranged in a circle.
    """
    # Create 4 coherent states arranged in a square
    n_states = 4
    radius = 2.0
    angles = np.linspace(0, 2*np.pi, n_states, endpoint=False)
    
    # Combine density matrices with equal weights
    rho_total = np.zeros((n_max, n_max), dtype=complex)
    
    for angle in angles:
        alpha = radius * np.exp(1j * angle)
        state = CoherentState(alpha, n_max=n_max)
        rho_total += state.density_matrix() / n_states
    
    return rho_total


def create_rotated_squeezed_states(n_max=40):
    """Create superposition of squeezed states with different orientations."""
    n_angles = 3
    angles = np.linspace(0, np.pi, n_angles, endpoint=False)
    
    rho_total = np.zeros((n_max, n_max), dtype=complex)
    
    for angle in angles:
        state = SqueezedState(r=1.0, phi=angle, n_max=n_max)
        rho_total += state.density_matrix() / n_angles
    
    return rho_total


def main():
    """Demonstrate custom quantum state engineering."""
    
    # Define phase space grid
    xvec = np.linspace(-5, 5, 120)
    yvec = np.linspace(-5, 5, 120)
    
    print("Engineering custom quantum states...")
    
    # 1. Displaced Fock states
    print("Creating displaced Fock states...")
    displaced_fock_0 = DisplacedFockState(n=0, alpha=2, n_max=30)
    displaced_fock_1 = DisplacedFockState(n=1, alpha=2, n_max=30)
    displaced_fock_3 = DisplacedFockState(n=3, alpha=2+1j, n_max=40)
    
    W_df_0 = wigner_function(displaced_fock_0.density_matrix(), xvec, yvec)
    W_df_1 = wigner_function(displaced_fock_1.density_matrix(), xvec, yvec)
    W_df_3 = wigner_function(displaced_fock_3.density_matrix(), xvec, yvec)
    
    # 2. Custom superposition: Four coherent states in a square
    print("Creating custom superposition (4 coherent states)...")
    rho_square = create_custom_superposition(n_max=40)
    W_square = wigner_function(rho_square, xvec, yvec)
    
    # 3. Rotated squeezed states
    print("Creating rotated squeezed state superposition...")
    rho_rotated = create_rotated_squeezed_states(n_max=40)
    W_rotated = wigner_function(rho_rotated, xvec, yvec)
    
    # 4. Create a high-amplitude coherent state
    print("Creating high-amplitude states...")
    high_amp = CoherentState(4 + 2j, n_max=60)
    W_high = wigner_function(high_amp.density_matrix(), 
                            np.linspace(-7, 7, 120),
                            np.linspace(-7, 7, 120))
    
    # Plot displaced Fock states
    print("Plotting displaced Fock states...")
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    
    for ax, W, title in zip(axes,
                           [W_df_0, W_df_1, W_df_3],
                           ['D(2)|0⟩', 'D(2)|1⟩', 'D(2+i)|3⟩']):
        vmax = np.max(np.abs(W))
        cf = ax.contourf(xvec, yvec, W, levels=25, cmap='RdBu',
                        vmin=-vmax, vmax=vmax)
        ax.contour(xvec, yvec, W, levels=25, colors='k',
                  alpha=0.3, linewidths=0.5)
        ax.set_xlabel('Re(α)')
        ax.set_ylabel('Im(α)')
        ax.set_title(f'Displaced Fock State: {title}')
        ax.set_aspect('equal')
        plt.colorbar(cf, ax=ax, label='W(x,y)')
    
    plt.tight_layout()
    plt.savefig('displaced_fock_states.png', dpi=150, bbox_inches='tight')
    print("Saved: displaced_fock_states.png")
    
    # Plot custom engineered states
    print("Plotting custom engineered states...")
    fig, axes = plt.subplots(2, 2, figsize=(12, 11))
    axes = axes.flatten()
    
    states_data = [
        (W_square, xvec, yvec, 'Four Coherent States\n(|2⟩+|2i⟩+|-2⟩+|-2i⟩)/2'),
        (W_rotated, xvec, yvec, 'Rotated Squeezed States\n(φ=0°, 60°, 120°)'),
        (W_high, np.linspace(-7, 7, 120), np.linspace(-7, 7, 120), 
         'High-Amplitude Coherent\n|4+2i⟩'),
        (W_df_3, xvec, yvec, 'Displaced Fock\nD(2+i)|3⟩'),
    ]
    
    for ax, (W, xv, yv, title) in zip(axes, states_data):
        vmax = np.max(np.abs(W))
        cf = ax.contourf(xv, yv, W, levels=25, cmap='RdBu',
                        vmin=-vmax, vmax=vmax)
        ax.contour(xv, yv, W, levels=25, colors='k',
                  alpha=0.3, linewidths=0.5)
        ax.set_xlabel('Re(α)')
        ax.set_ylabel('Im(α)')
        ax.set_title(title)
        ax.set_aspect('equal')
        plt.colorbar(cf, ax=ax, label='W(x,y)')
    
    plt.tight_layout()
    plt.savefig('custom_engineered_states.png', dpi=150, bbox_inches='tight')
    print("Saved: custom_engineered_states.png")
    
    # Create 3D visualization for one state
    print("Creating 3D visualization...")
    from wignersim import plot_wigner_3d
    
    fig, ax = plot_wigner_3d(W_square, xvec, yvec, 
                             title='3D Wigner Function: Four Coherent States')
    plt.savefig('wigner_3d.png', dpi=150, bbox_inches='tight')
    print("Saved: wigner_3d.png")
    
    plt.close('all')
    
    print("\nDone! Generated plots demonstrating custom state engineering.")
    print("\nKey features demonstrated:")
    print("- Displaced Fock states (photon-added coherent states)")
    print("- Superpositions of multiple coherent states")
    print("- Rotated squeezed state mixtures")
    print("- High-amplitude coherent states")
    print("- 3D visualization of Wigner functions")


if __name__ == "__main__":
    main()
