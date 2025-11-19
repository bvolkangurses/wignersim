"""
Example: Basic quantum states and their Wigner functions.

This script demonstrates how to create and visualize Wigner functions
for various fundamental quantum states.
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
import os

# Add parent directory to path to import wignersim
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from wignersim import (
    FockState, CoherentState, SqueezedState, ThermalState,
    wigner_function, plot_wigner
)


def main():
    """Demonstrate basic quantum states."""
    
    # Define phase space grid
    xvec = np.linspace(-5, 5, 100)
    yvec = np.linspace(-5, 5, 100)
    
    # 1. Fock state (vacuum and number states)
    print("Creating Fock states...")
    vacuum = FockState(0, n_max=20)
    fock_1 = FockState(1, n_max=20)
    fock_5 = FockState(5, n_max=20)
    
    W_vacuum = wigner_function(vacuum.density_matrix(), xvec, yvec)
    W_fock_1 = wigner_function(fock_1.density_matrix(), xvec, yvec)
    W_fock_5 = wigner_function(fock_5.density_matrix(), xvec, yvec)
    
    # 2. Coherent state
    print("Creating coherent states...")
    coherent_0 = CoherentState(0, n_max=20)
    coherent_2 = CoherentState(2, n_max=20)
    coherent_complex = CoherentState(2 + 1j, n_max=20)
    
    W_coh_0 = wigner_function(coherent_0.density_matrix(), xvec, yvec)
    W_coh_2 = wigner_function(coherent_2.density_matrix(), xvec, yvec)
    W_coh_complex = wigner_function(coherent_complex.density_matrix(), xvec, yvec)
    
    # 3. Squeezed state
    print("Creating squeezed states...")
    squeezed_0 = SqueezedState(0, n_max=20)
    squeezed_1 = SqueezedState(1, phi=0, n_max=30)
    
    W_sq_0 = wigner_function(squeezed_0.density_matrix(), xvec, yvec)
    W_sq_1 = wigner_function(squeezed_1.density_matrix(), xvec, yvec)
    
    # 4. Thermal state
    print("Creating thermal states...")
    thermal_1 = ThermalState(1, n_max=20)
    thermal_5 = ThermalState(5, n_max=30)
    
    W_th_1 = wigner_function(thermal_1.density_matrix(), xvec, yvec)
    W_th_5 = wigner_function(thermal_5.density_matrix(), xvec, yvec)
    
    # Plot results
    print("Plotting results...")
    
    # Fock states
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    for ax, W, title in zip(axes, 
                           [W_vacuum, W_fock_1, W_fock_5],
                           ['Vacuum |0⟩', 'Fock |1⟩', 'Fock |5⟩']):
        vmax = np.max(np.abs(W))
        cf = ax.contourf(xvec, yvec, W, levels=20, cmap='RdBu', 
                        vmin=-vmax, vmax=vmax)
        ax.contour(xvec, yvec, W, levels=20, colors='k', 
                  alpha=0.3, linewidths=0.5)
        ax.set_xlabel('Re(α)')
        ax.set_ylabel('Im(α)')
        ax.set_title(title)
        ax.set_aspect('equal')
        plt.colorbar(cf, ax=ax)
    
    plt.tight_layout()
    plt.savefig('fock_states.png', dpi=150, bbox_inches='tight')
    print("Saved: fock_states.png")
    
    # Coherent states
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    for ax, W, title in zip(axes,
                           [W_coh_0, W_coh_2, W_coh_complex],
                           ['Coherent |0⟩', 'Coherent |2⟩', 'Coherent |2+i⟩']):
        vmax = np.max(np.abs(W))
        cf = ax.contourf(xvec, yvec, W, levels=20, cmap='RdBu',
                        vmin=-vmax, vmax=vmax)
        ax.contour(xvec, yvec, W, levels=20, colors='k',
                  alpha=0.3, linewidths=0.5)
        ax.set_xlabel('Re(α)')
        ax.set_ylabel('Im(α)')
        ax.set_title(title)
        ax.set_aspect('equal')
        plt.colorbar(cf, ax=ax)
    
    plt.tight_layout()
    plt.savefig('coherent_states.png', dpi=150, bbox_inches='tight')
    print("Saved: coherent_states.png")
    
    # Squeezed states
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    
    for ax, W, title in zip(axes,
                           [W_sq_0, W_sq_1],
                           ['Squeezed (r=0)', 'Squeezed (r=1)']):
        vmax = np.max(np.abs(W))
        cf = ax.contourf(xvec, yvec, W, levels=20, cmap='RdBu',
                        vmin=-vmax, vmax=vmax)
        ax.contour(xvec, yvec, W, levels=20, colors='k',
                  alpha=0.3, linewidths=0.5)
        ax.set_xlabel('Re(α)')
        ax.set_ylabel('Im(α)')
        ax.set_title(title)
        ax.set_aspect('equal')
        plt.colorbar(cf, ax=ax)
    
    plt.tight_layout()
    plt.savefig('squeezed_states.png', dpi=150, bbox_inches='tight')
    print("Saved: squeezed_states.png")
    
    # Thermal states
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    
    for ax, W, title in zip(axes,
                           [W_th_1, W_th_5],
                           ['Thermal (n̄=1)', 'Thermal (n̄=5)']):
        vmax = np.max(np.abs(W))
        cf = ax.contourf(xvec, yvec, W, levels=20, cmap='RdBu',
                        vmin=-vmax, vmax=vmax)
        ax.contour(xvec, yvec, W, levels=20, colors='k',
                  alpha=0.3, linewidths=0.5)
        ax.set_xlabel('Re(α)')
        ax.set_ylabel('Im(α)')
        ax.set_title(title)
        ax.set_aspect('equal')
        plt.colorbar(cf, ax=ax)
    
    plt.tight_layout()
    plt.savefig('thermal_states.png', dpi=150, bbox_inches='tight')
    print("Saved: thermal_states.png")
    
    print("\nDone! Generated 4 plots demonstrating various quantum states.")


if __name__ == "__main__":
    main()
