"""
Example: Schrödinger cat states and their Wigner functions.

This script demonstrates cat states, which are superpositions of
macroscopically distinct coherent states showing quantum interference
at a macroscopic scale.
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
import os

# Add parent directory to path to import wignersim
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from wignersim import CatState, wigner_function, plot_wigner


def main():
    """Demonstrate cat states with different parameters."""
    
    # Define phase space grid
    xvec = np.linspace(-6, 6, 120)
    yvec = np.linspace(-6, 6, 120)
    
    # Create different types of cat states
    print("Creating cat states...")
    
    # Even cat states with different amplitudes
    even_cat_1 = CatState(1.5, kind='even', n_max=30)
    even_cat_2 = CatState(2.5, kind='even', n_max=40)
    even_cat_3 = CatState(3.0, kind='even', n_max=50)
    
    # Odd cat state
    odd_cat = CatState(2.0, kind='odd', n_max=40)
    
    # Yurke cat state
    yurke_cat = CatState(2.0, kind='yurke', n_max=40)
    
    # Complex amplitude cat state
    complex_cat = CatState(2 + 1j, kind='even', n_max=40)
    
    print("Computing Wigner functions...")
    W_even_1 = wigner_function(even_cat_1.density_matrix(), xvec, yvec)
    W_even_2 = wigner_function(even_cat_2.density_matrix(), xvec, yvec)
    W_even_3 = wigner_function(even_cat_3.density_matrix(), xvec, yvec)
    W_odd = wigner_function(odd_cat.density_matrix(), xvec, yvec)
    W_yurke = wigner_function(yurke_cat.density_matrix(), xvec, yvec)
    W_complex = wigner_function(complex_cat.density_matrix(), xvec, yvec)
    
    # Plot even cat states with different amplitudes
    print("Plotting even cat states...")
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    
    for ax, W, alpha, title in zip(axes,
                                   [W_even_1, W_even_2, W_even_3],
                                   [1.5, 2.5, 3.0],
                                   ['Even Cat (α=1.5)', 'Even Cat (α=2.5)', 'Even Cat (α=3.0)']):
        vmax = np.max(np.abs(W))
        cf = ax.contourf(xvec, yvec, W, levels=25, cmap='RdBu',
                        vmin=-vmax, vmax=vmax)
        ax.contour(xvec, yvec, W, levels=25, colors='k',
                  alpha=0.3, linewidths=0.5)
        ax.set_xlabel('Re(α)')
        ax.set_ylabel('Im(α)')
        ax.set_title(title)
        ax.set_aspect('equal')
        plt.colorbar(cf, ax=ax, label='W(x,y)')
        
        # Mark the coherent state positions
        ax.plot([alpha], [0], 'ko', markersize=8, label=f'|{alpha}⟩')
        ax.plot([-alpha], [0], 'ko', markersize=8, label=f'|-{alpha}⟩')
        ax.legend(loc='upper right', fontsize=8)
    
    plt.tight_layout()
    plt.savefig('cat_states_even.png', dpi=150, bbox_inches='tight')
    print("Saved: cat_states_even.png")
    
    # Plot different types of cat states
    print("Plotting cat state types...")
    fig, axes = plt.subplots(2, 2, figsize=(12, 11))
    axes = axes.flatten()
    
    states_data = [
        (W_even_2, 'Even Cat: (|α⟩ + |-α⟩)/√2', 2.5),
        (W_odd, 'Odd Cat: (|α⟩ - |-α⟩)/√2', 2.0),
        (W_yurke, 'Yurke Cat: (|α⟩ + i|-α⟩)/√2', 2.0),
        (W_complex, 'Complex α Cat: |2+i⟩ + |-2-i⟩', None),
    ]
    
    for ax, (W, title, alpha_val) in zip(axes, states_data):
        vmax = np.max(np.abs(W))
        cf = ax.contourf(xvec, yvec, W, levels=25, cmap='RdBu',
                        vmin=-vmax, vmax=vmax)
        ax.contour(xvec, yvec, W, levels=25, colors='k',
                  alpha=0.3, linewidths=0.5)
        ax.set_xlabel('Re(α)')
        ax.set_ylabel('Im(α)')
        ax.set_title(title)
        ax.set_aspect('equal')
        plt.colorbar(cf, ax=ax, label='W(x,y)')
        
        # Mark coherent state positions
        if alpha_val is not None:
            ax.plot([alpha_val], [0], 'ko', markersize=6)
            ax.plot([-alpha_val], [0], 'ko', markersize=6)
    
    plt.tight_layout()
    plt.savefig('cat_states_types.png', dpi=150, bbox_inches='tight')
    print("Saved: cat_states_types.png")
    
    # Print state properties
    print("\nState properties:")
    print(f"Even cat (α=2.5): Purity = {even_cat_2.purity():.4f}")
    print(f"Odd cat (α=2.0): Purity = {odd_cat.purity():.4f}")
    print(f"Yurke cat (α=2.0): Purity = {yurke_cat.purity():.4f}")
    
    # Show interference fringes
    print("\nNote: The interference fringes in the center are a signature")
    print("of quantum superposition. Negative values of the Wigner function")
    print("(blue regions) indicate non-classical quantum behavior.")
    
    print("\nDone! Generated plots showing various cat state configurations.")


if __name__ == "__main__":
    main()
