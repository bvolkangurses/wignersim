"""
Visualization tools for Wigner functions.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D


def plot_wigner(W, xvec, yvec, figsize=(8, 6), cmap='RdBu', 
                contours=True, colorbar=True, title=None):
    """
    Plot 2D contour map of the Wigner function.
    
    Parameters
    ----------
    W : ndarray
        Wigner function values (2D array)
    xvec : ndarray
        x-coordinates
    yvec : ndarray
        y-coordinates
    figsize : tuple, optional
        Figure size (default: (8, 6))
    cmap : str, optional
        Colormap name (default: 'RdBu')
    contours : bool, optional
        Whether to show contour lines (default: True)
    colorbar : bool, optional
        Whether to show colorbar (default: True)
    title : str, optional
        Plot title (default: None)
        
    Returns
    -------
    fig : matplotlib.figure.Figure
        The figure object
    ax : matplotlib.axes.Axes
        The axes object
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Find symmetric limits for colormap
    vmax = np.max(np.abs(W))
    vmin = -vmax
    
    # Create contour plot
    if contours:
        contour_levels = np.linspace(vmin, vmax, 20)
        cf = ax.contourf(xvec, yvec, W, levels=contour_levels, 
                        cmap=cmap, vmin=vmin, vmax=vmax)
        ax.contour(xvec, yvec, W, levels=contour_levels, 
                  colors='k', alpha=0.3, linewidths=0.5)
    else:
        cf = ax.pcolormesh(xvec, yvec, W, cmap=cmap, 
                          vmin=vmin, vmax=vmax, shading='auto')
    
    if colorbar:
        plt.colorbar(cf, ax=ax, label='W(x, y)')
    
    ax.set_xlabel('Re(α) / x')
    ax.set_ylabel('Im(α) / y')
    ax.set_aspect('equal')
    
    if title:
        ax.set_title(title)
    
    plt.tight_layout()
    return fig, ax


def plot_wigner_3d(W, xvec, yvec, figsize=(10, 8), cmap='RdBu',
                   elev=30, azim=45, title=None):
    """
    Plot 3D surface of the Wigner function.
    
    Parameters
    ----------
    W : ndarray
        Wigner function values (2D array)
    xvec : ndarray
        x-coordinates
    yvec : ndarray
        y-coordinates
    figsize : tuple, optional
        Figure size (default: (10, 8))
    cmap : str, optional
        Colormap name (default: 'RdBu')
    elev : float, optional
        Elevation angle for 3D view (default: 30)
    azim : float, optional
        Azimuth angle for 3D view (default: 45)
    title : str, optional
        Plot title (default: None)
        
    Returns
    -------
    fig : matplotlib.figure.Figure
        The figure object
    ax : matplotlib.axes.Axes3D
        The 3D axes object
    """
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection='3d')
    
    # Create meshgrid
    X, Y = np.meshgrid(xvec, yvec)
    
    # Find symmetric limits for colormap
    vmax = np.max(np.abs(W))
    vmin = -vmax
    
    # Plot surface
    surf = ax.plot_surface(X, Y, W, cmap=cmap, 
                          vmin=vmin, vmax=vmax,
                          linewidth=0, antialiased=True,
                          alpha=0.9)
    
    # Add colorbar
    fig.colorbar(surf, ax=ax, label='W(x, y)', shrink=0.5)
    
    # Set labels
    ax.set_xlabel('Re(α) / x')
    ax.set_ylabel('Im(α) / y')
    ax.set_zlabel('W(x, y)')
    
    # Set viewing angle
    ax.view_init(elev=elev, azim=azim)
    
    if title:
        ax.set_title(title)
    
    plt.tight_layout()
    return fig, ax


def plot_comparison(states_dict, xvec, yvec, figsize=None):
    """
    Plot multiple Wigner functions for comparison.
    
    Parameters
    ----------
    states_dict : dict
        Dictionary mapping state names to (W, title) tuples
    xvec : ndarray
        x-coordinates
    yvec : ndarray
        y-coordinates
    figsize : tuple, optional
        Figure size (auto-calculated if None)
        
    Returns
    -------
    fig : matplotlib.figure.Figure
        The figure object
    axes : list of matplotlib.axes.Axes
        List of axes objects
    """
    n_states = len(states_dict)
    
    # Calculate grid dimensions
    n_cols = min(3, n_states)
    n_rows = (n_states + n_cols - 1) // n_cols
    
    if figsize is None:
        figsize = (6 * n_cols, 5 * n_rows)
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    
    # Flatten axes array for easier indexing
    if n_states == 1:
        axes = [axes]
    else:
        axes = axes.flatten() if n_states > 1 else [axes]
    
    # Find global vmax for consistent coloring
    all_W = [item[0] for item in states_dict.values()]
    vmax = max(np.max(np.abs(W)) for W in all_W)
    vmin = -vmax
    
    # Plot each state
    for idx, (name, (W, title)) in enumerate(states_dict.items()):
        ax = axes[idx]
        
        contour_levels = np.linspace(vmin, vmax, 20)
        cf = ax.contourf(xvec, yvec, W, levels=contour_levels,
                        cmap='RdBu', vmin=vmin, vmax=vmax)
        ax.contour(xvec, yvec, W, levels=contour_levels,
                  colors='k', alpha=0.3, linewidths=0.5)
        
        ax.set_xlabel('Re(α) / x')
        ax.set_ylabel('Im(α) / y')
        ax.set_aspect('equal')
        ax.set_title(title if title else name)
    
    # Hide extra subplots
    for idx in range(n_states, len(axes)):
        axes[idx].axis('off')
    
    # Add shared colorbar
    fig.colorbar(cf, ax=axes, label='W(x, y)', 
                orientation='vertical', fraction=0.046, pad=0.04)
    
    plt.tight_layout()
    return fig, axes


def animate_wigner_evolution(W_sequence, xvec, yvec, times, 
                             figsize=(8, 6), interval=50, 
                             save_path=None):
    """
    Create animation of Wigner function evolution.
    
    Parameters
    ----------
    W_sequence : list of ndarray
        Sequence of Wigner function snapshots
    xvec : ndarray
        x-coordinates
    yvec : ndarray
        y-coordinates
    times : ndarray
        Time points for each snapshot
    figsize : tuple, optional
        Figure size (default: (8, 6))
    interval : int, optional
        Delay between frames in milliseconds (default: 50)
    save_path : str, optional
        Path to save animation (default: None, just display)
        
    Returns
    -------
    anim : matplotlib.animation.FuncAnimation
        Animation object
    """
    from matplotlib.animation import FuncAnimation
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # Find global limits
    vmax = max(np.max(np.abs(W)) for W in W_sequence)
    vmin = -vmax
    
    contour_levels = np.linspace(vmin, vmax, 20)
    
    # Initialize with first frame
    cf = ax.contourf(xvec, yvec, W_sequence[0], levels=contour_levels,
                    cmap='RdBu', vmin=vmin, vmax=vmax)
    ct = ax.contour(xvec, yvec, W_sequence[0], levels=contour_levels,
                   colors='k', alpha=0.3, linewidths=0.5)
    
    ax.set_xlabel('Re(α) / x')
    ax.set_ylabel('Im(α) / y')
    ax.set_aspect('equal')
    
    plt.colorbar(cf, ax=ax, label='W(x, y)')
    
    time_text = ax.text(0.02, 0.95, '', transform=ax.transAxes,
                       bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    def animate(frame):
        """Update function for animation."""
        # Clear previous contours
        for coll in cf.collections:
            coll.remove()
        for coll in ct.collections:
            coll.remove()
        
        # Plot new frame
        cf_new = ax.contourf(xvec, yvec, W_sequence[frame], 
                            levels=contour_levels,
                            cmap='RdBu', vmin=vmin, vmax=vmax)
        ct_new = ax.contour(xvec, yvec, W_sequence[frame], 
                           levels=contour_levels,
                           colors='k', alpha=0.3, linewidths=0.5)
        
        time_text.set_text(f't = {times[frame]:.2f}')
        
        return cf_new.collections + ct_new.collections + [time_text]
    
    anim = FuncAnimation(fig, animate, frames=len(W_sequence),
                        interval=interval, blit=False)
    
    if save_path:
        anim.save(save_path, writer='pillow')
    
    return anim
