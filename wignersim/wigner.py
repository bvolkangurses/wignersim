"""
Wigner function calculation for quantum states.
"""

import numpy as np
from scipy.special import factorial, genlaguerre


def wigner_function(rho, xvec, yvec, g=1.0):
    """
    Calculate the Wigner function for a given density matrix.
    
    The Wigner function is a quasi-probability distribution that provides
    a phase-space representation of quantum states.
    
    Parameters
    ----------
    rho : ndarray
        Density matrix of the quantum state (NxN complex array)
    xvec : ndarray
        Array of x-coordinates in phase space (position quadrature)
    yvec : ndarray
        Array of y-coordinates in phase space (momentum quadrature)
    g : float, optional
        Scaling factor (default: 1.0 for standard normalization)
        
    Returns
    -------
    W : ndarray
        Wigner function values on the grid defined by xvec and yvec
        
    Notes
    -----
    This implementation uses the Fock state representation of the Wigner function.
    The Wigner function can take negative values, which is a signature of
    non-classical quantum behavior.
    
    For coherent states |alpha>, the Wigner function is Gaussian centered at
    (Re(alpha), Im(alpha)) in phase space.
    """
    M = np.meshgrid(xvec, yvec)
    X = M[0] + 1j * M[1]
    
    # Get the size of the Fock space
    n_max = rho.shape[0]
    
    # Initialize Wigner function
    W = np.zeros(X.shape, dtype=complex)
    
    # Calculate Wigner function using Fock state representation
    for n in range(n_max):
        for m in range(n_max):
            if np.abs(rho[n, m]) > 1e-12:  # Skip negligible terms
                W += rho[n, m] * _wigner_fock_element(n, m, X, g)
    
    return np.real(W)


def _wigner_fock_element(n, m, X, g):
    """
    Calculate the Wigner function element for Fock states |n> and |m>.
    
    Uses the formula:
    W_nm(alpha) = (2/pi) * (-1)^n * exp(-2|alpha|^2) * sqrt(n!/m!) * 
                  (2*alpha^*)^(m-n) * L_n^(m-n)(4|alpha|^2)
    
    Parameters
    ----------
    n, m : int
        Fock state indices
    X : ndarray
        Complex phase space coordinates (x + iy) representing alpha
    g : float
        Scaling factor (for units, typically 1.0)
        
    Returns
    -------
    W_nm : ndarray
        Contribution to Wigner function from <n|rho|m>
    """
    r2 = np.abs(X) ** 2
    
    # Ensure n <= m for the formula
    if n > m:
        n, m = m, n
        X = np.conj(X)
    
    # Prefactor and Gaussian
    prefactor = 2.0 / np.pi
    gaussian = np.exp(-2.0 * r2)
    
    if n == m:
        # Diagonal element
        L = genlaguerre(n, 0)
        W_nm = prefactor * (-1)**n * gaussian * L(4.0 * r2)
    else:
        # Off-diagonal element
        sqrt_fact = np.sqrt(factorial(n) / factorial(m))
        L = genlaguerre(n, m - n)
        phase_term = (2.0 * np.conj(X)) ** (m - n)
        
        W_nm = (prefactor * (-1)**n * gaussian * sqrt_fact * 
                phase_term * L(4.0 * r2))
    
    return W_nm


def wigner_clenshaw(rho, xvec, yvec, g=1.0):
    """
    Alternative Wigner function calculation using Clenshaw algorithm.
    
    This is an alternative, potentially more efficient method for calculating
    the Wigner function for large Hilbert spaces.
    
    Parameters
    ----------
    rho : ndarray
        Density matrix of the quantum state
    xvec : ndarray
        Array of x-coordinates in phase space
    yvec : ndarray
        Array of y-coordinates in phase space
    g : float, optional
        Scaling factor (default: sqrt(2))
        
    Returns
    -------
    W : ndarray
        Wigner function values on the grid
    """
    M = np.meshgrid(xvec, yvec)
    X = M[0]
    Y = M[1]
    
    n_max = rho.shape[0]
    W = np.zeros((len(yvec), len(xvec)), dtype=float)
    
    # Use direct calculation for small systems
    for i in range(len(xvec)):
        for j in range(len(yvec)):
            x, y = X[j, i], Y[j, i]
            r2 = x**2 + y**2
            
            # Gaussian envelope
            envelope = (2.0 / np.pi) * np.exp(-2 * r2 / g**2)
            
            # Sum over Fock state contributions
            W_point = 0.0
            for n in range(n_max):
                for m in range(n_max):
                    if np.abs(rho[n, m]) > 1e-12:
                        if n == m:
                            L = genlaguerre(n, 0)
                            contrib = rho[n, m].real * L(4 * r2 / g**2)
                            W_point += (-1)**n * contrib
                        else:
                            # Off-diagonal terms
                            z = (x + 1j * y) / g
                            min_nm = min(n, m)
                            max_nm = max(n, m)
                            
                            L = genlaguerre(min_nm, max_nm - min_nm)
                            phase_factor = z ** (max_nm - min_nm)
                            if n > m:
                                phase_factor = np.conj(phase_factor)
                                
                            contrib = (2 * np.real(rho[n, m] * phase_factor) * 
                                     np.sqrt(factorial(min_nm) / factorial(max_nm)) *
                                     L(4 * r2 / g**2))
                            W_point += (-1)**min_nm * contrib
            
            W[j, i] = envelope * W_point
    
    return W
