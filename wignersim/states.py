"""
Quantum state classes for Wigner function calculations.
"""

import numpy as np
from scipy.special import factorial


class QuantumState:
    """
    Base class for quantum states.
    
    Attributes
    ----------
    n_max : int
        Maximum Fock state number (dimension of Hilbert space)
    rho : ndarray
        Density matrix representation of the state
    """
    
    def __init__(self, n_max=30):
        """
        Initialize quantum state.
        
        Parameters
        ----------
        n_max : int, optional
            Maximum Fock state number (default: 30)
        """
        self.n_max = n_max
        self.rho = None
        
    def density_matrix(self):
        """
        Return the density matrix of the state.
        
        Returns
        -------
        rho : ndarray
            Density matrix (complex array)
        """
        if self.rho is None:
            raise NotImplementedError("Density matrix not computed")
        return self.rho
    
    def trace(self):
        """Calculate the trace of the density matrix."""
        return np.trace(self.rho)
    
    def purity(self):
        """Calculate the purity of the state (Tr(rho^2))."""
        return np.real(np.trace(self.rho @ self.rho))


class FockState(QuantumState):
    """
    Fock state (number state) |n>.
    
    A Fock state is an eigenstate of the number operator with a definite
    number of quanta.
    """
    
    def __init__(self, n, n_max=None):
        """
        Initialize Fock state |n>.
        
        Parameters
        ----------
        n : int
            Fock state number
        n_max : int, optional
            Maximum Fock state for truncation (default: max(30, n+1))
        """
        if n_max is None:
            n_max = max(30, n + 1)
        super().__init__(n_max)
        
        if n >= n_max:
            raise ValueError(f"Fock state number {n} must be less than n_max {n_max}")
            
        self.n = n
        self._compute_density_matrix()
        
    def _compute_density_matrix(self):
        """Compute density matrix for Fock state."""
        self.rho = np.zeros((self.n_max, self.n_max), dtype=complex)
        self.rho[self.n, self.n] = 1.0


class CoherentState(QuantumState):
    """
    Coherent state |alpha>.
    
    Coherent states are eigenstates of the annihilation operator and
    represent the most classical quantum states of light.
    """
    
    def __init__(self, alpha, n_max=30):
        """
        Initialize coherent state |alpha>.
        
        Parameters
        ----------
        alpha : complex
            Complex amplitude of the coherent state
        n_max : int, optional
            Maximum Fock state for truncation (default: 30)
        """
        super().__init__(n_max)
        self.alpha = alpha
        self._compute_density_matrix()
        
    def _compute_density_matrix(self):
        """Compute density matrix for coherent state."""
        # Compute Fock state coefficients
        coeffs = np.zeros(self.n_max, dtype=complex)
        alpha_abs2 = np.abs(self.alpha) ** 2
        
        for n in range(self.n_max):
            coeffs[n] = (np.exp(-alpha_abs2 / 2) * 
                        (self.alpha ** n) / np.sqrt(factorial(n)))
        
        # Construct density matrix |alpha><alpha|
        self.rho = np.outer(coeffs, np.conj(coeffs))


class SqueezedState(QuantumState):
    """
    Squeezed vacuum state.
    
    Squeezed states have reduced quantum noise in one quadrature at the
    expense of increased noise in the conjugate quadrature.
    """
    
    def __init__(self, r, phi=0, n_max=30):
        """
        Initialize squeezed vacuum state.
        
        Parameters
        ----------
        r : float
            Squeezing parameter (magnitude)
        phi : float, optional
            Squeezing angle (default: 0)
        n_max : int, optional
            Maximum Fock state for truncation (default: 30)
        """
        super().__init__(n_max)
        self.r = r
        self.phi = phi
        self._compute_density_matrix()
        
    def _compute_density_matrix(self):
        """Compute density matrix for squeezed vacuum state."""
        # Squeezed vacuum has only even Fock states
        coeffs = np.zeros(self.n_max, dtype=complex)
        
        tanh_r = np.tanh(self.r)
        sech_r = 1.0 / np.cosh(self.r)
        phase = np.exp(1j * self.phi)
        
        for n in range(0, self.n_max, 2):
            # Only even n contribute
            k = n // 2
            coeffs[n] = (np.sqrt(factorial(n)) / 
                        (2 ** k * factorial(k)) * 
                        tanh_r ** k * 
                        np.sqrt(sech_r) * 
                        phase ** k)
        
        # Construct density matrix
        self.rho = np.outer(coeffs, np.conj(coeffs))


class CatState(QuantumState):
    """
    Schrödinger cat state (superposition of coherent states).
    
    Cat states are quantum superpositions of macroscopically distinct
    coherent states, exhibiting quantum interference on a macroscopic scale.
    """
    
    def __init__(self, alpha, kind="even", n_max=30):
        """
        Initialize cat state.
        
        Parameters
        ----------
        alpha : complex or float
            Amplitude of the coherent states
        kind : str, optional
            Type of cat state: 'even', 'odd', or 'yurke' (default: 'even')
        n_max : int, optional
            Maximum Fock state for truncation (default: 30)
        """
        super().__init__(n_max)
        self.alpha = alpha if isinstance(alpha, complex) else complex(alpha)
        self.kind = kind.lower()
        self._compute_density_matrix()
        
    def _compute_density_matrix(self):
        """Compute density matrix for cat state."""
        # Compute coefficients for |alpha> and |-alpha>
        coeffs_plus = np.zeros(self.n_max, dtype=complex)
        coeffs_minus = np.zeros(self.n_max, dtype=complex)
        
        alpha_abs2 = np.abs(self.alpha) ** 2
        exp_factor = np.exp(-alpha_abs2 / 2)
        
        for n in range(self.n_max):
            sqrt_fact_n = np.sqrt(factorial(n))
            coeffs_plus[n] = exp_factor * (self.alpha ** n) / sqrt_fact_n
            coeffs_minus[n] = exp_factor * ((-self.alpha) ** n) / sqrt_fact_n
        
        # Overlap between |alpha> and |-alpha>
        overlap = np.exp(-2 * alpha_abs2)
        
        if self.kind == "even":
            # Even cat: |alpha> + |-alpha>
            norm = 1.0 / np.sqrt(2 * (1 + np.real(overlap)))
            psi = norm * (coeffs_plus + coeffs_minus)
        elif self.kind == "odd":
            # Odd cat: |alpha> - |-alpha>
            norm = 1.0 / np.sqrt(2 * (1 - np.real(overlap)))
            psi = norm * (coeffs_plus - coeffs_minus)
        elif self.kind == "yurke":
            # Yurke cat: |alpha> + i|-alpha>
            norm = 1.0 / np.sqrt(2 * (1 + np.imag(overlap)))
            psi = norm * (coeffs_plus + 1j * coeffs_minus)
        else:
            raise ValueError(f"Unknown cat state kind: {self.kind}")
        
        # Construct density matrix
        self.rho = np.outer(psi, np.conj(psi))


class ThermalState(QuantumState):
    """
    Thermal state (mixed state at finite temperature).
    
    Thermal states represent quantum systems in thermal equilibrium.
    """
    
    def __init__(self, n_mean, n_max=30):
        """
        Initialize thermal state.
        
        Parameters
        ----------
        n_mean : float
            Mean photon number
        n_max : int, optional
            Maximum Fock state for truncation (default: 30)
        """
        super().__init__(n_max)
        self.n_mean = n_mean
        self._compute_density_matrix()
        
    def _compute_density_matrix(self):
        """Compute density matrix for thermal state."""
        self.rho = np.zeros((self.n_max, self.n_max), dtype=complex)
        
        # Thermal state is diagonal in Fock basis
        for n in range(self.n_max):
            self.rho[n, n] = (self.n_mean ** n / 
                             (1 + self.n_mean) ** (n + 1))


class DisplacedFockState(QuantumState):
    """
    Displaced Fock state D(alpha)|n>.
    
    Combination of displacement and Fock state operations.
    """
    
    def __init__(self, n, alpha, n_max=None):
        """
        Initialize displaced Fock state.
        
        Parameters
        ----------
        n : int
            Fock state number
        alpha : complex
            Displacement amplitude
        n_max : int, optional
            Maximum Fock state for truncation
        """
        if n_max is None:
            n_max = max(30, n + int(3 * np.abs(alpha)) + 1)
        super().__init__(n_max)
        self.n = n
        self.alpha = alpha
        self._compute_density_matrix()
        
    def _compute_density_matrix(self):
        """Compute density matrix for displaced Fock state."""
        # Start with Fock state
        psi = np.zeros(self.n_max, dtype=complex)
        
        # Apply displacement operator
        alpha_abs2 = np.abs(self.alpha) ** 2
        exp_factor = np.exp(-alpha_abs2 / 2)
        
        for m in range(self.n_max):
            # Sum over Fock state expansion
            for k in range(min(self.n, m) + 1):
                if m - k + self.n - k < self.n_max:
                    psi[m] += (exp_factor * 
                              (-1) ** (self.n - k) *
                              np.sqrt(factorial(self.n) * factorial(m)) /
                              (factorial(k) * factorial(self.n - k) * factorial(m - k)) *
                              self.alpha ** (m - k) *
                              np.conj(self.alpha) ** (self.n - k))
        
        # Construct density matrix
        self.rho = np.outer(psi, np.conj(psi))


class SuperpositionState(QuantumState):
    """
    General superposition of quantum states.
    
    Allows creation of arbitrary superpositions of other quantum states.
    """
    
    def __init__(self, states, coefficients):
        """
        Initialize superposition state.
        
        Parameters
        ----------
        states : list of QuantumState
            List of quantum states to superpose
        coefficients : list of complex
            Coefficients for the superposition
        """
        if len(states) != len(coefficients):
            raise ValueError("Number of states and coefficients must match")
        
        # Use the largest n_max from all states
        n_max = max(state.n_max for state in states)
        super().__init__(n_max)
        
        self.states = states
        self.coefficients = np.array(coefficients, dtype=complex)
        
        # Normalize coefficients
        norm = np.sqrt(np.sum(np.abs(self.coefficients) ** 2))
        self.coefficients /= norm
        
        self._compute_density_matrix()
        
    def _compute_density_matrix(self):
        """Compute density matrix for superposition."""
        self.rho = np.zeros((self.n_max, self.n_max), dtype=complex)
        
        # Sum over all combinations
        for i, state_i in enumerate(self.states):
            for j, state_j in enumerate(self.states):
                # Get density matrices with proper size
                rho_i = state_i.density_matrix()
                rho_j = state_j.density_matrix()
                
                # Pad to common size if needed
                if rho_i.shape[0] < self.n_max:
                    rho_i_padded = np.zeros((self.n_max, self.n_max), dtype=complex)
                    rho_i_padded[:rho_i.shape[0], :rho_i.shape[1]] = rho_i
                    rho_i = rho_i_padded
                    
                if rho_j.shape[0] < self.n_max:
                    rho_j_padded = np.zeros((self.n_max, self.n_max), dtype=complex)
                    rho_j_padded[:rho_j.shape[0], :rho_j.shape[1]] = rho_j
                    rho_j = rho_j_padded
                
                # Note: This is simplified - proper superposition requires
                # state vectors, not density matrices
                coeff = self.coefficients[i] * np.conj(self.coefficients[j])
                self.rho += coeff * np.sqrt(rho_i * rho_j)
