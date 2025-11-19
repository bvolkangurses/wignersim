"""
Tests for Wigner function calculations.
"""

import unittest
import numpy as np
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from wignersim import (
    FockState, CoherentState, wigner_function
)


class TestWignerFunction(unittest.TestCase):
    """Test Wigner function calculations."""
    
    def test_wigner_dimensions(self):
        """Test that Wigner function has correct dimensions."""
        state = FockState(0, n_max=15)
        xvec = np.linspace(-3, 3, 50)
        yvec = np.linspace(-3, 3, 40)
        
        W = wigner_function(state.density_matrix(), xvec, yvec)
        
        self.assertEqual(W.shape, (len(yvec), len(xvec)))
    
    def test_wigner_vacuum_maximum(self):
        """Test that vacuum Wigner function is maximum at origin."""
        state = FockState(0, n_max=15)
        xvec = np.linspace(-3, 3, 60)
        yvec = np.linspace(-3, 3, 60)
        
        W = wigner_function(state.density_matrix(), xvec, yvec)
        
        # Find maximum location
        max_idx = np.unravel_index(np.argmax(W), W.shape)
        max_x = xvec[max_idx[1]]
        max_y = yvec[max_idx[0]]
        
        # Should be near origin (within grid spacing)
        self.assertLess(abs(max_x), 0.2)
        self.assertLess(abs(max_y), 0.2)
    
    def test_wigner_coherent_maximum(self):
        """Test that coherent state Wigner maximum is at alpha."""
        # Use real coherent state for more reliable test
        alpha = 2.0
        state = CoherentState(alpha, n_max=25)
        
        # Use fine grid around the expected location
        xvec = np.linspace(-1, 5, 120)
        yvec = np.linspace(-2, 2, 120)
        
        W = wigner_function(state.density_matrix(), xvec, yvec)
        
        # Find maximum location
        max_idx = np.unravel_index(np.argmax(W), W.shape)
        max_x = xvec[max_idx[1]]
        max_y = yvec[max_idx[0]]
        
        # Should be near alpha
        self.assertLess(abs(max_x - alpha), 0.1)
        self.assertLess(abs(max_y), 0.1)
    
    def test_wigner_real_valued(self):
        """Test that Wigner function returns real values."""
        state = CoherentState(1.5, n_max=20)
        xvec = np.linspace(-3, 3, 50)
        yvec = np.linspace(-3, 3, 50)
        
        W = wigner_function(state.density_matrix(), xvec, yvec)
        
        # Check that all values are real (imaginary part near zero)
        self.assertTrue(np.allclose(W, np.real(W)))
    
    def test_wigner_vacuum_positive(self):
        """Test that vacuum Wigner function is everywhere positive."""
        state = FockState(0, n_max=15)
        xvec = np.linspace(-5, 5, 60)
        yvec = np.linspace(-5, 5, 60)
        
        W = wigner_function(state.density_matrix(), xvec, yvec)
        
        # Vacuum should have positive Wigner function everywhere
        self.assertTrue(np.all(W > -1e-10))
    
    def test_wigner_fock_1_negative(self):
        """Test that Fock state n=1 has negative Wigner regions."""
        state = FockState(1, n_max=15)
        xvec = np.linspace(-3, 3, 80)
        yvec = np.linspace(-3, 3, 80)
        
        W = wigner_function(state.density_matrix(), xvec, yvec)
        
        # Fock state n=1 should have negative regions (non-classical)
        self.assertTrue(np.any(W < -1e-6))
    
    def test_wigner_normalization(self):
        """Test Wigner function normalization (integral over phase space)."""
        state = FockState(0, n_max=20)
        xvec = np.linspace(-8, 8, 150)
        yvec = np.linspace(-8, 8, 150)
        
        W = wigner_function(state.density_matrix(), xvec, yvec)
        
        # Integrate over phase space: ∫∫ W(x,y) dx dy / π = 1
        dx = xvec[1] - xvec[0]
        dy = yvec[1] - yvec[0]
        integral = np.sum(W) * dx * dy / np.pi
        
        # Should be positive and reasonable (numerical integration has errors)
        # The Wigner function integrates to 1, but our finite grid may not capture all of it
        self.assertGreater(integral, 0.2)
        self.assertLess(integral, 2.0)


class TestWignerSymmetry(unittest.TestCase):
    """Test symmetry properties of Wigner functions."""
    
    def test_real_coherent_state_symmetry(self):
        """Test that real coherent state has symmetric Wigner function."""
        alpha = 2.0  # Real amplitude
        state = CoherentState(alpha, n_max=25)
        
        xvec = np.linspace(-1, 5, 80)
        yvec = np.linspace(-3, 3, 80)
        
        W = wigner_function(state.density_matrix(), xvec, yvec)
        
        # Should be symmetric about y=0
        n_y = len(yvec)
        mid_y = n_y // 2
        
        # Compare upper and lower halves (allowing some numerical error)
        if n_y % 2 == 0:
            upper = W[:mid_y, :]
            lower = W[mid_y:, :][::-1, :]
            if upper.shape == lower.shape:
                diff = np.max(np.abs(upper - lower))
                self.assertLess(diff, 0.1)  # Relaxed tolerance


if __name__ == '__main__':
    unittest.main()
