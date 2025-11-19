"""
Tests for quantum state classes.
"""

import unittest
import numpy as np
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from wignersim import (
    FockState, CoherentState, SqueezedState, CatState, ThermalState
)


class TestQuantumStates(unittest.TestCase):
    """Test quantum state classes."""
    
    def test_fock_state_trace(self):
        """Test that Fock state has unit trace."""
        state = FockState(0, n_max=10)
        trace = state.trace()
        self.assertAlmostEqual(trace, 1.0, places=10)
        
        state = FockState(5, n_max=20)
        trace = state.trace()
        self.assertAlmostEqual(trace, 1.0, places=10)
    
    def test_fock_state_purity(self):
        """Test that Fock state is pure (purity = 1)."""
        state = FockState(3, n_max=15)
        purity = state.purity()
        self.assertAlmostEqual(purity, 1.0, places=10)
    
    def test_coherent_state_trace(self):
        """Test that coherent state has unit trace."""
        state = CoherentState(0, n_max=20)
        trace = state.trace()
        self.assertAlmostEqual(trace, 1.0, places=8)
        
        state = CoherentState(2 + 1j, n_max=30)
        trace = state.trace()
        self.assertAlmostEqual(trace, 1.0, places=8)
    
    def test_coherent_state_purity(self):
        """Test that coherent state is pure."""
        state = CoherentState(1.5, n_max=25)
        purity = state.purity()
        self.assertAlmostEqual(purity, 1.0, places=6)
    
    def test_squeezed_state_trace(self):
        """Test that squeezed state has unit trace."""
        state = SqueezedState(r=0, n_max=20)
        trace = state.trace()
        self.assertAlmostEqual(trace, 1.0, places=8)
        
        state = SqueezedState(r=1.0, phi=0, n_max=30)
        trace = state.trace()
        # Squeezed states may have lower trace due to truncation
        self.assertGreater(abs(trace), 0.999)
    
    def test_squeezed_state_purity(self):
        """Test that squeezed state is pure."""
        state = SqueezedState(r=0.5, n_max=25)
        purity = state.purity()
        self.assertAlmostEqual(purity, 1.0, places=4)
    
    def test_cat_state_trace(self):
        """Test that cat state has unit trace."""
        state = CatState(2.0, kind='even', n_max=30)
        trace = state.trace()
        self.assertAlmostEqual(trace, 1.0, places=6)
        
        state = CatState(1.5, kind='odd', n_max=30)
        trace = state.trace()
        self.assertAlmostEqual(trace, 1.0, places=6)
    
    def test_cat_state_purity(self):
        """Test that cat state is pure."""
        state = CatState(2.0, kind='even', n_max=30)
        purity = state.purity()
        self.assertAlmostEqual(purity, 1.0, places=4)
    
    def test_thermal_state_trace(self):
        """Test that thermal state has unit trace."""
        state = ThermalState(n_mean=1, n_max=20)
        trace = state.trace()
        self.assertGreater(abs(trace), 0.999)
        
        state = ThermalState(n_mean=5, n_max=50)
        trace = state.trace()
        # Thermal states have lower trace due to truncation
        self.assertGreater(abs(trace), 0.99)
    
    def test_thermal_state_mixed(self):
        """Test that thermal state is mixed (purity < 1)."""
        state = ThermalState(n_mean=2, n_max=30)
        purity = state.purity()
        self.assertLess(purity, 1.0)
        self.assertGreater(purity, 0.0)
    
    def test_density_matrix_hermitian(self):
        """Test that density matrices are Hermitian."""
        states = [
            FockState(2, n_max=15),
            CoherentState(1.5, n_max=20),
            SqueezedState(0.5, n_max=20),
            CatState(2.0, kind='even', n_max=25),
        ]
        
        for state in states:
            rho = state.density_matrix()
            # Check Hermiticity: rho = rho^dagger
            hermitian_diff = np.max(np.abs(rho - rho.conj().T))
            self.assertLess(hermitian_diff, 1e-10)
    
    def test_density_matrix_positive_semidefinite(self):
        """Test that density matrices have non-negative eigenvalues."""
        states = [
            FockState(1, n_max=10),
            CoherentState(1.0, n_max=15),
            ThermalState(n_mean=1, n_max=15),
        ]
        
        for state in states:
            rho = state.density_matrix()
            eigenvalues = np.linalg.eigvalsh(rho)
            self.assertTrue(np.all(eigenvalues >= -1e-10))


class TestStateDimensions(unittest.TestCase):
    """Test state dimensions and parameters."""
    
    def test_fock_state_dimension(self):
        """Test Fock state dimension."""
        n_max = 25
        state = FockState(5, n_max=n_max)
        rho = state.density_matrix()
        self.assertEqual(rho.shape, (n_max, n_max))
    
    def test_coherent_state_dimension(self):
        """Test coherent state dimension."""
        n_max = 30
        state = CoherentState(2, n_max=n_max)
        rho = state.density_matrix()
        self.assertEqual(rho.shape, (n_max, n_max))
    
    def test_cat_state_kinds(self):
        """Test different cat state types can be created."""
        kinds = ['even', 'odd', 'yurke']
        for kind in kinds:
            state = CatState(2.0, kind=kind, n_max=25)
            self.assertIsNotNone(state.density_matrix())
            self.assertEqual(state.kind, kind)


if __name__ == '__main__':
    unittest.main()
