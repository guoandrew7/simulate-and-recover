import unittest
import numpy as np
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))
from ez_diffusion import EZDiffusionSimulator

class TestEZDiffusionSimulator(unittest.TestCase):
    
    def setUp(self):
        #test initialization
        self.simulator = EZDiffusionSimulator()

    def test_generate_true_parameters_range(self):
        #test that true parametes are within bounds
        a, v, t = self.simulator.generate_true_parameters()
        self.assertGreaterEqual(a, 0.5)
        self.assertLessEqual(a, 2.0)
        self.assertGreaterEqual(v, 0.5)
        self.assertLessEqual(v, 2.0)
        self.assertGreaterEqual(t, 0.1)
        self.assertLessEqual(t, 0.5)

    def test_generate_predicted_summary_statistics(self):
        #test that predicted statistics are valid
        a, v, t = 1.0, 1.0, 0.3  # Test with fixed parameters
        r_pred, m_pred, v_pred = self.simulator.generate_predicted_summary_statistics(a, v, t)
        self.assertTrue(0 < r_pred < 1)
        self.assertGreater(m_pred, 0)
        self.assertGreater(v_pred, 0)

    def test_zero_variance_estimation(self):
        #test that there is variance due to estimation
        n = 10
        a, v, t = 1.0, 1.0, 0.3
        r_pred, m_pred, v_pred = self.simulator.generate_predicted_summary_statistics(a, v, t)
        a_est, v_est, t_est = self.simulator.compute_estimated_parameters(n, r_pred, m_pred, v_pred)
        self.assertGreater(v_est, 0)  # Ensure that variance is never exactly zero

    def test_squared_error_decrease(self):
        #test that squared error decreases when n increases
        n_values = [10, 40, 400]
        b_squared_values = []
        for n in n_values:
            b, b_squared = self.simulator.calculate_bias(n)
            b_squared_values.append(b_squared)

        self.assertGreater(np.mean(b_squared_values[0]), np.mean(b_squared_values[1]))
        self.assertGreater(np.mean(b_squared_values[1]), np.mean(b_squared_values[2]))
                       
# To run the tests
if __name__ == '__main__':
    unittest.main()
