import unittest
import numpy as np
import math
from parameterized import parameterized

import conseal.simulate as sim


class TestLambdaEstimation(unittest.TestCase):
    """Unit tests for λ estimation methods in conseal.simulate._lambda_estimation"""

    @classmethod
    def setUpClass(cls):
        # Fix random seed so that expected_distortion uses the same _rho0/_rho1
        np.random.seed(42)

    def test_expected_distortion_monotonicity(self):
        """
        expected_distortion(λ) should strictly decrease as λ increases,
        since its derivative is always negative.
        """
        lambdas = [0.0, 0.5, 1.0, 2.0, 5.0]
        ed_values = [sim.expected_distortion(l) for l in lambdas]
        for i in range(len(ed_values) - 1):
            self.assertLess(
                ed_values[i + 1],
                ed_values[i],
                msg=(
                    f"E[D](λ={lambdas[i + 1]:.2f}) = {ed_values[i + 1]:.6f} is not "
                    f"less than E[D](λ={lambdas[i]:.2f}) = {ed_values[i]:.6f}"
                )
            )

    @parameterized.expand([
        (0.1,),
        (0.5,),
        (1.0,),
        (3.0,),
        (10.0,),
    ])
    def test_derivative_negative(self, lmbda):
        """
        d_expected_distortion(λ) should always be negative
        (the expected distortion is strictly decreasing in λ).
        """
        deriv = sim.d_expected_distortion(lmbda)
        self.assertLess(
            deriv,
            0.0,
            msg=f"d_expected_distortion(λ={lmbda:.2f}) = {deriv:.6f} is not negative"
        )

    def test_binary_search_lambda_accuracy(self):
        """
        If we pick a 'true' λ, compute its expected_distortion,
        then binary_search_lambda(target) should recover approximately that λ (within ~1e-3).
        """
        true_lambda = 1.234
        target = sim.expected_distortion(true_lambda)
        est_lambda = sim.binary_search_lambda(target, tol=1e-5)
        self.assertAlmostEqual(
            est_lambda,
            true_lambda,
            places=3,
            msg=(
                f"binary_search_lambda returned λ={est_lambda:.6f}, "
                f"expected around λ={true_lambda:.6f}"
            )
        )

    def test_newton_lambda_accuracy(self):
        """
        newton_lambda(target) should converge to the true λ (within ~1e-4)
        when given a reasonable initial guess and tight tolerance.
        """
        true_lambda = 0.789
        target = sim.expected_distortion(true_lambda)
        est_lambda = sim.newton_lambda(target, initial=1.0, tol=1e-8, max_iter=50)
        self.assertAlmostEqual(
            est_lambda,
            true_lambda,
            places=4,
            msg=(
                f"newton_lambda returned λ={est_lambda:.6f}, "
                f"expected around λ={true_lambda:.6f}"
            )
        )

    def test_polynomial_proxy_lambda_valid_root(self):
        """
        polynomial_proxy_lambda(target) should return a positive λ that yields
        expected_distortion(λ) ≈ target within ~5% relative error.
        """
        true_lambda = 2.5
        target = sim.expected_distortion(true_lambda)
        proxy_lambda = sim.polynomial_proxy_lambda(target, samples=(0.5, 2.0, 5.0))

        # It must not be NaN and must be positive
        self.assertFalse(math.isnan(proxy_lambda), "polynomial_proxy_lambda returned NaN")
        self.assertGreater(
            proxy_lambda,
            0.0,
            msg=f"polynomial_proxy_lambda returned non-positive λ={proxy_lambda:.6f}"
        )

        recovered_dist = sim.expected_distortion(proxy_lambda)
        rel_error = abs(recovered_dist - target) / target
        self.assertLess(
            rel_error,
            0.05,
            msg=(
                f"Polynomial proxy gave E[D]={recovered_dist:.6f}, "
                f"target was {target:.6f}, relative error {rel_error:.4f}"
            )
        )

    @parameterized.expand([
        (1.2, 0.10),  # allow 10% relative error for Taylor inverse
        (0.8, 0.10),
    ])
    def test_taylor_inverse_lambda_accuracy(self, true_lambda, rel_tol):
        """
        taylor_inverse_lambda(target) should approximate the true λ within ~10%,
        since it’s only a single‐step linear Taylor inversion around λ0=1.0.
        """
        target = sim.expected_distortion(true_lambda)
        est_lambda = sim.taylor_inverse_lambda(target, lambda0=1.0)
        rel_error = abs(est_lambda - true_lambda) / true_lambda
        self.assertLess(
            rel_error,
            rel_tol,
            msg=(
                f"taylor_inverse_lambda returned λ={est_lambda:.6f}, "
                f"true λ={true_lambda:.6f}, relative error {rel_error:.4f}"
            )
        )

    def test_compare_methods_structure(self):
        """
        compare_methods_with_taylor() should return a list of 5 dictionaries,
        each containing keys for binary, newton, polynomial, and taylor results.
        """
        results = sim.compare_methods()
        # Expect exactly 5 records
        self.assertEqual(len(results), 5, f"Expected 5 records, got {len(results)}")
        # Check keys in the first record
        expected_keys = {
            'target_distortion', 'true_lambda',
            'binary_lambda',   'binary_time_ms',   'binary_error',
            'newton_lambda',   'newton_time_ms',   'newton_error',
            'poly_lambda',     'poly_time_ms',     'poly_error',
            'taylor_lambda',   'taylor_time_ms',   'taylor_error',
        }
        first_keys = set(results[0].keys())
        self.assertTrue(
            expected_keys.issubset(first_keys),
            msg=f"Missing keys: {expected_keys - first_keys}"
        )


if __name__ == "__main__":
    unittest.main()
