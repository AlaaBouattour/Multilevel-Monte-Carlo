import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import math
import numpy as np

from mlmc import MLMC
from mlmc.bs_model import make_bs_level_fn, bs_exact_call
from mlmc.payoffs import call_payoff

def test_mlmc_price_accuracy():
    print("\nRunning MLMC price accuracy test...")

    # Setup
    S0, K, r, sigma, T = 100, 100, 0.05, 0.2, 1.0
    analytic = bs_exact_call(S0, K, r, sigma, T)
    payoff = lambda S: call_payoff(S, K)
    level_fn = make_bs_level_fn(S0, r, sigma, T, payoff)
    mlmc = MLMC(level_fn)

    # Target precision
    eps = 0.02

    # Estimate
    price, *_ = mlmc.estimate(eps)

    print(f"Analytic price : {analytic:.6f}")
    print(f"MLMC estimate  : {price:.6f}")
    print(f"Error          : {abs(price - analytic):.6f}")

    # Quick pass/fail logic
    if abs(price - analytic) < 2 * eps:
        print("PASSED: MLMC within 2×eps target")
    else:
        print("FAILED: MLMC too far from analytic value")

def run_all_tests():
    test_mlmc_price_accuracy()

if __name__ == "__main__":
    run_all_tests()