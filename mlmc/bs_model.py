"""Black‑Scholes tools for MLMC."""
import numpy as np, math
from typing import Callable

__all__ = ["make_bs_level_fn", "bs_exact_call", "bs_exact_put"]

# analytic prices -------------------------------------------------------------

def _norm_cdf(x):
    return 0.5*(1+math.erf(x/math.sqrt(2)))

def bs_exact_call(S0, K, r, sigma, T):
    d1 = (math.log(S0/K)+(r+0.5*sigma**2)*T)/(sigma*math.sqrt(T))
    d2 = d1 - sigma*math.sqrt(T)
    return S0*_norm_cdf(d1) - K*math.exp(-r*T)*_norm_cdf(d2)

def bs_exact_put(S0, K, r, sigma, T):
    return bs_exact_call(S0, K, r, sigma, T) - S0 + K*math.exp(-r*T)

# low‑level Y_l generator ------------------------------------------------------

def make_bs_level_fn(S0: float, r: float, sigma: float, T: float,
                     payoff: Callable[[np.ndarray], np.ndarray]):
    """
    Return a function level_fn(l, N, return_details=False) that:
     - if return_details=False: returns (sumY, sumY2), cost
     - if return_details=True: returns Y_array, details_list, cost

    `details_list` is a list of dicts with keys 'S_fine', 'pf_value'
    (and for l>0 also 'S_coarse', 'pc_value').
    """
    def level_fn(l: int, N: int, return_details: bool = False):
        M = 2**l
        h = T / M
        dW = np.random.randn(N, M) * math.sqrt(h)
        drift = (r - 0.5 * sigma**2) * h

        # Simulate fine path
        S_f = np.full(N, S0, dtype=float)
        for i in range(M):
            S_f *= np.exp(drift + sigma * dW[:, i])
        pf = np.exp(-r * T) * payoff(S_f)

        # Simulate coarse path if needed
        if l == 0:
            pc = np.zeros_like(pf)
        else:
            dWc = dW[:, 0::2] + dW[:, 1::2]
            hc = 2 * h
            S_c = np.full(N, S0, dtype=float)
            drift_c = (r - 0.5 * sigma**2) * hc
            for i in range(M // 2):
                S_c *= np.exp(drift_c + sigma * dWc[:, i])
            pc = np.exp(-r * T) * payoff(S_c)

        # Level increment
        Y = pf if l == 0 else (pf - pc)
        cost = N * M

        if not return_details:
            sumY  = Y.sum()
            sumY2 = (Y**2).sum()
            return np.array([sumY, sumY2]), cost
        else:
            details = []
            for i in range(N):
                d = {'S_fine': S_f[i], 'pf_value': pf[i]}
                if l > 0:
                    d['S_coarse'] = S_c[i]
                    d['pc_value']   = pc[i]
                details.append(d)
            return Y, details, cost

    return level_fn
