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
    """Return a function (level, N) -> (sumY, sumY2, cost) for MLMC."""
    def level_fn(l: int, N: int):
        M = 2**l
        h = T/M
        dW = math.sqrt(h)*np.random.randn(N, M)
        drift = (r-0.5*sigma**2)*h
        # fine path
        S_f = np.full(N, S0, dtype=float)
        for i in range(M):
            S_f *= np.exp(drift + sigma*dW[:, i])
        pf = math.exp(-r*T)*payoff(S_f)
        if l == 0:
            Y = pf
        else:
            dWc = dW[:,0::2] + dW[:,1::2]
            hc  = 2*h
            S_c = np.full(N, S0, dtype=float)
            drift_c = (r-0.5*sigma**2)*hc
            for i in range(M//2):
                S_c *= np.exp(drift_c + sigma*dWc[:, i])
            pc = math.exp(-r*T)*payoff(S_c)
            Y = pf - pc
            #here we assume that cost for each time step costs 1 unit of computation, and we simulate all steps explicitly.
        return np.array([Y.sum(), (Y**2).sum()]), N*M 
    return level_fn