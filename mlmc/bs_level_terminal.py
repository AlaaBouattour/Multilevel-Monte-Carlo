import numpy as np, math
from typing import Callable

# analytic prices 

def _norm_cdf(x):
    return 0.5*(1+math.erf(x/math.sqrt(2)))

def bs_exact_call(S0, K, r, sigma, T):
    d1 = (math.log(S0/K)+(r+0.5*sigma**2)*T)/(sigma*math.sqrt(T))
    d2 = d1 - sigma*math.sqrt(T)
    return S0*_norm_cdf(d1) - K*math.exp(-r*T)*_norm_cdf(d2)

def bs_exact_put(S0, K, r, sigma, T):
    return bs_exact_call(S0, K, r, sigma, T) - S0 + K*math.exp(-r*T)

class BSLevelFunction:
    def __init__(self, S0: float, r: float, sigma: float, T: float, payoff: Callable[[np.ndarray], np.ndarray],verbose=True):
        self.S0 = S0
        self.r = r
        self.sigma = sigma
        self.T = T
        self.payoff = payoff
        self.verbose=verbose

    def simulate(self, l: int, N: int, return_details: bool = False, rng=None):
        raise NotImplementedError("Subclasses must implement simulate()")


class EulerBSLevelFunction(BSLevelFunction):
    def simulate(self, l: int, N: int, return_details: bool = False, rng=None):
        rng = rng or np.random.default_rng()
        M = 2**l
        h = self.T / M
        dW = rng.normal(0.0, np.sqrt(h), size=(N, M))

        S_f = np.full(N, self.S0, dtype=float)
        for i in range(M):
            S_f += S_f * ((self.r - 0.5 * self.sigma**2) * h + self.sigma * dW[:, i])
        pf = np.exp(-self.r * self.T) * self.payoff(S_f)

        if l == 0:
            pc = np.zeros_like(pf)
        else:
            dWc = dW[:, 0::2] + dW[:, 1::2]
            hc = 2 * h
            S_c = np.full(N, self.S0, dtype=float)
            for i in range(M // 2):
                S_c += S_c * ((self.r - 0.5 * self.sigma**2) * hc + self.sigma * dWc[:, i])
            pc = np.exp(-self.r * self.T) * self.payoff(S_c)

        Y = pf if l == 0 else (pf - pc)
        cost = N * M

        if self.verbose:
            print(f"Level {l}: mean={Y.mean():.4e}, std={Y.std():.4e}")

        if not return_details:
            sumY = Y.sum()
            sumY2 = (Y**2).sum()
            return np.array([sumY, sumY2]), cost
        else:
            details = [{'S_fine': S_f[i], 'pf_value': pf[i],
                        **({'S_coarse': S_c[i], 'pc_value': pc[i]} if l > 0 else {})} for i in range(N)]
            return Y, details, cost


class GBMExactBSLevelFunction(BSLevelFunction):
    def simulate(self, l: int, N: int, return_details: bool = False, rng=None):
        rng = rng or np.random.default_rng()
        M = 2**l
        h = self.T / M
        dW = rng.normal(0.0, np.sqrt(h), size=(N, M))

        S_f = np.full(N, self.S0, dtype=float)
        for i in range(M):
            S_f *= np.exp((self.r - 0.5 * self.sigma**2) * h + self.sigma * dW[:, i])
        pf = np.exp(-self.r * self.T) * self.payoff(S_f)

        if l == 0:
            pc = np.zeros_like(pf)
        else:
            dWc = dW[:, 0::2] + dW[:, 1::2]
            hc = 2 * h
            S_c = np.full(N, self.S0, dtype=float)
            for i in range(M // 2):
                S_c *= np.exp((self.r - 0.5 * self.sigma**2) * hc + self.sigma * dWc[:, i])
            pc = np.exp(-self.r * self.T) * self.payoff(S_c)

        Y = pf if l == 0 else (pf - pc)
        cost = N * M

        if self.verbose:
            print(f"Level {l}: mean={Y.mean():.4e}, std={Y.std():.4e}")

        if not return_details:
            sumY = Y.sum()
            sumY2 = (Y**2).sum()
            return np.array([sumY, sumY2]), cost
        else:
            details = [{'S_fine': S_f[i], 'pf_value': pf[i],
                        **({'S_coarse': S_c[i], 'pc_value': pc[i]} if l > 0 else {})} for i in range(N)]
            return Y, details, cost


class MilsteinBSLevelFunction(BSLevelFunction):
    def simulate(self, l: int, N: int, return_details: bool = False, rng=None):
        rng = rng or np.random.default_rng()
        M = 2**l
        h = self.T / M
        dW = rng.normal(0.0, np.sqrt(h), size=(N, M))

        S_f = np.full(N, self.S0, dtype=float)
        for i in range(M):
            Z = dW[:, i]
            S_f += S_f * (self.r * h + self.sigma * Z + 0.5 * self.sigma**2 * (Z**2 - h))
        pf = np.exp(-self.r * self.T) * self.payoff(S_f)

        if l == 0:
            pc = np.zeros_like(pf)
        else:
            dWc = dW[:, 0::2] + dW[:, 1::2]
            hc = 2 * h
            S_c = np.full(N, self.S0, dtype=float)
            for i in range(M // 2):
                Zc = dWc[:, i]
                S_c += S_c * (self.r * hc + self.sigma * Zc + 0.5 * self.sigma**2 * (Zc**2 - hc))
            pc = np.exp(-self.r * self.T) * self.payoff(S_c)

        Y = pf if l == 0 else (pf - pc)
        cost = N * M

        if self.verbose:
            print(f"Level {l}: mean={Y.mean():.4e}, std={Y.std():.4e}")

        if not return_details:
            sumY = Y.sum()
            sumY2 = (Y**2).sum()
            return np.array([sumY, sumY2]), cost
        else:
            details = [{'S_fine': S_f[i], 'pf_value': pf[i],
                        **({'S_coarse': S_c[i], 'pc_value': pc[i]} if l > 0 else {})} for i in range(N)]
            return Y, details, cost
