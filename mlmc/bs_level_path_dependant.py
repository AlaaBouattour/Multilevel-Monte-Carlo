import numpy as np

class BSLevelFunction:
    def __init__(self, S0, K, r, sigma, T, payoff_fn, verbose=False):
        self.S0 = S0
        self.K = K
        self.r = r
        self.sigma = sigma
        self.T = T
        self.payoff = payoff_fn
        self.verbose = verbose

    def simulate(self, l: int, N: int, return_details: bool = False, rng=None):
        raise NotImplementedError("Implement in subclass")


class EulerBSLevelFunction(BSLevelFunction):
    def simulate(self, l: int, N: int, return_details: bool = False, rng=None):
        rng = rng or np.random.default_rng()
        M = 2 ** l
        h = self.T / M
        dW = rng.normal(0.0, np.sqrt(h), size=(N, M))

        # === Fine path ===
        S_f = np.full(N, self.S0)
        S_f_path = np.zeros((N, M + 1))
        S_f_path[:, 0] = self.S0
        for i in range(M):
            S_f += self.r * S_f * h + self.sigma * S_f * dW[:, i]
            S_f_path[:, i + 1] = S_f

        pf = np.exp(-self.r * self.T) * self.payoff(S_f, S_f_path)

        if l == 0:
            pc = np.zeros_like(pf)
        else:
            # === Coarse path ===
            dWc = dW[:, 0::2] + dW[:, 1::2]
            hc = 2 * h
            S_c = np.full(N, self.S0)
            S_c_path = np.zeros((N, M // 2 + 1))
            S_c_path[:, 0] = self.S0
            for i in range(M // 2):
                S_c += self.r * S_c * hc + self.sigma * S_c * dWc[:, i]
                S_c_path[:, i + 1] = S_c
            pc = np.exp(-self.r * self.T) * self.payoff(S_c, S_c_path)

        Y = pf if l == 0 else pf - pc
        cost = N * M

        if self.verbose:
            print(f"[Euler L={l}] mean={Y.mean():.4e}, std={Y.std():.4e}")

        if not return_details:
            return np.array([Y.sum(), (Y ** 2).sum()]), cost
        else:
            details = [
                {
                    'S_fine': S_f[i],
                    'pf_value': pf[i],
                    'avg_fine': S_f_path[i].mean(),
                    'min_fine': S_f_path[i].min(),
                    **({
                        'S_coarse': S_c[i],
                        'pc_value': pc[i],
                        'avg_coarse': S_c_path[i].mean(),
                        'min_coarse': S_c_path[i].min()
                    } if l > 0 else {})
                }
                for i in range(N)
            ]
            return Y, details, cost


class MilsteinBSLevelFunction(BSLevelFunction):
    def simulate(self, l: int, N: int, return_details: bool = False, rng=None):
        rng = rng or np.random.default_rng()
        M = 2 ** l
        h = self.T / M
        dW = rng.normal(0.0, np.sqrt(h), size=(N, M))

        # === Fine path ===
        S_f = np.full(N, self.S0)
        S_f_path = np.zeros((N, M + 1))
        S_f_path[:, 0] = self.S0
        for i in range(M):
            Z = dW[:, i]
            S_f += S_f * (self.r * h + self.sigma * Z + 0.5 * self.sigma**2 * (Z**2 - h))
            S_f_path[:, i + 1] = S_f

        pf = np.exp(-self.r * self.T) * self.payoff(S_f, S_f_path)

        if l == 0:
            pc = np.zeros_like(pf)
        else:
            dWc = dW[:, 0::2] + dW[:, 1::2]
            hc = 2 * h
            S_c = np.full(N, self.S0)
            S_c_path = np.zeros((N, M // 2 + 1))
            S_c_path[:, 0] = self.S0
            for i in range(M // 2):
                Zc = dWc[:, i]
                S_c += S_c * (self.r * hc + self.sigma * Zc + 0.5 * self.sigma**2 * (Zc**2 - hc))
                S_c_path[:, i + 1] = S_c
            pc = np.exp(-self.r * self.T) * self.payoff(S_c, S_c_path)

        Y = pf if l == 0 else pf - pc
        cost = N * M

        if self.verbose:
            print(f"[Milstein L={l}] mean={Y.mean():.4e}, std={Y.std():.4e}")

        if not return_details:
            return np.array([Y.sum(), (Y ** 2).sum()]), cost
        else:
            details = [
                {
                    'S_fine': S_f[i],
                    'pf_value': pf[i],
                    'avg_fine': S_f_path[i].mean(),
                    'min_fine': S_f_path[i].min(),
                    **({
                        'S_coarse': S_c[i],
                        'pc_value': pc[i],
                        'avg_coarse': S_c_path[i].mean(),
                        'min_coarse': S_c_path[i].min()
                    } if l > 0 else {})
                }
                for i in range(N)
            ]
            return Y, details, cost
