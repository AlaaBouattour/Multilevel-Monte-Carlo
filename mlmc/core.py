from __future__ import annotations
import numpy as np
import math

class WeakConvergenceError(RuntimeError):
    """Raised when the weak‑error remainder cannot be reduced under the target."""

class MLMC:
    """Generic Multilevel Monte‑Carlo estimator (Giles, 2008).

    Parameters
    ----------
    sde_step_fn : callable(level, n_paths) -> (sumY, sumY2, cost)
        Returns the sums for the level‑corrector Yₗ.
    Lmin, Lmax  : minimum / maximum refinement levels (Lmin ≥ 2).
    alpha0, beta0, gamma0 : initial rate guesses (≤0 → automatic regression).
    """

    def __init__(self, sde_step_fn, *, Lmin: int = 2, Lmax: int = 10,
                 alpha0: float = 0.0, beta0: float = 0.0, gamma0: float = 0.0):
        if Lmin < 2 or Lmax < Lmin:
            raise ValueError("Need Lmin ≥ 2 and Lmax ≥ Lmin")
        self._f = sde_step_fn
        self.Lmin, self.Lmax = Lmin, Lmax
        self.alpha0, self.beta0, self.gamma0 = alpha0, beta0, gamma0

    # ------------------------------------------------------------------
    def estimate(self, eps: float, N0: int = 1024):
        """Return (price, Nl, Cl, total_cost) for the given RMS tolerance `eps`."""
        alpha = max(0, self.alpha0); beta = max(0, self.beta0); gamma = max(0, self.gamma0)
        theta = 0.25  # split of error budget between bias/MC
        L = self.Lmin
        Nl   = np.zeros(L+1)
        suml = np.zeros((2, L+1))
        costl = np.zeros(L+1)
        dNl  = N0*np.ones(L+1)

        while dNl.sum() > 0:
            # 1) Simulate extra samples level by level
            for l in range(L+1):
                if dNl[l] <= 0: continue
                sums, cost = self._f(l, int(dNl[l]))
                Nl[l]      += dNl[l]
                suml[:, l] += sums
                costl[l]   += cost

            # 2) Empirical stats
            ml = np.abs(suml[0]/Nl)
            Vl = np.maximum(0.0, suml[1]/Nl - ml**2)
            Cl = costl/Nl
            for l in range(3, L+2):  # guard against zero variance on small samples
                ml[l-1] = max(ml[l-1], 0.5*ml[l-2]/2**alpha)
                Vl[l-1] = max(Vl[l-1], 0.5*Vl[l-2]/2**beta)

            # 3) Regression for alpha, beta, gamma if needed
            if self.alpha0 <= 0 and L >= 2:
                A = np.vstack([np.arange(1, L+1), np.ones(L)]).T
                alpha = max(0.5, -np.linalg.lstsq(A, np.log2(ml[1:]), rcond=None)[0][0])
            if self.beta0 <= 0 and L >= 2:
                A = np.vstack([np.arange(1, L+1), np.ones(L)]).T
                beta  = max(0.5, -np.linalg.lstsq(A, np.log2(Vl[1:]), rcond=None)[0][0])
            if self.gamma0 <= 0 and L >= 2:
                A = np.vstack([np.arange(1, L+1), np.ones(L)]).T
                gamma = max(0.5,  np.linalg.lstsq(A, np.log2(Cl[1:]), rcond=None)[0][0])

            # 4) Optimal allocation
            Ns  = np.ceil(np.sqrt(Vl/Cl) * np.sum(np.sqrt(Vl*Cl)) / ((1-theta)*eps**2))
            dNl = np.maximum(0, Ns - Nl)

            # 5) Weak‑error check – add level if bias too large
            if (dNl > 0.01*Nl).sum() == 0:
                tail = list(range(min(3, L)))
                remainder = (np.max(ml[[L-x for x in tail]]/2**(np.array(tail)*alpha))
                             / (2**alpha - 1))
                if remainder > np.sqrt(theta)*eps:
                    if L == self.Lmax:
                        raise WeakConvergenceError("Increase Lmax to reach tolerance")
                    L += 1
                    Vl    = np.append(Vl, Vl[-1]/2**beta)
                    Nl    = np.append(Nl, 0.0)
                    suml  = np.column_stack([suml, [0.0, 0.0]])
                    Cl    = np.append(Cl, Cl[-1]*2**gamma)
                    costl = np.append(costl, 0.0)
                    Ns  = np.ceil(np.sqrt(Vl/Cl) * np.sum(np.sqrt(Vl*Cl)) / ((1-theta)*eps**2))
                    dNl = np.maximum(0, Ns - Nl)

        price = np.sum(suml[0]/Nl)
        return price, Nl.astype(int), Cl, costl.sum()