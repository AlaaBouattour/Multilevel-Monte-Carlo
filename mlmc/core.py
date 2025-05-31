from __future__ import annotations
import numpy as np
import math
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


class WeakConvergenceError(RuntimeError):
    """Raised when the weak‑error remainder cannot be reduced under the target."""

class MLMC:
    """Generic Multilevel Monte‑Carlo estimator (Giles, 2008)."""

    def __init__(self, sde_step_fn, *, Lmin=2, Lmax=10,
                 alpha0=0.0, beta0=0.0, gamma0=0.0, N0=1000, theta=0.25):
        if Lmin < 2 or Lmax < Lmin:
            raise ValueError("Need Lmin ≥ 2 and Lmax ≥ Lmin")
        self._f = sde_step_fn
        self.Lmin, self.Lmax = Lmin, Lmax
        self.alpha0, self.beta0, self.gamma0 = alpha0, beta0, gamma0
        self.N0 = N0
        self.theta = theta

    def estimate(self, eps: float):
        alpha = max(0, self.alpha0)
        beta  = max(0, self.beta0)
        gamma = max(0, self.gamma0)
        L = self.Lmin

        Nl    = np.zeros(L+1)
        suml  = np.zeros((2, L+1))
        costl = np.zeros(L+1)

        # Initial N0 samples on each level
        for l in range(L+1):
            sums, cost = self._f(l, self.N0)
            Nl[l] = self.N0
            suml[:, l] = sums
            costl[l] = cost

        while True:
            ml = np.abs(suml[0] / Nl)
            Vl = np.maximum(0.0, suml[1] / Nl - (suml[0]/Nl)**2)
            Cl = costl / Nl

            # Exclude level 0 from regression and decay clamping
            ml_safe = np.maximum(ml[1:], 1e-12)
            Vl_safe = np.maximum(Vl[1:], 1e-12)

            for l in range(2, L+1):
                ml[l] = max(ml[l], 0.5 * ml[l-1] / 2**alpha)
                Vl[l] = max(Vl[l], 0.5 * Vl[l-1] / 2**beta)

            # Estimate convergence rates
            if self.alpha0 <= 0 and L >= 2:
                A = np.vstack([np.arange(1, L+1), np.ones(L)]).T
                alpha = max(0.5, -np.linalg.lstsq(A, np.log2(ml_safe), rcond=None)[0][0])
            if self.beta0 <= 0 and L >= 2:
                A = np.vstack([np.arange(1, L+1), np.ones(L)]).T
                beta = max(0.5, -np.linalg.lstsq(A, np.log2(Vl_safe), rcond=None)[0][0])
            if self.gamma0 <= 0 and L >= 2:
                A = np.vstack([np.arange(1, L+1), np.ones(L)]).T
                gamma = max(0.5, np.linalg.lstsq(A, np.log2(Cl[1:]), rcond=None)[0][0])

            # Compute optimal allocation
            Ns = np.ceil(np.sqrt(Vl/Cl) * np.sum(np.sqrt(Vl * Cl)) / ((1 - self.theta) * eps**2))
            dNl = np.maximum(0, Ns - Nl)

            # Check convergence using extrapolated bias remainder (Giles-style)
            if (dNl > 0.01 * Nl).sum() == 0:
                tail_range = np.array([L - i for i in range(min(3, L+1))])
                extrapolated = ml[tail_range] * 2**(np.arange(len(tail_range)) * alpha)
                remainder = np.max(extrapolated) / (2**alpha - 1)
                print(f"[Bias Check] L={L}, extrapolated remainder={remainder:.4e}, threshold={np.sqrt(self.theta)*eps:.4e}")
                if remainder > np.sqrt(self.theta) * eps:
                    if L == self.Lmax:
                        raise WeakConvergenceError("Increase Lmax to reach tolerance")
                    L += 1
                    Nl    = np.append(Nl, 0.0)
                    suml  = np.column_stack([suml, [0.0, 0.0]])
                    costl = np.append(costl, 0.0)

                    # Immediately simulate on new level
                    sums, cost = self._f(L, self.N0)
                    Nl[L] = self.N0
                    suml[:, L] = sums
                    costl[L] = cost

                    continue  # re-enter loop to use updated stats

                break  # convergence passed → exit loop

            # Simulate additional samples
            for l in range(L+1):
                if dNl[l] <= 0: continue
                n_new = int(dNl[l])
                sums, cost = self._f(l, n_new)
                Nl[l]      += n_new
                suml[:, l] += sums
                costl[l]   += cost

        price = np.sum(suml[0] / Nl)
        return price, Nl.astype(int), Cl, costl.sum()


class C_MLMC:
    """Clustered Multilevel Monte Carlo with proper stratified sampling."""

    def __init__(
        self,
        sde_step_fn,
        feature_fn,
        *,
        Lmin: int = 2,
        Lmax: int = 10,
        alpha0: float = 0.0,
        beta0: float = 0.0,
        gamma0: float = 0.0,
        n_clusters: int = 4,
        N0: int = 1000,
        theta: float = 0.25,
        scale_features: bool = True,
    ):
        if Lmin < 2 or Lmax < Lmin:
            raise ValueError("Need Lmin ≥ 2 and Lmax ≥ Lmin")
        self._f = sde_step_fn
        self.feature_fn = feature_fn
        self.n_clusters = n_clusters
        self.N0 = N0
        self.Lmin, self.Lmax = Lmin, Lmax
        self.alpha0, self.beta0, self.gamma0 = alpha0, beta0, gamma0
        self.theta = theta
        self.scale_features = scale_features
        self.scalers = [None] * (self.Lmax + 1) if scale_features else None

    def estimate(self, eps: float):
        alpha = max(0, self.alpha0)
        beta = max(0, self.beta0)
        gamma = max(0, self.gamma0)
        L = self.Lmin

        Nl = np.zeros(L + 1)
        suml = np.zeros((2, L + 1))
        costl = np.zeros(L + 1)

        self.kmeans_levels = [None] * (self.Lmax + 1)
        self.cluster_probs = [None] * (self.Lmax + 1)
        self.cluster_vars = [None] * (self.Lmax + 1)

        for l in range(L + 1):
            Y_arr, details, cost = self._f(l, self.N0, return_details=True)

            if l == 0:
                suml[0, l] = Y_arr.sum()
                suml[1, l] = (Y_arr ** 2).sum()
                Nl[l] = self.N0
                costl[l] = cost
                continue

            feats = np.vstack([self.feature_fn(d) for d in details])

            if self.scale_features:
                scaler = StandardScaler()
                feats = scaler.fit_transform(feats)
                self.scalers[l] = scaler

            km = KMeans(n_clusters=self.n_clusters).fit(feats)
            labels = km.predict(feats)

            Vc = np.array([
                np.var(Y_arr[labels == c], ddof=1) if np.any(labels == c) else 0.0
                for c in range(self.n_clusters)
            ])
            Pc = np.array([np.mean(labels == c) for c in range(self.n_clusters)])

            Vc = np.clip(Vc, 1e-6, None)
            Pc = np.clip(Pc, 1e-6, None)
            Pc /= Pc.sum()

            suml[0, l] = Y_arr.sum()
            suml[1, l] = (Y_arr ** 2).sum()
            Nl[l] = self.N0
            costl[l] = cost

            self.kmeans_levels[l] = km
            self.cluster_probs[l] = Pc
            self.cluster_vars[l] = Vc

        while True:
            ml = np.abs(suml[0] / Nl)
            Vl = np.maximum(0.0, suml[1] / Nl - (suml[0] / Nl) ** 2)
            Cl = costl / Nl

            ml_safe = np.maximum(ml[1:], 1e-12)
            Vl_safe = np.maximum(Vl[1:], 1e-12)

            for l in range(2, L + 1):
                ml[l] = max(ml[l], 0.5 * ml[l - 1] / 2 ** alpha)
                Vl[l] = max(Vl[l], 0.5 * Vl[l - 1] / 2 ** beta)

            if self.alpha0 <= 0 and L >= 2:
                A = np.vstack([np.arange(1, L + 1), np.ones(L)]).T
                alpha = max(0.5, -np.linalg.lstsq(A, np.log2(ml_safe), rcond=None)[0][0])
            if self.beta0 <= 0 and L >= 2:
                A = np.vstack([np.arange(1, L + 1), np.ones(L)]).T
                beta = max(0.5, -np.linalg.lstsq(A, np.log2(Vl_safe), rcond=None)[0][0])
            if self.gamma0 <= 0 and L >= 2:
                A = np.vstack([np.arange(1, L + 1), np.ones(L)]).T
                gamma = max(0.5, np.linalg.lstsq(A, np.log2(Cl[1:]), rcond=None)[0][0])

            Ns = np.ceil(np.sqrt(Vl / Cl) * np.sum(np.sqrt(Vl * Cl)) / ((1 - self.theta) * eps ** 2))
            dNl = np.maximum(0, Ns - Nl)

            if (dNl > 0.01 * Nl).sum() == 0:
                tail_range = np.array([L - i for i in range(min(3, L + 1))])
                extrapolated = ml[tail_range] * 2 ** (np.arange(len(tail_range)) * alpha)
                remainder = np.max(extrapolated) / (2 ** alpha - 1)
                print(f"[Bias Check] L={L}, extrapolated remainder={remainder:.4e}, threshold={np.sqrt(self.theta) * eps:.4e}")
                if remainder > np.sqrt(self.theta) * eps:
                    if L == self.Lmax:
                        raise RuntimeError("Increase Lmax to reach tolerance")
                    L += 1
                    Nl = np.append(Nl, 0.0)
                    suml = np.column_stack([suml, [0.0, 0.0]])
                    costl = np.append(costl, 0.0)

                    Y_arr, details, cost = self._f(L, self.N0, return_details=True)
                    feats = np.vstack([self.feature_fn(d) for d in details])
                    km = KMeans(n_clusters=self.n_clusters).fit(feats)
                    labels = km.predict(feats)

                    Vc = np.array([
                        np.var(Y_arr[labels == c], ddof=1) if np.any(labels == c) else 0.0
                        for c in range(self.n_clusters)
                    ])
                    Pc = np.array([np.mean(labels == c) for c in range(self.n_clusters)])

                    Vc = np.clip(Vc, 1e-12, None)
                    Pc = np.clip(Pc, 1e-12, None)
                    Pc /= Pc.sum()

                    suml[0, L] = Y_arr.sum()
                    suml[1, L] = (Y_arr ** 2).sum()
                    Nl[L] = self.N0
                    costl[L] = cost

                    self.kmeans_levels[L] = km
                    self.cluster_probs[L] = Pc
                    self.cluster_vars[L] = Vc
                    continue
                break

            for l in range(L + 1):
                if dNl[l] <= 0:
                    continue
                if l == 0:
                    n_add = int(dNl[l])
                    Y_arr, _, cost = self._f(0, n_add, return_details=True)
                    suml[0, 0] += Y_arr.sum()
                    suml[1, 0] += (Y_arr ** 2).sum()
                    Nl[0] += n_add
                    costl[0] += cost
                    continue

                n_add = int(dNl[l])
                km = self.kmeans_levels[l]
                Vc = self.cluster_vars[l]
                Pc = self.cluster_probs[l]
                weights = np.sqrt(Vc * Pc)
                if not np.isfinite(weights).all() or weights.sum() == 0:
                    weights = np.ones_like(weights)
                weights /= weights.sum()

                alloc = np.floor(n_add * weights).astype(int)
                alloc[-1] += n_add - alloc.sum()

                needed = alloc.copy()
                collected = [[] for _ in range(self.n_clusters)]
                cost_total = 0
                max_iter = 30
                iter_count = 0

                while needed.sum() > 0 and iter_count < max_iter:
                    Y_arr, details, cost_b = self._f(l, self.N0, return_details=True)
                    feats = np.vstack([self.feature_fn(d) for d in details])
                    labs = km.predict(feats)
                    for y, lab in zip(Y_arr, labs):
                        if needed[lab] > 0:
                            collected[lab].append(y)
                            needed[lab] -= 1
                    cost_total += cost_b
                    iter_count += 1

                if needed.sum() > 0:
                    print(f"[Warning] Not enough samples collected for level {l} after {max_iter} iterations.")

                for c in range(self.n_clusters):
                    y = np.array(collected[c])
                    if len(y) > 0:
                        suml[0, l] += y.sum()
                        suml[1, l] += (y ** 2).sum()
                        Nl[l] += len(y)

                costl[l] += cost_total

        price = sum(
            np.sum(self.cluster_probs[l] * (suml[0, l] / Nl[l])) if l > 0 else suml[0, l] / Nl[l]
            for l in range(L + 1)
        )
        return price, Nl.astype(int), Cl, costl.sum()

    def get_cluster_info(self, level: int):
        if level < 0 or level > self.Lmax:
            raise ValueError(f"Level must be between 0 and {self.Lmax}.")

        km = self.kmeans_levels[level]
        pc = self.cluster_probs[level]
        vc = self.cluster_vars[level]

        if km is None or pc is None or vc is None:
            raise ValueError(f"No cluster info available at level {level}. Did you run `.estimate()`?")

        return {
            "centers": km.cluster_centers_,
            "probs": pc,
            "vars": vc
        }

