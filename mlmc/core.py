from __future__ import annotations
import numpy as np
import math
from sklearn.cluster import KMeans


class WeakConvergenceError(RuntimeError):
    """Raised when the weak‑error remainder cannot be reduced under the target."""

class MLMC:
    """Generic Multilevel Monte‑Carlo estimator (Giles, 2008)."""

    def __init__(self, sde_step_fn, *, Lmin=2, Lmax=10,
                 alpha0=0.0, beta0=0.0, gamma0=0.0, N0=1024, theta=0.25):
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
    """Clustered Multilevel Monte-Carlo estimator (Giles, 2008 + KMeans stratification)."""

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
        N0: int = 1024,
        theta: float = 0.25,
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

    def _pilot_cluster(self, level: int):
        rng = np.random.default_rng()
        Y_arr, details, cost_pilot = self._f(level, self.N0, return_details=True, rng=rng)
        sums_pilot = np.array([Y_arr.sum(), (Y_arr**2).sum()])
        feats = np.vstack([self.feature_fn(d) for d in details])
        kmeans = KMeans(n_clusters=self.n_clusters).fit(feats)
        labels = kmeans.labels_
        V_cluster = np.array([
            np.var(Y_arr[labels == c], ddof=1) if np.any(labels == c) else 0.0
            for c in range(self.n_clusters)
        ])
        return sums_pilot, cost_pilot, V_cluster, kmeans, feats, labels

    def estimate(self, eps: float):
        alpha = max(0, self.alpha0)
        beta  = max(0, self.beta0)
        gamma = max(0, self.gamma0)
        L = self.Lmin

        Nl    = np.zeros(L+1)
        suml  = np.zeros((2, L+1))
        costl = np.zeros(L+1)

        self.V_cluster_levels = [None] * (self.Lmax + 1)
        self.kmeans_levels = [None] * (self.Lmax + 1)
        self._cluster_features = [None] * (self.Lmax + 1)
        self._cluster_labels = [None] * (self.Lmax + 1)

        for l in range(L+1):
            sums_p, cost_p, V_c, km, feats, labs = self._pilot_cluster(l)
            Nl[l] = self.N0
            suml[:, l] = sums_p
            costl[l] = cost_p
            self.V_cluster_levels[l] = V_c
            self.kmeans_levels[l] = km
            self._cluster_features[l] = feats
            self._cluster_labels[l] = labs

        while True:
            ml = np.abs(suml[0] / Nl)
            Vl_total = np.array([self.V_cluster_levels[l].sum() for l in range(L+1)])
            Cl = costl / Nl

            ml_safe = np.maximum(ml[1:], 1e-12)
            Vl_safe = np.maximum(Vl_total[1:], 1e-12)

            for l in range(2, L+1):
                ml[l] = max(ml[l], 0.5 * ml[l-1] / 2**alpha)
                Vl_total[l] = max(Vl_total[l], 0.5 * Vl_total[l-1] / 2**beta)

            if self.alpha0 <= 0 and L >= 2:
                A = np.vstack([np.arange(1, L+1), np.ones(L)]).T
                alpha = max(0.5, -np.linalg.lstsq(A, np.log2(ml_safe), rcond=None)[0][0])
            if self.beta0 <= 0 and L >= 2:
                A = np.vstack([np.arange(1, L+1), np.ones(L)]).T
                beta = max(0.5, -np.linalg.lstsq(A, np.log2(Vl_safe), rcond=None)[0][0])
            if self.gamma0 <= 0 and L >= 2:
                A = np.vstack([np.arange(1, L+1), np.ones(L)]).T
                gamma = max(0.5, np.linalg.lstsq(A, np.log2(Cl[1:]), rcond=None)[0][0])

            Ns = np.ceil(np.sqrt(Vl_total/Cl) * np.sum(np.sqrt(Vl_total*Cl)) / ((1 - self.theta) * eps**2))
            dNl = np.maximum(0, Ns - Nl)

            if (dNl > 0.01 * Nl).sum() == 0:
                tail_range = np.array([L - i for i in range(min(3, L+1))])
                extrapolated = ml[tail_range] * 2**(np.arange(len(tail_range)) * alpha)
                remainder = np.max(extrapolated) / (2**alpha - 1)
                print(f"[Bias Check] L={L}, extrapolated remainder={remainder:.4e}, threshold={np.sqrt(self.theta)*eps:.4e}")
                if remainder > np.sqrt(self.theta) * eps:
                    if L == self.Lmax:
                        raise RuntimeError("Increase Lmax to reach tolerance")
                    L += 1
                    Nl = np.append(Nl, 0.0)
                    suml = np.column_stack([suml, [0.0, 0.0]])
                    costl = np.append(costl, 0.0)
                    self.V_cluster_levels.append(None)
                    self.kmeans_levels.append(None)
                    self._cluster_features.append(None)
                    self._cluster_labels.append(None)
                    sums_p, cost_p, V_c, km, feats, labs = self._pilot_cluster(L)
                    Nl[L] = self.N0
                    suml[:, L] = sums_p
                    costl[L] = cost_p
                    self.V_cluster_levels[L] = V_c
                    self.kmeans_levels[L] = km
                    self._cluster_features[L] = feats
                    self._cluster_labels[L] = labs
                    continue
                break

            dNlc = {}
            for l in range(L+1):
                n_extra = int(dNl[l])
                if n_extra <= 0:
                    continue
                V_c = self.V_cluster_levels[l]
                if V_c.sum() == 0:
                    weights = np.ones(self.n_clusters) / self.n_clusters
                else:
                    weights = V_c / V_c.sum()
                alloc = np.floor(n_extra * weights).astype(int)
                alloc[-1] += n_extra - alloc.sum()
                dNlc[l] = alloc

            for l, alloc in dNlc.items():
                km = self.kmeans_levels[l]
                for c, n_c in enumerate(alloc):
                    if n_c <= 0:
                        continue
                    accepted = 0
                    sums = np.zeros(2, float)
                    cost = 0.0
                    while accepted < n_c:
                        batch = min(n_c - accepted, self.N0)
                        rng = np.random.default_rng()
                        Y_arr, details, cost_b = self._f(l, batch, return_details=True, rng=rng)
                        feats = np.vstack([self.feature_fn(d) for d in details])
                        labs = km.predict(feats)
                        for Yi, lab in zip(Y_arr, labs):
                            if lab == c and accepted < n_c:
                                sums[0] += Yi
                                sums[1] += Yi**2
                                accepted += 1
                        cost += cost_b
                    Nl[l] += n_c
                    suml[:, l] += sums
                    costl[l] += cost

        price = np.sum(suml[0] / Nl)
        return price, Nl.astype(int), Cl, costl.sum()

    def get_clusters(self):
        if self.kmeans_levels is None:
            raise ValueError("No clustering information available. Run estimate() first.")
        clusters_info = {}
        for lvl, km in enumerate(self.kmeans_levels):
            if km is not None:
                clusters_info[lvl] = {
                    'model': km,
                    'features': self._cluster_features[lvl],
                    'labels': self._cluster_labels[lvl]
                }
        return clusters_info
