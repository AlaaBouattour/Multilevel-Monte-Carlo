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
        Lmin=2,
        Lmax=10,
        alpha0=0.0,
        beta0=0.0,
        gamma0=0.0,
        n_clusters=4,
        N0=1000,
        theta=0.25,
        scale_features=True,
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

        self.kmeans_levels = [None] * (self.Lmax + 1)
        self.cluster_probs = [None] * (self.Lmax + 1)
        self.cluster_vars = [None] * (self.Lmax + 1)
        self.cluster_sums = [None] * (self.Lmax + 1)
        self.cluster_counts = [None] * (self.Lmax + 1)

    def estimate(self, eps: float):
        alpha = max(0, self.alpha0)
        beta = max(0, self.beta0)
        gamma = max(0, self.gamma0)
        L = self.Lmin

        Nl = np.zeros(L + 1)
        suml = np.zeros((2, L + 1))
        costl = np.zeros(L + 1)

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
            labels = km.labels_

            Vc, Pc, sums, counts = [], [], [], []
            for c in range(self.n_clusters):
                mask = labels == c
                vals = Y_arr[mask]
                if vals.size > 0:
                    Vc.append(np.var(vals, ddof=1))
                    Pc.append(len(vals) / len(Y_arr))
                    sums.append(vals.sum())
                    counts.append(len(vals))
                else:
                    Vc.append(1e-6)
                    Pc.append(1e-6)
                    sums.append(0.0)
                    counts.append(0)

            Pc = np.array(Pc)
            Pc /= Pc.sum()
            Vc = np.clip(Vc, 1e-6, None)

            suml[0, l] = np.sum(sums)
            suml[1, l] = np.sum(Y_arr ** 2)
            Nl[l] = self.N0
            costl[l] = cost

            self.kmeans_levels[l] = km
            self.cluster_probs[l] = Pc
            self.cluster_vars[l] = Vc
            self.cluster_sums[l] = np.array(sums)
            self.cluster_counts[l] = np.array(counts)

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
                tail = np.array([L - i for i in range(min(3, L + 1))])
                extrapolated = ml[tail] * 2 ** (np.arange(len(tail)) * alpha)
                remainder = np.max(extrapolated) / (2 ** alpha - 1)
                if remainder > np.sqrt(self.theta) * eps:
                    if L == self.Lmax:
                        raise RuntimeError("Increase Lmax to reach tolerance")
                    L += 1
                    Nl = np.append(Nl, 0.0)
                    suml = np.column_stack([suml, [0.0, 0.0]])
                    costl = np.append(costl, 0.0)

                    Y_arr, details, cost = self._f(L, self.N0, return_details=True)
                    feats = np.vstack([self.feature_fn(d) for d in details])
                    if self.scale_features:
                        feats = StandardScaler().fit_transform(feats)
                    km = KMeans(n_clusters=self.n_clusters).fit(feats)
                    labels = km.labels_

                    sums, counts, Vc, Pc = [], [], [], []
                    for c in range(self.n_clusters):
                        mask = labels == c
                        vals = Y_arr[mask]
                        if vals.size > 0:
                            Vc.append(np.var(vals, ddof=1))
                            Pc.append(len(vals) / len(Y_arr))
                            sums.append(vals.sum())
                            counts.append(len(vals))
                        else:
                            Vc.append(1e-6)
                            Pc.append(1e-6)
                            sums.append(0.0)
                            counts.append(0)

                    Pc = np.array(Pc)
                    Pc /= Pc.sum()
                    Vc = np.clip(Vc, 1e-6, None)

                    suml[0, L] = np.sum(sums)
                    suml[1, L] = np.sum(Y_arr ** 2)
                    Nl[L] = self.N0
                    costl[L] = cost

                    self.kmeans_levels[L] = km
                    self.cluster_probs[L] = Pc
                    self.cluster_vars[L] = Vc
                    self.cluster_sums[L] = np.array(sums)
                    self.cluster_counts[L] = np.array(counts)
                    continue
                break

            for l in range(L + 1):
                if dNl[l] <= 0:
                    continue
                n_add = int(dNl[l])
                if l == 0:
                    Y_arr, _, cost = self._f(0, n_add, return_details=True)
                    suml[0, 0] += Y_arr.sum()
                    suml[1, 0] += (Y_arr ** 2).sum()
                    Nl[0] += n_add
                    costl[0] += cost
                    continue

                km = self.kmeans_levels[l]
                Vc = self.cluster_vars[l]
                Pc = self.cluster_probs[l]
                weights = Pc * np.sqrt(Vc)
                weights /= weights.sum()
                alloc = np.floor(n_add * weights).astype(int)
                alloc[-1] += n_add - alloc.sum()

                needed = alloc.copy()
                collected = [[] for _ in range(self.n_clusters)]
                cost_total = 0

                for _ in range(30):
                    if needed.sum() == 0:
                        break
                    Y_arr, details, cost_b = self._f(l, self.N0, return_details=True)
                    feats = np.vstack([self.feature_fn(d) for d in details])
                    if self.scale_features and self.scalers[l] is not None:
                        feats = self.scalers[l].transform(feats)
                    labels = km.predict(feats)

                    for y, lab in zip(Y_arr, labels):
                        if needed[lab] > 0:
                            collected[lab].append(y)
                            needed[lab] -= 1
                    cost_total += cost_b

                if self.cluster_sums[l] is None:
                    self.cluster_sums[l] = np.zeros(self.n_clusters)
                    self.cluster_counts[l] = np.zeros(self.n_clusters, dtype=int)

                for c in range(self.n_clusters):
                    y = np.array(collected[c])
                    if y.size > 0:
                        self.cluster_sums[l][c] += y.sum()
                        self.cluster_counts[l][c] += len(y)

                total_new = sum(len(y) for y in collected)
                suml[0, l] += sum(np.sum(np.array(y)) for y in collected)
                suml[1, l] += sum(np.sum(np.square(np.array(y))) for y in collected)
                Nl[l] += total_new
                costl[l] += cost_total

        # === Final price (corrected version) ===
        price = suml[0, 0] / Nl[0]
        for l in range(1, L + 1):
            S_lc = self.cluster_sums[l]
            N_lc = self.cluster_counts[l]
            P_lc = self.cluster_probs[l]
            mask = N_lc > 0
            price += np.sum((S_lc[mask] / N_lc[mask]) * P_lc[mask])

        return price, Nl.astype(int), Cl, costl.sum()

    def get_cluster_info(self, level: int):
        if level < 0 or level > self.Lmax:
            raise ValueError(f"Level must be between 0 and {self.Lmax}.")
        if self.kmeans_levels[level] is None:
            raise ValueError("Run `.estimate()` first.")
        return {
            "centers": self.kmeans_levels[level].cluster_centers_,
            "probs": self.cluster_probs[level],
            "vars": self.cluster_vars[level],
            "sums": self.cluster_sums[level],
            "counts": self.cluster_counts[level],
        }
