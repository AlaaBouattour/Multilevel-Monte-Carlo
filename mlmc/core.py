from __future__ import annotations
import numpy as np
import math
from sklearn.cluster import KMeans


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
                 alpha0: float = 0.0, beta0: float = 0.0, gamma0: float = 0.0, N0: int = 1024,
                 theta: float=0.25):
        if Lmin < 2 or Lmax < Lmin:
            raise ValueError("Need Lmin ≥ 2 and Lmax ≥ Lmin")
        self._f = sde_step_fn
        self.Lmin, self.Lmax = Lmin, Lmax
        self.alpha0, self.beta0, self.gamma0 = alpha0, beta0, gamma0
        self.N0 = N0
        self.theta = theta

    # ------------------------------------------------------------------
    def estimate(self, eps: float):
        """Return (price, Nl, Cl, total_cost) for the given RMS tolerance `eps`."""
        alpha = max(0, self.alpha0); beta = max(0, self.beta0); gamma = max(0, self.gamma0)
        L = self.Lmin
        Nl   = np.zeros(L+1)
        suml = np.zeros((2, L+1))
        costl = np.zeros(L+1)
        dNl  = self.N0*np.ones(L+1)

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

            # ——————————————————————————————————————
            # Prepare safe copies for the log-regression (avoid log2(0))
            eps_floor = 1e-12
            ml_safe = np.maximum(ml[1:], eps_floor)
            Vl_safe = np.maximum(Vl[1:], eps_floor)
            # ——————————————————————————————————————

            for l in range(3, L+2):  # guard against zero variance on small samples
                ml[l-1] = max(ml[l-1], 0.5*ml[l-2]/2**alpha)
                Vl[l-1] = max(Vl[l-1], 0.5*Vl[l-2]/2**beta)

            # 3) Regression for alpha, beta, gamma if needed
            if self.alpha0 <= 0 and L >= 2:
                A = np.vstack([np.arange(1, L+1), np.ones(L)]).T
                alpha = max(0.5, -np.linalg.lstsq(A, np.log2(ml_safe), rcond=None)[0][0])
            if self.beta0 <= 0 and L >= 2:
                A = np.vstack([np.arange(1, L+1), np.ones(L)]).T
                beta  = max(0.5, -np.linalg.lstsq(A, np.log2(Vl_safe), rcond=None)[0][0])
            if self.gamma0 <= 0 and L >= 2:
                A = np.vstack([np.arange(1, L+1), np.ones(L)]).T
                gamma = max(0.5,
                            np.linalg.lstsq(A, np.log2(Cl[1:]), rcond=None)[0][0])

            # 4) Optimal allocation
            Ns  = np.ceil(np.sqrt(Vl/Cl) * np.sum(np.sqrt(Vl*Cl)) / ((1-self.theta)*eps**2))
            dNl = np.maximum(0, Ns - Nl)

            # 5) Weak‑error check – add level if bias too large
            if (dNl > 0.01*Nl).sum() == 0:
                tail = list(range(min(3, L)))
                remainder = (np.max(ml[[L-x for x in tail]]/2**(np.array(tail)*alpha))
                             / (2**alpha - 1))
                if remainder > np.sqrt(self.theta)*eps:
                    if L == self.Lmax:
                        raise WeakConvergenceError("Increase Lmax to reach tolerance")
                    L += 1
                    Vl    = np.append(Vl, Vl[-1]/2**beta)
                    Nl    = np.append(Nl, 0.0)
                    suml  = np.column_stack([suml, [0.0, 0.0]])
                    Cl    = np.append(Cl, Cl[-1]*2**gamma)
                    costl = np.append(costl, 0.0)
                    Ns  = np.ceil(np.sqrt(Vl/Cl) * np.sum(np.sqrt(Vl*Cl)) / ((1-self.theta)*eps**2))
                    dNl = np.maximum(0, Ns - Nl)

        price = np.sum(suml[0]/Nl)
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

        # placeholders for clustering info
        self.V_cluster_levels = None
        self.kmeans_levels = None

    def _pilot_cluster(self, level: int):
        # Run N0 pilot samples with details
        Y_arr, details, cost_pilot = self._f(level, self.N0, return_details=True)
        # Bulk sums for pilot
        sums_pilot = np.array([Y_arr.sum(), (Y_arr**2).sum()])
        # Build feature matrix
        feats = np.vstack([self.feature_fn(d) for d in details])
        # Cluster features
        kmeans = KMeans(n_clusters=self.n_clusters).fit(feats)
        labels = kmeans.labels_
        # Per-cluster variance
        V_cluster = np.array([
            np.var(Y_arr[labels == c], ddof=1) if np.any(labels == c) else 0.0
            for c in range(self.n_clusters)
        ])
        return sums_pilot, cost_pilot, V_cluster, kmeans, feats, labels

    def estimate(self, eps: float):
        """Return (price, Nl, Cl, total_cost) for RMS tolerance eps."""
        # Initialization
        alpha = max(0, self.alpha0)
        beta  = max(0, self.beta0)
        gamma = max(0, self.gamma0)
        L = self.Lmin
        Nl   = np.zeros(L+1, dtype=float)
        suml = np.zeros((2, L+1), dtype=float)
        costl= np.zeros(L+1, dtype=float)
        dNl  = self.N0 * np.ones(L+1, dtype=float)

        # storage for pilot clustering info
        self.V_cluster_levels = [None] * (self.Lmax + 1)
        self.kmeans_levels   = [None] * (self.Lmax + 1)
        # also store pilot features and labels for visualization
        self._cluster_features = [None] * (self.Lmax + 1)
        self._cluster_labels   = [None] * (self.Lmax + 1)

        while dNl.sum() > 0:
            # 1) Pilot & fold into sums
            for l in range(L+1):
                if self.V_cluster_levels[l] is None:
                    sums_p, cost_p, V_c, km, feats, labs = self._pilot_cluster(l)
                    Nl[l]    += self.N0
                    suml[:,l]+= sums_p
                    costl[l] += cost_p
                    self.V_cluster_levels[l] = V_c
                    self.kmeans_levels[l]    = km
                    self._cluster_features[l] = feats
                    self._cluster_labels[l]   = labs

            # 2) Empirical stats
            ml = np.abs(suml[0] / Nl)
            Vl_total = np.array([self.V_cluster_levels[l].sum() for l in range(L+1)])
            Cl = costl / Nl

            # safe copies for regression
            eps_floor = 1e-12
            ml_safe = np.maximum(ml[1:], eps_floor)
            Vl_safe = np.maximum(Vl_total[1:], eps_floor)

            # clamping to theoretical decay
            for l in range(3, L+2):
                ml[l-1] = max(ml[l-1], 0.5 * ml[l-2] / 2**alpha)
                Vl_total[l-1] = max(Vl_total[l-1], 0.5 * Vl_total[l-2] / 2**beta)

            # 3) Regression for rates if needed
            if self.alpha0 <= 0 and L >= 2:
                A = np.vstack([np.arange(1, L+1), np.ones(L)]).T
                alpha = max(0.5, -np.linalg.lstsq(A, np.log2(ml_safe), rcond=None)[0][0])
            if self.beta0 <= 0 and L >= 2:
                A = np.vstack([np.arange(1, L+1), np.ones(L)]).T
                beta  = max(0.5, -np.linalg.lstsq(A, np.log2(Vl_safe), rcond=None)[0][0])
            if self.gamma0 <= 0 and L >= 2:
                A = np.vstack([np.arange(1, L+1), np.ones(L)]).T
                gamma = max(0.5, np.linalg.lstsq(A, np.log2(Cl[1:]), rcond=None)[0][0])

            # 4) Optimal total allocation per level
            Ns  = np.ceil(np.sqrt(Vl_total/Cl) * np.sum(np.sqrt(Vl_total*Cl))
                          / ((1-self.theta)*eps**2))
            dNl = np.maximum(0, Ns - Nl)

            # 5) Stratify extra samples by cluster
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

            # 6) Simulate extra samples cluster-wise
            for l, alloc in dNlc.items():
                km      = self.kmeans_levels[l]
                for c, n_c in enumerate(alloc):
                    if n_c <= 0:
                        continue
                    accepted = 0
                    sums = np.zeros(2, float)
                    cost = 0.0
                    while accepted < n_c:
                        batch = min(n_c-accepted, self.N0)
                        Y_arr, details, cost_b = self._f(l, batch, return_details=True)
                        feats = np.vstack([self.feature_fn(d) for d in details])
                        labs = km.predict(feats)
                        for Yi, lab in zip(Y_arr, labs):
                            if lab == c and accepted < n_c:
                                sums[0] += Yi
                                sums[1] += Yi**2
                                accepted += 1
                        cost += cost_b
                    Nl[l]      += n_c
                    suml[:,l]  += sums
                    costl[l]   += cost

            # 7) Weak-error check → possibly extend L
            if (dNl > 0.01* Nl).sum() == 0:
                tail = list(range(min(3, L)))
                remainder = (np.max(ml[[L-x for x in tail]]/2**(np.array(tail)*alpha))
                             / (2**alpha - 1))
                if remainder > math.sqrt(self.theta)*eps:
                    if L == self.Lmax:
                        raise WeakConvergenceError("Increase Lmax to reach tolerance")
                    # extend arrays
                    L += 1
                    Nl   = np.pad(Nl,   (0,1), 'constant')
                    suml = np.pad(suml, ((0,0),(0,1)), 'constant')
                    costl= np.pad(costl,(0,1), 'constant')
                    self.V_cluster_levels.append(None)
                    self.kmeans_levels.append(None)
                    self._cluster_features.append(None)
                    self._cluster_labels.append(None)
                    continue

        # Final price
        price = np.sum(suml[0] / Nl)
        return price, Nl.astype(int), Cl, costl.sum()

    def get_clusters(self):
        """
        Return clustering information for each level after estimate().
        Returns a dict:
          level -> {
            'model': KMeans instance,
            'features': array of pilot features,
            'labels': array of pilot labels
          }
        """
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
