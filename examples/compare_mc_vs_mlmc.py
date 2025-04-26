"""Compare plain MC with MLMC for a European call option."""

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import time, math, numpy as np, pandas as pd, matplotlib.pyplot as plt
from mlmc import MLMC
from mlmc.bs_model import make_bs_level_fn, bs_exact_call
from mlmc.payoffs import call_payoff

# ---- Setup Black-Scholes parameters ----
S0, K, r, sigma, T = 100, 100, 0.05, 0.2, 1.0
exact = bs_exact_call(S0, K, r, sigma, T)

payoff = lambda S: call_payoff(S, K)
level_fn = make_bs_level_fn(S0, r, sigma, T, payoff)
mlmc = MLMC(level_fn)

EPS = [0.05, 0.025, 0.0125, 0.00625, 0.003125, 0.003125/2,0.003125/4]
rows = []

for eps in EPS:
    # MLMC estimate
    t0 = time.perf_counter()
    price_ml, *_ = mlmc.estimate(eps)
    t_ml = time.perf_counter() - t0
    rows.append(("MLMC", eps, price_ml, t_ml))

    # Plain MC estimate
    Z_pilot = np.random.randn(50_000)
    ST_pilot = S0*np.exp((r-0.5*sigma**2)*T + sigma*math.sqrt(T)*Z_pilot)
    var_pay = np.var(math.exp(-r*T)*payoff(ST_pilot), ddof=1)
    N_mc = math.ceil(var_pay/eps**2)

    t0 = time.perf_counter()
    Z = np.random.randn(N_mc)
    ST = S0*np.exp((r-0.5*sigma**2)*T + sigma*math.sqrt(T)*Z)
    price_mc = math.exp(-r*T)*payoff(ST).mean()
    t_mc = time.perf_counter() - t0
    rows.append(("MC", eps, price_mc, t_mc))

# ---- Print results ----
print("\nComparison MC vs MLMC")
print(f"{'Method':<6} | {'eps':<8} | {'Estimate':<12} | {'Time (s)':<10}")
print("-"*46)

for method, eps, estimate, timing in rows:
    print(f"{method:<6} | {eps:<8.5f} | {estimate:<12.6f} | {timing:<10.4f}")

# ---- Plot error and time ----
eps_list = []
errors_mc = []
times_mc = []
errors_mlmc = []
times_mlmc = []

for method, eps, estimate, timing in rows:
    if method == "MC":
        eps_list.append(eps)
        errors_mc.append(abs(estimate - exact))
        times_mc.append(timing)
    else:  # MLMC
        errors_mlmc.append(abs(estimate - exact))
        times_mlmc.append(timing)

# Create plots
plt.figure(figsize=(10, 4))

# Plot error
plt.subplot(1, 2, 1)
plt.plot(eps_list, errors_mc, marker='o', label='MC')
plt.plot(eps_list, errors_mlmc, marker='o', label='MLMC')
plt.xlabel('ε (target RMS error)')
plt.ylabel('Actual error')
plt.title('Error vs ε')
plt.xscale('log')
plt.yscale('log')
plt.grid(True, which="both", ls="--")
plt.legend()

# Plot time
plt.subplot(1, 2, 2)
plt.plot(eps_list, times_mc, marker='o', label='MC')
plt.plot(eps_list, times_mlmc, marker='o', label='MLMC')
plt.xlabel('ε (target RMS error)')
plt.ylabel('Time (seconds)')
plt.title('Time vs ε')
plt.xscale('log')
plt.yscale('log')
plt.grid(True, which="both", ls="--")
plt.legend()

plt.tight_layout()
plt.show()
