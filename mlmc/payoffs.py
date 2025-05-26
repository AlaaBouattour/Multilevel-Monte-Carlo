import numpy as np
__all__ = ["call_payoff", "put_payoff", "digital_payoff"]

def call_payoff(S, K):
    return np.maximum(S-K, 0.0)

def put_payoff(S, K):
    return np.maximum(K-S, 0.0)

def digital_payoff(S, K):
    return (S > K).astype(float)