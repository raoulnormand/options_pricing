"""
Analytic formulas for option pricing.
"""

import numpy as np
from scipy.stats import norm


def european_call_price(params, K):
    """
    Returns the price of a European call struck at K.

    Here params = dictionary with keys
    _ 'S0' = price of the underlying at t = 0
    _ 'r' = interest_rate
    _ 'sigma' = volatility
    _ 'T' = maturity
    """
    # Unpack parameters for simplicity
    S0 = params["S0"]
    r = params["r"]
    sigma = params["sigma"]
    T = params["T"]
    # Input the classical formula
    d1 = (np.log(S0 / K) + (r + sigma**2 / 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return S0 * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)


def barrier_call_price(params, K, H):
    """
    Returns the price of a down-and-out call option
    struck at K with barrier H < K.

    Here params = dictionary with keys
    _ 'S0' = price of the underlying at t = 0
    _ 'r' = interest_rate
    _ 'sigma' = volatility
    _ 'T' = maturity
    """
    # Unpack parameters for simplicity
    S0 = params["S0"]
    r = params["r"]
    sigma = params["sigma"]
    T = params["T"]
    # The formula is the European call price + a corrective term, see e.g. Joshi's book.
    h1 = (np.log(H**2 / (S0 * K)) + (r + sigma**2 / 2) * T) / (sigma * np.sqrt(T))
    h2 = h1 - sigma * np.sqrt(T)
    return (
        european_call_price(params, K)
        - (H / S0) ** (1 + 2 * r / sigma**2) * S0 * norm.cdf(h1)
        + (H / S0) ** (-1 + 2 * r / sigma**2) * K * np.exp(-r * T) * norm.cdf(h2)
    )
