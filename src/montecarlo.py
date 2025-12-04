"""
Helper functions to do Monte Carlo simulations.
"""

import numpy as np


def get_samples_final(n_samples, params, antithetic=False, moment_matching=False):
    """
    Provides n_samples samples of the final price of a stock S_T under
    the risk-neutral measure.

    Here params = dictionary with keys
    _ 'S0' = price at t = 0
    _ 'r' = interest_rate
    _ 'sigma' = volatility
    _ 'T' = final time (maturity for European options)

    Additionally, two extra variance reductions techniques can be usedL
    _ antithetic: adds the negative of each realization to the sample
    _ moment_matching: ensures the sample has mean 0 and variance 1.
    """
    # Get standard normal samples
    samples = np.random.normal(loc=0, scale=1, size=n_samples)
    # Use variance reduction techniques if desired
    if antithetic:
        samples = np.concatenate((samples, -samples))
    if moment_matching:
        # The centering does nothing if antithetic
        samples = (samples - np.mean(samples)) / np.std(samples)
    # Extract parameters for clarity
    S0 = params["S0"]
    r = params["r"]
    sigma = params["sigma"]
    T = params["T"]
    # Return the correct disribution.
    return S0 * np.exp((r - sigma**2 / 2) * T + sigma * np.sqrt(T) * samples)


def get_samples_trajectory(
    n_samples, n_intervals, params, antithetic=False, moment_matching=False
):
    """
    Provides n_samples samples of the whole geometric Brownian motion (S_t) under the risk-neutral measure, with the interval [0, T] discretized in
    n_intervals intervals.

    Same arguments as above.
    """
    # Get standard normal samples
    samples = np.random.normal(loc=0, scale=1, size=(n_samples, n_intervals))
    # Use variance reduction techniques if desired
    if antithetic:
        samples = np.concatenate((samples, -samples))
    if moment_matching:
        # The centering / rescaling should be done independently for each row.
        samples = (samples - np.mean(samples, axis=1)) / np.std(samples, axis=1)
    # Extract parameters for clarity
    S0 = params["S0"]
    r = params["r"]
    sigma = params["sigma"]
    T = params["T"]
    # The cumulative sum returns a Brownian motion, but we need to rescale.
    BM_samples = np.sqrt(T / n_intervals) * np.cumsum(samples, axis=1)
    # The drift is just a linear function, discretized n_intervals times
    drift = np.linspace(0, T, n_intervals)
    # Return the correct distribution
    return S0 * np.exp(sigma * BM_samples + (r - sigma**2 / 2) * drift)
