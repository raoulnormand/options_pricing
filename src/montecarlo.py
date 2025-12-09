"""
Helper functions to do Monte Carlo simulations.
"""

import numpy as np

# Functions to obtain samples


def get_samples_final_value(n_samples, params, antithetic=False, moment_matching=False):
    """
    Provides samples of the final price of a stock S_T under
    the risk-neutral measure. Note that n_samples can be an int or a tuple, where axes are as follows:
    _ axis 0 is used to estimate expectations;
    _ axis 1 is used to obtain different estimates of the expectation to compute a variance;
    _ axis 2 is to compute for different parameters.

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
    # Moment matching could be done on the log-normal distribution below, but this works well.
    if moment_matching:
        samples = (samples - np.mean(samples, axis=0)) / np.std(samples, axis=0)
    # Extract parameters for clarity
    S0 = params["S0"]
    r = params["r"]
    sigma = params["sigma"]
    T = params["T"]
    # Return the correct distribution.
    return S0 * np.exp((r - sigma**2 / 2) * T + sigma * np.sqrt(T) * samples)


def get_samples_trajectory(n_samples, params, antithetic=False, moment_matching=False):
    """
    Provides samples of the whole geometric Brownian motion (S_t) under the risk-neutral measure, with the interval [0, T] discretized. Note that n_samples should be a tuple with lenth at least 2, where axes are as follows:
    _ axis 0 is used to estimate expectations;
    _ axis 1 is the time axis (for trajectories).
    _ axis 2 is used to obtain different estimates of the expectation to compute a variance;
    _ axis 3 is to compute for different parameters.

    The other arguments are as above.
    """
    # Get standard normal samples
    samples = np.random.normal(loc=0, scale=1, size=n_samples)
    # Use variance reduction techniques if desired
    if antithetic:
        samples = np.concatenate((samples, -samples))
    # Add inital value of 0
    zeroes = np.zeros((samples.shape[0], 1, *samples.shape[2:]))
    samples = np.concat((zeroes, samples), axis=1)
    # Get Brownian motion samples
    BM_samples = np.cumsum(samples, axis=1)
    # Extract parameters for clarity
    S0 = params["S0"]
    r = params["r"]
    sigma = params["sigma"]
    T = params["T"]
    # Compute the drift
    drift = np.ones(samples.shape)
    drift[:, 0] = 0
    drift = np.cumsum(drift, axis=1)
    # Obtain the geometric BM by rescaling
    GBM_samples = np.exp(
        sigma * np.sqrt(T / n_samples[1]) * BM_samples
        + (r - sigma**2 / 2) * T / n_samples[1] * drift
    )
    # Use moment matching with geometric BM, as described in Art Owen's book, Ch.8. Note E(S_t) = e^(rt).
    if moment_matching:
        GBM_samples = (
            GBM_samples
            * np.exp(r * T / n_samples[1] * drift)
            / np.mean(GBM_samples, axis=0)
        )
    # Return the correct distribution
    return S0 * GBM_samples


# Functions to compute prices using MC


def MC_european_call_price(
    n_samples, params, K, antithetic=False, moment_matching=False
):
    """
    Computes the price a European call option struck at K, using
    Monte Carlo.

    Same arguments as above.
    """
    # Get samples of the final value S_T
    samples = get_samples_final_value(n_samples, params, antithetic, moment_matching)
    # Compute f(S_T) for the pay_off f
    samples = np.maximum(samples - K, 0)
    # Return the average, which gives an estimator of the expectation.
    return np.exp(-params["r"] * params["T"]) * np.mean(samples, axis=0)


def MC_barrier_call_price(
    n_samples, params, K, H, antithetic=False, moment_matching=False
):
    """
    Returns the price of a down-and-out call option
    struck at K with barrier H < K, using Monte Carlo.

    Same arguments as above.
    """
    # Get samples of the final value S_T
    samples = get_samples_trajectory(n_samples, params, antithetic, moment_matching)
    # Check if the barrier is hit
    hit_barrier = ~np.any(samples < H, axis=1)
    # Small trick: if hit, multiply payoff by False (0), otherwise by True (1)
    samples = hit_barrier * np.maximum(samples[:, -1, :] - K, 0)
    # Return the average, which gives an estimator of the expectation.
    return np.exp(-params["r"] * params["T"]) * np.mean(samples, axis=0)
