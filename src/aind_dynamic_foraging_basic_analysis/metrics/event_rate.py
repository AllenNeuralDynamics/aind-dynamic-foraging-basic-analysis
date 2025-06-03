"""Compute an event-rate timeseries by convolving discrete events"""

import numpy as np
from scipy.signal import fftconvolve


def compute_event_rate(
    event_times,
    t_start=None,
    t_end=None,
    dt=1 / 20,  # matching photometry frame rate
    tau=25.0,  # to be optimised
    kernel="exp",  # exp/hyperbolic
    hyper_p=1.0,  # only for hyperbolic
    normalize_kernel=True,
):
    """
    Compute an event-rate timeseries by convolving discrete events
    (e.g. rewards) with a decay kernel.

    Parameters
    ----------
    event_times : (N,) array-like
        1-D array of event (e.g. reward) timestamps [s].
    t_start, t_end : float, optional
        Output time range [s].  Defaults: t_start=0, t_end=max(event_times).
    dt : float, default 0.001
        Time step of the output signal [s].
        If too small, computation gets too slow, now matching photometry frame rate
    tau : float, default 1.0
        Time constant of the kernel [s].
        A parameter to be optimised using behavior relationship etc.
    kernel : {"exp", "hyperbolic"}, default "exp"
        * "exp": k(t) = exp(-t / tau)                              (classic EWMA)
        * "hyperbolic": k(t) = (1 + t / tau)^{-hyper_p}            (hyperbolic discounting)
    hyper_p : float, default 1.0
        Power *p* of the hyperbolic kernel. Ignored if kernel="exp".
    normalize_kernel : bool, default True
        If True, scale the kernel so its integral equals 1.

    Returns
    -------
    t : (T,) ndarray
        Time axis [s].
    e_rate : (T,) ndarray
        event rate (events · s⁻¹).

    Notes
    -----
    * Kernel is truncated at 5 × tau to keep computation fast.
    * fftconvolve gives O(N log N) performance.
    """

    event_times = np.asarray(event_times, dtype=float)
    if event_times.ndim != 1:
        raise ValueError("event_times must be a 1-D array of timestamps.")

    # ------------------------------------------------------------------
    # Time axis: start at 0 so the series is zero until the first reward
    # ------------------------------------------------------------------
    if t_start is None:
        t_start = 0.0
    if t_end is None:
        t_end = event_times.max()
    t = np.arange(t_start, t_end + dt, dt)

    # ------------------------------------------------------------------
    # Binary event series aligned to the timeline
    # ------------------------------------------------------------------
    event_series = np.zeros_like(t)
    idx = np.searchsorted(t, event_times - 1e-12)
    idx = idx[(idx >= 0) & (idx < len(t))]
    event_series[idx] = 1.0

    # ------------------------------------------------------------------
    # Build the decay kernel
    # ------------------------------------------------------------------
    t_kernel = np.arange(0, 5 * tau, dt)

    if kernel == "exp":
        k = np.exp(-t_kernel / tau)

    elif kernel == "hyperbolic":
        if hyper_p <= 0:
            raise ValueError("hyper_p must be > 0 for a valid hyperbolic kernel.")
        k = (1.0 + t_kernel / tau) ** (-hyper_p)

    else:
        raise ValueError("kernel must be 'exp' or 'hyperbolic'.")

    if normalize_kernel:
        k /= k.sum() * dt  # area = 1

    # ------------------------------------------------------------------
    # Convolution (truncate to match timeline length)
    # ------------------------------------------------------------------
    e_rate = fftconvolve(event_series, k, mode="full")[: len(t)]

    return t, e_rate
