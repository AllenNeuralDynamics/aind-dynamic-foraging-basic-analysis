"""
Utilities for signal quality metrics on 1D fluorescence traces.

This module provides:
- :func:`estimate_snr` — an SNR estimator using a derivative-based noise
  estimate and peak-based signal estimate.
- :func:`estimate_kurtosis` — excess kurtosis of the trace distribution.

Notes
-----
- The SNR function design was inspired by the AIND-OPhys SLAP2 team.
- Feed a dF/F preprocessed trace to :func:`estimate_snr`, as the peak
  height is interpreted from zero.
- Default sampling frequency (``fps``) is 20 Hz; adjust it if your data
  differ.
- NaNs are filled with the median of the trace prior to computation.

Example
-------
>>> import numpy as np
>>> t = np.linspace(0, 10, 200, dtype=float)  # 20 Hz sampling
>>> y = 0.1 * np.sin(2 * np.pi * 1.0 * t)     # small signal
>>> snr, noise, peaks = estimate_snr(y)       # doctest: +ELLIPSIS
>>> isinstance(snr, float) and isinstance(noise, float)
True
>>> isinstance(peaks, np.ndarray)
True
>>> isinstance(estimate_kurtosis(y), float)
True
"""

from __future__ import annotations

import warnings
from typing import Tuple

import numpy as np
from numpy.typing import NDArray
from scipy.signal import find_peaks
from scipy.stats import kurtosis

__all__ = ["estimate_snr", "estimate_kurtosis"]


def estimate_snr(
    trace: NDArray[np.floating], fps: float = 20.0
) -> Tuple[float, float, NDArray[np.intp]]:
    """
    Estimate the signal-to-noise ratio (SNR) of a 1D trace.

    Parameters
    ----------
    trace : numpy.ndarray
        1D input trace (e.g., dF/F). NaNs will be replaced with the
        median of ``trace`` before calculation.
    fps : float, optional
        Sampling frequency (frames per second), by default ``20.0``.

    Returns
    -------
    snr : float
        Estimated signal-to-noise ratio (dimensionless).
    noise : float
        Estimated noise level computed from the first difference of the trace
        (standard deviation of ``diff(trace)`` divided by ``sqrt(2)``).
    peaks : numpy.ndarray
        Indices of detected peaks in the trace.

    Notes
    -----
    - Noise is estimated from the derivative assuming white noise.
    - Signal is estimated from the 95th percentile of peak amplitudes.
    - Peak detection uses ``scipy.signal.find_peaks`` with sensible defaults.
    - If fewer than three peaks are found, ``snr`` and ``peaks`` are set to
      ``NaN`` and a :class:`warnings.WarningMessage` is issued.
    """
    # Replace NaNs with the median of the trace
    trace = np.nan_to_num(trace, nan=np.nanmedian(trace))

    # Noise estimation based on derivative, assuming random noise
    dfdt = np.diff(trace)
    noise = float(
        np.std(dfdt) / np.sqrt(2)
    )

    # Peak detection
    peaks, _ = find_peaks(
        trace,
        height=3 * noise,     # Minimum peak height (adjust for your scale)
        distance=fps * 0.1,   # Minimum number of samples between peaks
        prominence=0.05,      # How much a peak stands out relative to neighbors
        width=5,              # Optional: minimum width of peak
    )

    if len(peaks) < 3:
        warnings.warn(
            "Not enough peaks found to estimate SNR. Returning NaN values.",
            RuntimeWarning,
            stacklevel=2,
        )
        return float("nan"), noise, np.array(np.nan)

    # Signal estimate: 95th percentile of detected peak amplitudes
    amplitudes = np.sort(trace[peaks])
    signal = float(np.percentile(amplitudes, 95))

    # Calculate SNR
    snr = float(signal / noise) if noise > 0 else float("inf")

    return snr, noise, peaks


def estimate_kurtosis(trace: NDArray[np.floating]) -> float:
    """
    Compute the **excess kurtosis** of a 1D trace distribution.

    Parameters
    ----------
    trace : numpy.ndarray
        1D input trace. NaNs will be replaced with the median of ``trace``.

    Returns
    -------
    float
        Excess kurtosis of the distribution (Fisher definition):
        - Normal distribution → 0.0
        - Leptokurtic         → positive
        - Platykurtic         → negative
    """
    # Replace NaNs with the median of the trace
    trace = np.nan_to_num(trace, nan=np.nanmedian(trace))

    # Excess kurtosis (normal distribution = 0)
    return float(kurtosis(trace, fisher=True, bias=False))
