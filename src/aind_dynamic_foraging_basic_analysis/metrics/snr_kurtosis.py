import numpy as np
import warnings
from scipy.signal import find_peaks
from scipy.stats import kurtosis


def estimate_snr(trace, fps):
    """
    Estimate the signal-to-noise ratio (SNR) of a trace.

    Parameters
    ----------
    trace : np.ndarray
        The input trace.
    fps : float
        Frames per second of the trace.

    Returns
    -------
    snr : float
        Estimated signal-to-noise ratio.
    noise : float
        Estimated noise level.
    peaks : np.ndarray
        Indices of detected peaks in the trace.
    """
    # Replace NaNs with the median of the trace
    trace = np.nan_to_num(trace, nan=np.nanmedian(trace))

    # Noise estimation based on derivative, assuming random noise
    dfdt = np.diff(trace)
    noise = np.std(dfdt) / np.sqrt(2)

    # Estimate signal as the third peak using scipy's find_peaks
    peaks, _ = find_peaks(
        trace,
        height=3 * noise,      # Minimum peak height (adjust based on your signal scale)
        distance=fps * 0.1,    # Minimum number of samples between peaks
        prominence=0.05,       # How much a peak stands out relative to neighbors
        width=5                # Optional: minimum width of peak
    )

    if len(peaks) < 3:
        # Warning if not enough peaks are found
        warnings.warn("Not enough peaks found to estimate SNR. Returning NaN values.")
        return np.nan, noise, np.nan

    # Take the 95th percentile of peak amplitudes as the signal
    amplitudes = np.sort(trace[peaks])
    signal = np.percentile(amplitudes, 95)

    # Calculate SNR
    snr = signal / noise

    return snr, noise, peaks


def estimate_kurtosis(trace):
    """
    Estimate the kurtosis of a trace distribution.

    Parameters
    ----------
    trace : np.ndarray
        The input trace.

    Returns
    -------
    kurt : float
        Estimated excess kurtosis of the distribution.
        (Normal distribution = 0, leptokurtic > 0, platykurtic < 0)
    """
    # Replace NaNs with the median of the trace
    trace = np.nan_to_num(trace, nan=np.nanmedian(trace))

    # Excess kurtosis (normal distribution = 0)
    kurt = kurtosis(trace, fisher=True, bias=False)

    return kurt
