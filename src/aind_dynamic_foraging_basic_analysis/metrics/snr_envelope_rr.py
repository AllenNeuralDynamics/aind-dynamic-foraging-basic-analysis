"""
Envelope + rise-rate (Env+RR) signal-to-noise estimator for 1D
fluorescence (dF/F) traces.

This module provides :class:`EnvelopeRRSNR`, an alternative to a plain
derivative-based estimator (see e.g. ``estimate_snr`` in
``snr_kurtosis.py``). Rather than estimating noise from the sample-to-
sample derivative and signal from raw peak heights, it:

1. Tracks a valley-anchored **tonic envelope** beneath the trace, which
   is robust to slow drift/bleaching that would otherwise bias a
   derivative-based noise estimate.
2. Subtracts the tonic envelope to obtain a **phasic residual**.
3. Estimates the noise floor robustly from the below-median half of the
   residual (IQR-based sigma), which resists inflation by the phasic
   events themselves.
4. Detects candidate phasic peaks above ``peak_threshold_sd`` and
   refines them with a **rise-rate gate** that rejects slow drifts and
   keeps only fast-onset transients.
5. Reports SNR as a suprathreshold peak-amplitude statistic (median or
   95th percentile) divided by the estimated noise floor.

This is the "Env+RR" method benchmarked against a derivative-based
estimator across a synthetic SNR sweep in
``BWNM_Signal_Quality_Benchmark.ipynb``. That benchmark found Env+RR
detects events at least as well as the derivative method, but with a
small, consistent linear bias in its SNR estimate at high true SNR;
:class:`EnvelopeRRSNR` can optionally apply the fitted linear correction
for that bias (see ``bias_correction`` below and
``fit_bias_correction_from_benchmark``).

Class API
---------
:class:`EnvelopeRRSNR`
    Stateful estimator: construct once with configuration, call
    :meth:`~EnvelopeRRSNR.fit` per trace to populate result attributes
    (``snr_``, ``noise_``, ``peaks_``, ``tonic_``, ``residual_``, ...),
    call :meth:`~EnvelopeRRSNR.estimate` for a one-shot functional
    interface returning ``(snr, noise, peaks)`` -- the same 3-tuple
    shape as a plain ``estimate_snr(trace, fps)`` function, for drop-in
    comparison -- or call :meth:`~EnvelopeRRSNR.decompose` for just the
    ``(tonic, phasic)`` component arrays.

Notes
-----
- Feed a dF/F preprocessed trace, as the peak height is interpreted
  from zero.
- Default sampling frequency (``fps``) is 20 Hz; adjust it if your data
  differ.
- NaNs are filled with the median of the trace prior to computation.
- This class wraps
  :func:`bwnm_signal_utils.estimate_snr_components_envelope`; see that
  module's ``ENVELOPE_CONFIG`` for the full set of tunable parameters
  (envelope smoothing windows, ALS baseline alternative, etc.), any of
  which can be overridden via the ``config`` argument here.

Example
-------
>>> import numpy as np
>>> from snr_envelope_rr import EnvelopeRRSNR
>>> rng = np.random.default_rng(0)
>>> t = np.arange(1200) / 20.0                      # 60 s @ 20 Hz
>>> y = 0.05 * np.sin(2 * np.pi * t / 40.0) + 0.01 * rng.standard_normal(1200)
>>> estimator = EnvelopeRRSNR(fps=20.0)
>>> result = estimator.fit(y)
>>> isinstance(result.snr, float) and isinstance(result.noise, float)
True
>>> isinstance(result.peaks, np.ndarray)
True
>>> snr, noise, peaks = estimator.estimate(y)        # one-shot form
>>> snr == result.snr
True
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple

import numpy as np
from numpy.typing import NDArray

from bwnm_signal_utils import ENVELOPE_CONFIG, estimate_snr_components_envelope

__all__ = ["EnvelopeRRSNR", "EnvelopeRRResult"]

# Tuned defaults at the reference frame rate below. peak_threshold_sd is a
# sigma multiplier (unitless, does not scale with fps). The other three are
# window sizes in *samples*; scale_window() rescales them to preserve their
# real-world duration at a different fps.
_REFERENCE_FPS = 20.0
_TUNED_DEFAULTS = {
    "peak_threshold_sd":   1.5,
    "lower_min_distance":  20,   # 1.00 s at 20 fps
    "lower_smooth_window": 11,   # 0.55 s at 20 fps
    "rise_window":         5,    # 0.25 s at 20 fps
}


@dataclass
class EnvelopeRRResult:
    """Fitted result of :meth:`EnvelopeRRSNR.fit`.

    Attributes
    ----------
    snr : float
        Estimated signal-to-noise ratio (dimensionless), using
        ``signal_statistic`` on the suprathreshold phasic peak
        amplitudes, before any bias correction.
    snr_corrected : float or None
        ``snr`` after applying ``bias_correction`` (``None`` if no
        correction was configured).
    noise : float
        Estimated noise floor (IQR-based sigma of the tonic-subtracted
        residual).
    peaks : numpy.ndarray
        Indices of detected phasic event peaks (rise-rate gated).
    tonic : numpy.ndarray
        The fitted tonic (slow baseline) envelope, same length as the
        input trace.
    residual : numpy.ndarray
        Tonic-subtracted residual (``trace - tonic``).
    signal : float
        Suprathreshold peak-amplitude statistic used as the numerator
        of ``snr`` (median or 95th percentile, per
        ``signal_statistic``).
    config : dict
        The resolved configuration used for this fit (defaults merged
        with any overrides).
    """

    snr:           float
    noise:         float
    peaks:         NDArray[np.intp]
    tonic:         NDArray[np.floating]
    residual:      NDArray[np.floating]
    signal:        float
    config:        Dict = field(repr=False)
    snr_corrected: Optional[float] = None


class EnvelopeRRSNR:
    """Envelope + rise-rate SNR estimator for 1D fluorescence traces.

    Parameters
    ----------
    fps : float, optional
        Sampling frequency (frames per second), by default ``20.0``.
    peak_threshold_sd : float, optional
        Detection threshold for candidate phasic peaks, in units of the
        estimated noise sigma. Defaults to ``1.5``, a value tuned by
        grid search in ``BWNM_Signal_Quality_Benchmark.ipynb``. This is
        a unitless sigma multiplier, so it does *not* scale with
        ``fps``.
    signal_statistic : {'median', 'p95'}, optional
        Which statistic of the suprathreshold peak amplitudes to use as
        the SNR numerator, by default ``'median'``.
    bias_correction : tuple of (slope, intercept), optional
        If given, ``snr_corrected = (snr_raw - intercept) / slope`` is
        computed on every fit. Fit this once against ground-truth SNR
        (e.g. from a synthetic benchmark sweep) with
        :meth:`fit_bias_correction_from_benchmark` or
        ``numpy.polyfit(true_snr, snr_est, 1)``, then reuse it here to
        correct a consistent linear bias without re-running the
        benchmark on new data.
    config : dict, optional
        Overrides merged on top of the tuned window defaults below (which
        are themselves merged onto
        :data:`bwnm_signal_utils.ENVELOPE_CONFIG`). Anything not
        overridden here uses the tuned/scaled value; see that module for
        the full parameter list.

    Notes
    -----
    - Noise is estimated from the IQR of the tonic-subtracted residual,
      not the raw derivative, so it is robust to both slow drift and
      the phasic events themselves.
    - Signal is estimated from suprathreshold, rise-rate-gated peak
      amplitudes in the residual.
    - If fewer than one phasic peak is found, ``snr`` is ``NaN`` and a
      :class:`RuntimeWarning` is issued (mirroring the behaviour of a
      plain derivative-based estimator when too few peaks are found).
    - **Window defaults scale with fps.** ``lower_min_distance``,
      ``lower_smooth_window``, and ``rise_window`` are sample counts,
      tuned at 20 fps (1.00 s, 0.55 s, and 0.25 s respectively). At any
      other ``fps`` they are automatically rescaled by
      :meth:`scale_window` to preserve those durations, then rounded to
      the nearest integer (``lower_smooth_window`` is additionally
      forced odd, as required by the underlying Savitzky-Golay smoothing
      step). Pass an explicit value in ``config`` to opt out of scaling
      for a given parameter.
    """

    def __init__(
        self,
        fps: float = 20.0,
        peak_threshold_sd: float = _TUNED_DEFAULTS["peak_threshold_sd"],
        signal_statistic: str = "median",
        bias_correction: Optional[Tuple[float, float]] = None,
        config: Optional[Dict] = None,
    ) -> None:
        if signal_statistic not in ("median", "p95"):
            raise ValueError(
                f"signal_statistic must be 'median' or 'p95', got {signal_statistic!r}."
            )
        self.fps               = fps
        self.peak_threshold_sd = peak_threshold_sd
        self.signal_statistic  = signal_statistic
        self.bias_correction   = bias_correction

        scaled_defaults = {
            "lower_min_distance":  self.scale_window(
                _TUNED_DEFAULTS["lower_min_distance"], fps),
            "lower_smooth_window": self.scale_window(
                _TUNED_DEFAULTS["lower_smooth_window"], fps, make_odd=True),
            "rise_window":         self.scale_window(
                _TUNED_DEFAULTS["rise_window"], fps),
        }
        self.config = {
            **ENVELOPE_CONFIG,
            "fps": fps,
            "peak_threshold_sd": peak_threshold_sd,
            **scaled_defaults,
            **(config or {}),
        }

        # populated by fit()
        self.result_: Optional[EnvelopeRRResult] = None

    @staticmethod
    def scale_window(
        base_samples: int,
        fps: float,
        reference_fps: float = _REFERENCE_FPS,
        make_odd: bool = False,
    ) -> int:
        """Rescale a sample-count window to a new fps, preserving duration.

        Parameters
        ----------
        base_samples : int
            Window size in samples at ``reference_fps``.
        fps : float
            Target sampling frequency.
        reference_fps : float, optional
            The fps ``base_samples`` was tuned/specified at, by default
            ``20.0``.
        make_odd : bool, optional
            If True, increment the result by 1 if it comes out even
            (required by e.g. Savitzky-Golay smoothing windows).

        Returns
        -------
        int
            ``round(base_samples * fps / reference_fps)``, floored at 1.

        Example
        -------
        >>> EnvelopeRRSNR.scale_window(20, fps=20.0)
        20
        >>> EnvelopeRRSNR.scale_window(20, fps=40.0)
        40
        >>> EnvelopeRRSNR.scale_window(11, fps=40.0, make_odd=True)
        23
        """
        n = int(round(base_samples * fps / reference_fps))
        n = max(1, n)
        if make_odd and n % 2 == 0:
            n += 1
        return n

    # ------------------------------------------------------------------
    # Core API
    # ------------------------------------------------------------------

    def fit(self, trace: NDArray[np.floating]) -> EnvelopeRRResult:
        """Decompose ``trace`` and estimate its SNR.

        Parameters
        ----------
        trace : numpy.ndarray
            1D input trace (e.g. dF/F). NaNs are replaced with the
            median of ``trace`` before calculation.

        Returns
        -------
        EnvelopeRRResult
            Also stored on ``self.result_`` for later access via the
            convenience properties (``self.snr_``, ``self.noise_``,
            ``self.peaks_``, etc.).
        """
        trace = np.nan_to_num(trace, nan=float(np.nanmedian(trace)))

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            raw = estimate_snr_components_envelope(
                trace,
                detection_method="rise_rate",
                peak_threshold_sd=self.peak_threshold_sd,
                config=self.config,
            )

        snr_key = "phasic_snr_p95" if self.signal_statistic == "p95" else "phasic_snr_median"
        sig_key = "phasic_p95"     if self.signal_statistic == "p95" else "phasic_median"

        peaks = np.asarray(raw["event_maxima"], dtype=int)
        snr   = float(raw[snr_key])
        noise = float(raw["noise_sigma"])
        signal = float(raw[sig_key])

        if len(peaks) == 0:
            warnings.warn(
                "No phasic peaks detected above threshold. Returning NaN for snr.",
                RuntimeWarning,
                stacklevel=2,
            )
            snr = float("nan")

        snr_corrected = None
        if self.bias_correction is not None and not np.isnan(snr):
            slope, intercept = self.bias_correction
            snr_corrected = float((snr - intercept) / slope)

        self.result_ = EnvelopeRRResult(
            snr=snr,
            snr_corrected=snr_corrected,
            noise=noise,
            peaks=peaks,
            tonic=raw["tonic"],
            residual=raw["residual"],
            signal=signal,
            config=raw["config"],
        )
        return self.result_

    def estimate(
        self, trace: NDArray[np.floating]
    ) -> Tuple[float, float, NDArray[np.intp]]:
        """One-shot functional interface: ``fit`` and return a 3-tuple.

        Mirrors the ``(snr, noise, peaks)`` return signature of a plain
        derivative-based ``estimate_snr(trace, fps)`` function, so this
        class can be dropped in wherever that function is used. Returns
        the bias-corrected SNR if ``bias_correction`` was configured,
        otherwise the raw SNR.

        Parameters
        ----------
        trace : numpy.ndarray
            1D input trace (e.g. dF/F).

        Returns
        -------
        snr : float
        noise : float
        peaks : numpy.ndarray
        """
        result = self.fit(trace)
        snr = result.snr_corrected if result.snr_corrected is not None else result.snr
        return snr, result.noise, result.peaks

    def decompose(
        self, trace: Optional[NDArray[np.floating]] = None
    ) -> Tuple[NDArray[np.floating], NDArray[np.floating]]:
        """Return the (tonic, phasic) decomposition of a trace.

        Convenience wrapper around :meth:`fit` for callers who only want
        the two component arrays rather than the full SNR estimate.
        "Phasic" here is the tonic-subtracted residual (``trace -
        tonic``) — the same array as ``EnvelopeRRResult.residual`` /
        ``self.residual_`` — named ``phasic`` to match the signal/noise
        decomposition terminology used elsewhere (e.g.
        ``phasic_median``, ``phasic_snr_median``).

        Parameters
        ----------
        trace : numpy.ndarray, optional
            1D input trace (e.g. dF/F). If omitted, returns the
            decomposition from the most recent :meth:`fit` /
            :meth:`estimate` call instead of recomputing one; raises if
            this instance hasn't been fit yet.

        Returns
        -------
        tonic : numpy.ndarray
            The fitted tonic (slow baseline) envelope.
        phasic : numpy.ndarray
            The tonic-subtracted residual, i.e. the phasic component.

        Example
        -------
        >>> import numpy as np
        >>> rng = np.random.default_rng(0)
        >>> t = np.arange(1200) / 20.0
        >>> y = 0.05 * np.sin(2 * np.pi * t / 40.0) + 0.01 * rng.standard_normal(1200)
        >>> tonic, phasic = EnvelopeRRSNR(fps=20.0).decompose(y)
        >>> tonic.shape == y.shape and phasic.shape == y.shape
        True
        >>> np.allclose(phasic, y - tonic)
        True
        """
        if trace is not None:
            result = self.fit(trace)
        else:
            self._check_fitted()
            result = self.result_
        return result.tonic, result.residual

    # ------------------------------------------------------------------
    # Convenience read-only views onto the last fit
    # ------------------------------------------------------------------

    @property
    def snr_(self) -> float:
        """SNR from the most recent :meth:`fit` call (raw, uncorrected)."""
        self._check_fitted()
        return self.result_.snr

    @property
    def snr_corrected_(self) -> Optional[float]:
        """Bias-corrected SNR from the most recent :meth:`fit` call."""
        self._check_fitted()
        return self.result_.snr_corrected

    @property
    def noise_(self) -> float:
        """Noise floor from the most recent :meth:`fit` call."""
        self._check_fitted()
        return self.result_.noise

    @property
    def peaks_(self) -> NDArray[np.intp]:
        """Detected phasic peak indices from the most recent :meth:`fit` call."""
        self._check_fitted()
        return self.result_.peaks

    @property
    def tonic_(self) -> NDArray[np.floating]:
        """Fitted tonic envelope from the most recent :meth:`fit` call."""
        self._check_fitted()
        return self.result_.tonic

    @property
    def residual_(self) -> NDArray[np.floating]:
        """Tonic-subtracted residual from the most recent :meth:`fit` call."""
        self._check_fitted()
        return self.result_.residual

    def _check_fitted(self) -> None:
        if self.result_ is None:
            raise RuntimeError(
                "This EnvelopeRRSNR instance has not been fit yet. "
                "Call `.fit(trace)`, `.estimate(trace)`, or `.decompose(trace)` first."
            )

    # ------------------------------------------------------------------
    # Bias-correction helper
    # ------------------------------------------------------------------

    @staticmethod
    def fit_bias_correction_from_benchmark(
        true_snr: NDArray[np.floating], snr_est: NDArray[np.floating]
    ) -> Tuple[float, float]:
        """Fit a linear bias correction from benchmark ground truth.

        Convenience wrapper around ``numpy.polyfit`` for the common case
        of correcting a consistent linear bias identified by sweeping
        known SNR levels (e.g. via
        ``BWNM_Signal_Quality_Benchmark.ipynb``, Figure 2).

        Parameters
        ----------
        true_snr : numpy.ndarray
            Ground-truth SNR values used in the benchmark sweep.
        snr_est : numpy.ndarray
            Corresponding ``EnvelopeRRSNR``-estimated SNR values.

        Returns
        -------
        (slope, intercept) : tuple of float
            Pass directly as ``bias_correction=(slope, intercept)`` to
            the constructor.

        Example
        -------
        >>> import numpy as np
        >>> true_snr = np.array([5.0, 10.0, 20.0, 40.0])
        >>> snr_est  = 0.9 * true_snr + 1.5   # simulated linear bias
        >>> slope, intercept = EnvelopeRRSNR.fit_bias_correction_from_benchmark(
        ...     true_snr, snr_est)
        >>> estimator = EnvelopeRRSNR(bias_correction=(slope, intercept))
        """
        true_snr = np.asarray(true_snr, dtype=float)
        snr_est  = np.asarray(snr_est,  dtype=float)
        ok = np.isfinite(true_snr) & np.isfinite(snr_est)
        slope, intercept = np.polyfit(true_snr[ok], snr_est[ok], 1)
        return float(slope), float(intercept)
