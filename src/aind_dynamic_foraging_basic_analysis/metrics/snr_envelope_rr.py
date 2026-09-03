"""
Envelope + rise-rate (Env+RR) signal-to-noise estimator for 1D
fluorescence (dF/F) traces.

:class:`EnvelopeRRSNR` estimates SNR by:

1. Tracking a **tonic baseline** beneath the trace with arPLS
   (asymmetrically reweighted penalized least squares).
2. Subtracting it to get a **phasic residual**.
3. Estimating the noise floor from that residual (median-filtered
   residual + twice-trimmed scaled MAD).
4. Detecting phasic peaks above ``peak_threshold_sd`` and refining them
   with a rise-rate gate that rejects slow drifts.
5. Reporting **total SNR** as ``snr_tonic + snr_phasic``: ``snr_tonic``
   is the median peak-to-valley swing amplitude of the tonic curve
   divided by the noise floor, and ``snr_phasic`` is the median
   suprathreshold peak amplitude divided by the noise floor, optionally
   bias-corrected (``bias_correction``; fit against ``snr_phasic``
   specifically, see ``EnvelopeRRResult``).

Class API
---------
Construct once, call :meth:`~EnvelopeRRSNR.fit` per trace for the full
result (``snr_`` (total), ``snr_tonic_``, ``snr_phasic_``, ``noise_``,
``peaks_``, ``tonic_``, ``residual_``, ...), :meth:`~EnvelopeRRSNR.estimate`
for a one-shot ``(snr, noise, peaks)`` tuple (``snr`` is the total;
drop-in compatible with a plain ``estimate_snr(trace, fps)`` function),
:meth:`~EnvelopeRRSNR.estimate_components` for the one-shot
``(snr_total, snr_tonic, snr_phasic)`` breakdown, or
:meth:`~EnvelopeRRSNR.decompose` for just ``(tonic, phasic)``.

Notes
-----
- Feed a dF/F preprocessed trace (e.g., detrending, motion correction
should already be applied).
- Default ``fps`` is 20 Hz; NaNs are filled with the trace median.
- Needs only numpy/scipy.

Example
-------
>>> import numpy as np
>>> rng = np.random.default_rng(2)
>>> t = np.arange(1200) / 20.0
>>> y = 0.05 * np.sin(2 * np.pi * t / 40.0) + 0.01 * rng.standard_normal(1200)
>>> estimator = EnvelopeRRSNR(fps=20.0)
>>> result = estimator.fit(y)
>>> result.snr_tonic > 5  # true tonic_snr here is ~10
True
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple, Union

import numpy as np
from numpy.typing import NDArray
from scipy import ndimage, signal

__all__ = ["EnvelopeRRSNR", "EnvelopeRRResult"]

# Tuned defaults at the reference frame rate below; scale_window() rescales
# the window sizes to preserve their real-world duration at a different fps.
_REFERENCE_FPS = 20.0
_TUNED_PEAK_THRESHOLD_SD = 2.0  # sigma multiplier; unitless, does not scale with fps
_TUNED_RISE_WINDOW = 5  # 0.25 s at 20 fps
_DEFAULT_TONIC_ROBUST_MIN_DISTANCE = 100  # 5.0 s at 20 fps
_DEFAULT_MIN_SAMPLES = 62  # ~3.1 s at 20 fps; see EnvelopeRRSNR's min_samples docstring

# Linear bias correction fit against ground-truth SNR for the tuned defaults
# above: snr_corrected = (snr_raw - intercept) / slope. R^2=0.9462.
# Auto-applied as the constructor's default bias_correction only when
# peak_threshold_sd matches the value it was fit against; pass
# bias_correction=None to disable, or refit via
# fit_bias_correction_from_benchmark for a different peak_threshold_sd.
_TUNED_BIAS_CORRECTION: Tuple[float, float] = (1.8494, -0.5508)

_DEFAULT_CONFIG: Dict = {
    "arpls_lam": 1e7,
    "arpls_n_iter": 15,
    "arpls_ratio": 1e-6,
    "upper_min_distance": 3,  # min spacing enforced by find_peaks on the residual
}


class _UnsetType:
    """Sentinel distinguishing "bias_correction not specified" (auto-
    resolve) from an explicit ``bias_correction=None`` (disable)."""

    def __repr__(self) -> str:
        """Return a short, unambiguous marker for debugging/repr output."""
        return "<unset>"


_UNSET = _UnsetType()


# ======================================================================
# Private helpers
# ======================================================================


def _robust_std(x: NDArray[np.floating]) -> float:
    """Scaled median absolute deviation, assuming near-Gaussian noise."""
    if x.size == 0:
        return float("nan")
    med = float(np.median(x))
    return 1.4826 * float(np.median(np.abs(x - med)))


def _mad_noise_std(residual: NDArray[np.floating], filter_length: int = 31) -> float:
    """Robust noise sigma via a median-filtered residual + twice-trimmed
    scaled MAD.

    Median-filters the residual to remove any remaining slow baseline,
    takes what's left, trims positive-peak outliers, then trims any
    remaining outliers on either side, and returns the scaled MAD of
    that twice-trimmed remainder.

    Falls back to a less-trimmed robust std (rather than NaN) if a
    trimming step empties out completely -- this happens on short or
    heavily-anomalous windows, where the first trim's own scale estimate
    collapses to 0. Silently returning NaN there would poison any
    downstream aggregation (``np.median`` propagates a single NaN to
    the whole result).
    """
    residual = np.asarray(residual, dtype=np.float64)
    if np.any(np.isnan(residual)):
        return float("nan")
    baseline = ndimage.percentile_filter(residual, 50, size=filter_length, mode="reflect")
    noise = residual - baseline
    if noise.size == 0:
        return float("nan")
    filtered_0 = noise[noise < 1.5 * np.abs(noise.min())]
    rstd = _robust_std(filtered_0)
    filtered_1 = filtered_0[np.abs(filtered_0) < 2.5 * rstd] if rstd > 0 else np.array([])
    if filtered_1.size > 0:
        result = _robust_std(filtered_1)
        if result > 0:
            return result
    if rstd > 0:
        return rstd
    return _robust_std(noise)


def _arpls_baseline(y, lam: float = 1e7, n_iter: int = 15, ratio: float = 1e-6):
    """Asymmetrically Reweighted Penalized Least Squares (arPLS) smooth
    curve (Baek et al. 2015).

    The asymmetry weights are re-derived from the current residual's own
    below-baseline noise statistics at every iteration, instead of one
    fixed asymmetry chosen in advance. Iterates until the relative
    weight change drops below ``ratio`` or ``n_iter`` is reached.
    """
    from scipy import sparse
    from scipy.sparse.linalg import spsolve

    L = len(y)
    D = sparse.diags([1, -2, 1], [0, -1, -2], shape=(L, L - 2), dtype=float)
    D = lam * D.dot(D.transpose())
    w = np.ones(L)
    z = y.copy()
    for _ in range(n_iter):
        W = sparse.spdiags(w, 0, L, L)
        z = spsolve((W + D).tocsc(), w * y)
        d = y - z
        d_neg = d[d < 0]
        m = float(np.mean(d_neg)) if len(d_neg) else 0.0
        s = float(np.std(d_neg)) if len(d_neg) else 1e-12
        s = s if s > 1e-12 else 1e-12
        w_new = 1.0 / (1.0 + np.exp(np.clip(2 * (d - (2 * s - m)) / s, -500, 500)))
        if np.linalg.norm(w - w_new) / (np.linalg.norm(w) + 1e-12) < ratio:
            w = w_new
            break
        w = w_new
    return z


def _detect_peaks_rise_rate(residual, candidates, sigma, rise_window: int = 3):
    """Reject candidate peaks whose approach isn't fast enough to be a
    real transient onset (vs. a slow drift crossing threshold).

    Calibrates its slope threshold from a robust (MAD-based) scale
    estimate of the below-median ("noise-like") candidate slopes, not an
    ordinary standard deviation -- ordinary std is itself not robust: a
    single moderate anomaly elsewhere in the trace can spawn a few
    large-magnitude candidate slopes from its own decay tail, and even
    one or two of those landing in the "below-median" bucket can inflate
    an ordinary std by 2x+, silently suppressing genuine events'
    detection elsewhere in the trace.
    """
    if len(candidates) < 5:
        return candidates

    slopes = []
    for idx in candidates:
        lo = max(0, idx - rise_window)
        window = residual[lo:idx]
        if len(window) >= 2:
            slopes.append(float(np.mean(np.diff(window))))
        else:
            slopes.append(0.0)
    slopes = np.array(slopes)

    lower = slopes[slopes <= np.median(slopes)]
    if len(lower) >= 5:
        sigma_slope = 1.4826 * np.median(np.abs(lower - np.median(lower)))
        slope_thresh = max(sigma_slope * 3.0, sigma * 0.1)
    else:
        slope_thresh = np.percentile(slopes, 50)

    return candidates[slopes > slope_thresh]


def _robust_tonic_range(tonic: NDArray[np.floating], min_distance: int) -> Tuple[float, int]:
    """Median peak-to-valley swing amplitude of the tonic curve --
    mirrors how phasic amplitude is estimated (detect individual events,
    take the median across them) instead of ``ptp(tonic)``'s single
    global max-minus-min.

    Detects local maxima and minima in ``tonic``
    (``scipy.signal.argrelextrema``, ``order=min_distance`` -- the
    number of samples on each side a point must exceed to count as a
    local extremum), takes the absolute difference between each pair of
    temporally-adjacent extrema (each one a single peak-to-valley or
    valley-to-peak "swing"), and returns the median swing amplitude.

    A one-off contaminated region typically produces only one or two
    anomalous swings among many genuine ones, so the median has a real
    breakdown point -- unlike ``ptp(tonic)``, which *is* the
    contaminated region's own extreme value whenever that region happens
    to contain the global max or min.

    The trade-off: robustness here scales with how many genuine swings
    get detected, same as phasic's median scales with how many genuine
    events get detected. A short recording relative to the tonic's own
    drift period (few genuine cycles) gives the median little to work
    with, same limitation phasic would have with only a handful of true
    transients.

    Parameters
    ----------
    tonic : ndarray
        Fitted tonic curve.
    min_distance : int
        Minimum spacing (samples) between detected extrema -- should be
        well below the tonic's own characteristic drift period, but
        well above any remaining fast wiggle in the fitted curve.

    Returns
    -------
    tonic_range : float
        Median swing amplitude, or ``ptp(tonic)`` if fewer than 2
        extrema are detected (not enough structure for a meaningful
        median -- e.g. a very short trace or an almost perfectly flat
        tonic).
    n_swings : int
        Number of swings the median was computed over (0 if it fell
        back to ``ptp``).
    """
    peaks = signal.argrelextrema(tonic, np.greater, order=min_distance)[0]
    valleys = signal.argrelextrema(tonic, np.less, order=min_distance)[0]
    extrema_idx = np.sort(np.concatenate([peaks, valleys]))
    if len(extrema_idx) < 2:
        return float(np.ptp(tonic)), 0
    swings = np.abs(np.diff(tonic[extrema_idx]))
    return float(np.median(swings)), len(swings)


def _decompose_envelope_rr(x: NDArray[np.floating], cfg: Dict) -> Dict:
    """Tonic/phasic decomposition + rise-rate gated peak detection."""
    tonic = _arpls_baseline(
        x, lam=cfg["arpls_lam"], n_iter=cfg["arpls_n_iter"], ratio=cfg["arpls_ratio"]
    )
    residual = x - tonic

    sigma_fit = _mad_noise_std(residual)
    threshold = cfg["peak_threshold_sd"] * sigma_fit

    raw_peaks, _ = signal.find_peaks(residual, height=threshold, distance=cfg["upper_min_distance"])
    event_maxima = _detect_peaks_rise_rate(residual, raw_peaks, sigma_fit, cfg["rise_window"])
    event_maxima = np.asarray(event_maxima, dtype=int)
    event_maxima = event_maxima[(event_maxima >= 0) & (event_maxima < len(residual))]
    peak_amps = residual[event_maxima] if len(event_maxima) > 0 else np.array([])
    phasic_median = float(np.median(peak_amps)) if len(peak_amps) > 0 else 0.0

    tonic_range, n_tonic_swings = _robust_tonic_range(tonic, cfg["tonic_robust_min_distance"])

    return {
        "tonic": tonic,
        "residual": residual,
        "event_maxima": event_maxima,
        "peak_amps": peak_amps,
        "noise_sigma": float(sigma_fit),
        "peak_threshold": float(threshold),
        "tonic_range": tonic_range,
        "phasic_median": phasic_median,
        "phasic_snr_median": (float(phasic_median / sigma_fit) if sigma_fit > 0 else float("nan")),
        "tonic_snr_sd": float(tonic_range / sigma_fit) if sigma_fit > 0 else float("nan"),
        "n_tonic_swings": n_tonic_swings,
    }


@dataclass
class EnvelopeRRResult:
    """Fitted result of :meth:`EnvelopeRRSNR.fit`.

    Attributes
    ----------
    snr : float
        **Total** SNR: ``snr_tonic + snr_phasic``, before bias
        correction.
    snr_tonic : float
        Tonic-baseline SNR: the median peak-to-valley swing amplitude of
        the tonic curve divided by the noise floor. How much the slow
        baseline itself varies relative to the noise, independent of
        any phasic events.
    snr_phasic : float
        Phasic SNR: median suprathreshold phasic peak amplitude divided
        by the noise floor. This is what ``snr`` meant in earlier
        versions of this class, and is the quantity ``bias_correction``
        was actually fit against (see ``snr_phasic_corrected``).
    snr_phasic_corrected : float or None
        ``snr_phasic`` after applying ``bias_correction`` (``None`` if
        none configured, or if no phasic peaks were detected).
    snr_corrected : float or None
        **Total**, bias-corrected: ``snr_tonic + snr_phasic_corrected``
        (``None`` if ``bias_correction`` isn't configured). Since
        ``bias_correction`` was fit only against ``snr_phasic``, this is
        a derived combination, not itself a directly validated
        correction of the total.
    noise : float
        Estimated noise floor of the tonic-subtracted residual.
    peaks : numpy.ndarray
        Indices of detected phasic event peaks (rise-rate gated).
    tonic : numpy.ndarray
        Fitted tonic (slow baseline), same length as the input trace.
    residual : numpy.ndarray
        Tonic-subtracted residual (``trace - tonic``).
    signal : float
        Median suprathreshold peak amplitude -- ``snr_phasic``'s
        numerator.
    config : dict
        The resolved configuration used for this fit.
    n_tonic_swings : int
        Number of peak-to-valley swings the median tonic range was
        computed over (0 if it fell back to ``ptp`` -- fewer than 2
        extrema were detected; see :func:`_robust_tonic_range`).
    """

    snr: float
    snr_tonic: float
    snr_phasic: float
    noise: float
    peaks: NDArray[np.intp]
    tonic: NDArray[np.floating]
    residual: NDArray[np.floating]
    signal: float
    config: Dict = field(repr=False)
    snr_corrected: Optional[float] = None
    snr_phasic_corrected: Optional[float] = None
    n_tonic_swings: int = 0


class EnvelopeRRSNR:
    """Envelope + rise-rate SNR estimator for 1D fluorescence traces.

    Parameters
    ----------
    fps : float, optional
        Sampling frequency (frames per second), by default ``20.0``.
    peak_threshold_sd : float, optional
        Detection threshold for candidate phasic peaks, in units of the
        estimated noise sigma (unitless; does not scale with ``fps``).
        Defaults to ``2.0``, grid-search-optimal (score=0.762).
    bias_correction : tuple of (slope, intercept), or None, optional
        If given, ``snr_phasic_corrected = (snr_phasic - intercept) /
        slope`` is computed on every fit -- this was fit against
        ``snr_phasic`` specifically (the only quantity ever validated
        against ground truth), not the total ``snr``; ``snr_corrected``
        (the corrected total) is then ``snr_tonic + snr_phasic_corrected``,
        a derived combination rather than an independently-validated
        correction. Defaults to a tuned fit
        (``slope=1.8494, intercept=-0.5508, R^2=0.9462``), but only
        auto-applies when ``peak_threshold_sd`` still matches the
        default it was fit against; change it and this silently
        resolves to ``None`` instead of misapplying a correction fit
        for a different threshold. Pass ``bias_correction=None``
        explicitly to disable it even at the tuned default. Fit your
        own via :meth:`fit_bias_correction_from_benchmark` or
        ``numpy.polyfit(true_snr, snr_est, 1)`` for other
        configurations -- each sits at its own scale.
    config : dict, optional
        Overrides merged on top of the fps-scaled tuned defaults
        (``rise_window``) and this module's fixed defaults (see
        ``_DEFAULT_CONFIG``): ``arpls_lam``/``arpls_n_iter``/
        ``arpls_ratio`` (tonic smoothness), ``upper_min_distance`` (min
        spacing enforced on candidate peaks), ``tonic_robust_min_distance``
        (default 100 samples at 20fps, fps-scaled -- minimum spacing
        between detected tonic extrema; see :func:`_robust_tonic_range`),
        ``min_samples`` (default 62 samples at 20fps, fps-scaled -- see
        below).

    Notes
    -----
    - Noise is estimated from the tonic-subtracted residual, not the
      raw derivative, so it's robust to slow drift and phasic events.
    - If fewer than one phasic peak is found, ``snr_phasic`` is ``NaN``
      (and so is the total ``snr``, via propagation) and a
      ``RuntimeWarning`` is issued; ``snr_tonic`` is still computed.
    - **Window defaults scale with fps.** ``rise_window`` is tuned at
      20 fps and rescaled by :meth:`scale_window` to preserve
      real-world duration at other rates. Pass an explicit value in
      ``config`` to opt out.
    - **Empty and too-short traces never raise.** ``fit`` (and
      everything built on it -- ``estimate``, ``estimate_components``,
      ``decompose``) treats a trace with zero samples, or fewer than
      ``config["min_samples"]``, as un-fittable: rather than letting the
      arPLS baseline fit crash outright (its sparse difference matrix is
      degenerate below 2 samples) or silently returning a noise/tonic
      estimate computed from too little data to mean anything, it warns
      once via ``RuntimeWarning`` and returns an all-NaN
      :class:`EnvelopeRRResult`. ``min_samples`` defaults to 62 samples
      at 20 fps (fps-scaled elsewhere) -- twice ``_mad_noise_std``'s own
      50-sample median-filter width, the smallest of the module's fixed
      internal windows, so the floor is "the noise estimate's own
      smoothing kernel fits at least twice over," not an arbitrary
      round number. This is deliberately conservative: a pipeline
      processing many channels/sessions should get a clearly-flagged
      NaN for a too-short recording, not a plausible-looking number
      quietly fit from a handful of samples.
    """

    def __init__(
        self,
        fps: float = 20.0,
        peak_threshold_sd: float = _TUNED_PEAK_THRESHOLD_SD,
        bias_correction: Union[Tuple[float, float], None, "_UnsetType"] = _UNSET,
        config: Optional[Dict] = None,
    ) -> None:
        """Construct an estimator; see the class docstring for parameter details."""
        effective_peak_threshold_sd = (config or {}).get("peak_threshold_sd", peak_threshold_sd)

        if bias_correction is _UNSET:
            # Auto-apply the tuned correction only if peak_threshold_sd still
            # matches what it was fit against -- otherwise it'd silently
            # misapply a correction fit for a different threshold.
            matches_tuned_config = effective_peak_threshold_sd == _TUNED_PEAK_THRESHOLD_SD
            bias_correction = _TUNED_BIAS_CORRECTION if matches_tuned_config else None

        self.fps = fps
        self.peak_threshold_sd = effective_peak_threshold_sd
        self.bias_correction = bias_correction

        scaled_defaults = {
            "rise_window": self.scale_window(_TUNED_RISE_WINDOW, fps),
            "tonic_robust_min_distance": self.scale_window(_DEFAULT_TONIC_ROBUST_MIN_DISTANCE, fps),
            "min_samples": self.scale_window(_DEFAULT_MIN_SAMPLES, fps),
        }
        self.config = {
            **_DEFAULT_CONFIG,
            "peak_threshold_sd": peak_threshold_sd,
            **scaled_defaults,
            **(config or {}),
        }

        # arPLS's difference matrix is degenerate below 2 samples regardless
        # of any config -- never let a configured min_samples below that
        # silently re-open the crash this guard exists to prevent.
        self.config["min_samples"] = max(2, self.config["min_samples"])

        # populated by fit()
        self.result_: Optional[EnvelopeRRResult] = None

    @staticmethod
    def scale_window(base_samples: int, fps: float, reference_fps: float = _REFERENCE_FPS) -> int:
        """Rescale a sample-count window to a new fps, preserving duration.

        Returns ``round(base_samples * fps / reference_fps)``, floored
        at 1.

        Example
        -------
        >>> EnvelopeRRSNR.scale_window(20, fps=40.0)
        40
        """
        return max(1, int(round(base_samples * fps / reference_fps)))

    # ------------------------------------------------------------------
    # Core API
    # ------------------------------------------------------------------

    def _nan_result(self, n_input_samples: int, reason: str) -> EnvelopeRRResult:
        """Build the all-NaN :class:`EnvelopeRRResult` returned when a trace
        can't be fit (empty, or shorter than ``config["min_samples"]``).

        Shared by both cases so the "un-fittable" result always has the same
        shape -- callers checking ``np.isnan(result.snr)`` don't need to
        special-case *why* it's NaN.
        """
        warnings.warn(reason, RuntimeWarning, stacklevel=3)
        self.result_ = EnvelopeRRResult(
            snr=float("nan"),
            snr_tonic=float("nan"),
            snr_phasic=float("nan"),
            snr_corrected=None,
            snr_phasic_corrected=None,
            noise=float("nan"),
            peaks=np.array([], dtype=int),
            tonic=np.full(n_input_samples, np.nan, dtype=float),
            residual=np.full(n_input_samples, np.nan, dtype=float),
            signal=float("nan"),
            config=self.config,
            n_tonic_swings=0,
        )
        return self.result_

    def fit(self, trace: NDArray[np.floating]) -> EnvelopeRRResult:
        """Decompose ``trace`` and estimate its SNR.

        NaNs in ``trace`` are replaced with its median first. Returns
        an :class:`EnvelopeRRResult`, also stored on ``self.result_``
        for the convenience properties (``self.snr_``, etc.).

        Empty and too-short traces never raise -- see "Empty and
        too-short traces never raise" in the class docstring. Briefly:
        a trace with zero samples, or fewer than
        ``self.config["min_samples"]``, returns an all-NaN
        :class:`EnvelopeRRResult` and issues a ``RuntimeWarning``
        instead of being fit, so a pipeline processing many
        channels/sessions doesn't crash -- or silently get a
        statistically meaningless estimate -- from one short input.
        """
        trace = np.asarray(trace, dtype=float)
        n = trace.size

        if n == 0:
            return self._nan_result(
                n,
                "Empty trace passed to EnvelopeRRSNR.fit(); cannot estimate a "
                "tonic baseline or noise floor from zero samples. Returning "
                "NaN for all SNR/noise fields.",
            )

        min_samples = self.config["min_samples"]
        if n < min_samples:
            return self._nan_result(
                n,
                f"Trace has only {n} sample(s), below min_samples="
                f"{min_samples} (fps={self.fps}); a tonic/noise estimate "
                "from this few samples would not be reliable, and the "
                "arPLS baseline fit is undefined below 2 samples. "
                "Returning NaN for all SNR/noise fields. Pass a longer "
                "trace, or lower config['min_samples'] if you understand "
                "the reliability trade-off.",
            )

        trace = np.nan_to_num(trace, nan=float(np.nanmedian(trace)))

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            raw = _decompose_envelope_rr(trace, self.config)

        peaks = np.asarray(raw["event_maxima"], dtype=int)
        snr_phasic = float(raw["phasic_snr_median"])
        snr_tonic = float(raw["tonic_snr_sd"])
        noise = float(raw["noise_sigma"])
        signal_amplitude = float(raw["phasic_median"])

        if len(peaks) == 0:
            warnings.warn(
                "No phasic peaks detected above threshold. Returning NaN for snr_phasic.",
                RuntimeWarning,
                stacklevel=2,
            )
            snr_phasic = float("nan")

        snr_total = snr_tonic + snr_phasic  # NaN-propagates if snr_phasic is NaN

        snr_phasic_corrected = None
        snr_corrected = None
        if self.bias_correction is not None and not np.isnan(snr_phasic):
            # bias_correction was fit against snr_phasic specifically (the
            # only quantity ever validated against ground truth) -- see
            # EnvelopeRRResult's docstring for why snr_corrected (a total)
            # is therefore a derived combination, not itself validated.
            slope, intercept = self.bias_correction
            snr_phasic_corrected = float((snr_phasic - intercept) / slope)
            snr_corrected = snr_tonic + snr_phasic_corrected

        self.result_ = EnvelopeRRResult(
            snr=snr_total,
            snr_tonic=snr_tonic,
            snr_phasic=snr_phasic,
            snr_corrected=snr_corrected,
            snr_phasic_corrected=snr_phasic_corrected,
            noise=noise,
            peaks=peaks,
            tonic=raw["tonic"],
            residual=raw["residual"],
            signal=signal_amplitude,
            config=self.config,
            n_tonic_swings=raw["n_tonic_swings"],
        )
        return self.result_

    def estimate(
        self, trace: NDArray[np.floating], apply_correction: bool = False
    ) -> Tuple[float, float, NDArray[np.intp]]:
        """One-shot functional interface: ``fit`` and return a 3-tuple.

        Returns ``(snr, noise, peaks)``, the same shape as a plain
        derivative-based ``estimate_snr(trace, fps)`` function, for
        drop-in comparison. ``snr`` is the **total**
        (``snr_tonic + snr_phasic``). Use :meth:`estimate_components`
        for the tonic/phasic breakdown without giving up this shape.

        Returns the raw (uncorrected) total by default even if
        ``bias_correction`` is configured; pass ``apply_correction=True``
        to get ``snr_corrected`` instead. Use ``.fit(trace)`` directly
        for repeated calls to avoid recomputing.
        """
        result = self.fit(trace)
        if apply_correction and result.snr_corrected is not None:
            snr = result.snr_corrected
        else:
            snr = result.snr
        return snr, result.noise, result.peaks

    def estimate_components(
        self, trace: NDArray[np.floating], apply_correction: bool = False
    ) -> Tuple[float, float, float]:
        """One-shot SNR breakdown: ``fit`` and return
        ``(snr_total, snr_tonic, snr_phasic)``.

        Companion to :meth:`estimate`, which only returns the total (to
        keep that method's 3-tuple shape drop-in compatible with a
        plain ``estimate_snr(trace, fps)`` function). ``snr_total``
        here always equals :meth:`estimate`'s first element under the
        same ``apply_correction`` setting.

        Returns raw (uncorrected) values by default; pass
        ``apply_correction=True`` to get ``snr_corrected`` and
        ``snr_phasic_corrected`` for the 1st/3rd elements (``snr_tonic``
        has no correction to apply, so it's unaffected either way).
        """
        result = self.fit(trace)
        if apply_correction and result.snr_corrected is not None:
            return result.snr_corrected, result.snr_tonic, result.snr_phasic_corrected
        return result.snr, result.snr_tonic, result.snr_phasic

    def decompose(
        self, trace: Optional[NDArray[np.floating]] = None
    ) -> Tuple[NDArray[np.floating], NDArray[np.floating]]:
        """Return the (tonic, phasic) decomposition of a trace.

        Convenience wrapper around :meth:`fit`. "Phasic" is the tonic-
        subtracted residual (``trace - tonic``, same as
        ``EnvelopeRRResult.residual``). If ``trace`` is omitted, returns
        the decomposition from the most recent :meth:`fit`/
        :meth:`estimate` call (raises if not yet fit).

        Example
        -------
        >>> import numpy as np
        >>> y = 0.01 * np.random.default_rng(0).standard_normal(1200)
        >>> tonic, phasic = EnvelopeRRSNR(fps=20.0).decompose(y)
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
        """Total SNR (``snr_tonic_ + snr_phasic_``) from the most recent
        :meth:`fit` call (raw, uncorrected)."""
        self._check_fitted()
        return self.result_.snr

    @property
    def snr_corrected_(self) -> Optional[float]:
        """Bias-corrected total SNR from the most recent :meth:`fit` call."""
        self._check_fitted()
        return self.result_.snr_corrected

    @property
    def snr_tonic_(self) -> float:
        """Tonic-baseline SNR from the most recent :meth:`fit` call."""
        self._check_fitted()
        return self.result_.snr_tonic

    @property
    def snr_phasic_(self) -> float:
        """Phasic SNR from the most recent :meth:`fit` call (raw, uncorrected)."""
        self._check_fitted()
        return self.result_.snr_phasic

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
        """Raise RuntimeError if fit()/estimate()/decompose() hasn't been called yet."""
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

        Convenience wrapper around ``numpy.polyfit`` for correcting a
        consistent linear bias found by sweeping known SNR levels. Fit
        this against ``snr_phasic`` (not the total ``snr``), separately
        per ``peak_threshold_sd`` -- each sits at its own scale. Returns
        ``(slope, intercept)``; pass directly as
        ``bias_correction=(slope, intercept)`` to the constructor.

        Example
        -------
        >>> import numpy as np
        >>> true_snr = np.array([5.0, 10.0, 20.0, 40.0])
        >>> snr_est  = 0.9 * true_snr + 1.5   # simulated linear bias
        >>> slope, intercept = EnvelopeRRSNR.fit_bias_correction_from_benchmark(
        ...     true_snr, snr_est)
        """
        true_snr = np.asarray(true_snr, dtype=float)
        snr_est = np.asarray(snr_est, dtype=float)
        ok = np.isfinite(true_snr) & np.isfinite(snr_est)
        slope, intercept = np.polyfit(true_snr[ok], snr_est[ok], 1)
        return float(slope), float(intercept)
