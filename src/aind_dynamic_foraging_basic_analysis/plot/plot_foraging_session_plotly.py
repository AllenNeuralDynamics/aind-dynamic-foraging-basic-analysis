"""Interactive plotly figures that can be used in the Streamlit app as well as
Jupyter Notebook.

Two plotly counterparts of the matplotlib plotting functions are provided, each meant to
match its matplotlib sibling as closely as plotly allows:

- :func:`plot_foraging_session_plotly` mirrors
  :func:`plot_foraging_session.plot_foraging_session` -- a **trial-based** view (choice /
  reward raster on top, reward-probability schedule below).
- :func:`plot_session_in_time_plotly` mirrors
  :func:`plot_session_scroller.plot_session_scroller` -- a **time-based** view of the
  session (licks / rewards / go cues / reward-probability band in real time).
"""

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from aind_dynamic_foraging_basic_analysis.data_model.foraging_session import (
    ForagingSessionData,
    PhotostimData,
)
from aind_dynamic_foraging_basic_analysis.plot.plot_foraging_session import moving_average
from aind_dynamic_foraging_basic_analysis.plot.style import PHOTOSTIM_EPOCH_MAPPING

# Map the matplotlib single-letter colors used by the matplotlib versions to plotly names,
# so the two renderings line up. Anything not listed is passed through unchanged.
_MPL_COLORS = {
    "y": "gold",
    "m": "magenta",
    "g": "green",
    "b": "blue",
    "r": "red",
    "k": "black",
}


def _color(c):
    """Translate a matplotlib single-letter color to a plotly-friendly name."""
    return _MPL_COLORS.get(c, c)


def _vlines(segments):
    """Flatten ``[(x_array, y0, y1), ...]`` into x / y arrays of ``None``-separated segments.

    The standard plotly trick for drawing many vertical line ticks in a single trace: insert
    ``None`` between each ``(x, y0)->(x, y1)`` pair so plotly lifts the pen between ticks.
    """
    xs, ys = [], []
    for x_arr, y0, y1 in segments:
        for xi in np.asarray(x_arr):
            xs += [xi, xi, None]
            ys += [y0, y1, None]
    return xs, ys


def _vline_hover(x_arr, y0, y1, hover, gap=None):
    """Vertical ticks at ``x_arr`` (each y0->y1) plus a parallel ``customdata`` array.

    Like :func:`_vlines` for a single group, but also threads a per-tick ``hover`` value
    (repeated on both vertices, ``gap`` on the separator) so each tick can surface e.g. its
    trial / session via a ``hovertemplate``. Pass ``gap=(None, None)`` for 2-field customdata.
    """
    xs, ys, cd = [], [], []
    for xi, hi in zip(np.asarray(x_arr), hover):
        xs += [xi, xi, None]
        ys += [y0, y1, None]
        cd += [hi, hi, gap]
    return xs, ys, cd


def _nice_step(span, target=4):
    """A round tick step (1/2/5 x 10^k) giving roughly ``target`` ticks across ``span``."""
    if span <= 0:
        return 1.0
    raw = span / target
    mag = 10.0 ** np.floor(np.log10(raw))
    for m in (1, 2, 5, 10):
        if m * mag >= raw:
            return m * mag
    return 10.0 * mag


def _session_segments(session_id, n):
    """Contiguous per-session index segments and the boundary indices between them.

    Returns ``(segments, boundaries)`` where ``segments`` is a list of ``(start, end)``
    half-open index ranges (one per session, in order) and ``boundaries`` is the list of
    indices at which a new session starts (used to draw the dividing lines). With
    ``session_id=None`` the whole input is one segment.
    """
    if session_id is None:
        return [(0, n)], []
    sid = np.asarray(session_id)
    change = list(np.nonzero(sid[1:] != sid[:-1])[0] + 1)
    segments = list(zip([0, *change], [*change, n]))
    return segments, change


def _broken(x, y, segments):
    """Concatenate per-session slices of ``x``/``y`` with ``None`` gaps between sessions.

    Inserting a ``None`` at each boundary breaks the line so a continuous trace is not drawn
    across the (meaningless) jump from one session's last trial to the next session's first.
    """
    xs, ys = [], []
    for s, e in segments:
        xs += [*np.asarray(x)[s:e], None]
        ys += [*np.asarray(y)[s:e], None]
    return xs, ys


def plot_foraging_session_plotly(  # noqa: C901
    choice_history,
    reward_history,
    p_reward,
    autowater_offered=None,
    fitted_data=None,
    photostim=None,
    valid_range=None,
    smooth_factor=5,
    base_color="y",
    bias=None,
    bias_lower=None,
    bias_upper=None,
    plot_list=["choice", "finished", "reward_prob"],
    session_id=None,
):
    """Plotly version of :func:`plot_foraging_session.plot_foraging_session` (trial-based).

    Renders the same two stacked panels as the matplotlib version:

    - top: choice / reward raster (rewarded & unrewarded choices, ignored / autowater
      trials, smoothed choice, finished ratio, base reward probability, optional bias).
    - bottom: the per-trial left / right reward-probability schedule.

    Parameters mirror :func:`plot_foraging_session.plot_foraging_session` (minus the
    matplotlib-only ``ax`` / ``vertical``):

    Parameters
    ----------
    choice_history : list or np.ndarray
        Choice history (0 = left, 1 = right, np.nan = ignored).
    reward_history : list or np.ndarray
        Reward history (0 = unrewarded, 1 = rewarded).
    p_reward : list or np.ndarray
        Reward probability for both sides, shape (2, len(choice_history)).
    autowater_offered : list or np.ndarray, optional
        Boolean mask of trials where autowater was offered.
    fitted_data : list or np.ndarray, optional
        If not None, overlay fitted data (e.g. from an RL model).
    photostim : dict, optional
        Photostimulation trials, with keys "trial", "power" and optional "stim_epoch".
    valid_range : list, optional
        If not None, add two vertical lines marking the engaged range.
    smooth_factor : int, optional
        Smoothing window for the choice / finished traces, by default 5.
    base_color : str, optional
        Color for the base reward-probability line, by default "y" (gold).
    bias : list or np.ndarray, optional
        Side-bias trace; drawn (with the ``bias_lower`` / ``bias_upper`` band) when
        "bias" is in ``plot_list``.
    bias_lower, bias_upper : list or np.ndarray, optional
        Lower / upper confidence bounds for ``bias``.
    plot_list : list, optional
        Which optional traces to draw, from {"choice", "finished", "reward_prob", "bias"}.
    session_id : list or np.ndarray, optional
        Per-trial session label (same length as ``choice_history``). When given, multiple
        sessions are concatenated horizontally along the trial axis in order; smoothed traces
        reset at each session and a thick vertical line marks every session boundary.

    Returns
    -------
    plotly.graph_objects.Figure
        Figure with the choice/reward panel (row 1) over the reward-schedule panel (row 2).
    """
    # Formatting and sanity checks (reuse the shared validation, like the matplotlib version)
    data = ForagingSessionData(
        choice_history=choice_history,
        reward_history=reward_history,
        p_reward=p_reward,
        autowater_offered=autowater_offered,
        fitted_data=fitted_data,
        photostim=PhotostimData(**photostim) if photostim is not None else None,
    )
    choice_history = data.choice_history
    reward_history = data.reward_history
    p_reward = data.p_reward
    autowater_offered = data.autowater_offered
    fitted_data = data.fitted_data
    photostim = data.photostim

    n_trials = len(choice_history)
    p_reward_fraction = p_reward[1, :] / (np.sum(p_reward, axis=0))
    ignored = np.isnan(choice_history)

    if autowater_offered is None:
        rewarded_excluding_autowater = reward_history
        autowater_collected = np.full_like(choice_history, False, dtype=bool)
        autowater_ignored = np.full_like(choice_history, False, dtype=bool)
        unrewarded_trials = ~reward_history & ~ignored
    else:
        rewarded_excluding_autowater = reward_history & ~autowater_offered
        autowater_collected = autowater_offered & ~ignored
        autowater_ignored = autowater_offered & ignored
        unrewarded_trials = ~reward_history & ~ignored & ~autowater_offered

    # Per-session segments (multi-session concatenation); single segment when session_id None.
    segments, boundaries = _session_segments(session_id, n_trials)

    # Per-trial within-session index (resets to 0 each session) and session label -- used for
    # the (session, trial) hover and the per-session x tick labels.
    within = np.zeros(n_trials, dtype=int)
    sess_label = np.array([""] * n_trials, dtype=object)
    sid_arr = None if session_id is None else np.asarray(session_id)
    for s, e in segments:
        within[s:e] = np.arange(e - s)
        if sid_arr is not None:
            sess_label[s:e] = sid_arr[s]

    hovertemplate = "trial %%{customdata[0]}<br>session %%{customdata[1]}<extra>%s</extra>"

    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        row_heights=[0.83, 0.17],
        vertical_spacing=0.02,
    )

    def _raster(mask, up, down):
        """Vertical choice ticks for ``mask``, each carrying (within-session trial, session).

        Right choices (>0.5) use the ``up`` (y0, y1); left choices use ``down``. Returns
        ``x, y, customdata`` lists (``None`` gaps between ticks) for one Scattergl trace.
        """
        idx = np.nonzero(mask)[0]
        side = choice_history[idx]
        xs, ys, cd = [], [], []
        for i, sd in zip(idx, side):
            y0, y1 = up if sd > 0.5 else down
            xs += [i + 1, i + 1, None]
            ys += [y0, y1, None]
            cd += [(int(within[i]), sess_label[i])] * 2 + [(None, None)]
        return xs, ys, cd

    def _markers(mask):
        """(x, customdata) for marker traces (ignored / autowater-ignored), with hover."""
        idx = np.nonzero(mask)[0]
        return idx + 1, [(int(within[i]), sess_label[i]) for i in idx]

    # == Choice trace ==
    # Rewarded (real foraging, autowater excluded): tall black ticks just outside [0, 1]
    xs, ys, cd = _raster(rewarded_excluding_autowater, (1.05, 1.15), (-0.15, -0.05))
    fig.add_trace(
        go.Scattergl(x=xs, y=ys, customdata=cd, mode="lines", line=dict(color="black", width=1),
                     name="Rewarded choices", hovertemplate=hovertemplate % "Rewarded choices"),
        row=1, col=1,
    )

    # Unrewarded (real foraging): short gray ticks
    xs, ys, cd = _raster(unrewarded_trials, (1.05, 1.10), (-0.10, -0.05))
    fig.add_trace(
        go.Scattergl(x=xs, y=ys, customdata=cd, mode="lines", line=dict(color="gray", width=1),
                     name="Unrewarded choices", hovertemplate=hovertemplate % "Unrewarded choices"),
        row=1, col=1,
    )

    # Ignored trials: red x at the top
    xx, cd = _markers(ignored & ~autowater_ignored)
    fig.add_trace(
        go.Scattergl(x=xx, y=[1.2] * len(xx), customdata=cd, mode="markers",
                     marker=dict(symbol="x", color="red", size=4), name="Ignored",
                     hovertemplate=hovertemplate % "Ignored"),
        row=1, col=1,
    )

    # Autowater collected / ignored
    if autowater_offered is not None:
        xs, ys, cd = _raster(autowater_collected, (1.05, 1.15), (-0.15, -0.05))
        fig.add_trace(
            go.Scattergl(x=xs, y=ys, customdata=cd, mode="lines",
                         line=dict(color="royalblue", width=1), name="Autowater collected",
                         hovertemplate=hovertemplate % "Autowater collected"),
            row=1, col=1,
        )
        xx, cd = _markers(autowater_ignored)
        fig.add_trace(
            go.Scattergl(x=xx, y=[1.2] * len(xx), customdata=cd, mode="markers",
                         marker=dict(symbol="x", color="royalblue", size=4),
                         name="Autowater ignored",
                         hovertemplate=hovertemplate % "Autowater ignored"),
            row=1, col=1,
        )

    # Base reward probability (broken at session boundaries)
    if "reward_prob" in plot_list:
        xs, ys = _broken(np.arange(n_trials) + 1, p_reward_fraction, segments)
        fig.add_trace(
            go.Scattergl(x=xs, y=ys, mode="lines",
                         line=dict(color=_color(base_color), width=1.5), name="Base rew. prob."),
            row=1, col=1,
        )

    def _smoothed_trace(num, den):
        """Per-session smoothed series (resets at each session); x in 1-based trial coords."""
        xs, ys = [], []
        for s, e in segments:
            y = moving_average(num[s:e], smooth_factor)
            if den is not None:
                y = y / (moving_average(den[s:e], smooth_factor) + 1e-6)
            x = s + np.arange(len(y)) + int(smooth_factor / 2) + 1
            xs += [*x, None]
            ys += [*y, None]
        return xs, np.where(np.array(ys, dtype=float) > 100, np.nan, ys)

    # Smoothed choice history
    if "choice" in plot_list:
        xs, ys = _smoothed_trace(choice_history, ~np.isnan(choice_history))
        fig.add_trace(
            go.Scattergl(x=xs, y=ys, mode="lines", line=dict(color="black", width=1.5),
                         name=f"Choice (smooth = {smooth_factor})"),
            row=1, col=1,
        )

    # Finished ratio (only meaningful if there are ignored trials)
    if "finished" in plot_list and np.sum(np.isnan(choice_history)):
        xs, ys = _smoothed_trace(~np.isnan(choice_history), None)
        fig.add_trace(
            go.Scattergl(x=xs, y=ys, mode="lines", line=dict(color="magenta", width=0.8),
                         name=f"Finished (smooth = {smooth_factor})"),
            row=1, col=1,
        )

    # Bias trace + confidence band (broken at session boundaries)
    if ("bias" in plot_list) and (bias is not None):
        xx = np.arange(n_trials) + 1
        bias = (np.array(bias) + 1) / 2
        bias_lower = np.clip((np.array(bias_lower) + 1) / 2, 0, None)
        bias_upper = np.clip((np.array(bias_upper) + 1) / 2, None, 1)
        xb_up, y_up = _broken(xx, bias_upper, segments)
        xb_lo, y_lo = _broken(xx, bias_lower, segments)
        xb, y_bias = _broken(xx, bias, segments)
        # go.Scatter (not Scattergl) for the filled band -- Scattergl ignores fill.
        fig.add_trace(
            go.Scatter(x=xb_up, y=y_up, mode="lines", line=dict(width=0),
                       showlegend=False, hoverinfo="skip"),
            row=1, col=1,
        )
        fig.add_trace(
            go.Scatter(x=xb_lo, y=y_lo, mode="lines", line=dict(width=0),
                       fill="tonexty", fillcolor="rgba(0,128,0,0.25)",
                       showlegend=False, hoverinfo="skip"),
            row=1, col=1,
        )
        fig.add_trace(
            go.Scattergl(x=xb, y=y_bias, mode="lines", line=dict(color="green", width=1.5),
                         name="bias"),
            row=1, col=1,
        )

    # Valid (engaged) range
    if valid_range is not None:
        for vr in valid_range:
            fig.add_vline(x=vr, line=dict(color="magenta", dash="dash", width=1), row=1, col=1)

    # Fitted model overlay
    if fitted_data is not None:
        fig.add_trace(
            go.Scattergl(x=np.arange(n_trials), y=fitted_data, mode="lines",
                         line=dict(width=1.5), name="model"),
            row=1, col=1,
        )

    # Photostim markers
    if photostim is not None:
        trial = np.asarray(photostim.trial)
        power = np.asarray(photostim.power, dtype=float)
        if photostim.stim_epoch is not None:
            colors = [PHOTOSTIM_EPOCH_MAPPING[t] for t in photostim.stim_epoch]
        else:
            colors = "darkcyan"
        fig.add_trace(
            go.Scattergl(x=trial, y=np.ones_like(trial, dtype=float) + 0.4, mode="markers",
                         marker=dict(symbol="triangle-down", size=power * 2,
                                     color="rgba(0,0,0,0)",
                                     line=dict(color=colors, width=0.5)),
                         name="photostim"),
            row=1, col=1,
        )

    # == Reward schedule (bottom panel; broken at session boundaries) ==
    xx = np.arange(n_trials) + 1
    xr, y_pr = _broken(xx, p_reward[1, :], segments)
    fig.add_trace(
        go.Scattergl(x=xr, y=y_pr, mode="lines", line=dict(color="blue", width=1),
                     name="p_right"),
        row=2, col=1,
    )
    xl, y_pl = _broken(xx, p_reward[0, :], segments)
    fig.add_trace(
        go.Scattergl(x=xl, y=y_pl, mode="lines", line=dict(color="red", width=1),
                     name="p_left"),
        row=2, col=1,
    )

    # Thick vertical lines marking session boundaries (between trials b and b+1)
    for b in boundaries:
        for row in (1, 2):
            fig.add_vline(x=b + 0.5, line=dict(color="black", width=2), row=row, col=1)

    # Axes styling to match the matplotlib version
    fig.update_yaxes(
        tickvals=[0, 1, 1.2], ticktext=["Left", "Right", "Ignored"],
        range=[-0.15, 1.25], fixedrange=True, row=1, col=1,
    )
    fig.update_yaxes(title_text="p_reward", range=[0, 1], fixedrange=True, row=2, col=1)
    # Bottom x-axis: a rangeslider scroller (drag to pan/zoom), and -- for multiple sessions --
    # tick labels that restart at 0 each session.
    fig.update_xaxes(title_text="Trial number", row=2, col=1,
                     rangeslider=dict(visible=True, thickness=0.08))
    if len(segments) > 1:
        step = 250
        tickvals, ticktext = [], []
        for s, e in segments:
            for w in range(0, e - s, step):
                tickvals.append(s + 1 + w)
                ticktext.append(str(w))
        fig.update_xaxes(tickvals=tickvals, ticktext=ticktext, row=2, col=1)
    fig.update_layout(
        width=1300, height=440, template="simple_white",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        margin=dict(l=60, r=20, t=60, b=50),
    )
    return fig


def plot_session_in_time_plotly(  # noqa: C901 pragma: no cover
    df_events, df_trials=None, fip_df=None, adjust_time=True, title=None, smooth_factor=5
):
    """Plotly version of :func:`plot_session_scroller.plot_session_scroller` (time-based).

    Plots the session in real time (not in trial): left / right licks and rewards as ticks,
    go cues as vertical lines (red for ignored trials), smoothed overlays above the events,
    and -- when ``df_trials`` is supplied -- the left / right reward-probability band in the
    rangeslider "scroller" below.

    Multiple sessions: if ``df_events`` has a ``session_id`` column with more than one
    session, the sessions are concatenated end-to-end along time in order of appearance
    (each restarted at the running offset); a thick vertical line marks each boundary and
    per-trial / smoothed quantities reset per session. ``df_trials`` is matched per session
    by its own ``session_id`` column when present.

    Parameters
    ----------
    df_events : pandas.DataFrame
        Tidy dataframe of session events (``event`` + ``timestamps``; optional ``session_id``).
        Recognised events: ``left_lick_time``, ``right_lick_time``, ``left_reward_delivery_time``,
        ``right_reward_delivery_time`` and ``goCue_start_time``.
    df_trials : pandas.DataFrame, optional
        Per-trial dataframe for the reward-probability band / overlays / red ignored go cues
        (and a fallback source of go-cue times). Uses ``goCue_start_time``,
        ``reward_probabilityL/R`` and ``animal_response``; go-cue times must share the
        ``df_events`` time base. Matched per session via ``session_id`` when present.
    fip_df : pandas.DataFrame, optional
        Tidy FIP measurements (single-session only); each present channel is normalised and
        stacked above the behavior panel.
    adjust_time : bool, optional
        If True (default), shift time so the first event is at t = 0 (always shifted when
        concatenating multiple sessions).
    title : str, optional
        Figure title.
    smooth_factor : int, optional
        Smoothing window for the choice / lick-count overlays, by default 5.

    Returns
    -------
    plotly.graph_objects.Figure
    """
    df_events = df_events.copy()
    if df_trials is not None:
        df_trials = df_trials.copy()
    if fip_df is not None:
        fip_df = fip_df.copy()

    # y-layout, bottom -> top:
    #   * event rows in [0, 1]: rewards at the outer edges, licks inside (right pair near the
    #     top, left pair near the bottom), like the trial-based figure.
    #   * smoothed overlays in their own band [curve_bottom, curve_top] above the events.
    #   * the reward-probability band sits higher still -- it only shows in the rangeslider.
    params = {
        "behavior_bottom": 0.0, "behavior_top": 1.0,
        "curve_bottom": 1.1, "curve_top": 2.1,
        "probs_center": 2.75, "probs_half": 0.25,  # reward-prob lines (scroller), above main
    }
    row_centers = {"right_reward": 0.92, "right_lick": 0.78,
                   "left_lick": 0.22, "left_reward": 0.08}
    tick_half = 0.25 * 0.30 / 2.0

    def _to_curve(v):
        """Map a 0..1 per-trial value into the smoothed-overlay band above the events."""
        span = params["curve_top"] - params["curve_bottom"]
        return params["curve_bottom"] + np.asarray(v, dtype=float) * span

    yticks = [0.92, 0.78, 0.22, 0.08,
              params["curve_bottom"], (params["curve_bottom"] + params["curve_top"]) / 2,
              params["curve_top"]]
    ylabels = ["right reward", "right lick", "left lick", "left reward", "0", "0.5", "1"]

    # Sessions in order of appearance; concatenate end-to-end along time when more than one.
    has_sess = "session_id" in df_events.columns
    sessions = (list(dict.fromkeys(df_events["session_id"].tolist())) if has_sess else [None])
    shift_each = adjust_time or len(sessions) > 1

    fig = go.Figure()

    ev_meta = {
        "left_lick": ("left_lick_time", "gray", 1.5, "left lick"),
        "right_lick": ("right_lick_time", "gray", 1.5, "right lick"),
        "left_reward": ("left_reward_delivery_time", "black", 2, "left reward"),
        "right_reward": ("right_reward_delivery_time", "black", 2, "right reward"),
    }
    ev_acc = {k: {"x": [], "y": [], "cd": []} for k in ev_meta}
    gocue_acc = {"go cue": {"x": [], "y": [], "cd": []},
                 "go cue (ignored)": {"x": [], "y": [], "cd": []}}
    frac_x, frac_y, choice_x, choice_y, lick_x, lick_y = [], [], [], [], [], []
    probL_x, probL_y, probR_x, probR_y = [], [], [], []  # reward-prob lines (scroller)
    boundaries, sess_spans, has_prob = [], [], False
    cum, first_t0, first_gc, last_off = 0.0, None, None, 0.0

    for si, sess in enumerate(sessions):
        ev_s = df_events if sess is None else df_events[df_events["session_id"] == sess]
        ts = ev_s["timestamps"].to_numpy()
        t0 = np.nanmin(ts)
        if first_t0 is None:
            first_t0 = t0
        off = (cum - t0) if shift_each else 0.0
        last_off = off
        if si > 0:
            boundaries.append(cum)

        def _ev(name, _ev=ev_s, _off=off):
            return _ev.loc[_ev["event"] == name, "timestamps"].to_numpy() + _off

        tr_s = None
        if df_trials is not None:
            tr_s = (df_trials[df_trials["session_id"] == sess]
                    if (sess is not None and "session_id" in df_trials.columns) else df_trials)

        gc = _ev("goCue_start_time")
        if len(gc) == 0 and tr_s is not None and "goCue_start_time" in tr_s.columns:
            gc = tr_s["goCue_start_time"].to_numpy() + off
        gc = gc[~np.isnan(gc)]
        n_tr = len(gc)
        if n_tr and first_gc is None:
            first_gc = gc.min()

        def _trial_of(times, _gc=gc, _n=n_tr):
            if _n == 0:
                return np.zeros(len(times), dtype=int)
            return np.searchsorted(_gc, np.asarray(times), side="right")

        aligned = tr_s is not None and len(tr_s) == n_tr and n_tr > 0
        choice = None
        if aligned and "animal_response" in tr_s.columns:
            choice = tr_s["animal_response"].astype(float).to_numpy().copy()
            choice[choice == 2] = np.nan

        sess_disp = "" if sess is None else sess  # shown in (trial, session) hover

        for key, (name, _color, _width, _label) in ev_meta.items():
            c = row_centers[key]
            t = _ev(name)
            hov = [(int(tr), sess_disp) for tr in _trial_of(t)]
            xs, ys, cd = _vline_hover(t, c - tick_half, c + tick_half, hov, gap=(None, None))
            ev_acc[key]["x"] += xs
            ev_acc[key]["y"] += ys
            ev_acc[key]["cd"] += cd

        if n_tr:
            trial_no = np.arange(1, n_tr + 1)
            ign = np.isnan(choice) if choice is not None else np.zeros(n_tr, dtype=bool)
            for gname, mask in [("go cue", ~ign), ("go cue (ignored)", ign)]:
                if mask.any():
                    hov = [(int(tr), sess_disp) for tr in trial_no[mask]]
                    xs, ys, cd = _vline_hover(gc[mask], params["behavior_bottom"],
                                              params["behavior_top"], hov, gap=(None, None))
                    gocue_acc[gname]["x"] += xs
                    gocue_acc[gname]["y"] += ys
                    gocue_acc[gname]["cd"] += cd

        if n_tr:
            off_s = smooth_factor // 2
            if aligned and {"reward_probabilityL", "reward_probabilityR"} <= set(tr_s.columns):
                pL = tr_s["reward_probabilityL"].to_numpy()
                pR = tr_s["reward_probabilityR"].to_numpy()
                frac = np.divide(pR, pL + pR, out=np.full(n_tr, np.nan), where=(pL + pR) > 0)
                frac_x += [*gc, None]
                frac_y += [*_to_curve(frac), None]
            if choice is not None:
                sm = moving_average(choice, smooth_factor) / (
                    moving_average(~np.isnan(choice), smooth_factor) + 1e-6)
                sm[sm > 100] = np.nan
                xsm = gc[off_s: off_s + len(sm)]
                choice_x += [*xsm, None]
                choice_y += [*_to_curve(sm[: len(xsm)]), None]
            lt = np.concatenate([_ev("left_lick_time"), _ev("right_lick_time")])
            if len(lt):
                counts = np.bincount(_trial_of(lt), minlength=n_tr + 1)[1:n_tr + 1]
                sm = moving_average(counts.astype(float), smooth_factor)
                top = np.nanmax(sm) if len(sm) else 0
                if top > 0:
                    sm = sm / top
                xsm = gc[off_s: off_s + len(sm)]
                lick_x += [*xsm, None]
                lick_y += [*_to_curve(sm[: len(xsm)]), None]

        # Reward-probability as two lines (pL red, pR blue), like the trial-based schedule;
        # values 0..1 mapped into the scroller band and broken at session boundaries.
        if (tr_s is not None and n_tr and len(tr_s) == n_tr
                and {"reward_probabilityL", "reward_probabilityR"} <= set(tr_s.columns)):
            has_prob = True
            lo = params["probs_center"] - params["probs_half"]
            span = 2 * params["probs_half"]
            probL_x += [*gc, None]
            probL_y += [*(lo + tr_s["reward_probabilityL"].to_numpy() * span), None]
            probR_x += [*gc, None]
            probR_y += [*(lo + tr_s["reward_probabilityR"].to_numpy() * span), None]

        dur = np.nanmax(ts) - t0
        sess_spans.append((cum, dur))
        cum += dur

    # --- build one trace per type from the accumulators ---
    ht = "%%{x:.2f}s<br>trial %%{customdata[0]}<br>session %%{customdata[1]}<extra>%s</extra>"
    for key, (name, color, width, label) in ev_meta.items():
        a = ev_acc[key]
        fig.add_trace(go.Scattergl(
            x=a["x"], y=a["y"], customdata=a["cd"], mode="lines",
            line=dict(color=color, width=width), name=label, hovertemplate=ht % label))

    for gname, gcolor in [("go cue", "green"), ("go cue (ignored)", "red")]:
        a = gocue_acc[gname]
        if a["x"]:
            fig.add_trace(go.Scattergl(
                x=a["x"], y=a["y"], customdata=a["cd"], mode="lines",
                line=dict(color=gcolor, width=0.75), opacity=0.75, name=gname,
                hovertemplate=ht % gname))

    if has_prob:
        # pL (red) / pR (blue) as lines, like the trial-based reward schedule (scroller only).
        # go.Scatter (not Scattergl) -- WebGL traces do not render in the rangeslider.
        fig.add_trace(go.Scatter(x=probR_x, y=probR_y, mode="lines",
                                 line=dict(color="blue", width=1.2), name="pR"))
        fig.add_trace(go.Scatter(x=probL_x, y=probL_y, mode="lines",
                                 line=dict(color="red", width=1.2), name="pL"))

    # Smoothed overlays on top
    if frac_x:
        fig.add_trace(go.Scattergl(x=frac_x, y=frac_y, mode="lines",
                                   line=dict(color="gold", width=1.5), name="pR/(pL+pR)"))
    if choice_x:
        fig.add_trace(go.Scattergl(x=choice_x, y=choice_y, mode="lines",
                                   line=dict(color="black", width=1.5),
                                   name=f"choice (smooth = {smooth_factor})"))
    if lick_x:
        fig.add_trace(go.Scattergl(x=lick_x, y=lick_y, mode="lines",
                                   line=dict(color="black", width=1.2, dash="dash"),
                                   name=f"lick count (smooth = {smooth_factor})"))

    y_main_top = params["curve_top"]

    # FIP channels (single-session only), normalised and stacked above the behavior panel
    if fip_df is not None and len(sessions) == 1:
        fip_channels = ["G_1_preprocessed", "G_2_preprocessed",
                        "R_1_preprocessed", "R_2_preprocessed"]
        fip_colors = {"G_1": "green", "G_2": "darkgreen", "R_1": "red", "R_2": "darkred"}
        present = set(fip_df["event"].unique())
        band = 0
        for channel in fip_channels:
            if channel not in present:
                continue
            bottom = params["probs_center"] + params["probs_half"] + 0.1 + band
            C = fip_df.query("event == @channel").copy()
            d = C["data"].values - np.nanmin(C["data"].values)
            d = d / np.nanmax(d) + bottom
            color = fip_colors["_".join(channel.split("_")[:2])]
            fig.add_trace(go.Scattergl(x=C.timestamps.values + last_off, y=d, mode="lines",
                                       line=dict(color=color), name=channel))
            yticks.append(bottom + 0.5)
            ylabels.append(channel)
            band += 1
            y_main_top = bottom + 1.0

    # Thick vertical lines marking session boundaries
    for b in boundaries:
        fig.add_vline(x=b, line=dict(color="black", width=2))

    # Full extent + initial ~120 s window at the first go cue. The rangeslider scrubs the
    # whole session(s); when reward-prob lines are present pin the scroller y to their band so
    # they fill the scroller (x-dragging is unaffected); otherwise auto-fit.
    x_first = 0.0 if shift_each else (first_t0 if first_t0 is not None else 0.0)
    x_last = x_first + cum
    t0_view = first_gc if first_gc is not None else x_first
    if has_prob:
        slider_yaxis = dict(
            rangemode="fixed",
            range=[params["probs_center"] - params["probs_half"] - 0.05,
                   params["probs_center"] + params["probs_half"] + 0.05])
    else:
        slider_yaxis = dict(rangemode="auto")

    # Multi-session: x tick labels restart at 0 each session.
    xaxis = dict(range=[t0_view, t0_view + 120],
                 rangeslider=dict(visible=True, range=[x_first, x_last], yaxis=slider_yaxis))
    if len(sess_spans) > 1:
        tickvals, ticktext = [], []
        for start, dur in sess_spans:
            step = _nice_step(dur)
            w = 0.0
            while w <= dur:
                tickvals.append(start + w)
                ticktext.append(str(int(w)))
                w += step
        xaxis.update(tickvals=tickvals, ticktext=ticktext)

    fig.update_layout(
        title=title or "Session Scroller",
        xaxis_title="Time (s)",
        yaxis=dict(tickvals=yticks, ticktext=ylabels, fixedrange=True,
                   range=[params["behavior_bottom"] - 0.05, y_main_top + 0.25]),
        xaxis=xaxis,
        showlegend=True, height=600, width=1300, template="simple_white",
    )
    return fig
