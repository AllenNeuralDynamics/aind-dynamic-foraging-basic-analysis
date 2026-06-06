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

    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        row_heights=[0.83, 0.17],
        vertical_spacing=0.02,
    )

    def _side_segments(mask, up, down):
        """Split a boolean trial mask into right (>0.5) and left (<0.5) tick segments.

        ``up`` is ``(y0, y1)`` for right choices (drawn just above Right); ``down`` for
        left choices (just below Left). Returns the list consumed by :func:`_vlines`.
        """
        xx = np.nonzero(mask)[0] + 1
        side = choice_history[mask]
        return [
            (xx[side > 0.5], up[0], up[1]),
            (xx[side < 0.5], down[0], down[1]),
        ]

    # == Choice trace ==
    # Rewarded (real foraging, autowater excluded): tall black ticks just outside [0, 1]
    xs, ys = _vlines(_side_segments(rewarded_excluding_autowater, (1.05, 1.15), (-0.15, -0.05)))
    fig.add_trace(
        go.Scattergl(x=xs, y=ys, mode="lines", line=dict(color="black", width=1),
                     name="Rewarded choices"),
        row=1, col=1,
    )

    # Unrewarded (real foraging): short gray ticks
    xs, ys = _vlines(_side_segments(unrewarded_trials, (1.05, 1.10), (-0.10, -0.05)))
    fig.add_trace(
        go.Scattergl(x=xs, y=ys, mode="lines", line=dict(color="gray", width=1),
                     name="Unrewarded choices"),
        row=1, col=1,
    )

    # Ignored trials: red x at the top
    xx = np.nonzero(ignored & ~autowater_ignored)[0] + 1
    fig.add_trace(
        go.Scattergl(x=xx, y=[1.2] * len(xx), mode="markers",
                     marker=dict(symbol="x", color="red", size=4), name="Ignored"),
        row=1, col=1,
    )

    # Autowater collected / ignored
    if autowater_offered is not None:
        xs, ys = _vlines(_side_segments(autowater_collected, (1.05, 1.15), (-0.15, -0.05)))
        fig.add_trace(
            go.Scattergl(x=xs, y=ys, mode="lines", line=dict(color="royalblue", width=1),
                         name="Autowater collected"),
            row=1, col=1,
        )
        xx = np.nonzero(autowater_ignored)[0] + 1
        fig.add_trace(
            go.Scattergl(x=xx, y=[1.2] * len(xx), mode="markers",
                         marker=dict(symbol="x", color="royalblue", size=4),
                         name="Autowater ignored"),
            row=1, col=1,
        )

    # Base reward probability
    if "reward_prob" in plot_list:
        fig.add_trace(
            go.Scattergl(x=np.arange(n_trials) + 1, y=p_reward_fraction, mode="lines",
                         line=dict(color=_color(base_color), width=1.5), name="Base rew. prob."),
            row=1, col=1,
        )

    # Smoothed choice history
    if "choice" in plot_list:
        y = moving_average(choice_history, smooth_factor) / (
            moving_average(~np.isnan(choice_history), smooth_factor) + 1e-6
        )
        y[y > 100] = np.nan
        x = np.arange(0, len(y)) + int(smooth_factor / 2) + 1
        fig.add_trace(
            go.Scattergl(x=x, y=y, mode="lines", line=dict(color="black", width=1.5),
                         name=f"Choice (smooth = {smooth_factor})"),
            row=1, col=1,
        )

    # Finished ratio (only meaningful if there are ignored trials)
    if "finished" in plot_list and np.sum(np.isnan(choice_history)):
        y = moving_average(~np.isnan(choice_history), smooth_factor)
        x = np.arange(0, len(y)) + int(smooth_factor / 2) + 1
        fig.add_trace(
            go.Scattergl(x=x, y=y, mode="lines", line=dict(color="magenta", width=0.8),
                         name=f"Finished (smooth = {smooth_factor})"),
            row=1, col=1,
        )

    # Bias trace + confidence band
    if ("bias" in plot_list) and (bias is not None):
        xx = np.arange(n_trials) + 1
        bias = (np.array(bias) + 1) / 2
        bias_lower = np.clip((np.array(bias_lower) + 1) / 2, 0, None)
        bias_upper = np.clip((np.array(bias_upper) + 1) / 2, None, 1)
        # go.Scatter (not Scattergl) for the filled band -- Scattergl ignores fill.
        fig.add_trace(
            go.Scatter(x=xx, y=bias_upper, mode="lines", line=dict(width=0),
                       showlegend=False, hoverinfo="skip"),
            row=1, col=1,
        )
        fig.add_trace(
            go.Scatter(x=xx, y=bias_lower, mode="lines", line=dict(width=0),
                       fill="tonexty", fillcolor="rgba(0,128,0,0.25)",
                       showlegend=False, hoverinfo="skip"),
            row=1, col=1,
        )
        fig.add_trace(
            go.Scattergl(x=xx, y=bias, mode="lines", line=dict(color="green", width=1.5),
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

    # == Reward schedule (bottom panel) ==
    xx = np.arange(n_trials) + 1
    fig.add_trace(
        go.Scattergl(x=xx, y=p_reward[1, :], mode="lines", line=dict(color="blue", width=1),
                     name="p_right"),
        row=2, col=1,
    )
    fig.add_trace(
        go.Scattergl(x=xx, y=p_reward[0, :], mode="lines", line=dict(color="red", width=1),
                     name="p_left"),
        row=2, col=1,
    )

    # Axes styling to match the matplotlib version
    fig.update_yaxes(
        tickvals=[0, 1, 1.2], ticktext=["Left", "Right", "Ignored"],
        range=[-0.15, 1.25], row=1, col=1,
    )
    fig.update_yaxes(title_text="p_reward", range=[0, 1], row=2, col=1)
    fig.update_xaxes(title_text="Trial number", row=2, col=1)
    fig.update_layout(
        width=1300, height=400, template="simple_white",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        margin=dict(l=60, r=20, t=60, b=50),
    )
    return fig


def plot_session_in_time_plotly(  # noqa: C901 pragma: no cover
    df_events, df_trials=None, fip_df=None, adjust_time=True, session_id=None
):
    """Plotly version of :func:`plot_session_scroller.plot_session_scroller` (time-based).

    Plots the session in real time (not in trial): left / right licks and rewards as ticks,
    go cues as vertical lines, and -- when ``df_trials`` is supplied -- the left / right
    reward-probability band, laid out to match the matplotlib scroller.

    Parameters
    ----------
    df_events : pandas.DataFrame
        Tidy dataframe of session events, e.g. from
        ``aind_dynamic_foraging_data_utils.nwb_utils.create_df_events``. Needs ``event`` and
        ``timestamps`` columns; recognised events are ``left_lick_time``, ``right_lick_time``,
        ``left_reward_delivery_time``, ``right_reward_delivery_time`` and ``goCue_start_time``.
    df_trials : pandas.DataFrame, optional
        Per-trial dataframe used for the reward-probability band (and as a fallback source of
        go-cue times). Needs ``goCue_start_time`` and ``reward_probabilityL/R``. The go-cue
        times must share the same time base as ``df_events.timestamps``.
    fip_df : pandas.DataFrame, optional
        Tidy dataframe of FIP measurements (from ``create_df_fip(tidy=True)``); each present
        channel is normalised and stacked above the behavior panel.
    adjust_time : bool, optional
        If True (default), shift time so the first event is at t = 0.
    session_id : str, optional
        Title for the figure.

    Returns
    -------
    plotly.graph_objects.Figure
    """
    df_events = df_events.copy()
    if df_trials is not None:
        df_trials = df_trials.copy()
    if fip_df is not None:
        fip_df = fip_df.copy()

    if adjust_time:
        start_time = df_events.iloc[0]["timestamps"]
        df_events["timestamps"] = df_events["timestamps"] - start_time
        if df_trials is not None:
            df_trials["goCue_start_time"] = df_trials["goCue_start_time"] - start_time
        if fip_df is not None:
            fip_df["timestamps"] = fip_df["timestamps"] - start_time

    xmin = df_events.iloc[0]["timestamps"]
    xmax = df_events.iloc[-1]["timestamps"]
    x_first, x_last = xmin, xmax  # full extent (used for the rangeslider / "home")

    # y-layout. The four behavior rows are stacked contiguously (top -> bottom: right licks,
    # right reward, left reward, left licks) so a single go-cue line spans all of them; the
    # reward-probability band sits in its own block above, centered on probs_center.
    params = {
        "left_lick_bottom": 0.0, "left_lick_top": 0.25,
        "left_reward_bottom": 0.25, "left_reward_top": 0.5,
        "right_reward_bottom": 0.5, "right_reward_top": 0.75,
        "right_lick_bottom": 0.75, "right_lick_top": 1.0,
        "behavior_bottom": 0.0, "behavior_top": 1.0,  # go cue spans this contiguous block
        "probs_center": 1.4, "probs_half": 0.25,      # band: probs_center +/- probs_half
    }
    yticks = [
        0.875, 0.625, 0.375, 0.125,  # right licks, right reward, left reward, left licks
        params["probs_center"] - params["probs_half"],  # pL = 1
        params["probs_center"],                         # 0
        params["probs_center"] + params["probs_half"],  # pR = 1
    ]
    ylabels = ["right licks", "right reward", "left reward", "left licks", "pL = 1", "0", "pR = 1"]

    fig = go.Figure()

    def _event_times(name):
        return df_events.query("event == @name").timestamps.values

    # Licks (gray, like the scroller when no bout coloring)
    for name, lo, hi in [
        ("left_lick_time", params["left_lick_bottom"], params["left_lick_top"]),
        ("right_lick_time", params["right_lick_bottom"], params["right_lick_top"]),
    ]:
        t = _event_times(name)
        xs, ys = _vlines([(t, lo, hi)])
        fig.add_trace(go.Scattergl(x=xs, y=ys, mode="lines", line=dict(color="gray", width=1.5),
                                   name=name.replace("_time", "").replace("_", " ")))

    # Rewards (black)
    for name, lo, hi in [
        ("left_reward_delivery_time", params["left_reward_bottom"], params["left_reward_top"]),
        ("right_reward_delivery_time", params["right_reward_bottom"], params["right_reward_top"]),
    ]:
        t = _event_times(name)
        xs, ys = _vlines([(t, lo, hi)])
        fig.add_trace(go.Scattergl(x=xs, y=ys, mode="lines", line=dict(color="black", width=2),
                                   name=name.replace("_delivery_time", "").replace("_", " ")))

    # Go cues: prefer events, fall back to df_trials
    go_cue_times = _event_times("goCue_start_time")
    if len(go_cue_times) == 0 and df_trials is not None and "goCue_start_time" in df_trials:
        go_cue_times = df_trials["goCue_start_time"].dropna().values
    if len(go_cue_times):
        # A single line spanning the contiguous behavior block
        xs, ys = _vlines([(go_cue_times, params["behavior_bottom"], params["behavior_top"])])
        fig.add_trace(go.Scattergl(x=xs, y=ys, mode="lines",
                                   line=dict(color="blue", width=0.75), opacity=0.75,
                                   name="go cue"))

    # Reward-probability band (needs df_trials and go-cue times)
    if df_trials is not None and len(go_cue_times) == len(df_trials):
        x_doubled = np.repeat(go_cue_times, 2)[1:]
        center = params["probs_center"]
        pR = np.repeat(center + df_trials["reward_probabilityR"].values / 4, 2)[:-1]
        pL = np.repeat(center - df_trials["reward_probabilityL"].values / 4, 2)[:-1]
        base = np.full_like(x_doubled, center, dtype=float)
        # pR above center (red), pL below center (blue); fill toward the center baseline.
        # go.Scatter (not Scattergl) -- the WebGL trace ignores fill="tonexty".
        fig.add_trace(go.Scatter(x=x_doubled, y=base, mode="lines", line=dict(width=0),
                                 showlegend=False, hoverinfo="skip"))
        fig.add_trace(go.Scatter(x=x_doubled, y=pR, mode="lines", line=dict(width=0),
                                 fill="tonexty", fillcolor="rgba(255,0,0,0.4)",
                                 name="pR"))
        fig.add_trace(go.Scatter(x=x_doubled, y=base, mode="lines", line=dict(width=0),
                                 showlegend=False, hoverinfo="skip"))
        fig.add_trace(go.Scatter(x=x_doubled, y=pL, mode="lines", line=dict(width=0),
                                 fill="tonexty", fillcolor="rgba(0,0,255,0.4)",
                                 name="pL"))

    y_top = params["probs_center"] + params["probs_half"]  # top of the plotted content

    # FIP channels, normalised and stacked above the behavior panel
    if fip_df is not None:
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
            fig.add_trace(go.Scattergl(x=C.timestamps.values, y=d, mode="lines",
                                       line=dict(color=color), name=channel))
            yticks.append(bottom + 0.5)
            ylabels.append(channel)
            band += 1
            y_top = bottom + 1.0

    # Start zoomed to a readable ~120 s window at the first go cue (like the matplotlib
    # scroller's default window), with a rangeslider so the whole session can be scrubbed --
    # the plotly analog of the scroller's arrow-key panning.
    t0 = go_cue_times.min() if len(go_cue_times) else x_first
    fig.update_layout(
        title=session_id or "Session Scroller",
        xaxis_title="Time (s)",
        yaxis=dict(tickvals=yticks, ticktext=ylabels,
                   range=[params["behavior_bottom"] - 0.05, y_top + 0.1]),
        xaxis=dict(range=[t0, t0 + 120], rangeslider=dict(visible=True, range=[x_first, x_last])),
        showlegend=True, height=600, width=1300, template="simple_white",
    )
    return fig
