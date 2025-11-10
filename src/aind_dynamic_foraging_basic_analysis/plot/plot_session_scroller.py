"""Plot foraging session in a standard format.
This is supposed to be reused in plotting real data or simulation data to ensure
a consistent visual representation.
"""

import numpy as np
from matplotlib import pyplot as plt

from aind_dynamic_foraging_data_utils import nwb_utils as nu
from aind_dynamic_foraging_basic_analysis.licks import annotation as a
from aind_dynamic_foraging_basic_analysis.plot.style import (
    STYLE,
    FIP_COLORS,
)


def plot_session_scroller(  # noqa: C901 pragma: no cover
    nwb,
    ax=None,
    fig=None,
    metrics=[],
    plot_list=[
        "bouts",
        "go cue",
    ],
    fip=[],
):
    """
    Creates an interactive plot of the session.
    Plots left/right licks/rewards, and go cues

    pressing "left arrow" scrolls backwards in time
    pressing "right arrow" scrolls forwards in time
    pressing "up arrow" zooms out, in time
    pressing "down arrow" zooms in, in time
    prssing "h", for home, sets the xlimits to (first event, last event)

    nwb, an nwb like object that contains attributes: df_events, session_id
        and optionally contains attributes fip_df, df_licks

    ax is a list of pyplot figure axis. The list must be the correct length of
            1 + len(metrics) + len(fip). If provided, fig must also be provided.
            If None, a new figure is created.


    fig is a pyplot figure container. If provided, ax must also be provided.
             If None, a new figure is created.

    metrics, list of metrics to plot. Each metric must be a column of
        nwb.df_trials
        This can be formatted in several ways
        a string containing the name of the metric
            example: "response_rate"
        a list that contains the metric
            example: ["response_rate"]
        a list that contains the metric and ylimits
            example: ["response_rate",(0,1)]
        a list that contains multiple metrics and ylimits
            example: ["response_rate","side_bias",(-1,1)]
        If ylimits are not supplied, then the limits of the data are used
        The ylabel will be the metric name unless multiple metrics are used
            then the ylabel is "metrics", and a legend is added

    plot_list (list of strings), list of annotations and features to plot

    fip (list), FIP channels to plot. Must be present in nwb.df_fip

    EXAMPLES:
    plot_foraging_session.plot_session_scroller(nwb)
    """

    approved_plot_list = [
        "bouts",
        "go cue",
        "rewarded lick",
        "cue response",
        "baiting",
        "lick artifacts",
        "manual rewards",
        "auto rewards",
    ]
    unapproved = set(plot_list) - set(approved_plot_list)
    if len(unapproved) > 0:
        print("Unknown plot_list options: {}".format(list(unapproved)))
        return

    if not hasattr(nwb, "df_events"):
        print("computing df_events first")
        nwb.df_events = nu.create_df_events(nwb)
        df_events = nwb.df_events
    else:
        df_events = nwb.df_events
    if hasattr(nwb, "df_fip"):
        fip_df = nwb.df_fip
    if hasattr(nwb, "df_licks"):
        df_licks = nwb.df_licks
    elif ("bouts" in plot_list) or ("cue response" in plot_list) or ("rewarded lick" in plot_list):
        print("computing df_licks first")
        nwb.df_licks = a.annotate_licks(nwb)
        df_licks = nwb.df_licks
    else:
        df_licks = None
    if not hasattr(nwb, "df_trials"):
        print("computing df_trials")
        nwb.df_trials = nu.create_df_trials(nwb)
        df_trials = nwb.df_trials
    else:
        df_trials = nwb.df_trials

    num_plots = 1 + len(metrics) + len(fip)

    if (ax is None) or len(ax) != num_plots:
        fig, ax = plt.subplots(
            num_plots,
            1,
            sharex=True,
            figsize=(15, 2 * num_plots + 1),
            height_ratios=[2 / 3] * (num_plots - 1) + [1],
        )
        fig.subplots_adjust(hspace=0)
        if num_plots == 1:
            ax = [ax]

    xmin = df_events.iloc[0]["timestamps"]
    x_first = xmin
    x_last = df_events.iloc[-1]["timestamps"]
    xmax = xmin + 20
    ax[-1].set_xlim(xmin, xmax)

    params = {
        "left_lick_bottom": 0,
        "left_lick_top": 0.25,
        "right_lick_bottom": 1.25,
        "right_lick_top": 1.5,
        "left_reward_bottom": 0.25,
        "left_reward_top": 0.5,
        "right_reward_bottom": 1,
        "right_reward_top": 1.25,
        "probs_bottom": 0.75,
        "probs_top": 1.5,
        "go_cue_bottom": 0,
        "go_cue_top": 1.5,
    }
    yticks = [
        (params["left_lick_top"] - params["left_lick_bottom"]) / 2 + params["left_lick_bottom"],
        (params["right_lick_top"] - params["right_lick_bottom"]) / 2 + params["right_lick_bottom"],
        (params["left_reward_top"] - params["left_reward_bottom"]) / 2
        + params["left_reward_bottom"],
        (params["right_reward_top"] - params["right_reward_bottom"]) / 2
        + params["right_reward_bottom"],
        (params["probs_top"] - params["probs_bottom"]) * -0.33 + params["probs_bottom"],
        (params["probs_top"] - params["probs_bottom"]) * 0 + params["probs_bottom"],
        (params["probs_top"] - params["probs_bottom"]) * 0.33 + params["probs_bottom"],
    ]
    ylabels = [
        "left licks",
        "right licks",
        "left reward",
        "right reward",
        "pL = 1",
        "0",
        "pR = 1",
    ]
    ycolors = ["b", "r", "b", "r", "darkgray", "darkgray", "darkgray"]

    if ("bouts" not in plot_list) or (df_licks is None):
        left_licks = df_events.query('event == "left_lick_time"')
        left_times = left_licks.timestamps.values
        ax[-1].vlines(
            left_times,
            params["left_lick_bottom"],
            params["left_lick_top"],
            alpha=1,
            linewidth=2,
            color="gray",
        )

        right_licks = df_events.query('event == "right_lick_time"')
        right_times = right_licks.timestamps.values
        ax[-1].vlines(
            right_times,
            params["right_lick_bottom"],
            params["right_lick_top"],
            alpha=1,
            linewidth=2,
            color="gray",
        )
    else:
        cmap = plt.get_cmap("tab20")
        bouts = df_licks.bout_number.unique()
        for b in bouts:
            bout_left_licks = df_licks.query(
                '(bout_number == @b)&(event=="left_lick_time")'
            ).timestamps.values
            bout_right_licks = df_licks.query(
                '(bout_number == @b)&(event=="right_lick_time")'
            ).timestamps.values
            ax[-1].vlines(
                bout_left_licks,
                params["left_lick_bottom"],
                params["left_lick_top"],
                alpha=1,
                linewidth=2,
                color=cmap(np.mod(b, 20)),
            )
            ax[-1].vlines(
                bout_right_licks,
                params["right_lick_bottom"],
                params["right_lick_top"],
                alpha=1,
                linewidth=2,
                color=cmap(np.mod(b, 20)),
            )

    if ("rewarded lick" in plot_list) and (df_licks is not None):
        # plot licks that trigger left rewards
        left_rewarded_licks = df_licks.query(
            '(event == "left_lick_time")&(rewarded)'
        ).timestamps.values
        ax[-1].plot(
            left_rewarded_licks,
            [params["left_lick_top"]] * len(left_rewarded_licks),
            "ro",
            label="rewarded lick",
        )

        # Plot licks that trigger right rewards
        right_rewarded_licks = df_licks.query(
            '(event == "right_lick_time")&(rewarded)'
        ).timestamps.values
        ax[-1].plot(
            right_rewarded_licks, [params["right_lick_bottom"]] * len(right_rewarded_licks), "ro"
        )

    if ("cue response" in plot_list) and (df_licks is not None):
        # plot cue responsive licks
        left_cue_licks = df_licks.query(
            '(event == "left_lick_time")&(cue_response)'
        ).timestamps.values
        ax[-1].plot(
            left_cue_licks,
            [
                params["left_lick_bottom"]
                + (params["left_lick_top"] - params["left_lick_bottom"]) / 2
            ]
            * len(left_cue_licks),
            "bD",
            label="cue responsive lick",
        )
        right_cue_licks = df_licks.query(
            '(event == "right_lick_time")&(cue_response)'
        ).timestamps.values
        ax[-1].plot(
            right_cue_licks,
            [
                params["right_lick_bottom"]
                + (params["right_lick_top"] - params["right_lick_bottom"]) / 2
            ]
            * len(right_cue_licks),
            "bD",
        )

    if "baiting" in plot_list:
        # Plot baiting
        bait_right = df_trials.query("bait_right")["goCue_start_time_in_session"].values
        bait_left = df_trials.query("bait_left")["goCue_start_time_in_session"].values
        ax[-1].plot(
            bait_right, [params["right_lick_top"] - 0.05] * len(bait_right), "ms", label="baited"
        )
        ax[-1].plot(bait_left, [params["left_lick_bottom"] + 0.05] * len(bait_left), "ms")

    if "lick artifacts" in plot_list:
        artifacts_right = df_licks.query('likely_artifact and (event=="right_lick_time")')[
            "timestamps"
        ].values
        artifacts_left = df_licks.query('likely_artifact and (event=="left_lick_time")')[
            "timestamps"
        ].values
        ax[-1].plot(
            artifacts_right,
            [params["right_lick_top"]] * len(artifacts_right),
            "d",
            color="darkorange",
            label="lick artifact",
        )
        ax[-1].plot(
            artifacts_left,
            [params["left_lick_bottom"]] * len(artifacts_left),
            "d",
            color="darkorange",
        )

    left_reward_deliverys = df_events.query('event == "left_reward_delivery_time"')
    left_times = left_reward_deliverys.timestamps.values
    ax[-1].vlines(
        left_times,
        params["left_reward_bottom"],
        params["left_reward_top"],
        alpha=1,
        linewidth=2,
        color="black",
    )

    right_reward_deliverys = df_events.query('event == "right_reward_delivery_time"')
    right_times = right_reward_deliverys.timestamps.values
    ax[-1].vlines(
        right_times,
        params["right_reward_bottom"],
        params["right_reward_top"],
        alpha=1,
        linewidth=2,
        color="black",
    )

    if "manual rewards" in plot_list:
        manual_left_times = left_reward_deliverys.query('data == "manual"').timestamps.values
        ax[-1].vlines(
            manual_left_times,
            params["left_reward_bottom"],
            params["left_reward_top"],
            alpha=1,
            linewidth=2,
            color="cyan",
            label="manual reward",
        )
        manual_right_times = right_reward_deliverys.query('data == "manual"').timestamps.values
        ax[-1].vlines(
            manual_right_times,
            params["right_reward_bottom"],
            params["right_reward_top"],
            alpha=1,
            linewidth=2,
            color="cyan",
        )
    if "auto rewards" in plot_list:
        auto_left_times = left_reward_deliverys.query('data == "auto"').timestamps.values
        ax[-1].vlines(
            auto_left_times,
            params["left_reward_bottom"],
            params["left_reward_top"],
            alpha=1,
            linewidth=2,
            color="royalblue",
            label="auto reward",
        )
        auto_right_times = right_reward_deliverys.query('data == "auto"').timestamps.values
        ax[-1].vlines(
            auto_right_times,
            params["right_reward_bottom"],
            params["right_reward_top"],
            alpha=1,
            linewidth=2,
            color="royalblue",
        )

    go_cues = df_events.query('event == "goCue_start_time"')
    go_cue_times = go_cues.timestamps.values
    if "go cue" in plot_list:
        ax[-1].vlines(
            go_cue_times,
            params["left_lick_bottom"],
            params["left_reward_top"],
            alpha=0.75,
            linewidth=1,
            color="b",
            label="go cue",
        )
        ax[-1].vlines(
            go_cue_times,
            params["right_reward_bottom"],
            params["right_lick_top"],
            alpha=0.75,
            linewidth=1,
            color="b",
        )

    # plot metrics
    ax[-1].axhline(params["right_lick_top"], color="k", linewidth=0.5, alpha=0.25)
    go_cue_times_doubled = np.repeat(go_cue_times, 2)[1:]

    pR = params["probs_bottom"] + df_trials["reward_probabilityR"] / 4
    pR = np.repeat(pR, 2)[:-1]
    ax[-1].fill_between(go_cue_times_doubled, params["probs_bottom"], pR, color="r", alpha=0.4)

    pL = params["probs_bottom"] - df_trials["reward_probabilityL"] / 4
    pL = np.repeat(pL, 2)[:-1]

    ax[-1].fill_between(go_cue_times_doubled, pL, params["probs_bottom"], color="b", alpha=0.4)

    # plot metrics if they are available
    for index, metric in enumerate(metrics):
        plot_metric(df_trials, go_cue_times, metric, ax[len(fip) + index])

    # plot fip if they are available:
    for index, f in enumerate(fip):
        plot_fip(fip_df, f, ax[index])

    # Clean up plot
    if len(plot_list) > 0:
        ax[-1].legend(framealpha=1, loc="lower left", reverse=True)
    ax[-1].set_xlabel("time (s)", fontsize=STYLE["axis_fontsize"])
    ax[-1].set_ylim(0, 1.5)
    ax[-1].set_yticks(yticks)
    ax[-1].set_yticklabels(ylabels, fontsize=STYLE["axis_ticks_fontsize"])
    for tick, color in zip(ax[-1].get_yticklabels(), ycolors):
        tick.set_color(color)

    for my_ax in ax:
        my_ax.spines["top"].set_visible(False)
        my_ax.spines["right"].set_visible(False)

    ax[0].set_title(nwb.session_id)

    if num_plots == 1:
        plt.tight_layout()

    def on_key_press(event):
        """
        Define interaction resonsivity
        """
        x = ax[-1].get_xlim()
        xmin = x[0]
        xmax = x[1]
        xStep = (xmax - xmin) / 4
        if event.key == "<" or event.key == "," or event.key == "left":
            xmin -= xStep
            xmax -= xStep
        elif event.key == ">" or event.key == "." or event.key == "right":
            xmin += xStep
            xmax += xStep
        elif event.key == "up":
            xmin -= xStep
            xmax += xStep
        elif event.key == "down":
            xmin += xStep * (2 / 3)
            xmax -= xStep * (2 / 3)
        elif event.key == "h":
            xmin = x_first
            xmax = x_last
        ax[-1].set_xlim(xmin, xmax)
        plt.draw()

    kpid = fig.canvas.mpl_connect("key_press_event", on_key_press)  # noqa: F841

    return fig, ax


def plot_metric(df_trials, go_cue_times, metric, ax):
    """
    Plots a metric from df_trials

    df_trials, dataframe that contains the metric as a column
    go_cue_times, array of xvalue time points
    metric, what metric to plot. This can be formatted in several ways
        a string containing the name of the metric
            example: "response_rate"
        a list that contains the metric
            example: ["response_rate"]
        a list that contains the metric and ylimits
            example: ["response_rate",(0,1)]
        a list that contains multiple metrics and ylimits
            example: ["response_rate","side_bias",(-1,1)]
        If ylimits are not supplied, then the limits of the data are used
        The ylabel will be the metric name unless multiple metrics are used
            then the ylabel is "metrics", and a legend is added
    ax, the axis to plot on
    """

    # Parse input
    if isinstance(metric, str):
        metric_names = [metric]
        ylims = None
        ylabel = metric
    elif len(metric) == 1:
        metric_names = [metric[0]]
        ylims = None
        ylabel = metric[0]
    elif len(metric) == 2:
        metric_names = [metric[0]]
        ylims = metric[1]
        ylabel = metric[0]
    elif len(metric) > 2:
        metric_names = metric[0:-1]
        ylims = metric[-1]
        ylabel = "metric"
    else:
        raise Exception("Bad metric format, must be [metric_names,ylims]: {}".format(metric))

    # plot metrics for this axis
    for m in metric_names:
        if m in df_trials:
            ax.plot(go_cue_times, df_trials[m], label=m, alpha=0.7)
        else:
            raise Exception("metric not in df_trials: {}".format(m))

    # determine if we need a legend
    if len(metric_names) > 1:
        ax.legend()

    # Set ylims is supplied
    if ylims is not None:
        ax.set_ylim(ylims)

    # Set ylabel
    ax.set_ylabel(ylabel, fontsize=12)


def plot_fip(fip_df, channel, ax):
    """
    Plot an FIP channel
    """

    if fip_df is None:
        raise Exception("Cannot plot FIP, no FIP data")

    if channel not in fip_df["event"].unique():
        raise Exception("Cannot plot {}, no data".format(channel))

    color = get_fip_color(channel)
    C = fip_df.query("event == @channel")
    ax.plot(C.timestamps.values, C.data.values, color)
    ax.axhline(0, color="k", linewidth=0.5, alpha=0.25)
    ax.set_ylabel(channel, fontsize=12)


def get_fip_color(channel):
    """
    Gets the color for FIP
    if the channel is defined in style.FIP_COLORS, use that
    otherwise look for the root "G" in "G_0_dff"
    otherwise use black
    """
    if channel in FIP_COLORS:
        return FIP_COLORS.get(channel)

    root = channel.split("_")[0]
    if root in FIP_COLORS:
        return FIP_COLORS.get(root)

    return "k"
