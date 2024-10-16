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
    nwb, ax=None, fig=None, plot_bouts=False, processing="bright"
):
    """
    Creates an interactive plot of the session.
    Plots left/right licks/rewards, and go cues

    pressing "left arrow" scrolls backwards in time
    pressing "right arrow" scrolls forwards in time
    pressing "up arrow" zooms out, in time
    pressing "down arrow" zooms in, in time

    nwb, an nwb like object that contains attributes: df_events, session_id
        and optionally contains attributes fip_df, df_licks

    ax is a pyplot figure axis. If None, a new figure is created

    plot_bouts (bool), if True, plot licks colored by segmented lick bouts

    EXAMPLES:
    plot_foraging_session.plot_session_scroller(nwb)
    plot_foraging_session.plot_session_scroller(nwb, plot_bouts=True)
    """

    if not hasattr(nwb, "df_events"):
        print("computing df_events first")
        nwb.df_events = nu.create_events_df(nwb)
        df_events = nwb.df_events
    else:
        df_events = nwb.df_events
    if hasattr(nwb, "fip_df"):
        fip_df = nwb.fip_df
    else:
        fip_df = None
    if hasattr(nwb, "df_licks") & plot_bouts:
        df_licks = nwb.df_licks
    elif plot_bouts:
        print("computing df_licks first")
        nwb.df_licks = a.annotate_lick_bouts(nwb)
        nwb.df_licks = a.annotate_rewards(nwb)
        df_licks = nwb.df_licks
    else:
        df_licks = None

    if ax is None:
        if fip_df is None:
            fig, ax = plt.subplots(figsize=(15, 3))
        else:
            fig, ax = plt.subplots(figsize=(15, 8))

    xmin = df_events.iloc[0]["timestamps"]
    xmax = xmin + 20
    ax.set_xlim(xmin, xmax)

    params = {
        "left_lick_bottom": 0,
        "left_lick_top": 0.25,
        "right_lick_bottom": 0.75,
        "right_lick_top": 1,
        "left_reward_bottom": 0.25,
        "left_reward_top": 0.5,
        "right_reward_bottom": 0.5,
        "right_reward_top": 0.75,
        "go_cue_bottom": 0,
        "go_cue_top": 1,
        "G_1_dff-bright_bottom": 1,
        "G_1_dff-bright_top": 2,
        "G_2_dff-bright_bottom": 2,
        "G_2_dff-bright_top": 3,
        "R_1_dff-bright_bottom": 3,
        "R_1_dff-bright_top": 4,
        "R_2_dff-bright_bottom": 4,
        "R_2_dff-bright_top": 5,
        "G_1_dff-poly_bottom": 1,
        "G_1_dff-poly_top": 2,
        "G_2_dff-poly_bottom": 2,
        "G_2_dff-poly_top": 3,
        "R_1_dff-poly_bottom": 3,
        "R_1_dff-poly_top": 4,
        "R_2_dff-poly_bottom": 4,
        "R_2_dff-poly_top": 5,
        "G_1_dff-exp_bottom": 1,
        "G_1_dff-exp_top": 2,
        "G_2_dff-exp_bottom": 2,
        "G_2_dff-exp_top": 3,
        "R_1_dff-exp_bottom": 3,
        "R_1_dff-exp_top": 4,
        "R_2_dff-exp_bottom": 4,
        "R_2_dff-exp_top": 5,
    }
    yticks = [
        (params["left_lick_top"] - params["left_lick_bottom"]) / 2 + params["left_lick_bottom"],
        (params["right_lick_top"] - params["right_lick_bottom"]) / 2 + params["right_lick_bottom"],
        (params["left_reward_top"] - params["left_reward_bottom"]) / 2
        + params["left_reward_bottom"],
        (params["right_reward_top"] - params["right_reward_bottom"]) / 2
        + params["right_reward_bottom"],
    ]
    ylabels = ["left licks", "right licks", "left reward", "right reward"]
    ycolors = ["k", "k", "r", "r"]

    if fip_df is not None:
        fip_channels = [
            "G_2_dff-{}".format(processing),
            "G_1_dff-{}".format(processing),
            "R_2_dff-{}".format(processing),
            "R_1_dff-{}".format(processing),
        ]
        present_channels = fip_df["event"].unique()
        for index, channel in enumerate(fip_channels):
            if channel in present_channels:
                yticks.append(
                    (params[channel + "_top"] - params[channel + "_bottom"]) * 1
                    + params[channel + "_bottom"]
                )
                ylabels.append(channel)
                color = FIP_COLORS.get(channel, "k")
                ycolors.append(color)
                C = fip_df.query("event == @channel").copy()
                d_min = C["data"].min()
                d_max = C["data"].max()
                d_range = d_max - d_min
                t1 = 0.25
                t2 = 0.5
                t3 = 0.75
                p1 = d_min + t1 * d_range
                p2 = d_min + t2 * d_range
                p3 = d_min + t3 * d_range
                yticks.append(
                    params[channel + "_bottom"]
                    + (params[channel + "_top"] - params[channel + "_bottom"]) * t1
                )
                yticks.append(
                    params[channel + "_bottom"]
                    + (params[channel + "_top"] - params[channel + "_bottom"]) * t2
                )
                yticks.append(
                    params[channel + "_bottom"]
                    + (params[channel + "_top"] - params[channel + "_bottom"]) * t3
                )
                ylabels.append(str(round(p1, 3)))
                ylabels.append(str(round(p2, 3)))
                ylabels.append(str(round(p3, 3)))
                ycolors.append("darkgray")
                ycolors.append("darkgray")
                ycolors.append("darkgray")
                C["data"] = C["data"] - d_min
                C["data"] = C["data"].values / (d_max - d_min)
                C["data"] += params[channel + "_bottom"]
                ax.plot(C.timestamps.values, C.data.values, color)
                ax.axhline(params[channel + "_bottom"], color="k", linewidth=0.5, alpha=0.25)

    if df_licks is None:
        left_licks = df_events.query('event == "left_lick_time"')
        left_times = left_licks.timestamps.values
        ax.vlines(
            left_times,
            params["left_lick_bottom"],
            params["left_lick_top"],
            alpha=1,
            linewidth=2,
            color="k",
        )

        right_licks = df_events.query('event == "right_lick_time"')
        right_times = right_licks.timestamps.values
        ax.vlines(
            right_times,
            params["right_lick_bottom"],
            params["right_lick_top"],
            alpha=1,
            linewidth=2,
            color="k",
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
            ax.vlines(
                bout_left_licks,
                params["left_lick_bottom"],
                params["left_lick_top"],
                alpha=1,
                linewidth=2,
                color=cmap(np.mod(b, 20)),
            )
            ax.vlines(
                bout_right_licks,
                params["right_lick_bottom"],
                params["right_lick_top"],
                alpha=1,
                linewidth=2,
                color=cmap(np.mod(b, 20)),
            )
        left_rewarded_licks = df_licks.query(
            '(event == "left_lick_time")&(rewarded)'
        ).timestamps.values
        ax.plot(left_rewarded_licks, [params["left_lick_top"]] * len(left_rewarded_licks), "ro")
        right_rewarded_licks = df_licks.query(
            '(event == "right_lick_time")&(rewarded)'
        ).timestamps.values
        ax.plot(
            right_rewarded_licks, [params["right_lick_bottom"]] * len(right_rewarded_licks), "ro"
        )

    left_reward_deliverys = df_events.query('event == "left_reward_delivery_time"')
    left_times = left_reward_deliverys.timestamps.values
    ax.vlines(
        left_times,
        params["left_reward_bottom"],
        params["left_reward_top"],
        alpha=1,
        linewidth=2,
        color="r",
    )

    right_reward_deliverys = df_events.query('event == "right_reward_delivery_time"')
    right_times = right_reward_deliverys.timestamps.values
    ax.vlines(
        right_times,
        params["right_reward_bottom"],
        params["right_reward_top"],
        alpha=1,
        linewidth=2,
        color="r",
    )

    go_cues = df_events.query('event == "goCue_start_time"')
    go_cue_times = go_cues.timestamps.values
    ax.vlines(
        go_cue_times,
        params["go_cue_bottom"],
        params["go_cue_top"],
        alpha=0.75,
        linewidth=1,
        color="b",
    )

    # Clean up plot
    ax.set_yticks(yticks)
    ax.set_yticklabels(ylabels, fontsize=STYLE["axis_ticks_fontsize"])

    for tick, color in zip(ax.get_yticklabels(), ycolors):
        tick.set_color(color)
    ax.set_xlabel("time (s)", fontsize=STYLE["axis_fontsize"])
    if fip_df is None:
        ax.set_ylim(0, 1)
    else:
        ax.set_ylim(0, 5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    if fip_df is not None:
        ax.set_title(nwb.session_id + ", FIP processing: {}".format(processing))
    else:
        ax.set_title(nwb.session_id)
    plt.tight_layout()

    def on_key_press(event):
        """
        Define interaction resonsivity
        """
        x = ax.get_xlim()
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
        ax.set_xlim(xmin, xmax)
        plt.draw()

    kpid = fig.canvas.mpl_connect("key_press_event", on_key_press)  # noqa: F841

    return fig, ax
