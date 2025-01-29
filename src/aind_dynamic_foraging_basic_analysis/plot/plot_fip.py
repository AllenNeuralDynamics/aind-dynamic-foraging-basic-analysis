"""
Tools for plotting FIP data
"""

import matplotlib.pyplot as plt
import numpy as np

from aind_dynamic_foraging_data_utils import alignment as an
from aind_dynamic_foraging_data_utils import nwb_utils as nu
from aind_dynamic_foraging_basic_analysis.plot.style import STYLE, FIP_COLORS


def plot_fip_psth_compare_alignments(
    nwb, alignments, channel, tw=[-4, 4], censor=True, extra_colors={}
):
    """
    Compare the same FIP channel aligned to multiple event types
    nwb, nwb object for the session
    alignments, either a list of event types in df_events, or a dictionary
        whose keys are event types and values are a list of timepoints
    channel, (str) the name of the FIP channel
    tw, time window for the PSTH
    extra_colors (dict), a dictionary of extra colors.
        keys should be alignments, or colors are random

    EXAMPLE
    *******************
    plot_fip_psth_compare_alignments(nwb,['left_reward_delivery_time',
        'right_reward_delivery_time'],'G_1_preprocessed')
    """
    if not hasattr(nwb, "fip_df"):
        print("You need to compute the fip_df first")
        print("running `nwb.fip_df = create_fib_df(nwb,tidy=True)`")
        nwb.fip_df = nu.create_fib_df(nwb, tidy=True)
    if not hasattr(nwb, "df_events"):
        print("You need to compute the df_events first")
        print("run `nwb.df_events = create_events_df(nwb)`")
        nwb.df_events = nu.create_events_df(nwb)

    if channel not in nwb.fip_df["event"].values:
        print("channel {} not in fip_df".format(channel))

    if isinstance(alignments, list):
        align_dict = {}
        for a in alignments:
            if a not in nwb.df_events["event"].values:
                print("{} not found in the events table".format(a))
                return
            else:
                align_dict[a] = nwb.df_events.query("event == @a")["timestamps"].values
    elif isinstance(alignments, dict):
        align_dict = alignments
    else:
        print(
            "alignments must be either a list of events in nwb.df_events, "
            + "or a dictionary where each key is an event type, "
            + "and the value is a list of timepoints"
        )
        return

    censor_times = []
    for key in align_dict:
        censor_times.append(align_dict[key])
    censor_times = np.sort(np.concatenate(censor_times))

    align_label = "Time (s)"

    fig, ax = plt.subplots()

    colors = {**FIP_COLORS, **extra_colors}

    for alignment in align_dict:
        etr = fip_psth_inner_compute(
            nwb, align_dict[alignment], channel, True, tw, censor, censor_times
        )
        fip_psth_inner_plot(ax, etr, colors.get(alignment, ""), alignment)

    plt.legend()
    ax.set_xlabel(align_label, fontsize=STYLE["axis_fontsize"])
    ax.set_ylabel("df/f", fontsize=STYLE["axis_fontsize"])
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_xlim(tw)
    ax.axvline(0, color="k", alpha=0.2)
    ax.tick_params(axis="both", labelsize=STYLE["axis_ticks_fontsize"])
    ax.set_title(nwb.session_id, fontsize=STYLE["axis_fontsize"])
    plt.tight_layout()
    return fig, ax


def plot_fip_psth_compare_channels(
    nwb,
    align,
    tw=[-4, 4],
    channels=[
        "G_1_preprocessed",
        "G_2_preprocessed",
        "R_1_preprocessed",
        "R_2_preprocessed",
        "Iso_1_preprocessed",
        "Iso_2_preprocessed",
    ],
    censor=True,
):
    """
    nwb, the nwb object for the session of interest
    align should either be a string of the name of an event type in nwb.df_events,
        or a list of timepoints
    channels should be a list of channel names (strings)
    EXAMPLE
    ********************
    plot_fip_psth(nwb, 'goCue_start_time')
    """
    if not hasattr(nwb, "fip_df"):
        print("You need to compute the fip_df first")
        print("running `nwb.fip_df = create_fib_df(nwb,tidy=True)`")
        nwb.fip_df = nu.create_fib_df(nwb, tidy=True)
    if not hasattr(nwb, "df_events"):
        print("You need to compute the df_events first")
        print("run `nwb.df_events = create_events_df(nwb)`")
        nwb.df_events = nu.create_events_df(nwb)

    if isinstance(align, str):
        if align not in nwb.df_events["event"].values:
            print("{} not found in the events table".format(align))
            return
        align_timepoints = nwb.df_events.query("event == @align")["timestamps"].values
        align_label = "Time from {} (s)".format(align)
    else:
        align_timepoints = align
        align_label = "Time (s)"

    fig, ax = plt.subplots()

    colors = [FIP_COLORS.get(c, "") for c in channels]
    for dex, c in enumerate(channels):
        if c in nwb.fip_df["event"].values:
            etr = fip_psth_inner_compute(nwb, align_timepoints, c, True, tw, censor)
            fip_psth_inner_plot(ax, etr, colors[dex], c)
        else:
            print("No data for channel: {}".format(c))

    plt.legend()
    ax.set_xlabel(align_label, fontsize=STYLE["axis_fontsize"])
    ax.set_ylabel("df/f", fontsize=STYLE["axis_fontsize"])
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_xlim(tw)
    ax.axvline(0, color="k", alpha=0.2)
    ax.tick_params(axis="both", labelsize=STYLE["axis_ticks_fontsize"])
    ax.set_title(nwb.session_id)
    plt.tight_layout()
    return fig, ax


def fip_psth_inner_plot(ax, etr, color, label):
    """
    helper function that plots an event triggered response
    ax, the pyplot axis to plot on
    etr, the dataframe that contains the event triggered response
    color, the line color to plot
    label, the label for the etr
    """
    if color == "":
        cmap = plt.get_cmap("tab20")
        color = cmap(np.random.randint(20))
    ax.fill_between(etr.index, etr.data - etr["sem"], etr.data + etr["sem"], color=color, alpha=0.2)
    ax.plot(etr.index, etr.data, color=color, label=label)


def fip_psth_inner_compute(
    nwb, align_timepoints, channel, average, tw=[-1, 1], censor=True, censor_times=None
):
    """
    helper function that computes the event triggered response
    nwb, nwb object for the session of interest, should have fip_df attribute
    align_timepoints, an iterable list of the timepoints to compute the ETR aligned to
    channel, what channel in the fip_df dataframe to use
    average(bool), whether to return the average, or all individual traces
    tw, time window before and after each event
    """

    data = nwb.fip_df.query("event == @channel")
    etr = an.event_triggered_response(
        data,
        "timestamps",
        "data",
        align_timepoints,
        t_start=tw[0],
        t_end=tw[1],
        output_sampling_rate=40,
        censor=censor,
        censor_times=censor_times,
    )

    if average:
        mean = etr.groupby("time").mean()
        sem = etr.groupby("time").sem()
        mean["sem"] = sem["data"]
        return mean
    return etr


def plot_histogram(nwb, preprocessed=True, edge_percentile=2):
    """
    Generates a histogram of values of each FIP channel
    preprocessed (Bool), if True, uses the preprocessed channel
    edge_percentile (float), displays only the (2, 100-2) percentiles of the data

    EXAMPLE
    ***********************
    plot_histogram(nwb)
    """
    if not hasattr(nwb, "fip_df"):
        print("You need to compute the fip_df first")
        print("running `nwb.fip_df = create_fib_df(nwb,tidy=True)`")
        nwb.fip_df = nu.create_fib_df(nwb, tidy=True)
        return

    fig, ax = plt.subplots(3, 2, sharex=True)
    channels = ["G", "R", "Iso"]
    mins = []
    maxs = []
    for i, c in enumerate(channels):
        for j, count in enumerate(["1", "2"]):
            if preprocessed:
                dex = c + "_" + count + "_preprocessed"
            else:
                dex = c + "_" + count
            df = nwb.fip_df.query("event == @dex")
            ax[i, j].hist(df["data"], bins=1000, color=FIP_COLORS.get(dex, "k"))
            ax[i, j].spines["top"].set_visible(False)
            ax[i, j].spines["right"].set_visible(False)
            if preprocessed:
                ax[i, j].set_xlabel("df/f")
            else:
                ax[i, j].set_xlabel("f")
            ax[i, j].set_ylabel("count")
            ax[i, j].set_title(dex)
            mins.append(np.percentile(df["data"].values, edge_percentile))
            maxs.append(np.percentile(df["data"].values, 100 - edge_percentile))
    ax[0, 0].set_xlim(np.min(mins), np.max(maxs))
    fig.suptitle(nwb.session_id)
    plt.tight_layout()
