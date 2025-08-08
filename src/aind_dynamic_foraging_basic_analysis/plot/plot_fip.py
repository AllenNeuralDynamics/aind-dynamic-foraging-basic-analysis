"""
Tools for plotting FIP data
"""

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from aind_dynamic_foraging_data_utils import alignment as an
from aind_dynamic_foraging_data_utils import nwb_utils as nu
from aind_dynamic_foraging_basic_analysis.plot.style import STYLE, FIP_COLORS


def plot_fip_psth_compare_alignments(  # NOQA C901
    nwb,
    alignments,
    channel,
    tw=[-4, 4],
    ax=None,
    fig=None,
    censor=True,
    extra_colors={},
    data_column="data",
):
    """
    Compare the same FIP channel aligned to multiple event types
    nwb, nwb object for the session
    alignments, either a list of event types in df_events, or a dictionary
        whose keys are event types and values are a list of timepoints
    channel, (str) the name of the FIP channel
    tw, time window for the PSTH
    censor, censor important timepoints before and after aligned timepoints
    extra_colors (dict), a dictionary of extra colors.
        keys should be alignments, or colors are random
    data_column (string), name of data column in nwb.df_fip

    EXAMPLE
    *******************
    plot_fip_psth_compare_alignments(nwb,['left_reward_delivery_time',
        'right_reward_delivery_time'],'G_1_preprocessed')
    """
    # Check if nwb is a list, and if so, check only the first element for attributes and channel
    nwb_to_check = nwb[0] if isinstance(nwb, list) else nwb

    if not hasattr(nwb_to_check, "df_fip"):
        print("You need to compute the df_fip first")
        print("running `nwb.df_fip = create_fib_df(nwb,tidy=True)`")
        nwb_to_check.df_fip = nu.create_fib_df(nwb_to_check, tidy=True)
    if not hasattr(nwb_to_check, "df_events"):
        print("You need to compute the df_events first")
        print("run `nwb.df_events = create_events_df(nwb)`")
        nwb_to_check.df_events = nu.create_events_df(nwb_to_check)

    if channel not in nwb_to_check.df_fip["event"].values:
        print("channel {} not in df_fip".format(channel))

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
        align_dict_flat = alignments.copy()
    else:
        print(
            "alignments must be either a list of events in nwb.df_events, "
            + "or a dictionary where each key is an event type, "
            + "and the value is a list of timepoints"
        )
        return

    censor_times = []
    for key in align_dict:
        if isinstance(nwb, list):
            align_dict_flat[key] = np.concatenate(align_dict[key])
        censor_times.append(align_dict_flat[key])
    censor_times = np.sort(np.concatenate(censor_times))

    align_label = "Time (s)"
    if fig is None and ax is None:
        fig, ax = plt.subplots()

    colors = {**FIP_COLORS, **extra_colors}

    for alignment in align_dict:
        if isinstance(nwb, list):
            # Compute etr for every NWB object in the list and average
            etr = fip_psth_multiple_nwb_inner_compute(
                    nwb, align_dict[alignment], channel, True, tw, censor, censor_times, data_column
                )
            session_id_title = ', '.join([nwb_i.session_id for nwb_i in nwb])

        else:
            etr = fip_psth_inner_compute(
                    nwb, align_dict[alignment], channel, True, tw, censor, censor_times, data_column
                )
            session_id_title = nwb.session_id
        fip_psth_inner_plot(ax, etr, colors.get(alignment, ""), alignment, data_column)

    plt.legend()
    ax.set_xlabel(align_label, fontsize=STYLE["axis_fontsize"])
    ax.set_ylabel("df/f", fontsize=STYLE["axis_fontsize"])
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_xlim(tw)
    ax.axvline(0, color="k", alpha=0.2)
    ax.tick_params(axis="both", labelsize=STYLE["axis_ticks_fontsize"])
    ax.set_title(session_id_title, fontsize=STYLE["axis_fontsize"])
    plt.tight_layout()
    return fig, ax


def plot_fip_psth_compare_channels(
    nwb,
    align,
    tw=[-4, 4],
    ax=None,
    fig=None,
    channels=[
        "G_1_preprocessed",
        "G_2_preprocessed",
        "R_1_preprocessed",
        "R_2_preprocessed",
        "Iso_1_preprocessed",
        "Iso_2_preprocessed",
    ],
    censor=True,
    data_column="data",
):
    """
    nwb, the nwb object for the session of interest
    align should either be a string of the name of an event type in nwb.df_events,
        or a list of timepoints
    channels should be a list of channel names (strings)
    censor, censor important timepoints before and after aligned timepoints
    data_column (string), name of data column in nwb.df_fip

    EXAMPLE
    ********************
    plot_fip_psth(nwb, 'goCue_start_time')
    """
    if not hasattr(nwb, "df_fip"):
        print("You need to compute the df_fip first")
        print("running `nwb.df_fip = create_fib_df(nwb,tidy=True)`")
        nwb.df_fip = nu.create_fib_df(nwb, tidy=True)
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

    if fig is None and ax is None:
        fig, ax = plt.subplots()

    colors = [FIP_COLORS.get(c, "") for c in channels]
    for dex, c in enumerate(channels):
        if c in nwb.df_fip["event"].values:
            etr = fip_psth_inner_compute(nwb, align_timepoints, c, True, tw,
                                         censor, data_column=data_column)
            fip_psth_inner_plot(ax, etr, colors[dex], c, data_column)
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


def fip_psth_inner_plot(ax, etr, color, label, data_column):
    """
    helper function that plots an event triggered response
    ax, the pyplot axis to plot on
    etr, the dataframe that contains the event triggered response
    color, the line color to plot
    label, the label for the etr
    data_column (string), name of data_column
    """
    if color == "":
        cmap = plt.get_cmap("tab20")
        color = cmap(np.random.randint(20))
    ax.fill_between(etr.index, etr[data_column] - etr["sem"],
                    etr[data_column] + etr["sem"], color=color, alpha=0.2)
    ax.plot(etr.index, etr[data_column], color=color, label=label)

def fip_psth_multiple_nwb_inner_compute(
    nwbs_list,
    align_timepoints,
    channel,
    average,
    tw=[-1, 1],
    censor=True,
    censor_times=None,
    data_column="data",
):
    """
    helper function that computes the event triggered response
    nwb, nwb object for the session of interest, should have df_fip attribute
    align_timepoints, an iterable list of the timepoints to compute the ETR aligned to
    channel, what channel in the df_fip dataframe to use
    average(bool), whether to return the average, or all individual traces
    tw, time window before and after each event
    censor, censor important timepoints before and after aligned timepoints
    censor_times, timepoints to censor
    data_column (string), name of data column in nwb.df_fip

    """
    etr_list = []
    for (i,nwb) in enumerate(nwbs_list):
        data = nwb.df_fip.query("event == @channel")
        etr = an.event_triggered_response(
            data,
            "timestamps",
            data_column,
            align_timepoints[i],
            t_start=tw[0],
            t_end=tw[1],
            output_sampling_rate=40,
            censor=censor,
            censor_times=censor_times,
        )
        etr['ses_idx'] = data.ses_idx.values[0]
        etr_list.append(etr)

    etr_all = pd.concat(etr_list, axis=0).reset_index(drop=True)
    if average:
        # Average within each ses_idx for each time point
        mean_per_ses = etr_all.groupby(['ses_idx', 'time'])[data_column].mean().unstack('ses_idx')
        # Grand mean: average across ses_idx for each time point
        grand_mean = mean_per_ses.mean(axis=1)
        # SEM over ses_idx for each time point
        grand_sem = mean_per_ses.sem(axis=1)
        # Combine into a DataFrame
        result = grand_mean.to_frame(name=data_column)
        result['sem'] = grand_sem
        return result
    return etr_all


def fip_psth_inner_compute(
    nwb,
    align_timepoints,
    channel,
    average,
    tw=[-1, 1],
    censor=True,
    censor_times=None,
    data_column="data",
):
    """
    helper function that computes the event triggered response
    nwb, nwb object for the session of interest, should have df_fip attribute
    align_timepoints, an iterable list of the timepoints to compute the ETR aligned to
    channel, what channel in the df_fip dataframe to use
    average(bool), whether to return the average, or all individual traces
    tw, time window before and after each event
    censor, censor important timepoints before and after aligned timepoints
    censor_times, timepoints to censor
    data_column (string), name of data column in nwb.df_fip

    """

    data = nwb.df_fip.query("event == @channel")
    etr = an.event_triggered_response(
        data,
        "timestamps",
        data_column,
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
        mean["sem"] = sem[data_column]
        return mean
    return etr


def plot_histogram(nwb, preprocessed=True, edge_percentile=2, data_column="data"):
    """
    Generates a histogram of values of each FIP channel
    preprocessed (Bool), if True, uses the preprocessed channel
    edge_percentile (float), displays only the (2, 100-2) percentiles of the data
    data_column (string), name of data column in nwb.df_fip

    EXAMPLE
    ***********************
    plot_histogram(nwb)
    """
    if not hasattr(nwb, "df_fip"):
        print("You need to compute the df_fip first")
        print("running `nwb.df_fip = create_fib_df(nwb,tidy=True)`")
        nwb.df_fip = nu.create_fib_df(nwb, tidy=True)
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
            df = nwb.df_fip.query("event == @dex")
            ax[i, j].hist(df[data_column], bins=1000, color=FIP_COLORS.get(dex, "k"))
            ax[i, j].spines["top"].set_visible(False)
            ax[i, j].spines["right"].set_visible(False)
            if preprocessed:
                ax[i, j].set_xlabel("df/f")
            else:
                ax[i, j].set_xlabel("f")
            ax[i, j].set_ylabel("count")
            ax[i, j].set_title(dex)
            mins.append(np.percentile(df[data_column].values, edge_percentile))
            maxs.append(np.percentile(df[data_column].values, 100 - edge_percentile))
    ax[0, 0].set_xlim(np.min(mins), np.max(maxs))
    fig.suptitle(nwb.session_id)
    plt.tight_layout()
