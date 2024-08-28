"""
Tools for plotting FIP data
"""

import matplotlib.pyplot as plt
import numpy as np

from aind_dynamic_foraging_data_utils import alignment as an


def plot_fip_psth(nwb, align,tw=[-1,1]):
    '''
        TODO, need to censor by next event
        todo, clean up plots
        todo, should auto compute fip_df, and df_events if needed
        todo, figure out a modular system for comparing alignments, and channels
        todo, add error bars
        todo, annotate licks into bouts, start of bout, etc 
    '''
    if not hasattr(nwb, "fip_df"):
        print("You need to compute the fip_df first")
        print("run `nwb.fip_df = create_fib_df(nwb,tidy=True)`")
        return
    if not hasattr(nwb, "df_events"):
        print("You need to compute the df_events first")
        print("run `nwb.df_events = create_events_df(nwb,tidy=True)`")
        return

    if isinstance(align,str) and (align not in nwb.df_events["event"].values):
        print("{} not found in the events table".format(align))
        return

    fig, ax = plt.subplots()
    channels = [
        "G_1_preprocessed",
        "G_2_preprocessed",
        "R_1_preprocessed",
        "R_2_preprocessed",
        "Iso_1_preprocessed",
        "Iso_2_preprocessed",
    ]
    colors = ["g", "darkgreen", "r", "darkred", "black", "gray"]
    for dex, c in enumerate(channels):
        etr = fip_psth_inner(nwb, align, c, True,tw)
        ax.fill_between(etr.index, etr.data-etr['sem'], etr.data+etr['sem'], color=colors[dex], alpha=.2)
        ax.plot(etr.index, etr.data, colors[dex], label=c)

    plt.legend()
    ax.set_xlabel("Time from {} (s)".format(align))


def fip_psth_inner(nwb, align, channel, average,tw=[-1,1]):
    '''
        TODO, this should just take timepoints for align, and handle everything else elsewhere
    '''
    if isinstance(align, str):
        align_timepoints = nwb.df_events.query("event == @align")["timestamps"].values
    else:
        align_timepoints = align
    data = nwb.fip_df.query("event == @channel")
    etr = an.event_triggered_response(
        data, "timestamps", "data", align_timepoints, t_start=tw[0], t_end=tw[1], output_sampling_rate=40
    )

    if average:
        mean = etr.groupby("time").mean()
        sem = etr.groupby('time').sem()
        mean['sem'] = sem['data']
        return mean
    return etr


def plot_histogram(nwb, preprocessed=True, edge_percentile=2):
    """
    Generates a histogram of values of each FIP channel
    preprocessed (Bool), if True, uses the preprocessed channel
    edge_percentile (float), displays only the (2, 100-2) percentiles of the data
    """
    if not hasattr(nwb, "fip_df"):
        print("You need to compute the fip_df first")
        print("run `nwb.fip_df = create_fib_df(nwb,tidy=True)`")
        return
    fig, ax = plt.subplots(3, 2, sharex=True)
    channels = ["G", "R", "Iso"]
    colors = ["g", "r", "k"]
    mins = []
    maxs = []
    for i, c in enumerate(channels):
        for j, count in enumerate(["1", "2"]):
            if preprocessed:
                dex = c + "_" + count + "_preprocessed"
            else:
                dex = c + "_" + count
            df = nwb.fip_df.query("event == @dex")
            ax[i, j].hist(df["data"], bins=1000, color=colors[i])
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
