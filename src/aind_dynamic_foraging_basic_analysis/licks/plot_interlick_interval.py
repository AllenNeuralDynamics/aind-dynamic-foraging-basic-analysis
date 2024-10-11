"""
    Compute and plot interlick interval, determine appropriate threshold for bout segmentation
"""

import matplotlib.pyplot as plt
import numpy as np
from aind_dynamic_foraging_basic_analysis.licks import annotation as a
from aind_dynamic_foraging_basic_analysis.plot.style import STYLE


def plot_interlick_interval(licks_df, key="pre_ili", categories=None, nbins=80, xmax=10):
    """
    Plots a histogram of <key> split by unique values of <categories>
    licks_df (dataframe)
    key (string), column of licks_df
    categories (string), column of licks_df with discrete values
    nbins (int), number of bins for histogram
    xmax (float), the maximum interlick interval to plot

    Plot interlick interval of all licks
    >> plot_interlick_interval(df_licks)

    plot interlick interval for left and right licks separately
    >> plot_interlick_interval(df_licks, categories='event')
    """

    if key not in licks_df:
        print('key "{}" not in dataframe'.format(key))
        return
    if (categories is not None) and (categories not in licks_df):
        print('categories "{}" not in dataframe'.format(categories))
        return

    # Remove NaNs (from start of session) and limit to range defined by xmax
    licks_df = licks_df.dropna(subset=[key]).query("{} < @xmax".format(key)).copy()

    if categories is not None:
        licks_df = licks_df.dropna(subset=[categories]).copy()
    if key == "pre_ili":
        xlabel = "interlick interval (s)"
        yscale = 4
    else:
        xlabel = key
        yscale = 1.5

    # Plot Figure
    fig, ax = plt.subplots(figsize=(5, 4))
    counts, edges = np.histogram(licks_df[key].values, nbins)
    if categories is None:
        # We have only one group of data
        plt.hist(
            licks_df[key].values,
            bins=edges,
            color=STYLE["data_color_all"],
            alpha=STYLE["data_alpha"],
        )
    else:
        # We have multiple groups of data
        groups = licks_df[categories].unique()
        colors = {"left_lick_time": "b", "right_lick_time": "r"}
        for index, g in enumerate(groups):
            df = licks_df.query(categories + " == @g")
            if isinstance(g, bool) or isinstance(g, np.bool_):
                if g:
                    label = categories
                else:
                    label = "not " + categories
                label = label
            else:
                label = g
            plt.hist(df[key].values, bins=edges, alpha=0.5, color=colors[g], label=label)

    # Clean up
    plt.ylim(top=np.sort(counts)[-2] * yscale)

    plt.xlim(0, xmax)
    plt.axvline(
        a.BOUT_THRESHOLD,
        color=STYLE["axline_color"],
        linestyle=STYLE["axline_linestyle"],
        alpha=STYLE["axline_alpha"],
        label="Licking bout threshold",
    )
    ax.set_ylabel("count", fontsize=STYLE["axis_fontsize"])
    ax.set_xlabel(xlabel, fontsize=STYLE["axis_fontsize"])
    plt.xticks(fontsize=STYLE["axis_ticks_fontsize"])
    plt.yticks(fontsize=STYLE["axis_ticks_fontsize"])
    plt.legend(frameon=False, fontsize=STYLE["axis_ticks_fontsize"])
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    if categories is not None:
        plt.legend(frameon=False, fontsize=STYLE["axis_ticks_fontsize"])
    plt.tight_layout()
