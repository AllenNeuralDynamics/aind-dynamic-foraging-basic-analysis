import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from aind_dynamic_foraging_data_utils import nwb_utils as nu
import aind_dynamic_foraging_basic_analysis.metrics.trial_metrics as tm

"""

# TODO
Make specification for multisession trials df
Make some capsule to generate them, and make data assets, what metadata?
Add axis for licking/rewards
make bias/lickspout plots dynamically loaded
Investigate why some sessions crash on bias computation
Add some indicator for missing session
"""


def make_multisession_trials_df(nwb_list, DATA_DIR, AGG_DIR):
    """
    takes a list of NWBs
    loads each NWB file
    makes trials table
    adds metrics
    adds bias
    makes aggregate trials table
    saves aggregate trials table
    """
    nwbs = []
    for n in nwb_list:
        try:
            nwb = nu.load_nwb_from_filename(n)
            nwb.df_trials = nu.create_df_trials(nwb)
            nwb.df_trials = tm.compute_trial_metrics(nwb)
            nwb.df_trials = tm.compute_bias(nwb)
            nwbs.append(nwb)
        except:
            print("Bad {}".format(n))

    df = pd.concat([x.df_trials for x in nwbs])

    filename = os.path.join(AGG_DIR, nwb_list[0].split("/")[-1].split("_")[1] + ".csv")
    df.to_csv(filename)
    return nwbs, df


def plot_foraging_lifetime(lifetime_df, plot_list=["bias", "lickspout_position"]):
    """
    Takes a dataframe of the aggregate for all sessions from this animal

    """

    # Ensure dataframe is sorted by session then trial
    df = lifetime_df.copy()
    df = df.sort_values(by=["ses_idx", "trial"])
    df["lifetime_trial"] = df.reset_index().index

    # Set up figure
    fig, ax = plt.subplots(len(plot_list), 1, figsize=(14, 2 * len(plot_list)), sharex=True)

    # Plot each element
    for index, plot in enumerate(plot_list):
        plot_foraging_lifetime_inner(ax[index], plot, df)

    # Add session breaks to each axis
    session_breaks = list(df.query("trial == 0")["lifetime_trial"].values - 0.5) + [
        df["lifetime_trial"].values[-1]
    ]
    for a in ax:
        for x in session_breaks:
            a.axvline(x, color="gray", alpha=0.25, linestyle="--")

    # Determine xtick positions and labels
    ticks = list(df.query("trial == 0")["lifetime_trial"].values) + [
        df["lifetime_trial"].values[-1]
    ]
    ticks = ticks[:-1] + np.diff(ticks) / 2
    labels = [
        "-".join(x.split("_")[1].split("-")[1:]) for x in df.query("trial == 0")["ses_idx"].values
    ]

    # Add ticks to the bottom plot
    ax[-1].set_xticks(ticks, labels)
    ax[-1].set_xlabel("Session")
    ax[-1].set_xlim(df["lifetime_trial"].values[0], df["lifetime_trial"].values[-1])
    plt.suptitle(df["ses_idx"].values[0].split("_")[0])
    plt.tight_layout()

    def on_key_press(event):
        """
        Define interaction resonsivity
        """
        x = ax[0].get_xlim()
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
        ax[0].set_xlim(xmin, xmax)
        plt.draw()

    kpid = fig.canvas.mpl_connect("key_press_event", on_key_press)  # noqa: F841


def plot_foraging_lifetime_inner(ax, plot, df):
    ax.set_ylabel(plot)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    if plot == "bias":
        ax.plot(df["lifetime_trial"], df["bias"], label="bias")
        ax.axhline(0, linestyle="--", color="k", alpha=0.25)
        ax.set_ylim(-1, 1)
    elif plot == "lickspout_position":
        ax.plot(
            df["lifetime_trial"],
            df["lickspout_position_z"] - df["lickspout_position_z"].values[0],
            "k",
            label="z",
        )
        ax.plot(
            df["lifetime_trial"],
            df["lickspout_position_y"] - df["lickspout_position_y"].values[0],
            "r",
            label="y",
        )
        ax.plot(
            df["lifetime_trial"],
            df["lickspout_position_x"] - df["lickspout_position_x"].values[0],
            "b",
            label="x",
        )
        ax.set_ylabel("$\Delta$ lickspout")
    elif plot in df:
        ax.plot(df["lifetime_trial"], df[plot], label=plot)
    else:
        print("Unknown plot element: {}".format(plot))

    ax.legend(loc="upper left")
