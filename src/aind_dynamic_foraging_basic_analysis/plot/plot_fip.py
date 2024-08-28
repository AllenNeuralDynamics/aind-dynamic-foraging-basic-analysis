import matplotlib.pyplot as plt
import numpy as np


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
