import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from aind_ephys_utils import align
from pynwb import NWBHDF5IO


def loadnwb(nwb_file):
    io = NWBHDF5IO(nwb_file, mode="r")
    nwb = io.read()
    return nwb


def plotLickAnalysis(nwb):
    tblTrials = nwb.trials.to_dataframe()
    gs = gridspec.GridSpec(
        3,
        6,
        width_ratios=np.ones(6).tolist(),
        height_ratios=np.ones(3).tolist(),
    )
    fig = plt.figure(figsize=(15, 8))
    plt.subplots_adjust(hspace=0.5, wspace=0.5)

    leftLicks = nwb.acquisition["left_lick_time"].timestamps[:]
    rightLicks = nwb.acquisition["right_lick_time"].timestamps[:]
    allLicks = np.sort(np.concatenate((rightLicks, leftLicks)))
    sortInd = np.argsort(np.concatenate((rightLicks, leftLicks)))
    allLicksID = np.concatenate(
        (np.ones_like(rightLicks), np.zeros_like(leftLicks))
    )
    allLicksID = allLicksID[sortInd]
    allLickDiffs = np.diff(allLicks)

    ax = fig.add_subplot(gs[0, 0])
    ax.hist(
        1000 * allLickDiffs[(allLicksID[:-1] == 0) & (allLicksID[1:] == 0)],
        bins=np.arange(0, 2500, 20),
    )
    ax.set_title("ILI_L-L")
    ax.set_xlabel("ms")
    ax = fig.add_subplot(gs[0, 1])
    ax.hist(
        1000 * allLickDiffs[(allLicksID[:-1] == 1) & (allLicksID[1:] == 1)],
        bins=np.arange(0, 2500, 20),
    )
    ax.set_title("ILI_R-R")
    ax.set_xlabel("ms")
    ax = fig.add_subplot(gs[0, 2])
    ax.hist(
        1000 * allLickDiffs[(allLicksID[:-1] == 1) & (allLicksID[1:] == 0)],
        bins=np.arange(0, 2500, 20),
    )
    ax.set_title("ILI_R-L")
    ax.set_xlabel("ms")
    ax = fig.add_subplot(gs[0, 3])
    ax.hist(
        1000 * allLickDiffs[(allLicksID[:-1] == 0) & (allLicksID[1:] == 1)],
        bins=np.arange(0, 2500, 20),
    )
    ax.set_title("ILI_L-R")
    ax.set_xlabel("ms")
    ax = fig.add_subplot(gs[0, 4])
    ax.hist(1000 * allLickDiffs, bins=np.arange(0, 2500, 20))
    ax.set_title("ILI")
    ax.set_xlabel("ms")
    # pre trial lick punishment
    lickDelay = (
        tblTrials["goCue_start_time"]
        - tblTrials["start_time"]
        - tblTrials["ITI_duration"]
    )
    ax = fig.add_subplot(gs[0, 5])
    ax.hist(lickDelay[tblTrials["animal_response"] == 1], alpha=0.5, label="R")
    ax.hist(lickDelay[tblTrials["animal_response"] == 0], alpha=0.5, label="L")
    ax.hist(lickDelay[tblTrials["animal_response"] == 2], alpha=0.5, label="0")
    pEarly = 1 - np.mean(lickDelay < tblTrials["delay_duration"] + 0.05)
    ax.set_title(f"Time out punishment prop {pEarly: .2f}")
    ax.set_xlabel("s")
    ax.legend()
    # pre trial licks
    tb = -5
    tf = 10
    binSize = 100 / 1000
    stepSize = 50 / 1000
    edges = np.arange(tb + 0.5 * binSize, tf - 0.5 * binSize, stepSize)

    sortIndL = np.argsort(
        tblTrials.loc[tblTrials["animal_response"] == 0, "delay_duration"]
    )
    sortIndR = np.argsort(
        tblTrials.loc[tblTrials["animal_response"] == 1, "delay_duration"]
    )
    LAlign = tblTrials.loc[
        tblTrials["animal_response"] == 0, "goCue_start_time"
    ].values
    RAlign = tblTrials.loc[
        tblTrials["animal_response"] == 1, "goCue_start_time"
    ].values
    LAlign = LAlign[sortIndL.values]
    RAlign = RAlign[sortIndR.values]

    if len(LAlign) > 0:
        # left align to left
        ax = fig.add_subplot(gs[1, 0])
        df = align.to_events(leftLicks, LAlign, (tb, tf), return_df=True)
        ax.scatter(df.time, df.event_index, c="k", marker="|", s=1, zorder=2)
        ax.scatter(
            -np.sort(
                tblTrials.loc[
                    tblTrials["animal_response"] == 0, "delay_duration"
                ]
            ),
            range(
                len(
                    tblTrials.loc[
                        tblTrials["animal_response"] == 0, "delay_max"
                    ]
                )
            ),
            c="b",
            marker="|",
            s=1,
            zorder=1,
        )
        ax.axvline(x=0, c="r", ls="--", lw=1, zorder=3)
        ax.set_title("L licks on L choice trials")

        ax = fig.add_subplot(gs[2, 0])
        countsPre = np.searchsorted(
            np.sort(df.time.values), edges - 0.5 * binSize
        )
        countsPost = np.searchsorted(
            np.sort(df.time.values), edges + 0.5 * binSize
        )
        lickRate = (countsPost - countsPre) / (binSize * len(LAlign))
        ax.plot(edges, lickRate)
        ax.set_title("lickRate")
        ax.set_xlabel("Time from go cue (s)")

        # right align to left
        ax = fig.add_subplot(gs[1, 3])
        df = align.to_events(rightLicks, LAlign, (tb, tf), return_df=True)
        ax.scatter(df.time, df.event_index, c="k", marker="|", s=1, zorder=2)
        ax.scatter(
            -np.sort(
                tblTrials.loc[
                    tblTrials["animal_response"] == 0, "delay_duration"
                ]
            ),
            range(
                len(
                    tblTrials.loc[
                        tblTrials["animal_response"] == 0, "delay_max"
                    ]
                )
            ),
            c="b",
            marker="|",
            s=1,
            zorder=1,
        )
        ax.axvline(x=0, c="r", ls="--", lw=1, zorder=3)
        ax.set_title("R licks on L choice trials")

        ax = fig.add_subplot(gs[2, 3])
        countsPre = np.searchsorted(
            np.sort(df.time.values), edges - 0.5 * binSize
        )
        countsPost = np.searchsorted(
            np.sort(df.time.values), edges + 0.5 * binSize
        )
        lickRate = (countsPost - countsPre) / (binSize * len(LAlign))
        ax.plot(edges, lickRate)
        ax.set_title("lickRate")
        ax.set_xlabel("Time from go cue (s)")

    if len(RAlign) > 0:
        ax = fig.add_subplot(gs[1, 1])
        df = align.to_events(rightLicks, RAlign, (tb, tf), return_df=True)
        ax.scatter(df.time, df.event_index, c="k", marker="|", s=1, zorder=2)
        ax.scatter(
            -np.sort(
                tblTrials.loc[
                    tblTrials["animal_response"] == 1, "delay_duration"
                ]
            ),
            range(
                len(
                    tblTrials.loc[
                        tblTrials["animal_response"] == 1, "delay_max"
                    ]
                )
            ),
            c="b",
            marker="|",
            s=1,
            zorder=1,
        )
        ax.axvline(x=0, c="r", ls="--", lw=1, zorder=3)
        ax.set_title("R licks on R choice trials")

        ax = fig.add_subplot(gs[2, 1])
        countsPre = np.searchsorted(
            np.sort(df.time.values), edges - 0.5 * binSize
        )
        countsPost = np.searchsorted(
            np.sort(df.time.values), edges + 0.5 * binSize
        )
        lickRate = (countsPost - countsPre) / (binSize * len(RAlign))
        ax.plot(edges, lickRate)
        ax.set_title("lickRate")
        ax.set_xlabel("Time from go cue (s)")

        ax = fig.add_subplot(gs[1, 2])
        df = align.to_events(leftLicks, RAlign, (tb, tf), return_df=True)
        ax.scatter(df.time, df.event_index, c="k", marker="|", s=1, zorder=2)
        ax.scatter(
            -np.sort(
                tblTrials.loc[
                    tblTrials["animal_response"] == 1, "delay_duration"
                ]
            ),
            range(
                len(
                    tblTrials.loc[
                        tblTrials["animal_response"] == 1, "delay_max"
                    ]
                )
            ),
            c="b",
            marker="|",
            s=1,
            zorder=1,
        )
        ax.axvline(x=0, c="r", ls="--", lw=1, zorder=3)
        ax.set_title("L licks on R choice trials")

        ax = fig.add_subplot(gs[2, 2])
        countsPre = np.searchsorted(
            np.sort(df.time.values), edges - 0.5 * binSize
        )
        countsPost = np.searchsorted(
            np.sort(df.time.values), edges + 0.5 * binSize
        )
        lickRate = (countsPost - countsPre) / (binSize * len(RAlign))
        ax.plot(edges, lickRate)
        ax.set_title("lickRate")
        ax.set_xlabel("Time from go cue (s)")

    sessionID = nwb.session_id
    sessionID = sessionID.split(".")[0]
    box = nwb.scratch["metadata"][0].box.values
    plt.suptitle(f"{sessionID} in {box}")

    # response latency
    ax = fig.add_subplot(gs[1, 4])
    lickLat = tblTrials["reward_outcome_time"] - tblTrials["goCue_start_time"]
    bins = np.arange(0, 1, 0.02)
    ax.hist(
        lickLat[tblTrials["animal_response"] == 1],
        bins=bins,
        alpha=0.5,
        label="R",
    )
    ax.hist(
        lickLat[tblTrials["animal_response"] == 0],
        bins=bins,
        alpha=0.5,
        label="L",
    )
    ax.legend()
    ax.set_title("lickLat by lick side")
    ax.set_xlabel("s")
    plt.suptitle(sessionID)
    return fig, sessionID


# example use
if __name__ == "__main__":
    import os
    from pathlib import Path

    data_dir = Path(os.path.dirname(__file__)).parent.parent
    nwbfile = os.path.join(
        data_dir, "tests\\data\\689514_2024-02-01_18-06-43.nwb"
    )
    # use of loadnwb depends on data struture
    nwb = loadnwb(nwbfile)
    fig, sessionID = plotLickAnalysis(nwb)
    saveDir = os.path.join(data_dir, "tests\\data", sessionID)
    fig.savefig(saveDir)
