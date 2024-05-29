"""Load packages."""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from aind_ephys_utils import align
from pynwb import NWBHDF5IO
from scipy.stats import norm


def loadnwb(nwb_file):
    """Load nwb."""
    io = NWBHDF5IO(nwb_file, mode="r")
    nwb = io.read()
    return nwb


def plotLickAnalysis(nwb):
    """Plot lick distributions."""
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
    """Plot lickLat."""

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


class lickMetrics:
    """calculate lick quantiles"""

    def __init__(self, nwb):
        """Input: nwb of behavior data"""
        self.sessionID = nwb.session_id
        self.tblTrials = nwb.trials.to_dataframe()
        self.leftLicks = nwb.acquisition["left_lick_time"].timestamps[:]
        self.rightLicks = nwb.acquisition["right_lick_time"].timestamps[:]
        self.allLicks = np.sort(
            np.concatenate((self.rightLicks, self.leftLicks))
        )
        self.lickLat = (
            self.tblTrials["reward_outcome_time"]
            - self.tblTrials["goCue_start_time"]
        )
        self.lickLatR = self.lickLat[self.tblTrials["animal_response"] == 1]
        self.lickLatL = self.lickLat[self.tblTrials["animal_response"] == 0]
        self.thresh = [0.05, 0.5, 1.0]
        self.kernel = norm.pdf(np.arange(-2, 2.1, 0.5))
        self.binWidth = 0.2
        self.binSteps = np.arange(0, 1.5, 0.02)
        self.lickMet = None
        self.winGo = [self.thresh[0], 1.0]
        self.winBl = [-1, 0]

    def calMetrics(self):
        """calculate lick quantiles"""
        lickPercentL = [
            np.sum(self.lickLatL <= threshCurr) / np.shape(self.lickLatL)[0]
            for threshCurr in self.thresh
        ]
        lickPercentR = [
            np.sum(self.lickLatR <= threshCurr) / np.shape(self.lickLatR)[0]
            for threshCurr in self.thresh
        ]
        """ calculate motivation chage """
        finish = self.tblTrials["animal_response"] != 2
        ref = np.ones_like(finish)
        refKernel = np.convolve(ref, self.kernel)

        finishKernel = np.convolve(finish.astype(float), self.kernel)
        finishKernel = np.divide(finishKernel, refKernel)
        finishKernel = finishKernel[
            int(0.5 * len(self.kernel)):-int(0.5 * len(self.kernel))
        ]

        """ calculate mean lick rate """
        allGoNoRwd = self.tblTrials.loc[
            (self.tblTrials["animal_response"] != 2)
            & (self.tblTrials["rewarded_historyL"] == 0)
            & (self.tblTrials["rewarded_historyR"] == 0),
            "goCue_start_time",
        ].values
        allPreNolick = (
            self.tblTrials.loc[
                self.tblTrials["animal_response"] != 2, "goCue_start_time"
            ].values
            - self.tblTrials.loc[
                self.tblTrials["animal_response"] != 2, "delay_duration"
            ].values
        )

        respondMean = np.mean(rateAlign(self.allLicks, allGoNoRwd, self.winGo))
        blMean = np.mean(rateAlign(self.allLicks, allPreNolick, self.winBl))
        blMeanL = np.mean(rateAlign(self.leftLicks, allPreNolick, self.winBl))
        blMeanR = np.mean(rateAlign(self.rightLicks, allPreNolick, self.winBl))
        # calcualate mode of lick and 'concentration'
        Lmajor, LmajorPerc = slideMode(
            self.lickLatL, self.binWidth, self.binSteps
        )
        Rmajor, RmajorPerc = slideMode(
            self.lickLatR, self.binWidth, self.binSteps
        )

        self.lickMet = {
            "sessionID": self.sessionID,
            # sessions id
            "blLick": blMean,
            # baseline lick rate before start of no lick window
            "blLickLR": [blMeanL, blMeanR],
            # baseline lick rate on each side
            "respLick": respondMean,
            # lick rate after go cue without reward
            "blLickTrial": rateAlign(
                self.allLicks, allPreNolick, self.winBl
            ),  # baseline lick across time
            "respLickTrial": rateAlign(
                self.allLicks, allGoNoRwd, self.winGo
            ),  # resp lick across time
            "peakRatio": [
                LmajorPerc,
                RmajorPerc,
            ],  # percent of licks in 200ms window that covers the most licks
            "peakLat": [Lmajor, Rmajor],  # mode of licks in 200ms window
            "lickCDF": {
                "thresh": self.thresh,
                "L": lickPercentL,
                "R": lickPercentR,
            },  # percent of licks under threshold
            "respScore": respondMean - blMean,
            "consistencyScore": [
                LmajorPerc - blMeanL * self.binWidth / 1000,
                RmajorPerc - blMeanR * self.binWidth / 1000,
            ],
            "finishRatio": finishKernel,
        }

    def plot(self):
        """plot lick metrics"""
        edges = np.arange(
            np.min(self.lickLat[self.tblTrials["animal_response"] != 2]),
            np.max(self.lickLat[self.tblTrials["animal_response"] != 2]),
            0.02,
        )
        fig = plt.figure(figsize=(8, 4))
        gs = fig.add_gridspec(
            2,
            2,
            width_ratios=[1, 1],
            height_ratios=[2, 1],
            wspace=0.5,
            hspace=0.5,
        )
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.hist(self.lickLatL, bins=edges, alpha=0.5, density=True)
        ax1t = ax1.twinx()
        plotCdf(self.lickLatL)
        ax1t.plot(np.array(self.thresh), self.lickMet["lickCDF"]["L"], "k")
        Lmajor = self.lickMet["peakLat"][0]
        LmajorRatio = self.lickMet["peakRatio"][0]
        ax1t.fill(
            [
                Lmajor - 0.5 * self.binWidth,
                Lmajor + 0.5 * self.binWidth,
                Lmajor + 0.5 * self.binWidth,
                Lmajor - 0.5 * self.binWidth,
            ],
            [0, 0, 1, 1],
            "r",
            alpha=0.2,
        )
        tempL = self.lickMet["consistencyScore"][0]
        ax1t.set_title(f"L {LmajorRatio:.2f} {tempL:.2f}")
        ax1t.set_ylim(0, 1.1)
        ax1.plot(
            [self.winBl[0], 0],
            np.ones_like(self.winBl) * self.lickMet["blLick"],
            color=[0.3, 0.3, 0.3],
            lw=4,
        )
        ax1.plot(
            self.winGo,
            np.ones_like(self.winBl) * self.lickMet["respLick"],
            color=[0.3, 0.3, 0.3],
            lw=4,
        )

        ax2 = fig.add_subplot(gs[0, 1])
        ax2.hist(self.lickLatR, bins=edges, alpha=0.5, density=True)
        ax2t = ax2.twinx()
        plotCdf(self.lickLatR)
        ax2t.plot(np.array(self.thresh), self.lickMet["lickCDF"]["R"], "k")
        Rmajor = self.lickMet["peakLat"][1]
        RmajorRatio = self.lickMet["peakRatio"][1]
        ax2t.fill(
            [
                Rmajor - 0.5 * self.binWidth,
                Rmajor + 0.5 * self.binWidth,
                Rmajor + 0.5 * self.binWidth,
                Rmajor - 0.5 * self.binWidth,
            ],
            [0, 0, 1, 1],
            "r",
            alpha=0.2,
        )
        tempR = self.lickMet["consistencyScore"][1]
        ax2t.set_title(f"R {RmajorRatio:.2f} {tempR:.2f}")
        ax2t.set_ylim(0, 1.1)
        ax2.plot(
            [self.winBl[0], 0],
            np.ones_like(self.winBl) * self.lickMet["blLick"],
            color=[0.3, 0.3, 0.3],
            lw=4,
        )
        ax2.plot(
            self.winGo,
            np.ones_like(self.winBl) * self.lickMet["respLick"],
            color=[0.3, 0.3, 0.3],
            lw=4,
        )

        ax3 = fig.add_subplot(gs[1, :])
        allGoNoRwd = self.tblTrials.loc[
            (self.tblTrials["animal_response"] != 2)
            & (self.tblTrials["rewarded_historyL"] == 0)
            & (self.tblTrials["rewarded_historyR"] == 0),
            "goCue_start_time",
        ].values
        allGoRwd = self.tblTrials.loc[
            (self.tblTrials["animal_response"] != 2)
            & (self.tblTrials["rewarded_historyL"] == 1)
            | (self.tblTrials["rewarded_historyR"] == 1),
            "goCue_start_time",
        ].values
        allPreNolick = (
            self.tblTrials.loc[
                self.tblTrials["animal_response"] != 2, "goCue_start_time"
            ].values
            - self.tblTrials.loc[
                self.tblTrials["animal_response"] != 2, "delay_duration"
            ].values
        )
        rateGoRwd = rateAlign(self.allLicks, allGoRwd, self.winGo)

        ax3.plot(
            allGoNoRwd, self.lickMet["respLickTrial"], label="Resp", color="r"
        )
        ax3.plot(
            allPreNolick, self.lickMet["blLickTrial"], label="bl", color="grey"
        )
        ax3.plot(allGoRwd, rateGoRwd, label="RespRwd", color="b")
        ax3t = ax3.twinx()
        ax3t.plot(
            self.tblTrials["goCue_start_time"],
            self.lickMet["finishRatio"],
            color="g",
            label="finish",
        )
        temp = self.lickMet["respScore"]
        ax3.set_title(f"Resp score {temp:.2f}")
        ax3.legend()
        plt.suptitle(self.sessionID)
        return fig, self.sessionID


def plotCdf(x, *arg):
    """plot CDF given input x"""
    sorted_data = np.sort(x)
    cdf = np.arange(1, len(sorted_data) + 1) / len(sorted_data)
    plt.plot(sorted_data, cdf, *arg)


def slideMode(x, binSize, binStep):
    """find mode with sliding window"""
    x = np.sort(x)
    startInds = np.searchsorted(x, binStep - 0.5 * binSize)
    stopsInds = np.searchsorted(x, binStep + 0.5 * binSize)
    modeInd = np.argmax(np.array(stopsInds - startInds))
    mode = binStep[modeInd]
    modePerc = np.max(np.array(stopsInds - startInds)) / np.max(x.shape)
    return mode, modePerc


def rateAlign(x, events, win):
    """calculate rate of occurance aligned to events with fixed window."""
    x = np.sort(x)
    startInds = np.searchsorted(x, events + win[0])
    stopInds = np.searchsorted(x, events + win[1])
    rate = (stopInds - startInds) / (win[1] - win[0])
    return rate


# example use
if __name__ == "__main__":
    import os
    from pathlib import Path

    """Example."""
    data_dir = Path(os.path.dirname(__file__)).parent.parent
    nwbfile = os.path.join(
        data_dir, "tests\\data\\689514_2024-02-01_18-06-43.nwb"
    )
    # use of loadnwb depends on data struture
    nwb = loadnwb(nwbfile)
    fig, sessionID = plotLickAnalysis(nwb)
    saveDir = os.path.join(data_dir, "tests\\data", sessionID)
    fig.savefig(saveDir)

    lickSum = lickMetrics(nwb)
    lickSum.calMetrics()
    fig = lickSum.plot()
    saveDir = os.path.join(data_dir, "tests\\data", sessionID + "qc")
    fig.savefig(saveDir)
