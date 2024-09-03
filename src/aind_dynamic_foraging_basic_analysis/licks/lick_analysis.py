"""Load packages."""
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from aind_ephys_utils import align
from pynwb import NWBHDF5IO
from hdmf_zarr import NWBZarrIO
from scipy.stats import norm


def load_nwb(nwb_file):
    """Load NWB file."""
    if os.path.isdir(nwb_file):
        io = NWBZarrIO(nwb_file, mode="r")
        nwb = io.read()
        return nwb
    elif os.path.isfile(nwb_file):
        io = NWBHDF5IO(nwb_file, mode="r")
        nwb = io.read()
        return nwb
    else:
        print("nwb file does not exist.")
        return None


def plot_lick_analysis(nwb):
    """Plot lick distributions."""
    tbl_trials = nwb.trials.to_dataframe()
    session_id = nwb.session_id
    session_id = session_id.split(".")[0]
    gs = gridspec.GridSpec(
        2,
        6,
        width_ratios=np.ones(6).tolist(),
        height_ratios=[1, 2],
    )
    fig = plt.figure(figsize=(15, 8))
    plt.subplots_adjust(hspace=0.5, wspace=0.5)

    if len(tbl_trials) < 10:
        """No enough trials to plot lick analysis."""
        title_str = f"{session_id} has less than 10 trials"
        plt.suptitle(title_str)
    else:
        left_licks = nwb.acquisition["left_lick_time"].timestamps[:]
        right_licks = nwb.acquisition["right_lick_time"].timestamps[:]
        all_licks = np.sort(np.concatenate((right_licks, left_licks)))
        sort_ind = np.argsort(np.concatenate((right_licks, left_licks)))
        all_licks_id = np.concatenate((np.ones_like(right_licks), np.zeros_like(left_licks)))
        all_licks_id = all_licks_id[sort_ind]
        all_lick_diffs = np.diff(all_licks)
        # ILI_L-L
        plot_ILI(
            all_lick_diffs[(all_licks_id[:-1] == 0) & (all_licks_id[1:] == 0)],
            "ILI_L-L",
            gs[0, 0],
            fig,
        )
        # ILI_R-R
        plot_ILI(
            all_lick_diffs[(all_licks_id[:-1] == 1) & (all_licks_id[1:] == 1)],
            "ILI_R-R",
            gs[0, 1],
            fig,
        )
        # ILI_R-L
        plot_ILI(
            all_lick_diffs[(all_licks_id[:-1] == 1) & (all_licks_id[1:] == 0)],
            "ILI_R-l",
            gs[0, 2],
            fig,
        )
        # ILI_L-R
        plot_ILI(
            all_lick_diffs[(all_licks_id[:-1] == 0) & (all_licks_id[1:] == 1)],
            "ILI_l-R",
            gs[0, 3],
            fig,
        )
        # ILI_all
        plot_ILI(all_lick_diffs, "ILI_all", gs[0, 4], fig)
        # Pre-trial lick punishment
        lick_delay = (
            tbl_trials["goCue_start_time"] - tbl_trials["start_time"] - tbl_trials["ITI_duration"]
        )

        bins = np.arange(
            np.min(lick_delay),
            np.max(lick_delay),
            (np.max(lick_delay) - np.min(lick_delay)) / 20,
        )

        ax = fig.add_subplot(gs[0, 5])
        ax.hist(
            lick_delay[tbl_trials["animal_response"] == 1],
            alpha=0.5,
            label="R",
            bins=bins,
            # density=True,
        )
        ax.hist(
            lick_delay[tbl_trials["animal_response"] == 0],
            alpha=0.5,
            label="L",
            bins=bins,
            # density=True,
        )
        ax.hist(
            lick_delay[tbl_trials["animal_response"] == 2],
            alpha=0.5,
            label="0",
            bins=bins,
            # density=True,
        )
        p_early = 1 - np.mean(lick_delay < tbl_trials["delay_duration"] + 0.05)
        ax.set_title(f"Time out punishment prop {p_early:.2f}")
        ax.set_xlabel("s")
        ax.legend()

        # Pre-trial licks

        sort_ind_l = np.argsort(
            tbl_trials.loc[tbl_trials["animal_response"] == 0, "delay_duration"]
        )
        sort_ind_r = np.argsort(
            tbl_trials.loc[tbl_trials["animal_response"] == 1, "delay_duration"]
        )
        l_align = tbl_trials.loc[tbl_trials["animal_response"] == 0, "goCue_start_time"].values
        r_align = tbl_trials.loc[tbl_trials["animal_response"] == 1, "goCue_start_time"].values
        l_align = l_align[sort_ind_l.values]
        r_align = r_align[sort_ind_r.values]

        if len(l_align) > 0:
            # Left lick aligned to left choices
            fig, ax1, ax2 = plot_raster_rate(
                left_licks, l_align, fig, gs[1, 0], "L licks on L choice trials"
            )
            ax1.scatter(
                -np.sort(tbl_trials.loc[tbl_trials["animal_response"] == 0, "delay_duration"]),
                range(len(tbl_trials.loc[tbl_trials["animal_response"] == 0, "delay_max"])),
                c="b",
                marker="|",
                s=1,
                zorder=1,
            )
            ax2.set_ylabel("Licks/s")

            # Right licks aligned to left choices
            fig, ax1, _ = plot_raster_rate(
                right_licks, l_align, fig, gs[1, 2], "R licks on L choice trials"
            )
            ax1.scatter(
                -np.sort(tbl_trials.loc[tbl_trials["animal_response"] == 0, "delay_duration"]),
                range(len(tbl_trials.loc[tbl_trials["animal_response"] == 0, "delay_max"])),
                c="b",
                marker="|",
                s=1,
                zorder=1,
            )

        if len(r_align) > 0:
            # right licks aligned to right choices
            fig, ax1, _ = plot_raster_rate(
                right_licks, r_align, fig, gs[1, 1], "R licks on R choice trials"
            )

            ax1.scatter(
                -np.sort(tbl_trials.loc[tbl_trials["animal_response"] == 1, "delay_duration"]),
                range(len(tbl_trials.loc[tbl_trials["animal_response"] == 1, "delay_max"])),
                c="b",
                marker="|",
                s=1,
                zorder=1,
            )

            # left licks aligned to right choices
            fig, ax1, _ = plot_raster_rate(
                left_licks, r_align, fig, gs[1, 3], "L licks on R choice trials"
            )

            ax1.scatter(
                -np.sort(tbl_trials.loc[tbl_trials["animal_response"] == 1, "delay_duration"]),
                range(len(tbl_trials.loc[tbl_trials["animal_response"] == 1, "delay_max"])),
                c="b",
                marker="|",
                s=1,
                zorder=1,
            )

        box = nwb.scratch["metadata"][0].box.values
        plt.suptitle(f"{session_id} in {box}")

        # Response latency
        ax = fig.add_subplot(gs[1, 4:6])
        lick_lat = tbl_trials["reward_outcome_time"] - tbl_trials["goCue_start_time"]
        bins = np.arange(0, 1, 0.05)
        ax.hist(
            lick_lat[tbl_trials["animal_response"] == 1],
            bins=bins,
            alpha=0.5,
            label="R",
            density=True,
        )
        ax.hist(
            lick_lat[tbl_trials["animal_response"] == 0],
            bins=bins,
            alpha=0.5,
            label="L",
            density=True,
        )
        ax.legend()
        ax.set_title("lickLat by lick side")
        ax.set_xlabel("s")
        ax.set_ylabel("density %")
        plt.suptitle(session_id)

    return fig, session_id


def load_data(nwb):
    """Load data from NWB file."""
    session_id = nwb.session_id
    session_id = session_id.split(".")[0]
    tbl_trials = nwb.trials.to_dataframe()
    left_licks = nwb.acquisition["left_lick_time"].timestamps[:]
    right_licks = nwb.acquisition["right_lick_time"].timestamps[:]
    all_licks = np.sort(np.concatenate((right_licks, left_licks)))
    lick_lat = tbl_trials["reward_outcome_time"] - tbl_trials["goCue_start_time"]
    lick_lat_r = lick_lat[tbl_trials["animal_response"] == 1]
    lick_lat_l = lick_lat[tbl_trials["animal_response"] == 0]
    thresh = [0.05, 0.5, 1.0]
    kernel = norm.pdf(np.arange(-2, 2.1, 0.5))
    bin_width = 0.2
    bin_steps = np.arange(0, 1.5, 0.02)
    win_go = [thresh[0], 1.0]
    win_bl = [-1, 0]
    return {
        "session_id": session_id,
        "tbl_trials": tbl_trials,
        "left_licks": left_licks,
        "right_licks": right_licks,
        "all_licks": all_licks,
        "lick_lat": lick_lat,
        "lick_lat_r": lick_lat_r,
        "lick_lat_l": lick_lat_l,
        "thresh": thresh,
        "kernel": kernel,
        "bin_width": bin_width,
        "bin_steps": bin_steps,
        "win_go": win_go,
        "win_bl": win_bl,
    }


def cal_metrics(data):
    """Calculate lick metrics."""
    session_id = data["session_id"]
    tbl_trials = data["tbl_trials"]
    left_licks = data["left_licks"]
    right_licks = data["right_licks"]
    all_licks = data["all_licks"]
    lick_lat_r = data["lick_lat_r"]
    lick_lat_l = data["lick_lat_l"]
    thresh = data["thresh"]
    kernel = data["kernel"]
    bin_width = data["bin_width"]
    bin_steps = data["bin_steps"]
    win_go = data["win_go"]
    win_bl = data["win_bl"]

    if tbl_trials.shape[0] < 10:
        """Not enough trials to calculate lick metrics."""
        return {
            "session_id": session_id,
            "bl_lick": 0,
            "bl_lick_lr": [0, 0],
            "resp_lick": 0,
            "bl_lick_trial": 0,
            "resp_lick_trial": 0,
            "peak_ratio": [0, 0],
            "peak_lat": [0, 0],
            "lick_cdf": {
                "thresh": thresh,
                "l": [0, 0, 0],
                "r": [0, 0, 0],
            },
            "resp_score": 0,
            "finish_ratio": 0,
        }
    else:  # calculate lick metrics
        lick_percent_l = [
            np.sum(lick_lat_l <= thresh_curr) / np.shape(lick_lat_l)[0] for thresh_curr in thresh
        ]
        lick_percent_r = [
            np.sum(lick_lat_r <= thresh_curr) / np.shape(lick_lat_r)[0] for thresh_curr in thresh
        ]
        finish = tbl_trials["animal_response"] != 2
        ref = np.ones_like(finish)
        ref_kernel = np.convolve(ref, kernel)
        finish_kernel = np.convolve(finish.astype(float), kernel)
        finish_kernel = np.divide(finish_kernel, ref_kernel)
        finish_kernel = finish_kernel[int(0.5 * len(kernel)): -int(0.5 * len(kernel))]
        all_go_no_rwd = tbl_trials.loc[
            (tbl_trials["animal_response"] != 2)
            & (tbl_trials["rewarded_historyL"] == 0)
            & (tbl_trials["rewarded_historyR"] == 0),
            "goCue_start_time",
        ].values
        all_pre_nolick = (
            tbl_trials.loc[tbl_trials["animal_response"] != 2, "goCue_start_time"].values
            - tbl_trials.loc[tbl_trials["animal_response"] != 2, "delay_duration"].values
        )
        respond_mean = np.mean(rate_align(all_licks, all_go_no_rwd, win_go))
        bl_mean = np.mean(rate_align(all_licks, all_pre_nolick, win_bl))
        bl_mean_l = np.mean(rate_align(left_licks, all_pre_nolick, win_bl))
        bl_mean_r = np.mean(rate_align(right_licks, all_pre_nolick, win_bl))
        l_major, l_major_perc = slide_mode(lick_lat_l, bin_width, bin_steps)
        r_major, r_major_perc = slide_mode(lick_lat_r, bin_width, bin_steps)
        lick_met = {
            "session_id": session_id,
            "bl_lick": bl_mean,
            "bl_lick_lr": [bl_mean_l, bl_mean_r],
            "resp_lick": respond_mean,
            "bl_lick_trial": rate_align(all_licks, all_pre_nolick, win_bl),
            "resp_lick_trial": rate_align(all_licks, all_go_no_rwd, win_go),
            "peak_ratio": [l_major_perc, r_major_perc],
            "peak_lat": [l_major, r_major],
            "lick_cdf": {
                "thresh": thresh,
                "l": lick_percent_l,
                "r": lick_percent_r,
            },
            "resp_score": respond_mean - bl_mean,
            "finish_ratio": finish_kernel,
        }
        return lick_met


def plot_met(data, lick_met):
    """Plot lick  metrics."""
    lick_lat_l = data["lick_lat_l"]
    lick_lat_r = data["lick_lat_r"]
    bin_width = data["bin_width"]
    win_bl = data["win_bl"]
    win_go = data["win_go"]
    session_id = data["session_id"]
    all_licks = data["all_licks"]
    tbl_trials = data["tbl_trials"]

    fig = plt.figure(figsize=(8, 4))
    if data["tbl_trials"].shape[0] < 10:
        """Not enough trials to plot lick metrics."""
        plt.suptitle(f"{session_id} has less than 10 trials")
    else:
        edges = np.arange(
            np.min(lick_lat_l),
            np.max(lick_lat_l),
            0.02,
        )
        gs = fig.add_gridspec(
            2,
            2,
            width_ratios=[1, 1],
            height_ratios=[2, 1],
            wspace=0.5,
            hspace=0.5,
        )
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.hist(lick_lat_l, bins=edges, alpha=0.5, density=True, label="lick hist")
        ax1.set_xlabel("s")
        ax1.set_ylabel("density %")
        ax1.legend(loc="upper left", bbox_to_anchor=(0, 1), ncol=1)
        ax1t = ax1.twinx()
        ax1t.set_ylabel("CDF")
        plot_cdf(lick_lat_l, {"label": "lick cdf"})
        # ax1t.plot(np.array(thresh), lick_met["lick_cdf"]["l"], "k")
        l_major = lick_met["peak_lat"][0]
        l_major_ratio = lick_met["peak_ratio"][0]
        ax1t.fill(
            [
                l_major - 0.5 * bin_width,
                l_major + 0.5 * bin_width,
                l_major + 0.5 * bin_width,
                l_major - 0.5 * bin_width,
            ],
            [0, 0, 1, 1],
            "r",
            alpha=0.2,
        )
        ax1t.set_title(f"Lick Latency: L {l_major_ratio:.2f}")
        ax1t.set_ylim(0, 1.1)
        ax1t.legend(loc="upper left", bbox_to_anchor=(0, 0.8), ncol=1)
        ax1.plot(
            [win_bl[0], 0],
            np.ones_like(win_bl) * lick_met["bl_lick"],
            color=[0.3, 0.3, 0.3],
            lw=4,
        )
        ax1.plot(
            win_go,
            np.ones_like(win_bl) * lick_met["resp_lick"],
            color=[0.3, 0.3, 0.3],
            lw=4,
        )

        ax2 = fig.add_subplot(gs[0, 1])
        ax2.hist(lick_lat_r, bins=edges, alpha=0.5, density=True)
        ax2.set_xlabel("s")
        ax2t = ax2.twinx()
        plot_cdf(lick_lat_r, {"label": "lick cdf"})
        # ax2t.plot(np.array(thresh), lick_met["lick_cdf"]["r"], "k")
        r_major = lick_met["peak_lat"][1]
        r_major_ratio = lick_met["peak_ratio"][1]
        ax2t.fill(
            [
                r_major - 0.5 * bin_width,
                r_major + 0.5 * bin_width,
                r_major + 0.5 * bin_width,
                r_major - 0.5 * bin_width,
            ],
            [0, 0, 1, 1],
            "r",
            alpha=0.2,
        )
        ax2t.set_title(f"R {r_major_ratio:.2f}")
        ax2t.set_ylim(0, 1.1)
        ax2.plot(
            [win_bl[0], 0],
            np.ones_like(win_bl) * lick_met["bl_lick"],
            color=[0.3, 0.3, 0.3],
            lw=4,
        )
        ax2.plot(
            win_go,
            np.ones_like(win_bl) * lick_met["resp_lick"],
            color=[0.3, 0.3, 0.3],
            lw=4,
        )

        ax3 = fig.add_subplot(gs[1, :])
        all_go_no_rwd = tbl_trials.loc[
            (tbl_trials["animal_response"] != 2)
            & (tbl_trials["rewarded_historyL"] == 0)
            & (tbl_trials["rewarded_historyR"] == 0),
            "goCue_start_time",
        ].values
        all_go_rwd = tbl_trials.loc[
            (tbl_trials["animal_response"] != 2) & (tbl_trials["rewarded_historyL"] == 1)
            | (tbl_trials["rewarded_historyR"] == 1),
            "goCue_start_time",
        ].values
        all_pre_nolick = (
            tbl_trials.loc[tbl_trials["animal_response"] != 2, "goCue_start_time"].values
            - tbl_trials.loc[tbl_trials["animal_response"] != 2, "delay_duration"].values
        )
        rate_go_rwd = rate_align(all_licks, all_go_rwd, win_go)

        ax3.plot(
            all_go_no_rwd,
            lick_met["resp_lick_trial"],
            label="Resp-noRwd",
            color="r",
        )
        ax3.plot(
            all_pre_nolick,
            lick_met["bl_lick_trial"],
            label="baseline",
            color="grey",
        )
        ax3.plot(all_go_rwd, rate_go_rwd, label="Resp-Rwd", color="b")
        ax3.set_ylabel("lick rate")
        ax3t = ax3.twinx()
        ax3t.plot(
            tbl_trials["goCue_start_time"],
            lick_met["finish_ratio"],
            color="g",
            label="finish",
        )
        ax3t.set_ylabel("Ratio")
        temp = lick_met["resp_score"]
        ax3.set_title(f"Resp score {temp:.2f}")
        ax3.legend()
        ax3t.legend()
        plt.suptitle(session_id)
    return fig, session_id


def plot_cdf(x, args):
    """plot CDF given input x"""
    sorted_data = np.sort(x)
    cdf = np.arange(1, len(sorted_data) + 1) / len(sorted_data)
    plt.plot(sorted_data, cdf, **args)


def slide_mode(x, bin_size, bin_step):
    """find mode with sliding window"""
    x = np.sort(x)
    start_inds = np.searchsorted(x, bin_step - 0.5 * bin_size)
    stops_inds = np.searchsorted(x, bin_step + 0.5 * bin_size)
    mode_ind = np.argmax(np.array(stops_inds - start_inds))
    mode = bin_step[mode_ind]
    mode_perc = np.max(np.array(stops_inds - start_inds)) / np.max(x.shape)
    return mode, mode_perc


def rate_align(x, events, win):
    """calculate rate of occurrence aligned to events with fixed window."""
    x = np.sort(x)
    start_inds = np.searchsorted(x, events + win[0])
    stop_inds = np.searchsorted(x, events + win[1])
    rate = (stop_inds - start_inds) / (win[1] - win[0])
    return rate


def plot_ILI(ili, title, subplot_spec, fig):
    """Plot ILI."""
    nested_gs = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=subplot_spec)
    nested_gs = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=subplot_spec)
    bins = np.arange(0, 2500, 20)
    ax1 = fig.add_subplot(nested_gs[0, 0])
    ax2 = fig.add_subplot(nested_gs[1, 0])
    ax1.hist(
        1000 * ili,
        bins=bins,
    )
    ax1.set_title(title)
    ax2.hist(
        1000 * ili,
        bins=bins,
    )
    ax2.set_xlim([0, 300])
    ax2.set_xlabel("ms")


def plot_raster_rate(
    events,
    align_events,
    fig,
    subplot_spec,
    title,
    tb=-5,
    tf=10,
    bin_size=100 / 1000,
    step_size=50 / 1000,
):
    """Plot raster and rate aligned to events"""
    edges = np.arange(tb + 0.5 * bin_size, tf - 0.5 * bin_size, step_size)
    nested_gs = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=subplot_spec)
    nested_gs = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=subplot_spec)
    ax1 = fig.add_subplot(nested_gs[0, 0])
    ax2 = fig.add_subplot(nested_gs[1, 0])

    df = align.to_events(events, align_events, (tb, tf), return_df=True)
    ax1.scatter(df.time, df.event_index, c="k", marker="|", s=1, zorder=2)
    ax1.axvline(x=0, c="r", ls="--", lw=1, zorder=3)
    ax1.set_title(title)
    ax1.set_xlim(tb, tf)

    counts_pre = np.searchsorted(np.sort(df.time.values), edges - 0.5 * bin_size)
    counts_post = np.searchsorted(np.sort(df.time.values), edges + 0.5 * bin_size)
    counts_pre = np.searchsorted(np.sort(df.time.values), edges - 0.5 * bin_size)
    counts_post = np.searchsorted(np.sort(df.time.values), edges + 0.5 * bin_size)
    lick_rate = (counts_post - counts_pre) / (bin_size * len(align_events))
    ax2.plot(edges, lick_rate)
    ax2.set_title("lickRate")
    ax2.set_xlim(tb, tf)
    ax2.set_xlabel("Time from go cue (s)")

    return fig, ax1, ax2


# example use
if __name__ == "__main__":
    from pathlib import Path

    """Example."""
    data_dir = Path(os.path.dirname(__file__)).parent.parent
    nwbfile = os.path.join(data_dir, "tests/data/705599_2024-05-31_14-06-54.nwb")
    nwbfile = os.path.join(data_dir, "tests/data/705599_2024-05-31_14-06-54.nwb")
    # use of load_data depends on data structure
    nwb = load_nwb(nwbfile)
    fig, session_id = plot_lick_analysis(nwb)
    save_dir = os.path.join(data_dir, "tests/data", session_id)
    fig.savefig(save_dir)

    data = load_data(nwb)
    lick_sum = cal_metrics(data)
    fig, _ = plot_met(data, lick_sum)
    save_dir = os.path.join(data_dir, "tests/data", session_id + "qc")
    fig.savefig(save_dir)
