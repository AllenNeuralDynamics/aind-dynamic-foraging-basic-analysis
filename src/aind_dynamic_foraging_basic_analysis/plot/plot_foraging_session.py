"""Plot foraging session in a standard format.
This is supposed to be reused in plotting real data or simulation data to ensure
a consistent visual representation.
"""

from typing import List, Tuple, Union

import numpy as np
from matplotlib import pyplot as plt

from aind_dynamic_foraging_basic_analysis.data_model.foraging_session import (
    ForagingSessionData,
    PhotostimData,
)
from aind_dynamic_foraging_basic_analysis.plot.style import PHOTOSTIM_EPOCH_MAPPING


def moving_average(a, n=3):
    """
    Compute moving average of a list or array.
    """
    ret = np.nancumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[(n - 1) :] / n  # noqa: E203


def plot_foraging_session_nwb(nwb, **kwargs):
    """
    Wrapper function that extracts fields
    """

    if not hasattr(nwb, "df_trials"):
        print("You need to compute df_trials: nwb_utils.create_trials_df(nwb)")
        return

    if "side_bias" not in nwb.df_trials:
        fig, axes = plot_foraging_session(
            [np.nan if x == 2 else x for x in nwb.df_trials["animal_response"].values],
            nwb.df_trials["earned_reward"].values,
            [nwb.df_trials["reward_probabilityL"], nwb.df_trials["reward_probabilityR"]],
            **kwargs,
        )
    else:
        if "plot_list" not in kwargs:
            kwargs["plot_list"] = ["choice", "reward_prob", "bias"]
        fig, axes = plot_foraging_session(
            [np.nan if x == 2 else x for x in nwb.df_trials["animal_response"].values],
            nwb.df_trials["earned_reward"].values,
            [nwb.df_trials["reward_probabilityL"], nwb.df_trials["reward_probabilityR"]],
            bias=nwb.df_trials["side_bias"].values,
            bias_lower=[x[0] for x in nwb.df_trials["side_bias_confidence_interval"].values],
            bias_upper=[x[1] for x in nwb.df_trials["side_bias_confidence_interval"].values],
            autowater_offered=nwb.df_trials[["auto_waterL", "auto_waterR"]].any(axis=1),
            **kwargs,
        )

    # Add some text info
    # TODO, waiting for AIND metadata to get integrated before adding this info:
    # {df_session.metadata.rig.iloc[0]}, {df_session.metadata.user_name.iloc[0]}\n'
    # f'FORAGING finished {df_session.session_stats.finished_trials.iloc[0]} '
    # f'ignored {df_session.session_stats.ignored_trials.iloc[0]} + '
    # f'AUTOWATER collected {df_session.session_stats.autowater_collected.iloc[0]} '
    # f'ignored {df_session.session_stats.autowater_ignored.iloc[0]}\n'
    # f'FORAGING finished rate {df_session.session_stats.finished_rate.iloc[0]:.2%}, '
    axes[0].text(
        0,
        1.05,
        f"{nwb.session_id}\n"
        f'Total trials {len(nwb.df_trials)}, ignored {np.sum(nwb.df_trials["animal_response"]==2)},'
        f' left {np.sum(nwb.df_trials["animal_response"] == 0)},'
        f' right {np.sum(nwb.df_trials["animal_response"] == 1)}',
        fontsize=8,
        transform=axes[0].transAxes,
    )


def plot_foraging_session(  # noqa: C901
    choice_history: Union[List, np.ndarray],
    reward_history: Union[List, np.ndarray],
    p_reward: Union[List, np.ndarray],
    autowater_offered: Union[List, np.ndarray] = None,
    fitted_data: Union[List, np.ndarray] = None,
    photostim: dict = None,
    valid_range: List = None,
    smooth_factor: int = 5,
    base_color: str = "y",
    ax: plt.Axes = None,
    vertical: bool = False,
    bias: Union[List, np.ndarray] = None,
    bias_lower: Union[List, np.ndarray] = None,
    bias_upper: Union[List, np.ndarray] = None,
    plot_list: List = ["choice", "finished", "reward_prob"],
) -> Tuple[plt.Figure, List[plt.Axes]]:
    """Plot dynamic foraging session.

    Parameters
    ----------
    choice_history : Union[List, np.ndarray]
        Choice history (0 = left choice, 1 = right choice, np.nan = ignored).
    reward_history : Union[List, np.ndarray]
        Reward history (0 = unrewarded, 1 = rewarded).
    p_reward : Union[List, np.ndarray]
        Reward probability for both sides. The size should be (2, len(choice_history)).
    autowater_offered: Union[List, np.ndarray], optional
        If not None, indicates trials where autowater was offered.
    fitted_data : Union[List, np.ndarray], optional
        If not None, overlay the fitted data (e.g. from RL model) on the plot.
    photostim : Dict, optional
        If not None, indicates photostimulation trials. It should be a dictionary with the keys:
            - trial: list of trial numbers
            - power: list of laser power
            - stim_epoch: optional, list of stimulation epochs from
               {"after iti start", "before go cue", "after go cue", "whole trial"}
    valid_range : List, optional
        If not None, add two vertical lines to indicate the valid range where animal was engaged.
    smooth_factor : int, optional
        Smoothing factor for the choice history, by default 5.
    base_color : str, optional
        Base color for the reward probability, by default "yellow".
    ax : plt.Axes, optional
        If not None, use the provided axis to plot, by default None.
    vertical : bool, optional
        If True, plot the session vertically, by default False.

    Returns
    -------
    Tuple[plt.Figure, List[plt.Axes]]
        fig, [ax_choice_reward, ax_reward_schedule]
    """

    # Formatting and sanity checks
    data = ForagingSessionData(
        choice_history=choice_history,
        reward_history=reward_history,
        p_reward=p_reward,
        autowater_offered=autowater_offered,
        fitted_data=fitted_data,
        photostim=PhotostimData(**photostim) if photostim is not None else None,
    )

    choice_history = data.choice_history
    reward_history = data.reward_history
    p_reward = data.p_reward
    autowater_offered = data.autowater_offered
    fitted_data = data.fitted_data
    photostim = data.photostim

    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(15, 3) if not vertical else (3, 12), dpi=200)
        plt.subplots_adjust(left=0.1, right=0.8, bottom=0.05, top=0.8)

    if not vertical:
        gs = ax._subplotspec.subgridspec(2, 1, height_ratios=[1, 0.2], hspace=0.1)
        ax_choice_reward = ax.get_figure().add_subplot(gs[0, 0])
        ax_reward_schedule = ax.get_figure().add_subplot(gs[1, 0], sharex=ax_choice_reward)
    else:
        gs = ax._subplotspec.subgridspec(1, 2, width_ratios=[0.2, 1], wspace=0.1)
        ax_choice_reward = ax.get_figure().add_subplot(gs[0, 1])
        ax_reward_schedule = ax.get_figure().add_subplot(gs[0, 0], sharey=ax_choice_reward)

    # == Fetch data ==
    n_trials = len(choice_history)

    p_reward_fraction = p_reward[1, :] / (np.sum(p_reward, axis=0))

    ignored = np.isnan(choice_history)

    if autowater_offered is None:
        rewarded_excluding_autowater = reward_history
        autowater_collected = np.full_like(choice_history, False, dtype=bool)
        autowater_ignored = np.full_like(choice_history, False, dtype=bool)
        unrewarded_trials = ~reward_history & ~ignored
    else:
        rewarded_excluding_autowater = reward_history & ~autowater_offered
        autowater_collected = autowater_offered & ~ignored
        autowater_ignored = autowater_offered & ignored
        unrewarded_trials = ~reward_history & ~ignored & ~autowater_offered

    # == Choice trace ==
    # Rewarded trials (real foraging, autowater excluded)
    xx = np.nonzero(rewarded_excluding_autowater)[0] + 1
    yy = 0.5 + (choice_history[rewarded_excluding_autowater] - 0.5) * 1.4
    yy_temp = choice_history[rewarded_excluding_autowater]
    yy_right = yy_temp[yy_temp > 0.5] + 0.05
    xx_right = xx[yy_temp > 0.5]
    yy_left = yy_temp[yy_temp < 0.5] - 0.05
    xx_left = xx[yy_temp < 0.5]
    if not vertical:
        ax_choice_reward.vlines(
            xx_right,
            yy_right,
            yy_right + 0.1,
            alpha=1,
            linewidth=1,
            color="black",
            label="Rewarded choices",
        )
        ax_choice_reward.vlines(
            xx_left,
            yy_left - 0.1,
            yy_left,
            alpha=1,
            linewidth=1,
            color="black",
        )
    else:
        ax_choice_reward.plot(
            *(xx, yy) if not vertical else [*(yy, xx)],
            "|" if not vertical else "_",
            color="black",
            markersize=10,
            markeredgewidth=2,
            label="Rewarded choices",
        )

    # Unrewarded trials (real foraging; not ignored or autowater trials)
    xx = np.nonzero(unrewarded_trials)[0] + 1
    yy = 0.5 + (choice_history[unrewarded_trials] - 0.5) * 1.4
    yy_temp = choice_history[unrewarded_trials]
    yy_right = yy_temp[yy_temp > 0.5]
    xx_right = xx[yy_temp > 0.5]
    yy_left = yy_temp[yy_temp < 0.5]
    xx_left = xx[yy_temp < 0.5]
    if not vertical:
        ax_choice_reward.vlines(
            xx_right,
            yy_right + 0.05,
            yy_right + 0.1,
            alpha=1,
            linewidth=1,
            color="gray",
            label="Unrewarded choices",
        )
        ax_choice_reward.vlines(
            xx_left,
            yy_left - 0.1,
            yy_left - 0.05,
            alpha=1,
            linewidth=1,
            color="gray",
        )
    else:
        ax_choice_reward.plot(
            *(xx, yy) if not vertical else [*(yy, xx)],
            "|" if not vertical else "_",
            color="gray",
            markersize=6,
            markeredgewidth=1,
            label="Unrewarded choices",
        )

    # Ignored trials
    xx = np.nonzero(ignored & ~autowater_ignored)[0] + 1
    yy = [1.2] * sum(ignored & ~autowater_ignored)
    ax_choice_reward.plot(
        *(xx, yy) if not vertical else [*(yy, xx)],
        "x",
        color="red",
        markersize=3,
        markeredgewidth=0.5,
        label="Ignored",
    )

    # Autowater history
    if autowater_offered is not None:
        # Autowater offered and collected
        xx = np.nonzero(autowater_collected)[0] + 1
        yy = 0.5 + (choice_history[autowater_collected] - 0.5) * 1.4

        yy_temp = choice_history[autowater_collected]
        yy_right = yy_temp[yy_temp > 0.5] + 0.05
        xx_right = xx[yy_temp > 0.5]
        yy_left = yy_temp[yy_temp < 0.5] - 0.05
        xx_left = xx[yy_temp < 0.5]

        if not vertical:
            ax_choice_reward.vlines(
                xx_right,
                yy_right,
                yy_right + 0.1,
                alpha=1,
                linewidth=1,
                color="royalblue",
                label="Autowater collected",
            )
            ax_choice_reward.vlines(
                xx_left,
                yy_left - 0.1,
                yy_left,
                alpha=1,
                linewidth=1,
                color="royalblue",
            )
        else:
            ax_choice_reward.plot(
                *(xx, yy) if not vertical else [*(yy, xx)],
                "|" if not vertical else "_",
                color="royalblue",
                markersize=10,
                markeredgewidth=2,
                label="Autowater collected",
            )

        # Also highlight the autowater offered but still ignored
        xx = np.nonzero(autowater_ignored)[0] + 1
        yy = [1.2] * sum(autowater_ignored)
        ax_choice_reward.plot(
            *(xx, yy) if not vertical else [*(yy, xx)],
            "x",
            color="royalblue",
            markersize=3,
            markeredgewidth=0.5,
            label="Autowater ignored",
        )

    # Base probability
    xx = np.arange(0, n_trials) + 1
    yy = p_reward_fraction
    if "reward_prob" in plot_list:
        ax_choice_reward.plot(
            *(xx, yy) if not vertical else [*(yy, xx)],
            color=base_color,
            label="Base rew. prob.",
            lw=1.5,
        )

    # Smoothed choice history
    y = moving_average(choice_history, smooth_factor) / (
        moving_average(~np.isnan(choice_history), smooth_factor) + 1e-6
    )
    y[y > 100] = np.nan
    x = np.arange(0, len(y)) + int(smooth_factor / 2) + 1
    if "choice" in plot_list:
        ax_choice_reward.plot(
            *(x, y) if not vertical else [*(y, x)],
            linewidth=1.5,
            color="black",
            label="Choice (smooth = %g)" % smooth_factor,
        )

    # finished ratio
    if np.sum(np.isnan(choice_history)):
        x = np.arange(0, len(y)) + int(smooth_factor / 2) + 1
        y = moving_average(~np.isnan(choice_history), smooth_factor)
        if "finished" in plot_list:
            ax_choice_reward.plot(
                *(x, y) if not vertical else [*(y, x)],
                linewidth=0.8,
                color="m",
                alpha=1,
                label="Finished (smooth = %g)" % smooth_factor,
            )

    # Bias
    if ("bias" in plot_list) and (bias is not None):
        bias = (np.array(bias) + 1) / (2)
        bias_lower = (np.array(bias_lower) + 1) / (2)
        bias_upper = (np.array(bias_upper) + 1) / (2)
        bias_lower[bias_lower < 0] = 0
        bias_upper[bias_upper > 1] = 1
        ax_choice_reward.plot(xx, bias, color="g", lw=1.5, label="bias")
        ax_choice_reward.fill_between(xx, bias_lower, bias_upper, color="g", alpha=0.25)
        ax_choice_reward.plot(xx, [0.5] * len(xx), color="g", linestyle="--", alpha=0.2, lw=1)

    # add valid ranage
    if valid_range is not None:
        add_range = ax_choice_reward.axhline if vertical else ax_choice_reward.axvline
        add_range(valid_range[0], color="m", ls="--", lw=1, label="motivation good")
        add_range(valid_range[1], color="m", ls="--", lw=1)

    # For each session, if any fitted_data
    if fitted_data is not None:
        x = np.arange(0, n_trials)
        y = fitted_data
        ax_choice_reward.plot(*(x, y) if not vertical else [*(y, x)], linewidth=1.5, label="model")

    # == photo stim ==
    if photostim is not None:

        trial = data.photostim.trial
        power = data.photostim.power
        stim_epoch = data.photostim.stim_epoch

        if stim_epoch is not None:
            edgecolors = [PHOTOSTIM_EPOCH_MAPPING[t] for t in stim_epoch]
        else:
            edgecolors = "darkcyan"

        x = trial
        y = np.ones_like(trial) + 0.4
        _ = ax_choice_reward.scatter(
            *(x, y) if not vertical else [*(y, x)],
            s=np.array(power) * 2,
            edgecolors=edgecolors,
            marker="v" if not vertical else "<",
            facecolors="none",
            linewidth=0.5,
            label="photostim",
        )

    # p_reward
    xx = np.arange(0, n_trials) + 1
    ll = p_reward[0, :]
    rr = p_reward[1, :]
    ax_reward_schedule.plot(
        *(xx, rr) if not vertical else [*(rr, xx)], color="b", label="p_right", lw=1
    )
    ax_reward_schedule.plot(
        *(xx, ll) if not vertical else [*(ll, xx)], color="r", label="p_left", lw=1
    )
    ax_reward_schedule.legend(fontsize=5, ncol=1, loc="upper left", bbox_to_anchor=(0, 1))

    if not vertical:
        ax_choice_reward.set_yticks([0, 1, 1.2])
        ax_choice_reward.set_yticklabels(["Left", "Right", "Ignored"])
        ax_choice_reward.legend(fontsize=6, loc="upper left", bbox_to_anchor=(0.4, 1.3), ncol=3)

        ax_choice_reward.spines["top"].set_visible(False)
        ax_choice_reward.spines["right"].set_visible(False)
        ax_choice_reward.spines["bottom"].set_visible(False)
        ax_choice_reward.tick_params(labelbottom=False)
        ax_choice_reward.xaxis.set_ticks_position("none")
        ax_choice_reward.set_ylim([-0.15, 1.25])

        ax_reward_schedule.set_ylim([0, 1])
        ax_reward_schedule.spines["top"].set_visible(False)
        ax_reward_schedule.spines["right"].set_visible(False)
        ax_reward_schedule.spines["bottom"].set_bounds(0, n_trials)
        ax_reward_schedule.set(xlabel="Trial number")

    else:
        ax_choice_reward.set_xticks([0, 1])
        ax_choice_reward.set_xticklabels(["Left", "Right"])
        ax_choice_reward.invert_yaxis()
        ax_choice_reward.legend(fontsize=6, loc="upper left", bbox_to_anchor=(0, 1.05), ncol=3)

        # ax_choice_reward.set_yticks([])
        ax_choice_reward.spines["top"].set_visible(False)
        ax_choice_reward.spines["right"].set_visible(False)
        ax_choice_reward.spines["left"].set_visible(False)
        ax_choice_reward.tick_params(labelleft=False)
        ax_choice_reward.yaxis.set_ticks_position("none")

        ax_reward_schedule.set_xlim([0, 1])
        ax_reward_schedule.spines["top"].set_visible(False)
        ax_reward_schedule.spines["right"].set_visible(False)
        ax_reward_schedule.spines["left"].set_bounds(0, n_trials)
        ax_reward_schedule.set(ylabel="Trial number")

    ax.remove()
    plt.tight_layout()

    return ax_choice_reward.get_figure(), [ax_choice_reward, ax_reward_schedule]
