"""

    Consolidated session metric tool
    df_session_meta = compute_session_metadata(nwb)
    df_session = compute_session_metrics(nwb)

"""

import logging

import numpy as np
import pandas as pd

from aind_dynamic_foraging_basic_analysis import compute_foraging_efficiency

# Copied metrics are from process_nwbs.py in bonsai basic
# NOTE: finished_rate_with_autowater is the same as the calculated total response rate
LEFT, RIGHT, IGNORE = 0, 1, 2

logger = logging.getLogger(__name__)


def compute_session_metadata(nwb):
    """
    block structure metrics,        block, contrast, and effective
                                    probability metrics
    duration metrics,               gocue, delay period, and iti
                                    metrics
    reward size metrics,            left and right reward volumes
    lick spout movement,            movement of lickspout during session:
                                    range, initial position, median position
    autotrain categories,           curriculum version, name, schema,
                                    current_stage_actual, and if overriden
    """
    if not hasattr(nwb, "df_trials"):
        print("You need to compute df_trials: nwb_utils.create_trials_df(nwb)")
        return

    df_trials = nwb.df_trials.copy()

    # Block information
    def _get_block_starts(p_L, p_R):
        """Find the indices of block starts"""
        block_start_ind_left = np.where(np.hstack([True, np.diff(p_L) != 0]))[0]
        block_start_ind_right = np.where(np.hstack([True, np.diff(p_R) != 0]))[0]
        block_start_ind_effective = np.sort(
            np.unique(np.hstack([block_start_ind_left, block_start_ind_right]))
        )
        return block_start_ind_left, block_start_ind_right, block_start_ind_effective

    # -- Key meta data --
    session_start_time = nwb.session_start_time
    session_date = session_start_time.strftime("%Y-%m-%d")
    subject_id = nwb.subject.subject_id

    # -- Block and probability analysis --
    p_L = df_trials.reward_probabilityL.values
    p_R = df_trials.reward_probabilityR.values
    p_contrast = np.max([p_L, p_R], axis=0) / (np.min([p_L, p_R], axis=0) + 1e-6)
    p_contrast[p_contrast > 100] = 100  # Cap the contrast at 100

    # Parse effective block
    block_start_left, block_start_right, block_start_effective = _get_block_starts(p_L, p_R)
    if "uncoupled" not in nwb.protocol.lower():
        if not (
            len(block_start_left) == len(block_start_right)
            and all(block_start_left == block_start_right)
        ):
            logger.warning("Blocks are not fully aligned in a Coupled task!")

    # -- Metadata dictionary --
    dict_meta = {
        "subject_id": subject_id,
        "session_date": session_date,
        "user_name": nwb.experimenter[0],
        "task": nwb.protocol,
        "session_start_time": session_start_time,
        # Block structure metrics
        "p_reward_sum_mean": np.mean(p_L + p_R),
        "p_reward_sum_std": np.std(p_L + p_R),
        "p_reward_sum_median": np.median(p_L + p_R),
        "p_reward_contrast_mean": np.mean(p_contrast),
        "p_reware_contrast_median": np.median(p_contrast),
        "effective_block_length_mean": np.mean(np.diff(block_start_effective)),
        "effective_block_length_std": np.std(np.diff(block_start_effective)),
        "effective_block_length_median": np.median(np.diff(block_start_effective)),
        "effective_block_length_min": np.min(np.diff(block_start_effective)),
        "effective_block_length_max": np.max(np.diff(block_start_effective)),
        # Duration metrics
        "duration_gocue_stop_mean": df_trials.loc[:, "duration_gocue_stop"].mean(),
        "duration_gocue_stop_std": df_trials.loc[:, "duration_gocue_stop"].std(),
        "duration_gocue_stop_median": df_trials.loc[:, "duration_gocue_stop"].median(),
        "duration_gocue_stop_min": df_trials.loc[:, "duration_gocue_stop"].min(),
        "duration_gocue_stop_max": df_trials.loc[:, "duration_gocue_stop"].max(),
        "duration_delay_period_mean": df_trials.loc[:, "duration_delay_period"].mean(),
        "duration_delay_period_std": df_trials.loc[:, "duration_delay_period"].std(),
        "duration_delay_period_median": df_trials.loc[:, "duration_delay_period"].median(),
        "duration_delay_period_min": df_trials.loc[:, "duration_delay_period"].min(),
        "duration_delay_period_max": df_trials.loc[:, "duration_delay_period"].max(),
        "duration_iti_mean": df_trials.loc[:, "duration_iti"].mean(),
        "duration_iti_std": df_trials.loc[:, "duration_iti"].std(),
        "duration_iti_median": df_trials.loc[:, "duration_iti"].median(),
        "duration_iti_min": df_trials.loc[:, "duration_iti"].min(),
        "duration_iti_max": df_trials.loc[:, "duration_iti"].max(),
        # Reward size metrics
        "reward_volume_left_mean": df_trials.loc[df_trials.reward, "reward_size_left"].mean(),
        "reward_volume_right_mean": df_trials.loc[df_trials.reward, "reward_size_right"].mean(),
        # Lickspouts movement range (in um)
        **{
            f"lickspout_movement_range_{axis}": np.ptp(df_trials[f"lickspout_position_{axis}"])
            for axis in "xyz"
        },
        **{
            f"lickspout_initial_pos_{axis}": df_trials[f"lickspout_position_{axis}"][0]
            for axis in "xyz"
        },
        **{
            f"lickspout_median_pos_{axis}": np.median(df_trials[f"lickspout_position_{axis}"])
            for axis in "xyz"
        },
    }

    # Add flag for old bpod session
    if "bpod" in nwb.session_description:
        dict_meta["old_bpod_session"] = True

    # Create metadata DataFrame
    df_session_meta = pd.DataFrame(dict_meta, index=[0])

    # Add automatic training info
    if "auto_train_engaged" in df_trials.columns:
        df_session_meta["auto_train", "curriculum_name"] = (
            np.nan
            if df_trials.auto_train_curriculum_name.mode()[0].lower() == "none"
            else df_trials.auto_train_curriculum_name.mode()[0]
        )
        df_session_meta["auto_train", "curriculum_version"] = (
            np.nan
            if df_trials.auto_train_curriculum_version.mode()[0].lower() == "none"
            else df_trials.auto_train_curriculum_version.mode()[0]
        )
        df_session_meta["auto_train", "curriculum_schema_version"] = (
            np.nan
            if df_trials.auto_train_curriculum_schema_version.mode()[0].lower() == "none"
            else df_trials.auto_train_curriculum_schema_version.mode()[0]
        )
        df_session_meta["auto_train", "current_stage_actual"] = (
            np.nan
            if df_trials.auto_train_stage.mode()[0].lower() == "none"
            else df_trials.auto_train_stage.mode()[0]
        )
        df_session_meta["auto_train", "if_overriden_by_trainer"] = (
            np.nan
            if all(df_trials.auto_train_stage_overridden.isna())
            else df_trials.auto_train_stage_overridden.mode()[0]
        )
        # Check consistency of auto train settings
        df_session_meta["auto_train", "if_consistent_within_session"] = (
            len(df_trials.groupby([col for col in df_trials.columns if "auto_train" in col])) == 1
        )
    else:
        for field in [
            "curriculum_name",
            "curriculum_version",
            "curriculum_schema_version",
            "current_stage_actual",
            "if_overriden_by_trainer",
        ]:
            df_session_meta["auto_train", field] = None

    return df_session_meta


def compute_session_metrics(nwb):
    """
    Compute all session metadata and performance metrics

    basic performance metrics,      both autowater and non-autowater specific
                                    rates (total, finished, ignored, finished rate,
                                    ignored rate, reward rate)
    calculated metrics,             foraging efficiency, foraging performance
                                    (both normal and random seed), bias naive,
                                    chosen probability
    lick metrics,                   reaction mean and median, early lick rate, invalid
                                    lick ratio, double dipping finished rates (reward and total),
                                    lick consistency means (total, reward, and non-rewarded)

    New addition: chosen_probability - average difference between the chosen probability
    and non-chosen probability / the difference between the largest and smallest probability
    in the session
    """

    if not hasattr(nwb, "df_trials"):
        print("You need to compute df_trials: nwb_utils.create_trials_df(nwb)")
        return

    df_trials = nwb.df_trials.copy()

    # Add session Metdata
    df_session_meta = compute_session_metadata(nwb)

    # -- Performance Metrics --
    n_total_trials = len(df_trials)
    n_finished_trials = (df_trials.animal_response != IGNORE).sum()

    # Actual foraging trials (autowater excluded)
    n_total_trials_non_autowater = df_trials.non_autowater_trial.sum()
    n_finished_trials_non_autowater = df_trials.non_autowater_finished_trial.sum()

    n_reward_trials_non_autowater = df_trials.reward_non_autowater.sum()
    reward_rate_non_autowater_finished = (
        n_reward_trials_non_autowater / n_finished_trials_non_autowater
        if n_finished_trials_non_autowater > 0
        else np.nan
    )

    # Foraging efficiency
    foraging_eff, foraging_eff_random_seed = compute_foraging_efficiency(
        baited="without bait" not in nwb.protocol.lower(),
        choice_history=df_trials.animal_response.map({0: 0, 1: 1, 2: np.nan}).values,
        reward_history=df_trials.rewarded_historyL | df_trials.rewarded_historyR,
        p_reward=[
            df_trials.reward_probabilityL.values,
            df_trials.reward_probabilityR.values,
        ],
        random_number=[
            df_trials.reward_random_number_left.values,
            df_trials.reward_random_number_right.values,
        ],
        autowater_offered=(df_trials.auto_waterL == 1) | (df_trials.auto_waterR == 1),
    )

    all_lick_number = len(nwb.acquisition["left_lick_time"].timestamps) + len(
        nwb.acquisition["right_lick_time"].timestamps
    )

    # Naive bias calculation
    n_left = ((df_trials.animal_response == LEFT) & (df_trials.non_autowater_trial)).sum()
    n_right = ((df_trials.animal_response == RIGHT) & (df_trials.non_autowater_trial)).sum()
    bias_naive = 2 * (n_right / (n_left + n_right) - 0.5) if n_left + n_right > 0 else np.nan
    finished_rate = (
        n_finished_trials_non_autowater / n_total_trials_non_autowater
        if n_total_trials_non_autowater > 0
        else np.nan
    )

    # Probability chosen calculation
    probability_chosen = []
    probability_not_chosen = []

    for _, row in df_trials.iterrows():
        if row.animal_response == 2:
            probability_chosen.append(np.nan)
            probability_not_chosen.append(np.nan)
        elif (
            row.animal_response == 0
        ):  # Chosen = left choice left probability, not chosen = left choice right probability
            probability_chosen.append(row.reward_probabilityL)
            probability_not_chosen.append(row.reward_probabilityR)
        else:  # Chosen = right choice right probability, not chosen = right choice left probability
            probability_chosen.append(row.reward_probabilityR)
            probability_not_chosen.append(row.reward_probabilityL)

    df_trials["probability_chosen"] = probability_chosen
    df_trials["probability_not_chosen"] = probability_not_chosen

    # Calculate chosen probability
    average = df_trials["probability_chosen"] - df_trials["probability_not_chosen"]

    p_larger_global = max(
        df_trials["probability_chosen"].max(), df_trials["probability_not_chosen"].max()
    )

    p_smaller_global = min(
        df_trials["probability_chosen"].min(), df_trials["probability_not_chosen"].min()
    )

    mean_difference = average.mean()
    chosen_probability = mean_difference / (p_larger_global - p_smaller_global)

    # Performance dictionary
    dict_performance = {
        # Basic performance metrics
        "total_trials_with_autowater": n_total_trials,
        "finished_trials_with_autowater": n_finished_trials,
        "finished_rate_with_autowater": n_finished_trials / n_total_trials,
        "ignore_rate_with_autowater": 1 - n_finished_trials / n_total_trials,
        "autowater_collected": (
            ~df_trials.non_autowater_trial & (df_trials.animal_response != IGNORE)
        ).sum(),
        "autowater_ignored": (
            ~df_trials.non_autowater_trial & (df_trials.animal_response == IGNORE)
        ).sum(),
        "total_trials": n_total_trials_non_autowater,
        "finished_trials": n_finished_trials_non_autowater,
        "ignored_trials": n_total_trials_non_autowater - n_finished_trials_non_autowater,
        "finished_rate": finished_rate,
        "ignore_rate": 1 - finished_rate,
        "reward_trials": n_reward_trials_non_autowater,
        "reward_rate": reward_rate_non_autowater_finished,
        "foraging_eff": foraging_eff,
        "foraging_eff_random_seed": foraging_eff_random_seed,
        "foraging_performance": foraging_eff * finished_rate,
        "foraging_performance_random_seed": foraging_eff_random_seed * finished_rate,
        "bias_naive": bias_naive,
        # New Metrics
        "chosen_probability": chosen_probability,
        # Lick timing metrics
        "reaction_time_median": df_trials.loc[:, "reaction_time"].median(),
        "reaction_time_mean": df_trials.loc[:, "reaction_time"].mean(),
        "early_lick_rate": (df_trials.loc[:, "n_lick_all_delay_period"] > 0).sum() / n_total_trials,
        "invalid_lick_ratio": (all_lick_number - df_trials.loc[:, "n_lick_all_gocue_stop"].sum())
        / all_lick_number,
        # Lick consistency metrics
        "double_dipping_rate_finished_trials": (
            df_trials.loc[(df_trials.animal_response != IGNORE), "n_lick_switches_gocue_stop"] > 0
        ).sum()
        / (df_trials.animal_response != IGNORE).sum(),
        "double_dipping_rate_finished_reward_trials": (
            df_trials.loc[df_trials.reward, "n_lick_switches_gocue_stop"] > 0
        ).sum()
        / df_trials.reward.sum(),
        "double_dipping_rate_finished_noreward_trials": (
            df_trials.loc[
                (df_trials.animal_response != IGNORE) & (~df_trials.reward),
                "n_lick_switches_gocue_stop",
            ]
            > 0
        ).sum()
        / ((df_trials.animal_response != IGNORE) & (~df_trials.reward)).sum(),
        "lick_consistency_mean_finished_trials": df_trials.loc[
            (df_trials.animal_response != IGNORE), "n_lick_consistency_gocue_stop"
        ].mean(),
        "lick_consistency_mean_finished_reward_trials": df_trials.loc[
            df_trials.reward, "n_lick_consistency_gocue_stop"
        ].mean(),
        "lick_consistency_mean_finished_noreward_trials": df_trials.loc[
            (df_trials.animal_response != IGNORE) & (~df_trials.reward),
            "n_lick_consistency_gocue_stop",
        ].mean(),
    }

    # Create performance Dataframe
    df_session_performance = pd.DataFrame(dict_performance, index=[0])

    # Create session DataFrame
    df_session = pd.concat([df_session_meta, df_session_performance], axis=1).reset_index(drop=True)

    return df_session
