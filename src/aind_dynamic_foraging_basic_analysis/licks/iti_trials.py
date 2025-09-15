"""
Defines functions for analysis of intertrial licking as if they were trials

"""

import numpy as np
import pandas as pd

import aind_dynamic_foraging_basic_analysis.licks.annotation as annotation


def build_iti_trials_table(nwb):
    """
    TODO: fill in
    """
    # Ensure inputs exist and have been processed
    if not hasattr(nwb, "df_trials"):
        print("You need to compute df_trials: nwb_utils.create_df_trials(nwb)")
        return
    if (
        (not hasattr(nwb, "df_licks"))
        or (not "bout_start" in nwb.df_licks)
        or (not "bout_intertrial_choice" in nwb.df_licks)
    ):
        print("You need to annotate df_licks: annotation.annotate_licks(nwb)")
        return

    # Define cue trials and iti trials
    iti_licks = nwb.df_licks.query("bout_start and bout_intertrial_choice").copy()
    df_trials = nwb.df_trials.copy()

    # Adding trial information for iti_trials
    iti_trials = iti_licks[["trial"]].copy()

    # Convert labels in licks table to response
    iti_trials["animal_response"] = [0.0 if "left" in x else 1.0 for x in iti_licks["event"]]

    # Define baiting as always false for ITI trials
    iti_trials["bait_left"] = False
    iti_trials["bait_right"] = False

    # Define the generative reward sample as always 0 for ITI trials
    iti_trials["reward_random_number_left"] = 0
    iti_trials["reward_random_number_right"] = 0

    # Never autowater for ITI trials
    iti_trials["auto_waterL"] = 0
    iti_trials["auto_waterR"] = 0

    # Define goCue as time of the lick
    iti_trials["goCue_start_time_in_session"] = iti_licks["timestamps"]
    iti_trials["goCue_start_time_in_trial"] = 0
    iti_trials["goCue_start_time_raw"] = iti_licks["raw_timestamps"]

    # Define reward outcome time as time of lick
    iti_trials["reward_outcome_time_in_session"] = iti_licks["timestamps"]
    iti_trials["reward_outcome_time_in_trial"] = 0

    # Define choice time as time of lick
    iti_trials["choice_time_in_session"] = iti_licks["timestamps"]
    iti_trials["choice_time_in_trial"] = 0

    # columns we propagate after merge (all get NaNs now)
    to_propagate = [
        "base_reward_probability_sum",
        "reward_probabilityL",
        "reward_probabilityR",
        "left_valve_open_time",
        "right_valve_open_time",
        "block_beta",
        "block_min",
        "block_max",
        "min_reward_each_block",
        "delay_beta",
        "delay_min",
        "delay_max",
        "ITI_beta",
        "ITI_min",
        "ITI_max",
        "response_duration",
        "reward_consumption_duration",
        "auto_train_engaged",
        "auto_train_curriculum_name",
        "auto_train_curriculum_version",
        "auto_train_curriculum_schema_version",
        "auto_train_stage",
        "auto_train_stage_overridden",
        "lickspout_position_x",
        "lickspout_position_y",
        "lickspout_position_z",
        "reward_size_left",
        "reward_size_right",
        "ses_idx",
    ]

    # columns that will be undefined as NaN
    stay_nans = [
        "laser_on_trial",
        "laser_wavelength",
        "laser_location",
        "laser_power",
        "laser_duration",
        "laser_condition",
        "laser_condition_probability",
        "laser_start",
        "laser_start_offset",
        "laser_end",
        "laser_end_offset",
        "laser_protocol",
        "laser_frequency",
        "laser_rampingdown",
        "laser_pulse_duration",
        "bonsai_start_time_in_session",
        "bonsai_start_time_in_trial",
        "bonsai_stop_time_in_session",
        "bonsai_stop_time_in_trial",
        "delay_start_time_in_session",
        "delay_start_time_in_trial",
        "reward_time_in_session",
        "reward_time_in_trial",
        "right_reward_type",
        "left_reward_type",
        "earned_reward",
        "extra_reward",
        "ITI_duration",
        "delay_duration",
    ]

    for col in to_propagate:
        if col in df_trials:
            iti_trials[col] = np.nan
    for col in stay_nans:
        if col in df_trials:
            iti_trials[col] = np.nan

    # Merge into trials dataframe
    df_trials["cue_trial"] = True
    iti_trials["cue_trial"] = False
    df_trials = (
        pd.concat([df_trials, iti_trials])
        .sort_values(by="goCue_start_time_in_session")
        .reset_index(drop=True)
    )
    df_trials["trial"] = df_trials.index.values

    # Compute reward history
    # Propagate rewarded history backwards to fill non-cue trials
    pd.set_option("future.no_silent_downcasting", True)
    df_trials["rewarded_historyL"] = df_trials["rewarded_historyL"].bfill()
    df_trials["rewarded_historyR"] = df_trials["rewarded_historyR"].bfill()

    # set rewarded_history to false for all trials after non-cue trials
    index = df_trials[df_trials["cue_trial"] == False].index.values + 1
    df_trials.loc[index, "rewarded_historyL"] = False
    df_trials.loc[index, "rewarded_historyR"] = False

    # Propagate some columns forward
    index = df_trials[df_trials["cue_trial"] == True].index.values
    df_trials.loc[index, to_propagate] = df_trials.loc[index, to_propagate].ffill()

    return df_trials


# dev
