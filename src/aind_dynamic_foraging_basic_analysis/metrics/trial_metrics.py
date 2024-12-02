"""
    Tools for computing trial by trial metrics
    df_trials = compute_all_trial_metrics(nwb)

"""

import numpy as np

# TODO, we might want to make these parameters metric specific
WIN_DUR = 15
MIN_EVENTS = 2

LEFT, RIGHT, IGNORE = 0, 1, 2


def compute_trial_metrics(nwb):
    """
    Computes all trial by trial metrics

    response_rate,          fraction of trials with a response
    gocue_reward_rate,      fraction of trials with a reward
    response_reward_rate,   fraction of trials with a reward,
                            computed only on trials with a response
    choose_right_rate,      fraction of trials where chose right,
                            computed only on trials with a response

    reward columns,         boolean: trials where reward reward is
                            given, autowater and non_autowater trials
                            are included
    lick columns,           duration, delay, and iti periods in session,
                            right, left, and total lick counts, intertrial
                            choices: switches, consistency, and reaction time

    """
    if not hasattr(nwb, "df_trials"):
        print("You need to compute df_trials: nwb_utils.create_trials_df(nwb)")
        return

    df_trials = nwb.df_trials.copy()

    # --- Add reward-related columns ---
    df_trials["reward"] = False
    df_trials.loc[
        (
            df_trials.rewarded_historyL
            | df_trials.rewarded_historyR
            | df_trials.auto_waterR
            | df_trials.auto_waterL
        )
        & (df_trials.animal_response != IGNORE),
        "reward",
    ] = True

    df_trials["reward_non_autowater"] = False
    df_trials.loc[
        (df_trials.rewarded_historyL | df_trials.rewarded_historyR), "reward_non_autowater"
    ] = True

    df_trials["non_autowater_trial"] = False
    df_trials.loc[
        (df_trials.auto_waterL == 0) & (df_trials.auto_waterR == 0), "non_autowater_trial"
    ] = True

    df_trials["non_autowater_finished_trial"] = df_trials["non_autowater_trial"] & (
        df_trials["animal_response"] != IGNORE
    )
    df_trials["ignored_non_autowater"] = df_trials["non_autowater_trial"] & (
        df_trials["animal_response"] == IGNORE
    )
    df_trials["ignored_autowater"] = ~df_trials["non_autowater_trial"] & (
        df_trials["animal_response"] == IGNORE
    )

    # --- Lick-related stats ---
    all_left_licks = nwb.acquisition["left_lick_time"].timestamps[:]
    all_right_licks = nwb.acquisition["right_lick_time"].timestamps[:]

    # Define the start and stop time for each epoch
    # Using _in_session columns
    lick_stats_epochs = {
        "gocue_stop": ["goCue_start_time_in_session", "stop_time_in_session"],
        "delay_period": ["delay_start_time_in_session", "goCue_start_time_in_session"],
        "iti": ["start_time_in_session", "delay_start_time_in_session"],
    }

    # Trial-by-trial counts
    for i in range(len(df_trials)):
        for epoch_name, (start_time_name, stop_time_name) in lick_stats_epochs.items():
            start_time, stop_time = df_trials.loc[i, [start_time_name, stop_time_name]]

            # Lick analysis for the specific epoch
            left_licks = all_left_licks[
                (all_left_licks > start_time) & (all_left_licks < stop_time)
            ]
            right_licks = all_right_licks[
                (all_right_licks > start_time) & (all_right_licks < stop_time)
            ]
            all_licks = np.hstack([left_licks, right_licks])

            # Lick counts
            df_trials.loc[i, f"duration_{epoch_name}"] = stop_time - start_time
            df_trials.loc[i, f"n_lick_left_{epoch_name}"] = len(left_licks)
            df_trials.loc[i, f"n_lick_right_{epoch_name}"] = len(right_licks)
            df_trials.loc[i, f"n_lick_all_{epoch_name}"] = len(all_licks)

            # Lick switches
            if len(all_licks) > 1:
                _lick_identity = np.hstack(
                    [np.ones(len(left_licks)) * LEFT, np.ones(len(right_licks)) * RIGHT]
                )
                _lick_identity_sorted = [
                    x for x, _ in sorted(zip(_lick_identity, all_licks), key=lambda pairs: pairs[1])
                ]
                df_trials.loc[i, f"n_lick_switches_{epoch_name}"] = np.sum(
                    np.diff(_lick_identity_sorted) != 0
                )

                # Lick consistency
                choice = df_trials.loc[i, "animal_response"]
                df_trials.loc[i, f"n_lick_consistency_{epoch_name}"] = (
                    np.sum(_lick_identity_sorted == choice) / len(_lick_identity_sorted)
                    if len(_lick_identity_sorted) > 0
                    else np.nan
                )
            else:
                df_trials.loc[i, f"n_lick_switches_{epoch_name}"] = 0
                df_trials.loc[i, f"n_lick_consistency_{epoch_name}"] = np.nan

            # Special treatment for gocue to stop epoch
            if epoch_name == "gocue_stop":
                # Reaction time
                first_lick = all_licks.min() if len(all_licks) > 0 else np.nan
                df_trials.loc[i, "reaction_time"] = (
                    first_lick - df_trials.loc[i, "goCue_start_time_in_session"]
                    if not np.isnan(first_lick)
                    else np.nan
                )

                # Handle ignored trials
                if df_trials.loc[i, "animal_response"] == IGNORE:
                    df_trials.loc[i, "reaction_time"] = np.nan
                    df_trials.loc[i, "n_valid_licks_left"] = 0
                    df_trials.loc[i, "n_valid_licks_right"] = 0
                    df_trials.loc[i, "n_valid_licks_all"] = 0

    df_trials["RESPONDED"] = [x in [0, 1] for x in df_trials["animal_response"].values]
    # Rolling fraction of goCues with a response
    df_trials["response_rate"] = (
        df_trials["RESPONDED"].rolling(WIN_DUR, min_periods=MIN_EVENTS, center=True).mean()
    )

    # Rolling fraction of goCues with a response
    df_trials["gocue_reward_rate"] = (
        df_trials["earned_reward"].rolling(WIN_DUR, min_periods=MIN_EVENTS, center=True).mean()
    )

    # Rolling fraction of responses with a response
    df_trials["RESPONSE_REWARD"] = [
        x[0] if x[1] else np.nan for x in zip(df_trials["earned_reward"], df_trials["RESPONDED"])
    ]
    df_trials["response_reward_rate"] = (
        df_trials["RESPONSE_REWARD"].rolling(WIN_DUR, min_periods=MIN_EVENTS, center=True).mean()
    )

    # Rolling fraction of choosing right
    df_trials["WENT_RIGHT"] = [x if x in [0, 1] else np.nan for x in df_trials["animal_response"]]
    df_trials["choose_right_rate"] = (
        df_trials["WENT_RIGHT"].rolling(WIN_DUR, min_periods=MIN_EVENTS, center=True).mean()
    )

    # Clean up temp columns
    drop_cols = [
        "RESPONDED",
        "RESPONSE_REWARD",
        "WENT_RIGHT",
    ]
    df_trials = df_trials.drop(columns=drop_cols)

    return df_trials
