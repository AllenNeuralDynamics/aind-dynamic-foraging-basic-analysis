"""Util function for reading NWB files. (for dev only)"""

import numpy as np
from pynwb import NWBHDF5IO


def get_history_from_nwb(nwb_file):
    """Get choice and reward history from nwb file"""

    io = NWBHDF5IO(nwb_file, mode="r")
    nwb = io.read()
    df_trial = nwb.trials.to_dataframe()

    # Exclude autowater
    df_trial["non_autowater_trial"] = False
    df_trial.loc[
        (df_trial.auto_waterL == 0) & (df_trial.auto_waterR == 0),
        "non_autowater_trial",
    ] = True
    non_autowater = df_trial.non_autowater_trial

    choice_history = df_trial.animal_response[non_autowater]
    choice_history[choice_history == 2] = np.nan  # Recoding
    reward_history = (
        df_trial.rewarded_historyL[non_autowater] | df_trial.rewarded_historyR[non_autowater]
    )
    p_reward = [
        df_trial.reward_probabilityL[non_autowater].values,
        df_trial.reward_probabilityR[non_autowater].values,
    ]
    random_number = [
        df_trial.reward_random_number_left[non_autowater].values,
        df_trial.reward_random_number_right[non_autowater].values,
    ]

    baiting = False if "without baiting" in nwb.protocol.lower() else True

    return (
        choice_history,
        reward_history,
        p_reward,
        random_number,
        baiting,
    )
