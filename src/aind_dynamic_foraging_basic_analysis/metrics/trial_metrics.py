"""
    Tools for computing trial by trial metrics
    df_trials = compute_all_trial_metrics(nwb)

"""

import numpy as np

# TODO, we might want to make these parameters metric specific
WIN_DUR = 15
MIN_EVENTS = 2


def compute_all_trial_metrics(nwb):
    """
    Computes all trial by trial metrics

    response_rate,          fraction of trials with a response
    gocue_reward_rate,      fraction of trials with a reward
    response_reward_rate,   fraction of trials with a reward,
                            computed only on trials with a response
    choose_right_rate,      fraction of trials where chose right,
                            computed only on trials with a response

    """
    if not hasattr(nwb, "df_trials"):
        print("You need to compute df_trials: nwb_utils.create_trials_df(nwb)")
        return

    df = nwb.df_trials.copy()

    df["RESPONDED"] = [x in [0, 1] for x in df["animal_response"].values]
    # Rolling fraction of goCues with a response
    df["response_rate"] = (
        df["RESPONDED"].rolling(WIN_DUR, min_periods=MIN_EVENTS, center=True).mean()
    )

    # Rolling fraction of goCues with a response
    df["gocue_reward_rate"] = (
        df["earned_reward"].rolling(WIN_DUR, min_periods=MIN_EVENTS, center=True).mean()
    )

    # Rolling fraction of responses with a response
    df["RESPONSE_REWARD"] = [
        x[0] if x[1] else np.nan for x in zip(df["earned_reward"], df["RESPONDED"])
    ]
    df["response_reward_rate"] = (
        df["RESPONSE_REWARD"].rolling(WIN_DUR, min_periods=MIN_EVENTS, center=True).mean()
    )

    # Rolling fraction of choosing right
    df["WENT_RIGHT"] = [x if x in [0, 1] else np.nan for x in df["animal_response"]]
    df["choose_right_rate"] = (
        df["WENT_RIGHT"].rolling(WIN_DUR, min_periods=MIN_EVENTS, center=True).mean()
    )

    # Clean up temp columns
    drop_cols = ["RESPONDED", "RESPONSE_REWARD", "WENT_RIGHT"]
    df = df.drop(columns=drop_cols)

    return df
