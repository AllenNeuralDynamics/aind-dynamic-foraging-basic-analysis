"""
    Tools for computing trial by trial metrics
    df_trials = compute_trial_metrics(nwb)
    df_trials = compute_bias(nwb)

"""

import pandas as pd
import numpy as np

import aind_dynamic_foraging_models.logistic_regression.model as model

# TODO, we might want to make these parameters metric specific
WIN_DUR = 15
MIN_EVENTS = 2


def compute_trial_metrics(nwb):
    """
    Computes trial by trial metrics

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

    df_trials = nwb.df_trials.copy()

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

    # TODO, add from process_nwb
    # trial duration (stop-time - start-time) (start/stop time, or gocue to gocue?)
    # n_licks_left (# of left licks in response window)
    # n_licks_left_total (# of left licks from goCue to next go cue)
    # Same for Right, same for all
    # intertrial choices (boolean)
    # number of intertrial choices
    # number of intertrial switches
    # response switch or repeat

    # Clean up temp columns
    drop_cols = [
        "RESPONDED",
        "RESPONSE_REWARD",
        "WENT_RIGHT",
    ]
    df_trials = df_trials.drop(columns=drop_cols)

    return df_trials


def compute_bias(nwb):
    """
    Computes side bias by fitting a logistic regression model
    returns trials table with the following columns:
    bias, the side bias
    bias_ci_lower, the lower confidence interval on the bias
    bias_ci_upper, the uppwer confidence interval on the bias
    """

    # Parameters for computing bias
    n_trials_back = 5
    max_window = 200
    cv = 1
    compute_every = 10
    BIAS_LIMIT = 10

    # Make sure trials table has been computed
    if not hasattr(nwb, "df_trials"):
        print("You need to compute df_trials: nwb_utils.create_trials_df(nwb)")
        return

    # extract choice and reward
    df_trials = nwb.df_trials.copy()
    df_trials["choice"] = [np.nan if x == 2 else x for x in df_trials["animal_response"]]
    df_trials["reward"] = [
        any(x) for x in zip(df_trials["earned_reward"], df_trials["extra_reward"])
    ]

    # Set up lists to store results
    bias = []
    ci_lower = []
    ci_upper = []
    C = []

    # Iterate over trials and compute
    compute_on = np.arange(compute_every, len(df_trials), compute_every)
    for i in compute_on:
        # Determine interval to compute on
        start = np.max([0, i - max_window])
        end = i

        # extract choice and reward
        choice = df_trials.loc[start:end]["choice"].values
        reward = df_trials.loc[start:end]["reward"].values

        # Determine if we have valid data to fit model
        unique = np.unique(choice[~np.isnan(choice)])
        if len(unique) == 0:
            # no choices, report bias confidence as (-inf, +inf)
            bias.append(np.nan)
            ci_lower.append(-BIAS_LIMIT)
            ci_upper.append(BIAS_LIMIT)
            C.append(np.nan)
        elif len(unique) == 2:
            # Fit model
            out = model.fit_logistic_regression(
                choice, reward, n_trial_back=n_trials_back, cv=cv, fit_exponential=False
            )
            bias.append(out["df_beta"].loc["bias"]["bootstrap_mean"].values[0])
            ci_lower.append(out["df_beta"].loc["bias"]["bootstrap_CI_lower"].values[0])
            ci_upper.append(out["df_beta"].loc["bias"]["bootstrap_CI_upper"].values[0])
            C.append(out["C"])
        elif unique[0] == 0:
            # only left choices, report bias confidence as (-inf, 0)
            bias.append(-1)
            ci_lower.append(-BIAS_LIMIT)
            ci_upper.append(0)
            C.append(np.nan)
        elif unique[0] == 1:
            # only right choices, report bias confidence as (0, +inf)
            bias.append(+1)
            ci_lower.append(0)
            ci_upper.append(BIAS_LIMIT)
            C.append(np.nan)

    # Pack results into a dataframe
    df = pd.DataFrame()
    df["trial"] = compute_on
    df["bias"] = bias
    df["bias_ci_lower"] = ci_lower
    df["bias_ci_upper"] = ci_upper
    df["bias_C"] = C

    # merge onto trials dataframe
    df_trials = pd.merge(
        nwb.df_trials.drop(columns=["bias", "bias_ci_lower", "bias_ci_upper"], errors="ignore"),
        df[["trial", "bias", "bias_ci_lower", "bias_ci_upper"]],
        how="left",
        on=["trial"],
    )

    # fill in bias on non-computed trials
    df_trials["bias"] = df_trials["bias"].bfill().ffill()
    df_trials["bias_ci_lower"] = df_trials["bias_ci_lower"].bfill().ffill()
    df_trials["bias_ci_upper"] = df_trials["bias_ci_upper"].bfill().ffill()

    return df_trials
