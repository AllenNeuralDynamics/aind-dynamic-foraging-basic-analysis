"""
    Tools for computing trial by trial metrics
    df_trials = compute_trial_metrics(nwb)
    df_trials = compute_bias(nwb)

"""

import aind_dynamic_foraging_data_utils.nwb_utils as nu
import aind_dynamic_foraging_models.logistic_regression.model as model
from aind_dynamic_foraging_data_utils import alignment as an
import numpy as np
import pandas as pd
import warnings

import aind_dynamic_foraging_basic_analysis.licks.annotation as a

# We might want to make these parameters metric specific
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
    intertrial_choice,      whether there was an intertrial lick event
    intertrial_choice_rate,   rolling fraction of go cues with intertrial licking

    """
    if not hasattr(nwb, "df_events"):
        print("computing df_events first")
        nwb.df_events = nu.create_df_events(nwb)

    if not hasattr(nwb, "df_trials"):
        print("computing df_trials")
        nwb.df_trials = nu.create_df_trials(nwb)

    if not hasattr(nwb, "df_licks"):
        print("Annotating licks")
        nwb.df_licks = a.annotate_licks(nwb)

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

    # Add intertrial licking
    df_trials = add_intertrial_licking(df_trials, nwb.df_licks)

    # Clean up temp columns
    drop_cols = [
        "RESPONDED",
        "RESPONSE_REWARD",
        "WENT_RIGHT",
    ]
    df_trials = df_trials.drop(columns=drop_cols)

    return df_trials


def compute_side_bias(nwb):
    """
    Computes side bias by fitting a logistic regression model
    returns trials table with the following columns:
    side_bias, the side bias
    side_bias_confidence_interval, the [lower,upper] confidence interval on the bias
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
    df["side_bias"] = bias
    df["side_bias_confidence_interval_lower"] = ci_lower
    df["side_bias_confidence_interval_upper"] = ci_upper
    df["side_bias_C"] = C

    # merge onto trials dataframe
    df_trials = nwb.df_trials.copy()
    df_trials = pd.merge(
        df_trials.drop(
            columns=[
                "side_bias",
                "side_bias_confidence_interval_lower",
                "side_bias_confidence_interval_upper",
            ],
            errors="ignore",
        ),
        df[
            [
                "trial",
                "side_bias",
                "side_bias_confidence_interval_lower",
                "side_bias_confidence_interval_upper",
            ]
        ],
        how="left",
        on=["trial"],
    )

    # fill in side_bias on non-computed trials
    df_trials["side_bias"] = df_trials["side_bias"].bfill().ffill()
    df_trials["side_bias_confidence_interval_lower"] = (
        df_trials["side_bias_confidence_interval_lower"].bfill().ffill()
    )
    df_trials["side_bias_confidence_interval_upper"] = (
        df_trials["side_bias_confidence_interval_upper"].bfill().ffill()
    )
    df_trials["side_bias_confidence_interval"] = [
        x
        for x in zip(
            df_trials["side_bias_confidence_interval_lower"],
            df_trials["side_bias_confidence_interval_upper"],
        )
    ]

    df_trials = df_trials.drop(
        columns=["side_bias_confidence_interval_lower", "side_bias_confidence_interval_upper"]
    )

    return df_trials


def add_intertrial_licking(df_trials, df_licks):
    """
    Adds two metrics
    intertrial_choice (bool), whether there was an intertrial lick event
    intertrial_choice_rate (float), rolling fraction of go cues with intertrial licking
    """

    has_intertrial_choice = (
        df_licks.query("within_session").groupby("trial")["intertrial_choice"].any()
    )
    df_trials.drop(columns=["intertrial_choice", "intertrial_choice_rate"], errors="ignore")
    df_trials = pd.merge(df_trials, has_intertrial_choice, on="trial", how="left")
    with pd.option_context("future.no_silent_downcasting", True):
        df_trials["intertrial_choice"] = (
            df_trials["intertrial_choice"].fillna(False).infer_objects(copy=False)
        )

    # Rolling fraction of goCues with intertrial licking
    df_trials["intertrial_choice_rate"] = (
        df_trials["intertrial_choice"].rolling(WIN_DUR, min_periods=MIN_EVENTS, center=True).mean()
    )
    return df_trials


def get_average_signal_window_multi(
    nwbs,
    alignment_event,
    offsets,
    channel,
    data_column='data_z',
    censor=True,
    output_col=None
):
    """
    Wrapper for get_average_signal_window to process a
    list of nwb objects and concatenate the results.

    Parameters
    ----------
    nwbs : list
        List of nwb-like objects (each with .df_trials and .df_fip).
    alignment_event : str
        The event column in df_trials to align to.
    offsets : list or tuple of float
        [start, end] offsets (in seconds) relative to alignment_event.
    channel : str
        The value in df_fip['event'] to filter for.
    data_col : str
        Column in df_fip to extract (default 'data_z').
    censor, censor important timepoints before and after aligned timepoints
    output_col : str or None
        Name for the new column. If None, will be generated automatically.

    Returns
    -------
    pd.DataFrame
        Concatenated DataFrame of all trials with the new signal window column.
    """
    for nwb in nwbs:
        df_trials = get_average_signal_window(
            nwb,
            alignment_event=alignment_event,
            offsets=offsets,
            channel=channel,
            data_column=data_column,
            censor=censor,
            output_col=output_col
        )
        nwb.df_trials = df_trials
    return nwbs


def get_average_signal_window(
    nwb,
    alignment_event,
    offsets,
    channel,
    data_column='data_z',
    censor=True,
    output_col=None,
):
    """
    Returns a Series with the mean signal in a window around an alignment event,
    for each trial, for each session and a specific signal (event).

    Parameters
    ----------
    nwb : nwb object (or nwb-like object)
        nwb object with df_fip and df_trials attributes
    alignment_event : str
        The event column in df_trials to align to. must be given in_session, not in_trial
    offsets: list or tuple of float
        [start, end] offsets (in seconds) relative to alignment_event.
    channel : str
        The value in df_fip['event'] to filter for.
    data_column : str
        Column in df_fip to extract (default 'data_z').
    censor, censor important timepoints before and after aligned timepoints
    output_col : str or None
        Name for the new column. If None, will be generated as
        '<data_col>_<channel>_<start>_<end>_<alignment_event>'.


    Returns
    -------
    df_trial: pd.DataFrame
        DataFrame with a new column containing the mean signal
        in the specified window for each trial.

    EXAMPLE
    *******************
    df_trials = get_average_signal_window(nwb, alignment_event='choice_time_in_session',
                        offsets=[0.33,1],channel='G_0_dff-bright_mc-iso-IRLS',
                        data_column='data_z_norm')
    """

    # Check alignment_event ends with 'in_session'
    if not alignment_event.endswith('in_session'):
        raise ValueError(f"alignment_event '{alignment_event}' must end with 'in_session'.")

    if not hasattr(nwb, "df_trials"):
        raise ValueError("You need to compute df_trials: nwb_utils.create_trials_df(nwb)")

    if not hasattr(nwb, "df_fip"):
        raise ValueError("You need to compute df_fip: nwb_utils.create_df_fip(nwb)")

    # Check alignment_event is in df_trials columns
    if alignment_event not in nwb.df_trials.columns:
        raise ValueError(f"alignment_event '{alignment_event}' not found in df_trials columns.")

    if channel not in nwb.df_fip.event.unique():
        warnings.warn(f"{channel} channel not found in df_fip. Returning original df_trials.")
        return nwb.df_trials

    if data_column not in nwb.df_fip.columns:
        raise ValueError(f"data column '{data_column}' not found in df_trials columns.")

    # Get output column name
    if output_col is None:
        output_col = (
            f"{data_column}_{channel}_{offsets[0]}_"
            f"{offsets[1]}_{alignment_event.replace('_in_session','')}"
        )

    # copy df_trials, drops na values, sort trial by alignment event
    # sorting needed because censor in event_triggered_response sorts
    # this allows the trials to be matched with event_times
    df_trials = nwb.df_trials.dropna(subset=alignment_event, inplace=False)
    df_trials = df_trials.sort_values(by=alignment_event)

    data = nwb.df_fip.query("event == @channel")
    align_timepoints = df_trials[alignment_event].values

    etr = an.event_triggered_response(
        data,
        "timestamps",
        data_column,
        align_timepoints,
        t_start=offsets[0],
        t_end=offsets[1],
        output_sampling_rate=40,
        censor=censor,
        censor_times=None,
    )

    avg_activity = etr.groupby("event_number").mean()
    avg_activity['trial'] = df_trials.trial.values
    avg_activity = avg_activity.rename(columns={data_column: output_col})

    # Merge on 'trial'
    df_trials = df_trials.merge(avg_activity[['trial', output_col]], on='trial', how='left')

    return df_trials
