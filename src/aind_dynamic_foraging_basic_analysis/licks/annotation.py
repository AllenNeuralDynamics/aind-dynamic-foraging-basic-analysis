"""
    Tools for annotation of lick bouts
    df_licks = annotate_licks(nwb)
    df_licks = annotate_lick_bouts(nwb)
    df_licks = annotate_rewards(nwb)
    df_licks = annotate_cue_response(nwb)
    df_licks = annotate_intertrial_choices(nwb)
    df_licks = annotate_switches(nwb)
"""

import numpy as np

# Maximum time between a lick and a reward delivery to assign that reward to the lick
LICK_TO_REWARD_TOLERANCE = 0.25

# Maximum time between licks to label them as one bout
BOUT_THRESHOLD = 0.7

# Maximum time between go cue and first lick to label the lick as cue responsive
CUE_TO_LICK_TOLERANCE = 1

# Maximum time after the last go cue for a bout to start and be considered within the session
CUE_TO_SESSION_END_TOLERANCE = CUE_TO_LICK_TOLERANCE

# Minimum time between licks to be flagged as artifacts
ARTIFACT_TOLERANCE = 0.0001


def annotate_licks(nwb):
    """
    Adds all annotations
    nwb is an object that has df_events as an attribute
    """
    if not hasattr(nwb, "df_events"):
        print("You need to compute df_events: nwb_utils.create_df_events(nwb)")
        return
    nwb.df_licks = annotate_lick_bouts(nwb)
    nwb.df_licks = annotate_artifacts(nwb)
    nwb.df_licks = annotate_rewards(nwb)
    nwb.df_licks = annotate_cue_response(nwb)
    nwb.df_licks = annotate_intertrial_choices(nwb)
    nwb.df_licks = annotate_switches(nwb)
    nwb.df_licks = annotate_within_session(nwb)
    return nwb.df_licks


def annotate_lick_bouts(nwb):
    """
    returns a dataframe of lick times with annotations
        pre_ili, the elapsed time since the last lick (on either side)
        post_ili, the time until the next lick (on either side)
        bout_start (bool), whether this was the start of a lick bout
        bout_end (bool), whether this was the end of a lick bout)
        bout_number (int), what lick bout this was a part of

    nwb, an object with attributes: df_events
    """

    if not hasattr(nwb, "df_events"):
        print("You need to compute df_events: nwb_utils.create_df_events(nwb)")
        return
    df_licks = nwb.df_events.query('event in ["right_lick_time","left_lick_time"]').copy()
    df_licks.reset_index(drop=True, inplace=True)

    # Computing ILI for each lick
    df_licks["pre_ili"] = np.concatenate([[np.nan], np.diff(df_licks.timestamps.values)])
    df_licks["post_ili"] = np.concatenate([np.diff(df_licks.timestamps.values), [np.nan]])

    # Assign licks into bouts
    df_licks["bout_start"] = df_licks["pre_ili"] > BOUT_THRESHOLD
    df_licks["bout_end"] = df_licks["post_ili"] > BOUT_THRESHOLD
    df_licks.loc[df_licks["pre_ili"].isnull(), "bout_start"] = True
    df_licks.loc[df_licks["post_ili"].isnull(), "bout_end"] = True
    df_licks["bout_number"] = np.cumsum(df_licks["bout_start"])

    # Check that bouts start and stop
    num_bout_start = df_licks["bout_start"].sum()
    num_bout_end = df_licks["bout_end"].sum()
    num_bouts = df_licks["bout_number"].max()
    assert num_bout_start == num_bout_end, "Bout Starts and Bout Ends don't align"
    assert num_bout_start == num_bouts, "Number of bouts is incorrect"

    return df_licks


def annotate_artifacts(nwb):
    """
    annotates df_licks with which licks could be electrical artifacts
        likely_artifact (bool) was this lick likely an artifact
    nwb, an object with attributes: df_licks, df_events
    """
    if not hasattr(nwb, "df_events"):
        print("You need to compute df_events: nwb_utils.create_df_events(nwb)")
        return

    if not hasattr(nwb, "df_licks"):
        print("annotating lick bouts")
        nwb.df_licks = annotate_lick_bouts(nwb)

    # make a copy of df licks
    df_licks = nwb.df_licks.copy()

    # Find lick intervals less than tolerance that also switch sides
    # mark the second lick as a likely artifact
    df_licks["switch_lick"] = df_licks["event"] != df_licks.shift(1)["event"]
    df_licks.loc[0, "switch_lick"] = False
    df_licks["likely_artifact"] = [
        np.all(x) for x in zip(df_licks["switch_lick"], df_licks["pre_ili"] < ARTIFACT_TOLERANCE)
    ]

    # Clean up temporary column
    df_licks = df_licks.drop(columns=["switch_lick"])
    return df_licks


def annotate_rewards(nwb):
    """
    Annotates df_licks with which lick triggered each reward
        rewarded (bool) did this lick trigger a reward
        bout_rewarded (bool) did this lick bout trigger a reward
    nwb, an object with attributes: df_licks, df_events
    """

    if not hasattr(nwb, "df_events"):
        print("You need to compute df_events: nwb_utils.create_df_events(nwb)")
        return

    # ensure we have df_licks
    if not hasattr(nwb, "df_licks"):
        print("annotating lick bouts")
        nwb.df_licks = annotate_lick_bouts(nwb)

    # make a copy of df licks
    df_licks = nwb.df_licks.copy()

    # set default to false
    df_licks["rewarded"] = False

    # Iterate right rewards, and find most recent lick within tolerance
    right_rewards = nwb.df_events.query('event == "right_reward_delivery_time"').copy()
    for index, row in right_rewards.iterrows():
        this_reward_lick_times = np.where(
            (df_licks.timestamps <= row.timestamps)
            & (df_licks.timestamps > (row.timestamps - LICK_TO_REWARD_TOLERANCE))
            & (df_licks.event == "right_lick_time")
        )[0]
        if len(this_reward_lick_times) > 0:
            df_licks.at[this_reward_lick_times[-1], "rewarded"] = True
        # TODO, if we can't find a matching lick, should ensure this is manual or auto water

    # Iterate left rewards, and find most recent lick within tolerance
    left_rewards = nwb.df_events.query('event == "left_reward_delivery_time"').copy()
    for index, row in left_rewards.iterrows():
        this_reward_lick_times = np.where(
            (df_licks.timestamps <= row.timestamps)
            & (df_licks.timestamps > (row.timestamps - LICK_TO_REWARD_TOLERANCE))
            & (df_licks.event == "left_lick_time")
        )[0]
        if len(this_reward_lick_times) > 0:
            df_licks.at[this_reward_lick_times[-1], "rewarded"] = True

    # Annotate lick bouts as rewarded or unrewarded
    x = (
        df_licks.groupby("bout_number")
        .any("rewarded")
        .rename(columns={"rewarded": "bout_rewarded"})["bout_rewarded"]
    )
    df_licks["bout_rewarded"] = False
    temp = df_licks.reset_index().set_index("bout_number").copy()
    temp.update(x)
    temp = temp.reset_index().set_index("index")
    df_licks["bout_rewarded"] = temp["bout_rewarded"]

    return df_licks


def annotate_cue_response(nwb):
    """
    Annotates df_licks with which lick was immediately after a go cue
        cue_response (bool) was this lick immediately after a go cue
        bout_cue_response (bool) was this licking bout immediately after a go cue
    nwb, an object with attributes: df_licks, df_events
    """

    if not hasattr(nwb, "df_events"):
        print("You need to compute df_events: nwb_utils.create_df_events(nwb)")
        return

    # ensure we have df_licks
    if not hasattr(nwb, "df_licks"):
        print("annotating lick bouts")
        nwb.df_licks = annotate_lick_bouts(nwb)

    # make a copy of df licks
    df_licks = nwb.df_licks.copy()

    # set default to false
    df_licks["cue_response"] = False

    # Iterate go cues, and find most recent lick within tolerance
    cues = nwb.df_events.query('event == "goCue_start_time"').copy()
    for index, row in cues.iterrows():
        this_lick_times = np.where(
            (df_licks.timestamps > row.timestamps)
            & (df_licks.timestamps <= (row.timestamps + CUE_TO_LICK_TOLERANCE))
            & ((df_licks.event == "right_lick_time") | (df_licks.event == "left_lick_time"))
            & (df_licks.bout_start)
        )[0]
        if len(this_lick_times) > 0:
            df_licks.at[this_lick_times[0], "cue_response"] = True

    # Annotate lick bouts as cue_responsive, or unresponsive
    x = (
        df_licks.groupby("bout_number")
        .any("cue_response")
        .rename(columns={"cue_response": "bout_cue_response"})["bout_cue_response"]
    )
    df_licks["bout_cue_response"] = False
    temp = df_licks.reset_index().set_index("bout_number").copy()
    temp.update(x)
    temp = temp.reset_index().set_index("index")
    df_licks["bout_cue_response"] = temp["bout_cue_response"]

    return df_licks


def annotate_intertrial_choices(nwb):
    """
    annotate licks and lick bouts as intertrial choices if they are not cue_responsive
        intertrial_choice (bool) was this lick the start of a non-cue-responsive bout
        bout_intertrial_choice (bool) was this bout non-cue-responsive?
    """
    # Add lick_bout annotation, and cue_response if not already added
    if not hasattr(nwb, "df_events"):
        print("You need to compute df_events: nwb_utils.create_df_events(nwb)")
        return
    if not hasattr(nwb, "df_licks"):
        nwb.df_licks = annotate_lick_bouts(nwb)
    if "cue_response" not in nwb.df_licks:
        nwb.df_licks = annotate_cue_response(nwb)

    # Make a copy
    df_licks = nwb.df_licks.copy()

    # Define intertrial choices
    df_licks["intertrial_choice"] = df_licks["bout_start"] & ~df_licks["cue_response"]

    # Annotate lick bouts as intertrial_choice
    x = (
        df_licks.groupby("bout_number")
        .any("intertrial_choice")
        .rename(columns={"intertrial_choice": "bout_intertrial_choice"})["bout_intertrial_choice"]
    )
    df_licks["bout_intertrial_choice"] = False
    temp = df_licks.reset_index().set_index("bout_number").copy()
    temp.update(x)
    temp = temp.reset_index().set_index("index")
    df_licks["bout_intertrial_choice"] = temp["bout_intertrial_choice"]

    return df_licks


def annotate_switches(nwb):
    """
    cue_switch: this cue_choice differs from the previous cue_choice
    iti_switch: this intertrial_choice differs from the previous choice (iti or cue)
    """
    # Add lick_bout annotation, and cue_response if not already added
    if not hasattr(nwb, "df_events"):
        print("You need to compute df_events: nwb_utils.create_df_events(nwb)")
        return
    if not hasattr(nwb, "df_licks"):
        nwb.df_licks = annotate_lick_bouts(nwb)
    if "cue_response" not in nwb.df_licks:
        nwb.df_licks = annotate_cue_response(nwb)
    if "intertrial_choice" not in nwb.df_licks:
        nwb.df_licks = annotate_intertrial_choices(nwb)

    # Make a copy
    df_licks = nwb.df_licks.copy()

    # Compute cue_switch labels
    df_cue_bouts = df_licks.query("bout_start").query("cue_response").copy()
    if len(df_cue_bouts) > 0:
        df_cue_bouts["cue_switch"] = (
            df_cue_bouts["event"].shift(1, fill_value=df_cue_bouts.iloc[0]["event"])
            != df_cue_bouts["event"]
        )
    else:
        df_cue_bouts["cue_switch"] = []

    # Compute iti_switch labels
    df_bouts = df_licks.query("bout_start").copy()
    df_bouts["iti_switch"] = df_bouts["intertrial_choice"] & (
        df_bouts["event"].shift(1, fill_value=df_bouts.iloc[0]["event"]) != df_bouts["event"]
    )

    # Add columns to df_licks
    df_licks = df_licks.join(df_cue_bouts["cue_switch"], how="left")
    df_licks = df_licks.join(df_bouts["iti_switch"], how="left")

    # Fill NaNs as False
    df_licks["cue_switch"] = df_licks["cue_switch"] == True  # noqa: E712
    df_licks["iti_switch"] = df_licks["iti_switch"] == True  # noqa: E712

    # Annotate lick bouts as cue_switch
    x = (
        df_licks.groupby("bout_number")
        .any("cue_switch")
        .rename(columns={"cue_switch": "bout_cue_switch"})["bout_cue_switch"]
    )
    df_licks["bout_cue_switch"] = False
    temp = df_licks.reset_index().set_index("bout_number").copy()
    temp.update(x)
    temp = temp.reset_index().set_index("index")
    df_licks["bout_cue_switch"] = temp["bout_cue_switch"]

    # Annotate lick bouts as iti_switch
    x = (
        df_licks.groupby("bout_number")
        .any("iti_switch")
        .rename(columns={"iti_switch": "bout_iti_switch"})["bout_iti_switch"]
    )
    df_licks["bout_iti_switch"] = False
    temp = df_licks.reset_index().set_index("bout_number").copy()
    temp.update(x)
    temp = temp.reset_index().set_index("index")
    df_licks["bout_iti_switch"] = temp["bout_iti_switch"]

    return df_licks


def annotate_within_session(nwb):
    """
    within_session: this lick happened after the first go cue,
        or < CUE_TO_SESSION_END_TOLERANCE after the last go cue
    """

    if not hasattr(nwb, "df_events"):
        print("You need to compute df_events: nwb_utils.create_df_events(nwb)")
        return

    # ensure we have df_licks
    if not hasattr(nwb, "df_licks"):
        print("annotating lick bouts")
        nwb.df_licks = annotate_lick_bouts(nwb)

    # make a copy of df licks
    df_licks = nwb.df_licks.copy()

    # Test for no go cues
    goCues = nwb.df_events.query('event == "goCue_start_time"')
    if len(goCues) == 0:
        df_licks["within_session"] = False
    else:
        start_time = goCues.iloc[0]["timestamps"]
        end_time = goCues.iloc[-1]["timestamps"] + CUE_TO_SESSION_END_TOLERANCE
        df_licks["within_session"] = (start_time <= df_licks["timestamps"]) & (
            df_licks["timestamps"] < end_time
        )

    return df_licks
