import numpy as np

from aind_dynamic_foraging_data_utils import nwb_utils as nu

"""
    Tools for annotation of lick bouts
    df_licks = annotate_lick_bouts(nwb)
    df_licks = annotate_rewards(nwb)
    TODO
    annotate lick bouts with reward
        Should check for licks before the last goCue
        set parameter for lick/reward tolerance
        make a note that we should ensure a matching lick for non autowater or manual water licks
    annotate possible bad licks?
    annotate cross licks?
    use lick annotations for rewarded and unrewarded lick bout starts
        make example of how to do this, since you need to build dictionary
    maybe annotate rewarded and unrewarded go cues
    Is the PSTH code flexible enough to take different options?
    censor should take cross events, instead of just self events
    sync streamlit version with plot bouts
"""


def annotate_lick_bouts(nwb, bout_threshold=0.7):
    '''
        returns a dataframe of lick times with annotations
            pre_ili, the elapsed time since the last lick (on either side)
            post_ili, the time until the next lick (on either side)
            bout_start (bool), whether this was the start of a lick bout
            bout_end (bool), whether this was the end of a lick bout)
            bout_number (int), what lick bout this was a part of

        nwb, an nwb-like object with attributes: df_events
        bout_threshold is the ILI that determines bout segmentation
    '''

    if not hasattr(nwb, "df_events"):
        nwb.df_events = nu.create_events_df(nwb)
    df_licks = nwb.df_events.query('event in ["right_lick_time","left_lick_time"]').copy()
    df_licks.reset_index(drop=True, inplace=True)

    # Computing ILI for each lick
    df_licks["pre_ili"] = np.concatenate([[np.nan], np.diff(df_licks.timestamps.values)])
    df_licks["post_ili"] = np.concatenate([np.diff(df_licks.timestamps.values), [np.nan]])

    # Assign licks into bouts
    df_licks["bout_start"] = df_licks["pre_ili"] > bout_threshold
    df_licks["bout_end"] = df_licks["post_ili"] > bout_threshold
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


def annotate_rewards(nwb):
    """
    Annotates df_licks with which lick triggered each reward
    nwb, an nwb-lick object with attributes: df_licks, df_events 
    """

    LICK_TO_REWARD_TOLERANCE = 0.25

    # ensure we have df_licks
    if not hasattr(nwb, "df_licks"):
        nwb.df_licks = annotate_lick_bouts(nwb)

    # ensure we have df_events
    if not hasattr(nwb, "df_events"):
        print("compute df_events")
        return

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
        # TODO, should check for licks that happened before the last go cue
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
