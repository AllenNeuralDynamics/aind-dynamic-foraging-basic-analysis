import numpy as np
import pandas as pd

from aind_dynamic_foraging_data_utils import nwb_utils as nu
'''
    TODO
    update plot_session_scroller to take NWB file
        clean up, auto compute
    plot lick bouts in rotating colors
        clean up, let users toggle
    notice that there are rewards that come late with respect to licks
    annotate lick bouts with reward (most recent lick, unless the lick was more than .5 seconds, or a gocue happened before)
    annotate possible bad licks?
    annotate cross licks?
    use lick annotations for rewarded and unrewarded lick bout starts
    maybe annotate rewarded and unrewarded go cues
    Is the PSTH code flexible enough to take different options?   
'''


def annotate_lick_bouts(nwb, bout_threshold=.7):

    if not hasattr(nwb, 'df_events'):
        nwb.df_events = nu.create_events_df(nwb)
    df_licks = nwb.df_events.query('event in ["right_lick_time","left_lick_time"]').copy()        

    # Computing ILI for each lick 
    df_licks['pre_ili'] = np.concatenate([
        [np.nan],np.diff(df_licks.timestamps.values)])
    df_licks['post_ili'] = np.concatenate([
        np.diff(df_licks.timestamps.values),[np.nan]])

    # Assign licks into bouts
    df_licks['bout_start'] = df_licks['pre_ili'] > bout_threshold
    df_licks['bout_end'] = df_licks['post_ili'] > bout_threshold
    df_licks.loc[df_licks['pre_ili'].isnull(),'bout_start']=True
    df_licks.loc[df_licks['post_ili'].isnull(),'bout_end']=True
    df_licks['bout_number'] = np.cumsum(df_licks['bout_start'])

    # Check that bouts start and stop
    num_bout_start = df_licks['bout_start'].sum()
    num_bout_end = df_licks['bout_end'].sum()
    num_bouts = df_licks['bout_number'].max()
    assert num_bout_start==num_bout_end, "Bout Starts and Bout Ends don't align"
    assert num_bout_start == num_bouts, "Number of bouts is incorrect"

    return df_licks

def annotate_rewards(nwb):
    '''
       TODO, add in 
    '''
    
    if not hasattr(nwb, 'df_licks'):
        nwb.df_licks = annotate_lick_bouts(nwb)
        
    if not hasattr(nwb, 'df_events'):
        print('compute df_events')
        return

    right_rewards = nwb.df_events.query('event == "right_reward_delivery_time"').copy()
    left_rewards = nwb.df_events.query('event == "left_reward_delivery_time"').copy()
    
    #for dex, row in right_rewards.iterrows():
    #    t = row.timestamps:
    #     
    #nwb.df_licks['event'] == 'left_lick_time'








