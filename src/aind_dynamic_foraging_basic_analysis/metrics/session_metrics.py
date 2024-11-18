"""

    Consolidated session metric tool
    df_session = session_metrics(nwb)

"""

import numpy as np
import pandas as pd
import re
from aind_dynamic_foraging_basic_analysis import compute_foraging_efficiency

# Copied metrics are from process_nwbs.py in bonsai basic
# nwb.df_trials = nwb_utils.create_trials_df(your_nwb)
# NOTE: finished_rate_with_autowater is the same as the calculated total response rate 
LEFT, RIGHT, IGNORE = 0, 1, 2

def session_metrics(nwb):
    """ 
    Compute all session level metrics

    Includes session level metadata as a temporary organizer 
    Includes majority of metrics from process_nwbs.py

    New addition: chosen_probability - average difference between the chosen probability
    and non-chosen probability / the difference between the largest and smallest probability 
    in the session
    """

    if not hasattr(nwb, 'df_trials'):
        print('You need to compute df_trials: nwb_utils.create_trials_df(nwb)')
        return
    
    # METADATA PLACEHOLDER 
    session_start_time = nwb.session_start_time
    session_date = session_start_time.strftime("%Y-%m-%d")
    subject_id_from_meta = nwb.subject.subject_id
    
    # Parse the file name for suffix
    old_re = re.match(r"(?P<subject_id>\d+)_(?P<date>\d{4}-\d{2}-\d{2})(?:_(?P<n>\d+))?\.json", 
                nwb.session_id)
    
    if old_re is not None:
        subject_id, session_date, nwb_suffix = old_re.groups()
        nwb_suffix = int(nwb_suffix) if nwb_suffix is not None else 0
    else:
        subject_id, session_date, session_json_time = re.match(r"(?P<subject_id>\d+)_(?P<date>\d{4}-\d{2}-\d{2})(?:_(?P<time>.*))\.json", 
                            nwb.session_id).groups()
        nwb_suffix = int(session_json_time.replace('-', ''))
    
    # Verify metadata matches
    assert subject_id == subject_id_from_meta, f"Subject name from the metadata ({subject_id_from_meta}) does not match "\
                                               f"that from json name ({subject_id})!!"
    assert session_date == session_date, f"Session date from the metadata ({session_date}) does not match "\
                                                   f"that from json name ({session_date})!!"
    
    # Create session index
    session_index = pd.MultiIndex.from_tuples([(subject_id, session_date, nwb_suffix)], 
                                            names=['subject_id', 'session_date', 'nwb_suffix'])

    # Get metadata from nwb.scratch
    meta_dict = nwb.scratch['metadata'].to_dataframe().iloc[0].to_dict()
    
    # Calculate performance metrics
    n_total_trials = len(df_trial)
    n_finished_trials = (df_trial.animal_response != IGNORE).sum()
    
    # Actual foraging trials (autowater excluded)
    n_total_trials_non_autowater = df_trial.non_autowater_trial.sum()
    n_finished_trials_non_autowater = df_trial.non_autowater_finished_trial.sum()
    
    n_reward_trials_non_autowater = df_trial.reward_non_autowater.sum()
    reward_rate_non_autowater_finished = n_reward_trials_non_autowater / n_finished_trials_non_autowater if n_finished_trials_non_autowater > 0 else np.nan
    
    # Calculate foraging efficiency
    foraging_eff, foraging_eff_random_seed = compute_foraging_efficiency(
        baited='without bait' not in nwb.protocol.lower(),
        choice_history=df_trial.animal_response.map({0: 0, 1: 1, 2: np.nan}).values,
        reward_history=df_trial.rewarded_historyL | df_trial.rewarded_historyR,
        p_reward=[
            df_trial.reward_probabilityL.values,
            df_trial.reward_probabilityR.values,
        ],
        random_number=[
            df_trial.reward_random_number_left.values,
            df_trial.reward_random_number_right.values,
        ],
        autowater_offered=(df_trial.auto_waterL == 1) | (df_trial.auto_waterR == 1),
    )
    
    # Override foraging_eff_random_seed for old bpod sessions
    if 'bpod' in nwb.session_description:
        foraging_eff_random_seed = nwb.get_scratch('metadata')['foraging_efficiency_with_actual_random_seed'].values[0]
    
    finished_rate = n_finished_trials_non_autowater / n_total_trials_non_autowater if n_total_trials_non_autowater > 0 else np.nan
    
    # New Metrics

    # Probability chosen calculation 
    probability_chosen = []
    probability_not_chosen = []

    for _, row in df_trial.iterrows():
        if row.animal_response == 2:
            probability_chosen.append(np.nan)
            probability_not_chosen.append(np.nan)
        elif row.animal_response == 0:  # Chosen = left choice left probability, not chosen = left choice right probability 
            probability_chosen.append(row.reward_probabilityL)
            probability_not_chosen.append(row.reward_probabilityR)
        else: # Chosen = right choice right probability, not chosen = right choice left probability 
            probability_chosen.append(row.reward_probabilityR)
            probability_not_chosen.append(row.reward_probabilityL)

    df_trial['probability_chosen'] = probability_chosen
    df_trial['probability_not_chosen'] = probability_not_chosen

    # Calculate the chosen probability
    average = (df_trial['probability_chosen'] - df_trial['probability_not_chosen'])
    
    p_larger_global = max(df_trial['probability_chosen'].max(), df_trial['probability_not_chosen'].max())
    
    p_smaller_global = min(df_trial['probability_chosen'].min(), df_trial['probability_not_chosen'].min())
    
    mean_difference = average.mean()
    chosen_probability = mean_difference / (p_larger_global - p_smaller_global)


    # Pack all data
    dict_meta = {
        # Basic metadata
        'rig': meta_dict['box'],        
        'user_name': nwb.experimenter[0],
        'task': nwb.protocol,
        
        # New metric
        'chosen_probability': chosen_probability,

        # Trial counts and rates
        'total_trials': n_total_trials_non_autowater,
        'finished_trials': n_finished_trials_non_autowater,
        'finished_rate': finished_rate,
        'total_trials_with_autowater': n_total_trials,
        'finished_trials_with_autowater': n_finished_trials,
        'finished_rate_with_autowater': n_finished_trials / n_total_trials,
        
        # Reward and foraging metrics
        'reward_trials': n_reward_trials_non_autowater,
        'reward_rate': reward_rate_non_autowater_finished,
        'foraging_eff': foraging_eff,
        'foraging_eff_random_seed': foraging_eff_random_seed,
        'foraging_performance': foraging_eff * finished_rate,
        'foraging_performance_random_seed': foraging_eff_random_seed * finished_rate,
        
        # Timing metrics
        'reaction_time_median': df_trial.loc[:, 'reaction_time'].median(),
        'reaction_time_mean': df_trial.loc[:, 'reaction_time'].mean(),
        'early_lick_rate': (df_trial.loc[:, 'n_lick_all_delay_period'] > 0).sum() / n_total_trials,
        
        # Double dipping metrics
        'double_dipping_rate_finished_trials': 
            (df_trial.loc[(df_trial.animal_response != IGNORE), 'n_lick_switches_gocue_stop'] > 0).sum() 
            / (df_trial.animal_response != IGNORE).sum(),
        'double_dipping_rate_finished_reward_trials':
            (df_trial.loc[df_trial.reward, 'n_lick_switches_gocue_stop'] > 0).sum()  
            / df_trial.reward.sum(),
        'double_dipping_rate_finished_noreward_trials':
            (df_trial.loc[(df_trial.animal_response != IGNORE) & (~df_trial.reward), 'n_lick_switches_gocue_stop'] > 0).sum() 
            / ((df_trial.animal_response != IGNORE) & (~df_trial.reward)).sum(),
            
        # Lick consistency metrics
        'lick_consistency_mean_finished_trials': 
            df_trial.loc[(df_trial.animal_response != IGNORE), 'n_lick_consistency_gocue_stop'].mean(),
        'lick_consistency_mean_finished_reward_trials': 
            df_trial.loc[df_trial.reward, 'n_lick_consistency_gocue_stop'].mean(),
        'lick_consistency_mean_finished_noreward_trials': 
            df_trial.loc[(df_trial.animal_response != IGNORE) & (~df_trial.reward), 'n_lick_consistency_gocue_stop'].mean(),
    }
    
    # Create DataFrame with hierarchical columns
    df_meta = pd.DataFrame(dict_meta, 
                          index=session_index)
    df_meta.columns = pd.MultiIndex.from_product([['metadata'], dict_meta.keys()],
                                                names=['type', 'variable'])
    
    return df_meta