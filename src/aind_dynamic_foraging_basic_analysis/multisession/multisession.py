import os
import pandas as pd
import matplotlib.pyplot as plt

from aind_dynamic_foraging_data_utils import nwb_utils as nu
import aind_dynamic_foraging_basic_analysis.metrics.trial_metrics as tm

'''

# TODO
Make specification for multisession trials df
Make some capsule to generate them, and make data assets
Add axis for licking/rewards
make bias/lickspout plots dynamically loaded
'''


def make_multisession_trials_df(nwb_list, DATA_DIR, AGG_DIR):
    '''
        takes a list of NWBs
        loads each NWB file
        makes trials table
        adds metrics
        adds bias
        makes aggregate trials table
        saves aggregate trials table
    '''
    nwbs = []
    for n in nwb_list:
        try:
            nwb = nu.load_nwb_from_filename(n)
            nwb.df_trials = nu.create_df_trials(nwb)
            nwb.df_trials = tm.compute_trial_metrics(nwb)
            nwb.df_trials = tm.compute_bias(nwb)
            nwbs.append(nwb)
        except:
            print('Bad {}'.format(n))
    
    df = pd.concat([x.df_trials for x in nwbs])

    filename = os.path.join(AGG_DIR, nwb_list[0].split('/')[-1].split('_')[1]+'.csv')
    df.to_csv(filename)
    return nwbs, df


def plot_foraging_lifetime(lifetime_df):
    '''
        Takes a dataframe of the aggregate for all sessions from this animal
        
    '''
    # Set up figure
    # determine order of sessions
    # plot each

    df = lifetime_df.copy()
    df = df.sort_values(by=['ses_idx','trial'])
    df['lifetime_trial'] = df.reset_index().index 
    session_breaks = df.query('trial == 0')['lifetime_trial'].values[1:] - .5

    fig, ax = plt.subplots(2,1,figsize=(12,4),sharex=True)
    ax[0].plot(df['lifetime_trial'],df['bias'],label='bias')   
    ax[0].axhline(0, linestyle='--',color='k',alpha=.25)
    ax[0].set_ylabel('bias')
    ax[0].set_ylim(-1,1)
    ax[0].vlines(session_breaks, -1,1,color='gray',alpha=.25,linestyle='--')
    ax[0].set_xlim(df['lifetime_trial'].values[0], df['lifetime_trial'].values[-1])
    ax[0].legend()
    ax[0].spines["top"].set_visible(False)
    ax[0].spines["right"].set_visible(False)

    ax[1].plot(df['lifetime_trial'],df['lickspout_position_z']-df['lickspout_position_z'].values[0],'k',label='z')
    ax[1].plot(df['lifetime_trial'],df['lickspout_position_y']-df['lickspout_position_y'].values[0],'r',label='y')
    ax[1].plot(df['lifetime_trial'],df['lickspout_position_x']-df['lickspout_position_x'].values[0],'b',label='x')
    ax[1].set_ylabel('$\Delta$ lickspout')
    ax[1].legend()
    ax[1].set_xlabel('Trial #')
    ax[1].spines["top"].set_visible(False)
    ax[1].spines["right"].set_visible(False)
    for x in session_breaks:
        ax[1].axvline(x,color='gray',alpha=.25,linestyle='--')

    #labels = [int(x.get_text()) for x in ax[1].get_xticklabels()]
    #labels = [x for x in labels if x < len(df)]
    #new_labels = [df.set_index('lifetime_trial').loc[x]['trial'] for x in labels]
    #ax[1].set_xticklabels(new_labels)
    plt.tight_layout()


