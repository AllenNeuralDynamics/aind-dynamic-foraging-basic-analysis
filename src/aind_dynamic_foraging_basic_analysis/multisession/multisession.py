import os
import pandas as pd

from aind_dynamic_foraging_data_utils import nwb_utils as nu
import aind_dynamic_foraging_basic_analysis.metrics.trial_metrics as tm


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
        nwb = nu.load_nwb_from_filename(n)
        nwb.df_trials = nu.create_df_trials(nwb)
        nwb.df_trials = tm.compute_trial_metrics(nwb)
        nwb.df_trials = tm.compute_bias(nwb)
        nwbs.append(nwb)
    
    df = pd.concat([x.df_trials for x in nwbs])

    filename = os.path.join(AGG_DIR, nwb_list[0].split('/')[-1].split('_')[1]+'.csv')
    df.to_csv(filename)
    return nwbs, df
