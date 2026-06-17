"""
Tools for computing per session metrics
compute_auroc: compute auroc for one NWB given alignments
compute_auroc_multi: compute auroc for multiple NWB given alignments

"""

from sklearn.metrics import roc_auc_score
from aind_dynamic_foraging_basic_analysis.plot import plot_fip as pf
import warnings
import pandas as pd
import numpy as np


def compute_auroc(nwb, alignment_times, labels, channel, tw, bin_size=0.25, data_col="data_z"):
    """
    Compute the time-resolved area under the ROC curve (auROC) for a single NWB session.

    Parameters
    - nwb: object
        NWB session object expected to contain a DataFrame `df_fip` with
        FIP data and a `session_id`.
    - alignment_times: array-like, shape (n_trials,)
        Times to align trials to (seconds), given in session time
    - labels: array-like, shape (n_trials,)
        Binary labels (0/1) for each alignment time. Must have same
        length as alignment_times.
    - channel: str
        Channel name to select from `nwb.df_fip.event`.
    - tw: tuple (start, end)
        Time window (seconds) around the alignment to compute auROC over
        (centered bins will be between tw[0] and tw[1]).
    - bin_size: float, optional
        Width (seconds) of each time bin used to aggregate values
        before computing auROC. Default 0.25s.
    - data_col: str, optional
        Column name in the FIP data to use for values (default is z-scored data, 'data_z').

    Returns
    - pandas.DataFrame
        DataFrame with columns:
          - 'bin_center': center time of each bin (seconds)
          - 'auc': auROC value for that bin (NaN when computation failed)
        If the requested channel is not present in the NWB,
        returns an empty DataFrame with those columns.

    Notes
    - alignment_times and labels are sorted together before computing PSTHs.
    - Trials with NaNs in the aggregated bin are dropped;
       event_numbers that contain any NaNs across bins are removed.
    """
    if len(labels) != len(alignment_times):
        raise Exception("Alignment times must have same number of labels ")

    if np.unique(labels).size > 2:
        raise Exception("Labels must be binary for auROC computation")

    if channel not in nwb.df_fip.event.unique():
        warnings.warn("No channel found in this NWB, returning empty DataFrame")
        return pd.DataFrame(columns=["bin_center", "auc"])

    # sort labels and alignment times
    sorted_indices = np.argsort(alignment_times)
    alignment_times = alignment_times[sorted_indices]
    labels = labels[sorted_indices]

    tw_for_center_bin = [tw[0] - bin_size / 2, tw[1] + bin_size / 2]

    # get alignments
    aligns = pf.fip_psth_inner_compute(
        nwb, alignment_times, channel, average=False, tw=tw_for_center_bin, data_column=data_col
    )
    n_centers = int(round((tw[1] - tw[0]) / bin_size)) + 1

    # bin the time values into discrete bins and compute bin centers
    left0 = tw_for_center_bin[0]
    edges = left0 + np.arange(n_centers + 1) * bin_size
    aligns["time_bin"] = pd.cut(aligns["time"], bins=edges, right=False, include_lowest=True)
    aligns["bin_center"] = aligns["time_bin"].apply(
        lambda iv: (iv.left + float(bin_size) / 2.0) if pd.notnull(iv) else np.nan
    )

    aligns = aligns.dropna(subset=["bin_center", data_col]).copy()

    # average by bin_centers
    agg_align = (
        aligns.groupby(["bin_center", "event_number"], observed=True)[data_col]
        .mean()
        .unstack(["event_number"])
    )
    # drop any event_number with nan values for any bin_centers.
    agg_align = agg_align.dropna(how="any", axis=1)

    # calculate auROC
    aucs = []
    labels_valid = labels[agg_align.columns.values]
    for bin_center, row in agg_align.iterrows():
        try:
            auc_val = roc_auc_score(labels_valid, row.values)
        except Exception:
            auc_val = np.nan
        aucs.append(auc_val)

    curr_auc_df = pd.DataFrame(
        {"bin_center": agg_align.index.values, "auc": np.asarray(aucs, dtype=float)}
    )

    return curr_auc_df


def compute_auroc_multi(nwb_list, alignment_times_list, label_list, channel, tw, bin_size=0.25):
    """
    Compute auROC across multiple NWB sessions and return a session x time-bin table.

    Parameters
    - nwb_list: sequence of NWB objects
        Each element should provide FIP data and a `session_id`.
    - alignment_times_list: sequence of array-like
        Per-session alignment times; must be same length as nwb_list.
    - label_list: sequence of array-like
        Per-session labels corresponding to alignment times; must be same length as nwb_list.
    - channel: str
        Channel name to use in each NWB.
    - tw: tuple (start, end)
        Time window (seconds) around alignments to compute auROC over.
    - bin_size: float, optional
        Time bin width for aggregation (default 0.25s).

    Returns
    - pandas.DataFrame
        Concatenated DataFrame where each row is a session (index = session_id)
        and each column is a bin_center; cell values are the auROC for that session
        and bin. If no sessions produced results, an empty DataFrame is returned.
    """

    if len(nwb_list) != len(alignment_times_list) or len(nwb_list) != len(label_list):
        raise ValueError("nwb_list, alignment_times_list, label_list must have the same length")

    # across sessions, should alway use z-scored data to compare
    data_col = "data_z"

    auc_df_list = []
    for nwb, align_times, labels in zip(nwb_list, alignment_times_list, label_list):
        auc_df = compute_auroc(nwb, align_times, labels, channel, tw, bin_size, data_col)
        if auc_df.empty:
            continue
        auc_df["session_id"] = nwb.session_id
        # pivot to single-row DataFrame: index=session_id, columns=bin_center, values=auc
        row = auc_df.pivot(index="session_id", columns="bin_center", values="auc")
        auc_df_list.append(row)

    if len(auc_df_list) == 0:
        return pd.DataFrame()

    # Concatenate all DataFrames in the list
    return pd.concat(auc_df_list, axis=0)
