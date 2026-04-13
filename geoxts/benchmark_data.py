

__all__ = ['Well_grlog_cleaning', 'marker_signature', 'extract_dataset_Xy', 'extract_validation_Xy', 'window', 'get_markers', 'recall_tops', 'Predicted_well_depth', 'plot_result_distribution', 'get_UCR_data' ]


# Importing some basic packages

import statistics
import pandas as pd
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from tsai.all import get_predefined_splits
from fastcore.all import coll_repr



#cleaned tabel per well
def Well_grlog_cleaning(well_name: str
                        , df_log : pd.DataFrame
                        , input_variable: list
                        ):

    """
    The function gives a cleaned dataframe for a particular well name given in the input.
    Function input:
    - well_name: Name of the well(must be a string) needs convertions if well_name is float or int
    - df_log: The well log dataframe
    - input_variable: A list of column label for input variables required for the deep learning model.

    Output: It is a pandas dataframe with same time stamp, no Nan and consisiting only the input feature bvariables for the deep learing model.
    """

    # OLD: .where(condition).dropna() drops ALL rows where ANY column is NaN,
    # killing most data even when log NaN has been interpolated (bfill/ffill).
    # df_wgr = df_log.where(df_log['wellName'] == well_name ).dropna().copy(deep = True)
    # #df_wgr = df_log[df_log.index == well_name].dropna()
    # df_wgr['GR'].replace(-1, df_wgr['GR'].mean(), inplace = True)
    # df_wgr['DEPTH'] = df_wgr['DEPTH'].astype('int')
    # df_wgr = df_wgr.groupby('DEPTH').mean()
    # df_wgr['Depth'] = df_wgr.index
    # df_wgr = df_wgr[input_variable]

    # NEW: filter by well name, group by integer depth, keep all rows.
    df_wgr = df_log[df_log['wellName'] == well_name].copy()
    df_wgr['DEPTH'] = df_wgr['DEPTH'].astype('int')
    df_wgr = df_wgr.groupby('DEPTH').mean(numeric_only=True)
    df_wgr['Depth'] = df_wgr.index
    df_wgr = df_wgr[input_variable]

    return df_wgr



#function to take return marker signature
def marker_signature( df_wgr : pd.DataFrame
                     , marker : float
                     , wsize : int
                     ):

    """
    Function input: 
    - df_wgr: A smaller snipped of well logs, for a particular well (Output of the Well_grlog_cleaning function)
    - marker: A float value which in the stated marker depth from df_tops dataframe 
    - wsize: window size for the marker signature

    Output: A np.array consisting of marker signature for a particular well

    """

    hd = int(wsize*0.5)
    std = 2
    #marker depth 
    md = marker
    dseq = list(range(int(md - (hd+std)),int(md + (hd+std)))) #depth the sequence
    seq = df_wgr[df_wgr.index.isin(dseq)].to_numpy() #create the series with gamma rays
    #print(len(seq))
    mseg = []
    for i in range(len(seq) - wsize + 1):
        #np.append(mseg, np.transpose(seq[i: i + wsize]), axis = 0) 
        mseg.append(np.transpose(seq[i: i + wsize]))
        
    np_mseg = np.array(mseg)
        
    return np_mseg



def extract_dataset_Xy(logs : pd.DataFrame
                      ,tops : pd.DataFrame
                      ,well_list : list
                      ,input_variable : list
                      ,wsize : int
                      ,top_list_bool : int
                      ):

    """
    Function Input: 
    - logs: A pd.dataframe containing well logs (unprocessed)
    - tops: A pd.dataframe containing well top depths 
    - well_list: A list of well names used for training of the model
    - input_variable: A list of column label for input variables required for the deep learning model.
    - wsize : window size for the marker signature
    Output: Two numpy array, X and y. X is the input to model and y is the output.
    
    """

    
    if len(well_list) == 0:
            well_list = list(tops.index)
            well_list_bool = 0
            #print(well_list)

    # OLD: used np.append in a loop (slow), crashed on max(wtop[0]) when markers are NaN,
    # generated only 1 "None" sample per well, and "None" sample could overlap real marker depths.
    # X = np.empty(shape = (0,len(input_variable),wsize))
    # y = np.empty(shape = (0))
    # for well in tops.index:
    #     df_wgr = Well_grlog_cleaning(well, logs, input_variable)
    #     for i in range(0,len(tops.columns)):
    #         top = tops.columns[i]
    #         if top_list_bool:
    #             top_well_list = well_list[i][0]
    #         else:
    #             top_well_list = well_list
    #         if well in top_well_list:
    #             marker = tops.loc[well][i]
    #             np_mseg = marker_signature(df_wgr, marker, wsize)
    #             y_seq = np.full(len(np_mseg), i+1)
    #             if X.ndim == np_mseg.ndim:
    #                 X = np.append(X, np_mseg, axis = 0)
    #                 y = np.append(y, y_seq, axis = 0)
    #     wtop = tops[tops.index == well].values
    #     rd = np.random.randint(low = 1000, high = max(wtop[0]))
    #     while (rd in wtop):
    #         rd = np.random.randint(low = 1000, high = max(wtop[0]))
    #     np_mseg = marker_signature( df_wgr, rd, wsize)
    #     y_seq = np.full(len(np_mseg), 0)
    #     if X.ndim == np_mseg.ndim:
    #         X = np.append(X, np_mseg, axis = 0)
    #         y = np.append(y, y_seq, axis = 0)
    # return X, y

    # NEW: skips NaN markers, generates n_none "None" samples per well,
    # ensures "None" samples don't overlap with real marker depths.
    n_none = 5
    half_w = wsize // 2
    X_list, y_list = [], []

    for well in tops.index:
        df_wgr = Well_grlog_cleaning(well, logs, input_variable)

        #pass through the different markers to get their signature
        for i in range(len(tops.columns)):
            top = tops.columns[i]
            if top_list_bool:
                top_well_list = well_list[i][0]
            else:
                top_well_list = well_list

            if well in top_well_list:
                marker = tops.loc[well][i]
                if pd.isna(marker):
                    continue  # skip NaN markers — well still contributes other classes
                np_mseg = marker_signature(df_wgr, marker, wsize)
                y_seq = np.full(len(np_mseg), i + 1)
                if np_mseg.ndim == 3:
                    X_list.append(np_mseg)
                    y_list.append(y_seq)

        # Generate "None" samples using only valid (non-NaN) marker depths
        wtop = tops[tops.index == well].values
        valid_tops = wtop[0][~np.isnan(wtop[0].astype(float))]
        if len(valid_tops) == 0:
            continue
        high_val = int(np.max(valid_tops))
        if high_val <= 1000 + wsize:
            continue
        for _ in range(n_none):
            rd = np.random.randint(low=1000, high=high_val)
            attempts = 0
            while any(abs(rd - t) < half_w for t in valid_tops) and attempts < 50:
                rd = np.random.randint(low=1000, high=high_val)
                attempts += 1
            np_mseg = marker_signature(df_wgr, rd, wsize)
            y_seq = np.full(len(np_mseg), 0)
            if np_mseg.ndim == 3:
                X_list.append(np_mseg)
                y_list.append(y_seq)

    X = np.concatenate(X_list, axis=0) if X_list else np.empty(shape=(0, len(input_variable), wsize))
    y = np.concatenate(y_list, axis=0) if y_list else np.empty(shape=(0,))
    return X, y

def extract_validation_Xy(logs,tops,input_variable, wsize):

    
    well_list = list(tops.index)
    
    wsize = 201
    X = np.empty(shape = (0,len(input_variable),wsize))
    y = np.empty(shape = (0))

    for well in tops.index:
        
        #print(well)
        df_wgr = Well_grlog_cleaning(well, logs, input_variable)
        
        #pass trough the different markers to get their signature
        for i in range(0,len(tops.columns)):
            
            top = tops.columns[i]
            top_well_list = well_list
    
            #assign the well list for the marker and create good signatures for the marker
            if well in top_well_list:
    
                md = tops.loc[well][i]
                hd = int(wsize*0.5)
                dseq = list(range(int(md - (hd)),int(md + (hd+1)))) #depth the sequence
                np_mseg = np.expand_dims(df_wgr[df_wgr.index.isin(dseq)].to_numpy().T, axis=0)#create the series with gamma rays
                y_seq = [i+1]
                if np_mseg.shape[2] == wsize:
                    y = np.append(y, y_seq, axis = 0)
                    X = np.append(X, np_mseg, axis = 0)

        #for the non marker labels
        wtop = tops[tops.index == well].values
        rd = np.random.randint(low = 1000, high = max(wtop[0]))
        while (rd in wtop):
            rd = np.random.randint(low = 1000, high = max(wtop[0]))
    
        dseq = list(range(int(rd - (hd)),int(rd + (hd+1)))) #depth the sequence
        np_mseg = np.expand_dims(df_wgr[df_wgr.index.isin(dseq)].to_numpy().T, axis=0)#create the series with gamma rays
        #print(len(seq))
        #print(np_mseg.shape)
        y_seq = [0]
        if np_mseg.shape[2] == wsize:
                y = np.append(y, y_seq, axis = 0)
                X = np.append(X, np_mseg, axis = 0)
        
    return X, y


def window(df_wgr: pd.DataFrame, wsize: int):

    """
    Input: 
    - df_wgr: A smaller snipped of well logs, for a particular well (Output of the Well_grlog_cleaning function)
    - wsize: window size for the marker signature

    Output: An np.array of the window size from the well log for a single well. It is used during testing the sliding window approach

    """
    
    #std = 5
    #dseq = list(range(int(md - 30),int(md + 30)))
    
    seq = df_wgr.to_numpy() #create the series with gamma rays
    mseg = []
    dep_list = []
    for i in range(len(seq) - wsize + 1):
        dep = df_wgr.index[i]+int(wsize/2) #depth the sequence
        mseg.append(np.transpose(seq[i: i + wsize]))
        dep_list.append(dep)
        
    return np.array(mseg), dep_list



def get_markers(df_test_log : pd.DataFrame
                , learn #model
                , dls #tensor format 
                , well : str
                , pred_column: list
                , wsize: int
                , valid_dl #tensor
                , input_variable : list
                #, existing_markers : list = None  # OLD: parameter did not exist
                , existing_markers : list = None
                ):
    
    """
    Input: 
    df_test_log: The logs of for the testing set
    learn: The trained deep learning model
    well: well name (in str)
    pred_column: A list of column names for the predicted dataframe (usually [well_name, depth, none, marker names..])
    wsize: window size (same as traing)
    valid_dl: The validation tensor (this is to make sure the testing data is in the correct format for the processing of deep learing model)
    input_variable: a list of column names, to be used are input feature. (same as when creating the traing dataset)
    existing_markers: list of marker names that exist for this well. If None, all markers are predicted (backward compatible).

    Output:
    A list of predicted marker depths, in the sequence mentioned in the pred_column.
    A pd.dataframe with probability distribution for the markers
    
    """

    df_gr = Well_grlog_cleaning(well, df_test_log, input_variable)
    well_seq, dep_seq = window(df_gr, wsize)

    # OLD: no guard for empty wells — crashed inside dls.dataset.add_test()
    # NEW: early return with NaN predictions for wells with insufficient log coverage
    n_markers = len(pred_column) - 1  # exclude 'None'
    if len(well_seq) == 0:
        return [np.nan] * n_markers, pd.DataFrame(columns=pred_column + ['Depth'])

    test_ds_well = dls.dataset.add_test(well_seq)
    # OLD: well_seq kept alive until function return (~30-60 MB per well)
    del well_seq  # free large numpy array after add_test consumes it
    test_dl_well = valid_dl.new(test_ds_well)

    # OLD: learn.get_preds() accumulates ALL batch outputs on GPU before returning,
    # causing OOM on cloud VMs for large wells (11k+ windows).
    # test_probas, *_ = learn.get_preds(dl=test_dl_well, save_preds=None)
    # test_prob = test_probas.detach().cpu().numpy()

    # NEW: manual batch loop — move each batch result to CPU immediately.
    # Same speed as get_preds (identical forward passes), just doesn't hoard GPU memory.
    import torch as _torch
    _chunks = []
    learn.model.eval()
    with _torch.no_grad():
        for batch in test_dl_well:
            xb = batch[0] if isinstance(batch, (list, tuple)) else batch
            out = learn.model(xb)
            _chunks.append(out.detach().cpu().numpy())
            del out, xb
    test_prob = np.concatenate(_chunks, axis=0)
    del _chunks, test_ds_well, test_dl_well
    df_wm = pd.DataFrame(test_prob, columns=pred_column)
    # OLD: test_prob kept alive alongside df_wm (duplicate memory)
    del test_prob  # numpy array consumed by DataFrame
    df_wm['Depth'] = dep_seq

    # OLD: merged on the 'Depth' column — breaks when the depth-normalization patch
    # is active because df_gr['Depth'] is scaled to [0,1] while dep_seq stays in
    # original integer feet, producing zero matches and skipping every well.
    # if 'Depth' not in df_gr.columns:
    #     df_gr['Depth'] = df_gr.index
    # df_wm = df_gr.merge(df_wm, how = 'inner', left_on = 'Depth', right_on = 'Depth')

    # NEW: always reset df_gr['Depth'] to the integer index before merging, so the
    # join and downstream marker-depth lookups use real feet regardless of normalization.
    df_gr['Depth'] = df_gr.index
    df_wm = df_gr.merge(df_wm, how='inner', left_on='Depth', right_on='Depth')
    # OLD: df_gr kept alive until function return (duplicate of data now in df_wm)
    del df_gr  # merged into df_wm, no longer needed
    #print(df_wm.head())
    pred_m = []
    df_wmn = df_wm
    for top in pred_column:

        if top != 'None':

            # OLD: always predicted every marker, even non-existent ones
            # md = df_wmn[df_wmn[top] == df_wmn[top].max()].Depth
            # idx = md.index[0]
            # ym = statistics.median(md)
            # df_wmn = df_wm.iloc[idx: , :]
            # pred_m.append(ym)

            # NEW: skip markers that don't exist for this well
            if existing_markers is not None and top not in existing_markers:
                pred_m.append(np.nan)
                continue

            md = df_wmn[df_wmn[top] == df_wmn[top].max()].Depth
            idx = md.index[0]
            ym = statistics.median(md)
            df_wmn = df_wm.iloc[idx: , :]
            pred_m.append(ym)

    return pred_m, df_wm


def Predicted_well_depth(df_test_log : pd.DataFrame, df_test_tops: pd.DataFrame
                , dls #tensordefinition  
                , learn #Module
                , pred_column: list
                , wsize: int
                , valid_dl  #Tensor
                , input_variable : list):

    """
    Input: 
    df_test_log: The logs of for the testing set
    df_test_log: The formation top(marker) depths for the testing set
    learn: The trained deep learning model
    pred_column: A list of column names for the predicted dataframe (usually [well_name, depth, none, marker names..])
    wsize: window size (same as traing)
    valid_dl: The validation tensor (this is to make sure the testing data is in the correct format for the processing of deep learing model)
    input_variable: a list of column names, to be used are input feature. (same as when creating the traing dataset)
    
    """
    well_list = list(df_test_tops.index)
    col = df_test_tops.reset_index().columns.tolist()
    df_tops_pred = pd.DataFrame(pd.DataFrame(columns = col)) #try to generalise 
    
    for well in well_list:

        # OLD: did not pass existing_markers, ghost predictions corrupted monotonic search
        # pred_m, df_wm = get_markers(df_test_log, learn, dls, well, pred_column, wsize, valid_dl, input_variable)

        # NEW: determine which markers exist for this well and pass to get_markers
        well_tops = df_test_tops.loc[well]
        existing_markers = well_tops.dropna().index.tolist()
        pred_m, df_wm = get_markers(df_test_log, learn, dls, well, pred_column, wsize, valid_dl, input_variable, existing_markers)

        #print(pred_m)
        # OLD: hardcoded 3 markers
        # row = {col[0]:well, col[1]:pred_m[0], col[2]:pred_m[1], col[3]:pred_m[2]}

        # NEW: generalized to N markers
        marker_cols = col[1:]  # all columns except wellName
        row = {col[0]: well}
        for j, mcol in enumerate(marker_cols):
            row[mcol] = pred_m[j] if j < len(pred_m) else np.nan
        row_df = pd.DataFrame([row])
        df_tops_pred = pd.concat([df_tops_pred, row_df], axis = 0, ignore_index = "True")
    
    df_tops_pred['wellName']  = df_tops_pred['wellName'].astype(float)
    df_tops_pred = df_tops_pred.sort_values(by = ['wellName']).reset_index().drop(['index'], axis = 1)
    df_tops_pred = df_tops_pred.set_index('wellName')

    return df_tops_pred



def recall_tops(df_tops_true: pd.DataFrame, df_tops_pred: pd.DataFrame, tolerance: int):

    """
    Input:
    - df_tops_true: The tops dataframe for testing
    - df_tops_pred: The predicted tops depth dataframe
    - tolerence: an integer value for the tolerence in feet to be allowed between the predicted and given marker depths, to be considered right prediction.

    """

    # OLD: no alignment — missing wells in df_tops_pred inflated denominator, biasing recall and MAE low
    # if set(df_tops_true.columns) == set(df_tops_pred.columns) :
    #     concat_df = df_tops_true.copy()
    #     for col in df_tops_pred.columns:
    #         concat_df[col+"_pred"] = df_tops_pred[col]
    #     tp = 0
    #     p = 0
    #     mae = 0
    #     for col in df_tops_true:
    #         diffname = "{0}_ae".format(col)
    #         tpname = "{0}_tp".format(col)
    #         p += concat_df[col].count()
    #         concat_df[diffname] = concat_df[col]-concat_df[str(col + "_pred")]
    #         concat_df[diffname] = concat_df[diffname].abs()
    #         concat_df[tpname] = concat_df[diffname] <= tolerance
    #         tp += concat_df[tpname].sum()
    #         mae += concat_df[diffname].sum()
    #     return tp/p, mae/p, concat_df
    # else :
    #     print("the tops columns are not valid")
    #     return None,None,None

    # NEW: align on common wells only (inner join on index) to avoid denominator inflation from skipped wells
    if set(df_tops_true.columns) != set(df_tops_pred.columns):
        print("the tops columns are not valid")
        return None, None, None

    common_wells = df_tops_true.index.intersection(df_tops_pred.index)
    if len(common_wells) < len(df_tops_true.index):
        import warnings
        n_skipped = len(df_tops_true.index) - len(common_wells)
        warnings.warn(f"recall_tops: {n_skipped} wells in true tops not found in predictions; evaluating on {len(common_wells)} common wells only.")

    df_true_aligned = df_tops_true.loc[common_wells]
    df_pred_aligned = df_tops_pred.loc[common_wells]

    concat_df = df_true_aligned.copy()
    for col in df_pred_aligned.columns:
        concat_df[col + "_pred"] = df_pred_aligned[col]

    tp = 0
    p = 0
    mae = 0
    for col in df_true_aligned.columns:
        diffname = "{0}_ae".format(col)
        tpname = "{0}_tp".format(col)
        p += concat_df[col].count()
        concat_df[diffname] = (concat_df[col] - concat_df[col + "_pred"]).abs()
        concat_df[tpname] = concat_df[diffname] <= tolerance
        tp += concat_df[tpname].sum()
        mae += concat_df[diffname].sum()

    if p == 0:
        return 0.0, np.nan, concat_df

    return tp / p, mae / p, concat_df



def plot_result_distribution(true_depth: list 
                             ,start_depth: float
                             ,pred_m: list
                             ,top_list: list #string list
                             , df_wm : pd.DataFrame
                             , Industrial_baseline: bool
                             ):
    

    """"
    Plots the gamma ray with the marker depth, the predicted marker depths and its propability distribution curve.
    Input: 
    - true_depth: A list of given marker depths in a well, for which we make a prediction (from df_tops)
    - start_depth: the starting depth for the graph.
    - pred_m : a list of the predicted marker depths (model output)
    - top_list: a list of marker names
    - df_wm : a pandas dataframe containg the model output for a well
    - Industrial_baseline: A booliean to be True if using Industrial baseline and false if using Deep learning model.
    """


    td = true_depth

    if Industrial_baseline == False:

        fig, ax1 = plt.subplots(figsize = (10,5))
        
        ax1.set_xlabel('depth (ft)')
        ax1.set_ylabel('Probability')
        #plt.plot(df_gr[5500:]['GR'], color = 'black')
        ax1.fill_between(df_wm[start_depth:]['Depth'], df_wm[start_depth:][top_list[0]]*100,  color = 'lightcoral', alpha = 0.5, label= 'prob_M')
        ax1.fill_between(df_wm[start_depth:]['Depth'], df_wm[start_depth:][top_list[1]]*100, color = 'lightblue', alpha = 0.5, label= 'prob_S')
        ax1.fill_between(df_wm[start_depth:]['Depth'], df_wm[start_depth:][top_list[2]]*100, color = 'lightgreen', alpha = 0.5, label= 'prob_C')
        #ax2.legend()

        ax2 = ax1.twinx()
        ax2.set_ylabel('GR')
        ax2.plot(df_wm[start_depth:]['Depth'], df_wm[start_depth:]['GR'], color = 'black')
        ax2.axvline(td[0], color='red', label = top_list[0])
        ax2.axvline(td[1], color='blue', label = top_list[1])
        ax2.axvline(td[2], color='green', label = top_list[2])
        
        ax2.axvline(pred_m[0], color='red', linestyle = '--')
        ax2.axvline(pred_m[1], color='blue',linestyle = '--')
        ax2.axvline(pred_m[2], color='green', linestyle = '--')
        ax2.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=6)

    else :

        fig, ax2 = plt.subplots(figsize = (10,5))
    
        ax2.set_ylabel('GR')
        ax2.plot( df_wm[start_depth:], color = 'black')
        ax2.axvline(td[0], color='red', label = 'actual_M')
        ax2.axvline(td[1], color='blue', label = 'actual_S')
        ax2.axvline(td[2], color='green', label = 'actual_C')
        
        ax2.axvline(pred_m[0], color='red', linestyle = '--', label= 'pred_M')
        ax2.axvline(pred_m[1], color='blue',linestyle = '--', label= 'pred_S')
        ax2.axvline(pred_m[2], color='green', linestyle = '--', label= 'pred_C')
        ax2.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=6)

    plt.show()


UCR_list = [ 'ArticularyWordRecognition', 'PenDigits',
'Heartbeat', 'JapaneseVowels', 'Libras', 'Phoneme', 'SpokenArabicDigits',
'AtrialFibrillation', 'StandWalkJump' ,
 'FaceDetection', 'FingerMovements', 'HandMovementDirection',  'SelfRegulationSCP1', 'SelfRegulationSCP2',
'BasicMotions', 'Cricket', 'Epilepsy', 'Ering', 'Handwriting', 'NATOPS', 'RacketSports', 'UWaveGestureLibrary',
'EthanolConcentration', 'LSST', 'PEMS-SF']


def get_UCR_data(dsid, parent_dir='data/UCR', on_disk=True, mode='c', Xdtype='float32', ydtype=None, return_split=True, split_data=True, 
                 verbose=False):
    
    dsid_list = [ds for ds in UCR_list if ds.lower() == dsid.lower()]
    assert len(dsid_list) > 0, f'{dsid} is not a UCR available dataset'
    dsid = dsid_list[0]
    return_split = return_split and split_data

    #pv(f'Dataset: {dsid}', verbose)
    full_parent_dir = Path(parent_dir)
    full_tgt_dir = full_parent_dir/dsid

    #if not os.path.exists(full_tgt_dir): os.makedirs(full_tgt_dir)
    #full_tgt_dir.parent.mkdir(parents=True, exist_ok=True)

    mmap_mode = mode if on_disk else None
    X_train = np.load(f'{full_tgt_dir}/X_train.npy', mmap_mode=mmap_mode)
    y_train = np.load(f'{full_tgt_dir}/y_train.npy', mmap_mode=mmap_mode)
    X_valid = np.load(f'{full_tgt_dir}/X_valid.npy', mmap_mode=mmap_mode)
    y_valid = np.load(f'{full_tgt_dir}/y_valid.npy', mmap_mode=mmap_mode)

    if return_split:
        if Xdtype is not None: 
            X_train = X_train.astype(Xdtype)
            X_valid = X_valid.astype(Xdtype)
        if ydtype is not None: 
            y_train = y_train.astype(ydtype)
            y_valid = y_valid.astype(ydtype)
        if verbose:
            print('X_train:', X_train.shape)
            print('y_train:', y_train.shape)
            print('X_valid:', X_valid.shape)
            print('y_valid:', y_valid.shape, '\n')
        return X_train, y_train, X_valid, y_valid
    else:
        X = np.load(f'{full_tgt_dir}/X.npy', mmap_mode=mmap_mode)
        y = np.load(f'{full_tgt_dir}/y.npy', mmap_mode=mmap_mode)
        splits = get_predefined_splits(X_train, X_valid)
        if Xdtype is not None: 
            X = X.astype(Xdtype)
        if verbose:
            print('X      :', X .shape)
            print('y      :', y .shape)
            print('splits :', coll_repr(splits[0]), coll_repr(splits[1]), '\n')

    return X, y, splits