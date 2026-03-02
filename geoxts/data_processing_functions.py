__all__ = ['plot_and_extract_pattern_local','compute_distance_matrix_dtw', 'cluster_dtw_distance_hdbscan', 'training_well', 'MDS_dtw','clustering_scatterplot' ]

# Importing some basic packages
import pandas as pd
import numpy as np

#importing matplotlib for displaying of plots
import matplotlib.pyplot as plt

#importing the dtw and ed for claculating the DTW distance matrix
from dtaidistance import dtw,ed

#Importing clustering algorithms for sklearn
from sklearn.cluster import HDBSCAN, OPTICS
from scipy import signal

from sklearn.manifold import MDS

def plot_and_extract_pattern_local(df_logs: pd.DataFrame # A dataframe with the logs (well name, depth, GR) as columns
                             , df_tops : pd.DataFrame # A dataframe with wellname as index and the formation (top) as unique column
                             , top_list : list # A list of tops to plot (one top or more)
                             , wsize : int # The window size around each top
                             
                             )  -> tuple [list # A list of labels of the Gamma Ray values
                                         , np.ndarray # A numpy array with of Gamma Ray values
                                         ] :  
                             
    
    
    """
    
    - Plot the Gamma Ray values for the specified wells and tops with a window size (wsize) around each top.
    It also extracts the features (Gamma Ray values) and labels (tops) from these plots, pads the features to ensure
    equal length.
    
    The function takes:
    - df_logs: A dataframe with the logs (well name, depth, GR) as columns
    - df_tops : A dataframe with wellname as index and the formation (top) as unique column
    - top_list : list # A list of tops to plot (one top or more)
    - wsize : int # The window size around each top
    
    The functions returns:
    
    - features_np :  A numpy array with of Gamma Ray values 
    - labels_np : A list of labels of the Gamma Ray values
    - A plot of the logs with the extract marker pattern and the name of the marker.
    
    """
    df_logs_copy = df_logs.copy(deep = True)
    df_tops_copy = df_tops.copy(deep = True)

    wellist = df_tops_copy.index.tolist()
    # top_list=top_name
    prop_cycle = plt.rcParams['axes.prop_cycle']
    colors = prop_cycle.by_key()['color']
    
    # Initialize lists to store the features and labels
    list_features = []
    list_labels = []
    list_well = []
    n_plots = len(top_list)
    fig, ax = plt.subplots(n_plots, 1, figsize=(10, 5 * n_plots))
    if n_plots == 1:
        ax = [ax]  # Convert ax to a list with a single element

    colors = colors[:len(top_list)]
    
    # Loop over tops and wells to plot the logs and extract the features
    for top_id, top in enumerate(top_list):
        y_min, y_max = None, -np.inf  
        # Loop over wells
        for well_id, wellname in enumerate(wellist):  
            df_temp = df_logs_copy[df_logs_copy["wellName"] == wellname].copy(deep = True)
            df_temp['GR'].replace(-1, df_temp['GR'].mean(), inplace = True)
            df_temp['DEPTH'] = df_temp['DEPTH'].astype('int')
            df_temp = df_temp.groupby('DEPTH').mean()
            df_temp['DEPTH'] = df_temp.index
            #df_temp = df_temp.set_index('wellName')
            #print(df_temp)
            true_top = df_tops_copy.loc[wellname][top]
            
            # Skip well if top is missing
            if true_top > 0:
                if len(df_temp[df_temp["DEPTH"] == true_top]) > 0:
                    ctr = df_temp[df_temp["DEPTH"] == true_top].index[0]
                # If the top is not in the dataframe, find the closest depth    
                else:
                    tol = 4
                    mask1 = (df_temp["DEPTH"] > true_top - tol)
                    mask2 = (df_temp["DEPTH"] < true_top + tol)
                    diff = df_temp[mask1.values & mask2.values].copy()
                    if diff.empty:
                        continue
                    # Find the closest depth
                    diff['DIFF'] = diff['DEPTH'] - true_top
                    ctr = diff['DIFF'].idxmin()
                    #print(ctr)
                    
                # Extract the GR values around the top
                #print(ctr, wsize,  ctr-wsize, ctr+wsize)
                true_log = df_temp.loc[ctr-wsize:ctr+wsize+1]

                true_log_depth = true_log.DEPTH
                true_log_gr = true_log.GR

                # Plot the GR values
                if np.sum(true_log_gr.values < -8000) == 0:
                    ax[top_id].plot(true_log_gr.values, c=colors[top_id])
                    ax[top_id].set_xlabel('wsize')
                    ax[top_id].set_ylabel('Gamma Ray')
                    ax[top_id].grid(True)
                    features, label = true_log_gr, top
                    #print(features)

                    # Extract the features and labels
                    if label is not None:
                        if np.sum(features < -8000) > 0: 
                            continue
                        list_features.append(features)
                        list_labels.append(label)
                        list_well.append(int(wellname))

                # Set y-limits for the current subplot
                if well_id == 0:
                    y_min, y_max = ax[top_id].get_ylim()
                else:
                    cur_y_min, cur_y_max = ax[top_id].get_ylim()
                    if y_min is not None:
                        y_min = min(y_min, cur_y_min)
                    else:
                        y_min = cur_y_min
                    y_max = max(y_max, cur_y_max)

        y_range = y_max - y_min
        y_pad = y_range * 0.1
        ax[top_id].set_ylim(y_min - y_pad, y_max + y_pad)
        ax[top_id].set_title(top)

        # Increase vertical spacing between subplots
        plt.subplots_adjust(hspace=0.5)
        
        print("Progress: " + str(100*(top_id+1)/len(top_list)) + " %")
    
    # Pad shorter features with a constant value of 150 to ensure the same length
    #print(list_features)
    max_len = max([len(f) for f in list_features])
    for i, features in enumerate(list_features):
        #print(i)
        if len(features) < max_len:
            pad = np.full(max_len - len(features), 150)
            list_features[i] = np.concatenate((features, pad))

    features_np = np.asarray(list_features)
    labels_np = np.asarray(list_labels)
    well_np = np.asarray(list_well)

    fig.tight_layout()

    # Return the features and labels as numpy arrays
    return features_np, labels_np, well_np




def compute_distance_matrix_dtw(features_np_top : list #   A list of Gramma rays values
                            )->np.matrix : # A distance matrix
  
    
    """
    - Generate a distance matrix using Dynamic Time Warping (DTW) of the GR values
    - features_np_top : list # A list of Gramma rays values
    
    """
    
    # features_np_top_norm =(features_np_top - np.mean(features_np_top, axis=1, keepdims=True)) / np.std(features_np_top, axis=1, keepdims=True)

    distance_matrix = dtw.distance_matrix_fast(features_np_top)

    return distance_matrix



def cluster_dtw_distance_hdbscan(features_np : np.ndarray #  A numpy array with the formation GR values
                                  
                                  , labels_np : list # A list of formations name 
                                  , well_np : list
                                  , top_list: list # The list formation names.
                                  , max_c = int
                                  
                                  ) -> tuple[list # A list of distance matrices for each top
                                             , list # A list of templates for each top
                                             , list # list of clusters for each top
                                             ]  :

    
    """
    - Compute the distance matrix for each top in top_list based on their correlation, and then computes the clusters for each top using HDBSCAN.

    The function takes:
    - eatures_np :  A numpy array with the formation GR values
    - labels_np : A list of formations name 
    - well_np : A list of well names
    - top_list: A list formation top names
    - max_c = int defining the maximum cluster size for a marker signatures

    The functiion returns:
    
    - distance_matrix_per_top: A list of distance matrices for each top
    - template_list_per_top : A list of templates for each top
    - cluster_list_per_top : A list of clusters for each top
    - A plot of the distance matrix histogram
    - Subplots of each clusters  and the template extracted from each one
    
    """
    #creating empty lists

    distance_matrix_per_top = [] #A list of distance matrices for each top
    template_list_per_top = []  # A list of templates for each top
    cluster_list_per_top = [] # A list of clusters for each top it has index id stored
    cluster_well_list_per_top = [] # A list of well name, according to clusters for each top
    
    # top_list=[top_name]
    #loop per top
    for top_id in range(len(top_list)): 
        
        top = top_list[top_id]
#         print(top)
        template_list = []
        cluster_list = []
        cluster_well_list = []
        template_index=[]
        # Filter the features and labels for the current top
        mask = (labels_np == top)
        #print(mask)
        # Extract the features for the current top
        features_np_top = features_np[mask, :]
        well_np_top = well_np[mask]
        #print(well_np_top)
        #print(well_np_top.shape)
        print(f"Computing distance matrix for {top}...")
        #print(features_np_top )
        # Compute the distance matrix from the features using the the function compute_distance_matrix_dtw (DTW distance is computed)
        
        distance_matrix = compute_distance_matrix_dtw(features_np_top)
        #distance_matrix = np.zeros((len(features_np_top), len(features_np_top)))
        #create a shift matrix 
        shift_matrix = np.zeros((len(features_np_top), len(features_np_top)))
        
        total_iterations = int(0.5*len(features_np_top)*len(features_np_top))
        iterations_per_print = int(total_iterations*0.4)

        iteration = 0
        for id1 in range(len(features_np_top)):

            for id2 in range(id1+1, len(features_np_top)):

                features1 = features_np_top[id1, :]
                features2 = features_np_top[id2, :]

                features1_norm = (features1 - np.mean(features1))/np.std(features1)
                features2_norm = (features2 - np.mean(features2))/np.std(features2)

                corr = signal.correlate(features1_norm, features2_norm, mode='same') 
                corr = corr/len(features1_norm)

                corr_max = np.max(corr)
                shift = np.argmax(corr) - int(0.5*len(features1_norm)+0.5)


                if shift>0:
                    features1_new = np.copy(features1[shift:])
                    features2_new = features2_norm[:len(features1_new)]
                else:
                    features2_new = features2[-shift:]
                    features1_new = features1[:len(features2_new)]

                diff = np.abs(features1_new-features2_new)
                distance = 1/len(diff)*np.sum(diff)

                #distance_matrix[id1, id2] = distance#1-corr_max
                #distance_matrix[id2, id1] = distance#1-corr_max
                shift_matrix[id1, id2] = shift
                shift_matrix[id2, id1] = shift

                iteration = iteration + 1

                if iteration % iterations_per_print == 0:
                    print("Progress: " + str(40*(iteration//iterations_per_print)) + " %")    

        #distance_matrix_per_top.append(distance_matrix)
        #print(distance_matrix)
        
        print(f"Computing clusters for {top}...")
        #template_list = []
        #cluster_list = []
        fig, ax = plt.subplots(1, 1)
        ax.hist(distance_matrix.flatten(), bins=300)

        #clustering
        clustering = HDBSCAN( min_cluster_size=5, max_cluster_size = max_c, metric='precomputed', n_jobs=-1).fit(distance_matrix)  
        
        num_cluster = len(np.unique(clustering.labels_)) - 1
        fig, ax = plt.subplots(num_cluster+1, 3, figsize=(15, 10))
        fig.suptitle(top,fontweight ="bold")

        prop_cycle = plt.rcParams['axes.prop_cycle']
        colors = prop_cycle.by_key()['color']
        colors = np.tile(colors, 10)

        for clustering_label in range(-1, num_cluster):

            mask_cluster = (clustering.labels_ == clustering_label)
            num_in_cluster = np.sum(mask_cluster)
            #print(mask_cluster.shape)

            if num_in_cluster == 0: continue

            index_origin = np.where(mask_cluster)[0]
            well_index = well_np_top[mask_cluster]
            #print('Well name', well_index)

            feature_cluster = features_np_top[mask_cluster, :]
            #print(feature_cluster.shape)

            distance_cluster = distance_matrix[:, mask_cluster]
            distance_cluster = distance_cluster[mask_cluster, :]
            pattern_quality = np.sum(distance_cluster, axis=0)

            index_template = np.argmin(pattern_quality)
            print("Cluster " + str(clustering_label) + " with  " + str(np.sum(mask_cluster)) + " elements")

            for i in range(len(feature_cluster)):

                ax[clustering_label, 0].plot(feature_cluster[i, :], c=colors[clustering_label+1])
                if clustering_label == -1:
                    ax[clustering_label, 0].set_title("Noisy labels")
                else:
                    ax[clustering_label, 0].set_title("Cluster " + str(clustering_label))

                    features1 = feature_cluster[index_template, :]
                    features2 = feature_cluster[i, :]
                    shift = int(shift_matrix[index_origin[index_template], index_origin[i]])

                    fill_value = np.nan
                    num = len(features1)
                    if shift>0:
                        features2_new = fill_value*np.ones(num)
                        features2_new[shift:] = features1[:num-shift]
                    else:
                        features2_new = fill_value*np.ones(num) 
                        features2_new[:num+shift] = features2[-shift:]

                    ax[clustering_label, 2].plot(features2_new, c=colors[clustering_label+1])

            if clustering_label > -1:
                template_list.append(feature_cluster[index_template, :])
                cluster_list.append(index_origin)
                cluster_well_list.append(well_index)
                
                ax[clustering_label, 1].plot(feature_cluster[index_template, :], c=colors[clustering_label+1])
                ax[clustering_label, 1].set_title("Template")

        template_list_per_top.append(template_list)
        cluster_list_per_top.append(cluster_list)
        cluster_well_list_per_top.append(cluster_well_list)
     
    # Return the distance matrix, the template list and the cluster list    
    return distance_matrix_per_top, template_list_per_top, cluster_list_per_top, cluster_well_list_per_top



def clustering_scatterplot(points, cluster_idx , title): # plot the examples, i.e. the data points

    """
    - Plot the clusters in 2D to be able to viualize the diffrence. 

    The function takes:
    - points: a list;  well signature representation in 2D
    - cluster_idx: list; the cluster index or number assigned to each well
    - title: the title for the plot

    The funxtion returns a plot of clusters
    
    """

    colors = np.array(['orange', 'blue', 'purple', 'red', 'green','yellow','pink','grey','teal'])
    n_clusters = len(cluster_idx) 
    n_concat = set(np.concatenate(cluster_idx, axis=0 ))
    ln = list(set(range(0, max(n_concat))) - n_concat)
    plt.figure()
    h = plt.scatter(points[ln,0], points[ln,1],
                        c=colors[n_clusters%colors.size], label = 'Noisy cluster')
    for i in range(n_clusters):
            l  = cluster_idx[i]
            h = plt.scatter(points[l,0], points[l,1],
                            c=colors[i%colors.size], label = 'cluster '+str(i))
    plt.plot()        
    
    _ = plt.title(title) 
    _ = plt.legend()
    _ = plt.xlabel('x') 
    _ = plt.ylabel('y')


def MDS_dtw(top_list, labels_np, features_np, cluster_list_per_top):

    """
    MDS is used to project the signatures into a 2D space.

    Input features
    - top list: lit consiting of top names
    - labels_np: numpy array 
    - features_np: numpy array of top signatures of all well
    -cluster_list_per_top: a list of all clusters divided over the number of tops 

    Output is a 2D plot with clusters

    """
    for top_id in range(len(top_list)): 
            
        top = top_list[top_id]
        # Filter the features and labels for the current top
        mask = (labels_np == top)
        #print(mask)
        # Extract the features for the current top
        features_np_top = features_np[mask, :]
        distance_matrix = compute_distance_matrix_dtw(features_np_top)
        mds = MDS(random_state=0)
        X_transform = mds.fit_transform(features_np_top)
        mds_dtw = MDS(dissimilarity='precomputed', random_state=0)
        X_transform_dtw = mds_dtw.fit_transform(distance_matrix)
        clustering_scatterplot(X_transform, cluster_list_per_top[top_id], title = "MDS_"+top)
        clustering_scatterplot(X_transform_dtw, cluster_list_per_top[top_id], title = "MDS_dtw_"+top)


def training_well(top_list, cluster_well_list_per_top, best_template):

    """
    This function returns a list of all wells which will be part of the training list, after the cleaning process, for the classifier. 
    All signatures which are not part of th enoise clusters and the cluster to which they belong is among the most populated will be used as part of training well.

    Input: 
    - top_lit: a list of all tops
    - cluster_well_list_per_to: a list of all clusters with the well names divided over the number of tops
    - best template: a boolean (if true will only consider the cluster with maximum number of wells excluding the noise cluster, else more than one

    Output:
    - template_ids_list: it is teh template signature from the wells in the training dataset. We will require them for DTW baseline testing
    - training_well_list: a list of well names to be used for traing the classification model
    """
    
    training_well_list = np.empty((3, 1), dtype=object)
    template_ids_list = []
    
    for i in range(0,len(top_list)):
        
        top = top_list[i]
        cluster_well_list = cluster_well_list_per_top[i]
        
        # Get the lengths of each list
        llen = np.array([len(l) for l in cluster_well_list])
        # sort the list in the array and cluster well signatures to be taken as part of training wells
        si = np.flip(np.argsort(llen))
        print(top)
        print('The clusters taken for this top are ', si[0], si[1])
        template_ids_list.append([si[0], si[1]])
        if best_template == True:
            training_well_list[i,0] = (np.concatenate((cluster_well_list[si[0]],cluster_well_list[si[1]]), axis = 0))
        else:
            training_well_list[i,0] = (cluster_well_list[si[0]])


    return template_ids_list, training_well_list






