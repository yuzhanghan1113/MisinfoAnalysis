import pandas as pd
import json
import os
import getopt
import sys
import random
import pandas as pd
import traceback
import time
from datetime import datetime
from sklearn.cluster import AffinityPropagation
from sklearn import metrics
from sklearn.datasets import make_blobs
import numpy as np
from sklearn.cluster import MeanShift, estimate_bandwidth
from sklearn.datasets import make_blobs
from collections import Counter
from sklearn import preprocessing
from sklearn.cluster import AgglomerativeClustering
import numpy as np

from matplotlib import pyplot as plt
from scipy.cluster.hierarchy import dendrogram
from sklearn.datasets import load_iris
from sklearn.cluster import AgglomerativeClustering

def plot_dendrogram(model, **kwargs):
    # Create linkage matrix and then plot the dendrogram

    # create the counts of samples under each node
    counts = np.zeros(model.children_.shape[0])
    n_samples = len(model.labels_)
    for i, merge in enumerate(model.children_):
        current_count = 0
        for child_idx in merge:
            if child_idx < n_samples:
                current_count += 1  # leaf node
            else:
                current_count += counts[child_idx - n_samples]
        counts[i] = current_count

    linkage_matrix = np.column_stack([model.children_, model.distances_,
                                      counts]).astype(float)

    # Plot the corresponding dendrogram
    dendrogram(linkage_matrix, **kwargs)



def main(argv):
    random.seed(1113)

    # pd.set_option('display.height', 1000)
    pd.set_option('display.max_rows', 1000)
    pd.set_option('display.max_columns', 1000)
    # pd.set_option('display.width', 170)

    opts, args = getopt.getopt(argv, '', ["absFilename_input=", "path_output_base="])

    print(opts)

    for opt, arg in opts:
        if opt == '--absFilename_input':
            absFilename_input = arg
        if opt == '--path_output_base':
            path_output_base = arg
        

    #absFilename_input = "/gpfs_common/share03/uncghan/yhan5/TwitterDataAnalysis/data/repliesWithFeatures/1247287993095897088_reply_data.csv"
    absFilename_input = "/gpfs_common/share03/uncghan/yhan5/TwitterDataAnalysis/data/repliesWithFeatures/1239336564334960642_reply_data.csv"
    print('absFilename_input:')
    print(absFilename_input)
    #df_input = pd.read_csv(absFilename_input, dtype=str, quotechar='"', delimiter=',', escapechar='\\')
    df_input = pd.read_csv(absFilename_input, dtype=str)
        
    print("df_input:")
    print(df_input)
    print(len(df_input))
    
    list_features_used = ["followers_count", "friends_count", "listed_count", "favourites_count", "statuses_count"]
    
    
    #list_features_used = ["fear", "anger", "anticip", "trust", "surprise", "positive", "negative", "sadness", "disgust", "joy", "neg", "neu", "pos", "compound"]
    list_features_used += ["fear", "anger", "anticip", "trust", "surprise", "positive", "negative", "sadness", "disgust", "joy", "neg", "neu", "pos", "compound"]

    
    
    #"val"
    #account_age


    list_training_input = df_input[list_features_used].astype(float).values.tolist()
    X = list_training_input.copy()
    
    #print("list_training_input:")
    #print(list_training_input)
    #print(len(list_training_input))
    print("X:")
    print(X)
    print(len(X))
    
    X = preprocessing.normalize(X, norm='max')
    print("X:")
    print(X)
    print(len(X))
    """
    # #############################################################################
    # Generate sample data
    #centers = [[1, 1], [-1, -1], [1, -1]]
    #X, labels_true = make_blobs(n_samples=300, centers=centers, cluster_std=0.5,                                 random_state=0)

    # #############################################################################
    # Compute Affinity Propagation
    
    #print("X:")
    #print(X)
    
    #af = AffinityPropagation(preference=-50).fit(X)
    af = AffinityPropagation().fit(list_training_input)
    cluster_centers_indices = af.cluster_centers_indices_
    labels = af.labels_

    n_clusters_ = len(cluster_centers_indices)

    print('Estimated number of clusters: %d' % n_clusters_)
    #print("Homogeneity: %0.3f" % metrics.homogeneity_score(labels_true, labels))
    #print("Completeness: %0.3f" % metrics.completeness_score(labels_true, labels))
    #print("V-measure: %0.3f" % metrics.v_measure_score(labels_true, labels))
    #print("Adjusted Rand Index: %0.3f" % metrics.adjusted_rand_score(labels_true, labels))
    #print("Adjusted Mutual Information: %0.3f" % metrics.adjusted_mutual_info_score(labels_true, labels))
    #print("Silhouette Coefficient: %0.3f" % metrics.silhouette_score(X, labels, metric='sqeuclidean'))
    print("Silhouette Coefficient: %0.3f" % metrics.silhouette_score(list_training_input, labels, metric='sqeuclidean'))

    # #############################################################################
    # Plot result
    import matplotlib.pyplot as plt
    from itertools import cycle

    plt.close('all')
    plt.figure(1)
    plt.clf()
    """
    """
    # #############################################################################
    # Generate sample data
    #centers = [[1, 1], [-1, -1], [1, -1]]
    #X, _ = make_blobs(n_samples=10000, centers=centers, cluster_std=0.6)

    # #############################################################################
    # Compute clustering with MeanShift

    # The following bandwidth can be automatically detected using
    #bandwidth = estimate_bandwidth(X, quantile=0.2, n_samples=500)
    #bandwidth = estimate_bandwidth(X, n_samples=500, n_jobs=5)
    bandwidth = estimate_bandwidth(X, n_jobs=5)

    #ms = MeanShift(bandwidth=bandwidth, bin_seeding=True)
    ms = MeanShift(bandwidth=bandwidth)
    ms.fit(X)
    labels = ms.labels_
    cluster_centers = ms.cluster_centers_

    labels_unique = np.unique(labels)
    n_clusters_ = len(labels_unique)

    print("number of estimated clusters : %d" % n_clusters_)
    
    print("labels_unique:")
    print(labels_unique)
    
    print("Counter(labels):")
    print(Counter(labels))    
    
    #print("cluster_centers:")
    #print(cluster_centers)
    
    # results:
    
    # root tweet: 1247287993095897088 (1972 replies)
    
    # only engagement OR engagmeent + emotion:
    #number of estimated clusters : 46
    #labels_unique:
    #[ 0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20 21 22 23
    #24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 43 44 45]
    #Counter(labels):
    #Counter({0: 1679, 1: 66, 4: 58, 2: 34, 6: 23, 3: 17, 7: 12, 8: 9, 13: 8, 12: 7, 11: 7, 5: 6, 10: 4, 9: 3, 20: 2, 14: 2, 19: 2, 16: 2, 18: 2, 21: 2, 15: 2, 40: 1, 37: 1, 38: 1, 29: 1, 23: 1, 32: 1, 26: 1, 42: 1, 41: 1, 27: 1, 44: 1, 28: 1, 22: 1, 24: 1, 39: 1, 34: 1, 30: 1, 36: 1, 45: 1, 31: 1, 33: 1, 35: 1, 17: 1, 43: 1, 25: 1})
    
    # only emotion:
    #number of estimated clusters : 13
    #labels_unique:
    #[ 0  1  2  3  4  5  6  7  8  9 10 11 12]
    #Counter(labels):
    #Counter({0: 883, 1: 649, 2: 411, 3: 11, 6: 5, 4: 5, 5: 2, 11: 1, 10: 1, 9: 1, 12: 1, 8: 1, 7: 1})
    #Discussion: engagement is overriding emotion, engagement needs to be normalized?
    
    #After max-normalized all features:
    #engagment still dominating
    #engagement + emotion:
    #number of estimated clusters : 6
    #labels_unique:
    #[0 1 2 3 4 5]
    #Counter(labels):
    #Counter({0: 1229, 1: 615, 3: 70, 2: 39, 5: 10, 4: 9})
    
    # root tweet: 1239336564334960642    (4963 replies)
    
    #engagement + emotion:
    #number of estimated clusters : 10
    #labels_unique:
    #[0 1 2 3 4 5 6 7 8 9]
    #Counter(labels):
    #Counter({0: 3062, 1: 1755, 2: 53, 3: 22, 9: 16, 4: 16, 6: 13, 5: 12, 7: 11, 8: 3})
    
    #emotion:
    #number of estimated clusters : 23
    #labels_unique:
    #[ 0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20 21 22]
    #Counter(labels):
    #Counter({0: 2213, 1: 1876, 2: 711, 17: 52, 3: 28, 13: 26, 4: 17, 12: 8, 5: 6, 6: 5, 8: 3, 7: 3, 9: 3, 10: 2, 16: 2, 14: 1, 20: 1, 11: 1, 21: 1, 19: 1, 22: 1, 18: 1, 15: 1})
    """

    #clustering = AgglomerativeClustering(n_clusters=4).fit(X)
    #labels = clustering.labels_
    #labels_unique = np.unique(labels)
    #n_clusters_ = len(labels_unique)
    
    
    model = AgglomerativeClustering(distance_threshold=0, n_clusters=None)
    model = model.fit(X)
    
    labels = model.labels_
    labels_unique = np.unique(labels)
    n_clusters_ = len(labels_unique)
    
    print("number of estimated clusters : %d" % n_clusters_)
    
    print("labels_unique:")
    print(labels_unique)
    
    print("Counter(labels):")
    print(Counter(labels))   
    
        
    plt.title('Hierarchical Clustering Dendrogram')
    # plot the top three levels of the dendrogram
    plot_dendrogram(model, truncate_mode='level', p=3)
    plt.xlabel("Number of points in node (or index of point if no parenthesis).")
    plt.show()


    
    
    
    

if __name__ == "__main__":
    main(sys.argv[1:])