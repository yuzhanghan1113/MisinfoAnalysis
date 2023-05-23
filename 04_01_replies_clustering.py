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
from sklearn import preprocessing

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

    opts, args = getopt.getopt(argv, '', ["path_input=", "path_output=", "list_tweetIDs=", "list_featureSets=", "str_standardizationMethod="])

    print(opts)

    for opt, arg in opts:
        if opt == '--path_input':
            path_input = arg
        elif opt == '--path_output':
            path_output = arg
        elif opt == '--list_tweetIDs':
            list_tweetIDs = arg.split(" ")
        elif opt == '--list_featureSets':
            list_featureSets = arg.split(" ")
        elif opt == '--str_standardizationMethod':
            str_standardizationMethod = arg
            
    dict_mapping_tweetIDs2TweetLabels = {"1246159197621727234":"F1", "1239336564334960642":"F2", "1240682713817976833":"F3", "1261326944492281859":"F4", "1262482651333738500":"F5", "1247287993095897088":"F6", "1256641587708342274":"F7", "1268269994074345473":"T1", "1269111354520211458":"T2", "1268155500753027073":"T3", "1269320689003216896":"T4", "1269077273770156032":"T5", "1268346742296252418":"T6"} 
            
            
    list_features_engagement = ["account_age", "followers_count", "friends_count", "listed_count", "favourites_count", "statuses_count"]
        
    #list_features_emotion = ["fear", "anger", "anticip", "trust", "surprise", "positive", "negative", "sadness", "disgust", "joy", "neg", "neu", "pos", "compound", "val_bin"] 
    #list_features_emotion = ["fear", "anger", "anticip", "trust", "surprise", "positive", "negative", "sadness", "disgust", "joy", "neg", "neu", "pos", "compound"] 
    list_features_emotion = ["fear", "anger", "anticip", "trust", "surprise", "sadness", "disgust", "joy", "neg", "neu", "pos", "compound"]
    
    #"val"
    #account_age
    
    list_features_output_eval_noGroundTruth = ["eval_silhouette", "eval_calinskiHarabasz", "eval_bouldin"]
    
    list_features_output_eval_groundTruth = ["eval_adjustedRand_rootTweetVeracity", "eval_adjustedRand_rootTweetLabel", "eval_mutualInformation_rootTweetVeracity", "eval_mutualInformation_rootTweetLabel", "eval_homogeneity_rootTweetVeracity", "eval_homogeneity_rootTweetLabel", "eval_completeness_rootTweetVeracity", "eval_completeness_rootTweetLabel", "eval_fowlkesMallows_rootTweetVeracity", "eval_fowlkesMallows_rootTweetLabel"]

    list_features_output_eval = list_features_output_eval_noGroundTruth + list_features_output_eval_groundTruth

    
    #list_featureSets = ["engagement", "emotion", "engagementAndEmotion"]
    
    if str_standardizationMethod == "norm_max":
    
        str_norm = "max"
    
        #str_norm = "l2"
        #str_norm = "l1"
    
        
    for str_featureSet in list_featureSets:
    
        print("str_featureSet:")
        print(str_featureSet)
    
        df_output_metrics_featureSet = pd.DataFrame()

        
        index_row = 0
        
                
        absFilename_output_metrics_featureSet = path_output + "clusteringResults_metrics_featureSet=" + str_featureSet + ".csv"
        if not os.path.exists(os.path.dirname(absFilename_output_metrics_featureSet)):
            os.makedirs(os.path.dirname(absFilename_output_metrics_featureSet))
        print("absFilename_output_metrics_featureSet:")
        print(absFilename_output_metrics_featureSet)
        
        absFilename_output_raw_featureSet = path_output + "clusteringResults_raw_featureSet=" + str_featureSet + ".csv"
        if not os.path.exists(os.path.dirname(absFilename_output_raw_featureSet)):
            os.makedirs(os.path.dirname(absFilename_output_raw_featureSet))
        print("absFilename_output_raw_featureSet:")
        print(absFilename_output_raw_featureSet)
        
        str_range = "singleCascade"
        
        list_df_input = []
        list_df_output_raw = []
        
        for tweetID in list_tweetIDs:
        
            absFilename_input = path_input + tweetID + "_reply_data.csv"
            
            print('absFilename_input:')
            print(absFilename_input)
            
            df_input = pd.read_csv(absFilename_input, dtype=str)       
            
            df_input["val_bin"] = df_input["val"].map({"positive":"1", "neutral":"0", "negative":"-1"})
                        
            rootTweetLabel = dict_mapping_tweetIDs2TweetLabels[tweetID]
            df_input["rootTweetLabel"] = rootTweetLabel
            df_input["rootTweetVeracity"] = df_input["rootTweetLabel"].map({"F1":"F", "F2":"F", "F3":"F", "F4":"F", "F5":"F", "F6":"F", "F7":"F", "T1":"T", "T2":"T", "T3":"T", "T4":"T", "T5":"T", "T6":"T"})
            
            #print("df_input:")
            #print(df_input)
            print("len(df_input):")
            print(len(df_input))
            
            list_df_input += [df_input.copy()]
            
            
            if str_featureSet == "engagement":
                list_features_used = list_features_engagement
            elif str_featureSet == "emotion":
                list_features_used = list_features_emotion
            elif str_featureSet == "engagementAndEmotion":
                list_features_used = list_features_engagement + list_features_emotion
            
            print("len(list_features_used):")
            print(len(list_features_used))
            
            df_input[list_features_used] = df_input[list_features_used].astype(float)
            list_training_input = df_input[list_features_used].values.tolist()
            
            #print("list_training_input:")
            #print(list_training_input) 
            print("before normalization:")
            print("list_training_input[0:5]:")
            print(list_training_input[0:5])
            print("list_training_input[-5:]:")
            print(list_training_input[-5:])
            print("len(list_training_input):")
            print(len(list_training_input))
            
            """
            if str_featureSet == "emotion":
                print("normalization: do not normalize any (emotion) feature")
                
            elif str_featureSet == "engagement":
            
                print("normalization: normalize all (engagement) features")
                
                print("str_norm:")
                print(str_norm)
                
                list_training_input = preprocessing.normalize(list_training_input, norm=str_norm)
                print("list_training_input[0:5]:")
                print(list_training_input[0:5])
                print("list_training_input[-5:]:")
                print(list_training_input[-5:])
                print("len(list_training_input):")
                print(len(list_training_input))
                
            elif str_featureSet == "engagementAndEmotion":
            
                print("normalization: normalize only the engagement, not emotion features")
                
                
                list_training_input_engagement = df_input[list_features_engagement].values.tolist()
                
                print("str_norm:")
                print(str_norm)
                
                list_training_input_engagement = preprocessing.normalize(list_training_input_engagement, norm=str_norm)
                print("list_training_input_engagement[0:5]:")
                print(list_training_input_engagement[0:5])
                print("len(list_training_input_engagement):")
                print(len(list_training_input_engagement))
                print("list_training_input_engagement[-5:]:")
                print(list_training_input_engagement[-5:])
                
                list_training_input_emotion = df_input[list_features_emotion].values.tolist()
                
                list_training_input = np.concatenate((list_training_input_engagement, list_training_input_emotion), axis=1)
            
                print("list_training_input[0:5]:")
                print(list_training_input[0:5])
                print("len(list_training_input):")
                print(len(list_training_input))
                print("list_training_input[-5:]:")
                print(list_training_input[-5:])
            """
            
            """
            print("str_norm:")
            print(str_norm)
            
            #list_training_input = preprocessing.scale(list_training_input)
            
            #scaler = preprocessing.MinMaxScaler()
            #scaler = preprocessing.MaxAbsScaler()
            #scaler = preprocessing.RobustScaler()
            #scaler.fit(list_training_input)
            #list_training_input = scaler.transform(list_training_input)
            quantile_transformer = preprocessing.QuantileTransformer(random_state=0)
            list_training_input = quantile_transformer.fit_transform(list_training_input)
            """
            print("str_standardizationMethod:")
            print(str_standardizationMethod)
            
            if str_standardizationMethod in ["norm_l1", "norm_l2", "norm_max"]:
            
                print("normalizer selected")
                
                print("str_norm:")
                print(str_norm)
                
                list_training_input = preprocessing.normalize(list_training_input, norm=str_norm)
                
            elif str_standardizationMethod == "MaxAbsScaler":
            
                print("MaxAbsScaler selected")
            
                scaler = preprocessing.MaxAbsScaler()
                scaler.fit(list_training_input)
                list_training_input = scaler.transform(list_training_input)
            
            print("list_training_input[0:5]:")
            print(list_training_input[0:5])
            print("list_training_input[-5:]:")
            print(list_training_input[-5:])
            print("len(list_training_input):")
            print(len(list_training_input))
            
            
            # #############################################################################
            # Generate sample data
            #centers = [[1, 1], [-1, -1], [1, -1]]
            #X, _ = make_blobs(n_samples=10000, centers=centers, cluster_std=0.6)

            # #############################################################################
            # Compute clustering with MeanShift

            # The following bandwidth can be automatically detected using
            #bandwidth = estimate_bandwidth(X, quantile=0.2, n_samples=500)
            #bandwidth = estimate_bandwidth(X, n_samples=500, n_jobs=5)
            bandwidth = estimate_bandwidth(list_training_input, n_jobs=5)

            #model = MeanShift(bandwidth=bandwidth, bin_seeding=True)
            model = MeanShift(bandwidth=bandwidth)
            print("fit model:")
            
            model.fit(list_training_input)
            
            print("model:")
            print(model)
            
            list_labels = model.labels_
            #list_custerCenters = model.cluster_centers_
            
            df_input["clusterLabel"] = list_labels
            
            list_evaluationMetric_silhouette = metrics.silhouette_score(list_training_input, list_labels)
            df_input["eval_silhouette"] = list_evaluationMetric_silhouette
            
            list_evaluationMetric_calinskiHarabasz = metrics.calinski_harabasz_score(list_training_input, list_labels)
            df_input["eval_calinskiHarabasz"] = list_evaluationMetric_calinskiHarabasz
            
            list_evaluationMetric_bouldin = metrics.davies_bouldin_score(list_training_input, list_labels)
            df_input["eval_bouldin"] = list_evaluationMetric_bouldin
            
            list_labelCounts = Counter(list_labels).most_common()
            print("list_labelCounts:")
            print(list_labelCounts)

            num_clusters = len(list_labelCounts)
            print("number of estimated clusters :" + str(num_clusters))     

            df_output_raw = df_input.copy()          
            
            
            df_output_raw["range"] = str_range
            df_output_raw["featureSet_name"] = str_featureSet
            df_output_raw["featureSet_features"] = str(list_features_used)
            df_output_raw["featureSet_size"] = str(len(list_features_used))
            df_output_raw["standardizationMethod"] = str_standardizationMethod
            df_output_raw["rootTweetLabel"] = rootTweetLabel
            df_output_raw["rootTweetID"] = tweetID
                            
            totalSize = len(list_training_input)
            df_output_raw["totalSize"] = str(totalSize)
             

            for tuple_labelCount in list_labelCounts:
            
                
                #if tuple_labelCount[1] >= len(list_training_input)/5:
                #    print(tuple_labelCount)
                
                print("cluster:")
                print(tuple_labelCount)
                
                df_inCluster = df_input.loc[df_input["clusterLabel"]==tuple_labelCount[0],]
                print("len(df_inCluster)")
                print(len(df_inCluster))
                
                df_output_metrics_featureSet.loc[index_row, "range"] = str_range
                df_output_metrics_featureSet.loc[index_row, "featureSet_name"] = str_featureSet
                df_output_metrics_featureSet.loc[index_row, "featureSet_features"] = str(list_features_used)
                df_output_metrics_featureSet.loc[index_row, "featureSet_size"] = str(len(list_features_used))
                df_output_metrics_featureSet.loc[index_row, "standardizationMethod"] = str_standardizationMethod
                df_output_metrics_featureSet.loc[index_row, "rootTweetLabel"] = rootTweetLabel
                df_output_metrics_featureSet.loc[index_row, "rootTweetID"] = tweetID
                
                totalSize = len(list_training_input)
                df_output_metrics_featureSet.loc[index_row, "totalSize"] = str(totalSize)
                            
                df_output_metrics_featureSet.loc[index_row, "clusterLabel"] = str(tuple_labelCount[0])
                df_output_metrics_featureSet.loc[index_row, "clusterSize"] = str(tuple_labelCount[1])            
                df_output_metrics_featureSet.loc[index_row, "clusterProportion"] = str(tuple_labelCount[1]/totalSize)
                
                df_output_raw.loc[df_output_raw["clusterLabel"]==tuple_labelCount[0], "clusterSize"] = str(tuple_labelCount[1])
                df_output_raw.loc[df_output_raw["clusterLabel"]==tuple_labelCount[0], "clusterProportion"] = str(tuple_labelCount[1]/totalSize)
                
                
                for feature in (list_features_used + list_features_output_eval_noGroundTruth):
                
                    print("feature:")
                    print(feature)
                
                    list_values = df_inCluster[feature].tolist()
                    
                    #list_values = list(map(float, list_values))
                    
                    if feature in (["val_bin"] + list_features_output_eval_noGroundTruth):
                        print("list_values[0:5]:")
                        print(list_values[0:5])
                        print(len(list_values))

                    metric_max = np.max(list_values)
                    metric_min = np.min(list_values)
                    metric_mean = np.mean(list_values)
                    metric_median = np.median(list_values)
                    metric_std = np.std(list_values)
                    metric_rsd = metric_std/metric_mean
                    
                    df_output_metrics_featureSet.loc[index_row, feature+"_max"] = str(metric_max)
                    df_output_metrics_featureSet.loc[index_row, feature+"_min"] = str(metric_min)
                    df_output_metrics_featureSet.loc[index_row, feature+"_mean"] = str(metric_mean)
                    df_output_metrics_featureSet.loc[index_row, feature+"_median"] = str(metric_median)
                    df_output_metrics_featureSet.loc[index_row, feature+"_std"] = str(metric_std)
                    df_output_metrics_featureSet.loc[index_row, feature+"_rsd"] = str(metric_rsd)
                    
                    
                    
                
                index_row += 1
                
                df_output_metrics_featureSet.to_csv(absFilename_output_metrics_featureSet, index=False)
                
                
            
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
                """
        
            list_df_output_raw += [df_output_raw.copy()]
                
        print("len(list_df_input):")
        print(len(list_df_input))
                
        df_input_allCascades = pd.concat(list_df_input)
        df_input_allCascades = df_input_allCascades.reset_index(drop=True)
        print("len(df_input_allCascades):")
        print(len(df_input_allCascades))
        
        print("len(list_df_output_raw):")
        print(len(list_df_output_raw))
                
        df_input_allCascades["rootTweetVeracity"] = df_input_allCascades["rootTweetLabel"].map({"F1":"F", "F2":"F", "F3":"F", "F4":"F", "F5":"F", "F6":"F", "F7":"F", "T1":"T", "T2":"T", "T3":"T", "T4":"T", "T5":"T", "T6":"T"})
                        
        for str_range in ["falseCascades", "trueCascades", "allCascades"]:
        
            print("str_range:")
            print(str_range)
        
            df_input_allCascades_range = df_input_allCascades.copy()
            
            if str_range == "falseCascades":
                df_input_allCascades_range = df_input_allCascades_range[df_input_allCascades_range["rootTweetVeracity"]=="F"]
            elif str_range == "trueCascades":
                df_input_allCascades_range = df_input_allCascades_range[df_input_allCascades_range["rootTweetVeracity"]=="T"]
            elif str_range == "allCascades":
                pass
                
            df_input_allCascades_range = df_input_allCascades_range.reset_index(drop=True)
        
            print("df_input_allCascades_range[[\"rootTweetLabel\", \"rootTweetVeracity\"]:")
            print(df_input_allCascades_range[["rootTweetLabel", "rootTweetVeracity"]])
    
            print("len(df_input_allCascades_range):")
            print(len(df_input_allCascades_range))
        
            if str_featureSet == "engagement":
                list_features_used = list_features_engagement
            elif str_featureSet == "emotion":
                list_features_used = list_features_emotion
            elif str_featureSet == "engagementAndEmotion":
                list_features_used = list_features_engagement + list_features_emotion
            
            print("len(list_features_used):")
            print(len(list_features_used))

            df_input_allCascades_range[list_features_used] = df_input_allCascades_range[list_features_used].astype(float)
            list_training_input = df_input_allCascades_range[list_features_used].values.tolist()
            
            #print("list_training_input:")
            #print(list_training_input)        
            #print("list_training_input[0:5]:")
            #print(list_training_input[0:5])
            print("len(list_training_input):")
            print(len(list_training_input))
            
            """
            if str_featureSet == "emotion":
                print("normalization: do not normalize any (emotion) feature")
                
            elif str_featureSet == "engagement":
            
                print("normalization: normalize all (engagement) features")
                
                print("str_norm:")
                print(str_norm)
                
                list_training_input = preprocessing.normalize(list_training_input, norm=str_norm)
                print("list_training_input[0:5]:")
                print(list_training_input[0:5])
                print("list_training_input[-5:]:")
                print(list_training_input[-5:])
                print("len(list_training_input):")
                print(len(list_training_input))
                
            elif str_featureSet == "engagementAndEmotion":
            
                print("normalization: normalize only the engagement, not emotion features")
                
                
                list_training_input_engagement = df_input_allCascades_range[list_features_engagement].values.tolist()
                
                print("str_norm:")
                print(str_norm)
                
                list_training_input_engagement = preprocessing.normalize(list_training_input_engagement, norm=str_norm)
                print("list_training_input_engagement[0:5]:")
                print(list_training_input_engagement[0:5])
                print("list_training_input_engagement[-5:]:")
                print(list_training_input_engagement[-5:])
                print("len(list_training_input_engagement):")
                print(len(list_training_input_engagement))
                
                list_training_input_emotion = df_input_allCascades_range[list_features_emotion].values.tolist()
                
                list_training_input = []
                for item_engagement, item_emotion in zip(list_training_input_engagement, list_training_input_emotion):
                    list_training_input += item_engagement.extend(item_emotion)
            
                print("list_training_input[0:5]:")
                print(list_training_input[0:5])
                print("list_training_input[-5:]:")
                print(list_training_input[-5:])
                print("len(list_training_input):")
                print(len(list_training_input))
            """
            """    
            print("str_norm:")
            print(str_norm)
            
            #list_training_input = preprocessing.scale(list_training_input)
            
            #scaler = preprocessing.MinMaxScaler()
            #scaler = preprocessing.MaxAbsScaler()
            #scaler = preprocessing.RobustScaler()
            #scaler.fit(list_training_input)
            #list_training_input = scaler.transform(list_training_input)
            quantile_transformer = preprocessing.QuantileTransformer(random_state=0)
            list_training_input = quantile_transformer.fit_transform(list_training_input)
            
            print("list_training_input[0:5]:")
            print(list_training_input[0:5])
            print("list_training_input[-5:]:")
            print(list_training_input[-5:])
            print("len(list_training_input):")
            print(len(list_training_input))
            """
            
            print("str_standardizationMethod:")
            print(str_standardizationMethod)
            
            if str_standardizationMethod in ["norm_l1", "norm_l2", "norm_max"]:
            
                print("normalizer selected")
                
                print("str_norm:")
                print(str_norm)
                
                list_training_input = preprocessing.normalize(list_training_input, norm=str_norm)
                
            elif str_standardizationMethod == "MaxAbsScaler":
            
                print("MaxAbsScaler selected")
            
                scaler = preprocessing.MaxAbsScaler()
                scaler.fit(list_training_input)
                list_training_input = scaler.transform(list_training_input)
            
            print("list_training_input[0:5]:")
            print(list_training_input[0:5])
            print("list_training_input[-5:]:")
            print(list_training_input[-5:])
            print("len(list_training_input):")
            print(len(list_training_input))
            
            # #############################################################################
            # Generate sample data
            #centers = [[1, 1], [-1, -1], [1, -1]]
            #X, _ = make_blobs(n_samples=10000, centers=centers, cluster_std=0.6)

            # #############################################################################
            # Compute clustering with MeanShift

            # The following bandwidth can be automatically detected using
            #bandwidth = estimate_bandwidth(X, quantile=0.2, n_samples=500)
            #bandwidth = estimate_bandwidth(X, n_samples=500, n_jobs=5)
            bandwidth = estimate_bandwidth(list_training_input, n_jobs=5)

            #model = MeanShift(bandwidth=bandwidth, bin_seeding=True)
            model = MeanShift(bandwidth=bandwidth)
            print("fit model:")
            
            model.fit(list_training_input)
            
            print("model:")
            print(model)
            
            list_labels = model.labels_
            #list_custerCenters = model.cluster_centers_
            
            df_input_allCascades_range["clusterLabel"] = list_labels    
            
            list_evaluationMetric_silhouette = metrics.silhouette_score(list_training_input, list_labels)
            df_input_allCascades_range["eval_silhouette"] = list_evaluationMetric_silhouette
            
            list_evaluationMetric_calinskiHarabasz = metrics.calinski_harabasz_score(list_training_input, list_labels)
            df_input_allCascades_range["eval_calinskiHarabasz"] = list_evaluationMetric_calinskiHarabasz
            
            list_evaluationMetric_bouldin = metrics.davies_bouldin_score(list_training_input, list_labels)
            df_input_allCascades_range["eval_bouldin"] = list_evaluationMetric_bouldin
            
            list_evaluationMetric_adjustedRand = metrics.adjusted_rand_score(df_input_allCascades_range["rootTweetVeracity"].tolist(), list_labels)
            df_input_allCascades_range["eval_adjustedRand_rootTweetVeracity"] = list_evaluationMetric_adjustedRand
            
            list_evaluationMetric_adjustedRand = metrics.adjusted_rand_score(df_input_allCascades_range["rootTweetLabel"].tolist(), list_labels)
            df_input_allCascades_range["eval_adjustedRand_rootTweetLabel"] = list_evaluationMetric_adjustedRand
            
            list_evaluationMetric_mutualInformation = metrics.adjusted_mutual_info_score(df_input_allCascades_range["rootTweetVeracity"].tolist(), list_labels)
            df_input_allCascades_range["eval_mutualInformation_rootTweetVeracity"] = list_evaluationMetric_mutualInformation
            
            list_evaluationMetric_mutualInformation = metrics.adjusted_mutual_info_score(df_input_allCascades_range["rootTweetLabel"].tolist(), list_labels)
            df_input_allCascades_range["eval_mutualInformation_rootTweetLabel"] = list_evaluationMetric_mutualInformation
            
            list_evaluationMetric_homogeneity = metrics.homogeneity_score(df_input_allCascades_range["rootTweetVeracity"].tolist(), list_labels)
            df_input_allCascades_range["eval_homogeneity_rootTweetVeracity"] = list_evaluationMetric_homogeneity
            
            list_evaluationMetric_homogeneity = metrics.homogeneity_score(df_input_allCascades_range["rootTweetLabel"].tolist(), list_labels)
            df_input_allCascades_range["eval_homogeneity_rootTweetLabel"] = list_evaluationMetric_homogeneity
            
            list_evaluationMetric_completeness = metrics.completeness_score(df_input_allCascades_range["rootTweetVeracity"].tolist(), list_labels)
            df_input_allCascades_range["eval_completeness_rootTweetVeracity"] = list_evaluationMetric_completeness
            
            list_evaluationMetric_completeness = metrics.completeness_score(df_input_allCascades_range["rootTweetLabel"].tolist(), list_labels)
            df_input_allCascades_range["eval_completeness_rootTweetLabel"] = list_evaluationMetric_completeness
            
            list_evaluationMetric_fowlkesMallows = metrics.fowlkes_mallows_score(df_input_allCascades_range["rootTweetVeracity"].tolist(), list_labels)
            df_input_allCascades_range["eval_fowlkesMallows_rootTweetVeracity"] = list_evaluationMetric_fowlkesMallows
            
            list_evaluationMetric_fowlkesMallows = metrics.fowlkes_mallows_score(df_input_allCascades_range["rootTweetLabel"].tolist(), list_labels)
            df_input_allCascades_range["eval_fowlkesMallows_rootTweetLabel"] = list_evaluationMetric_fowlkesMallows
            
                        
            list_labelCounts = Counter(list_labels).most_common()
            print("list_labelCounts:")
            print(list_labelCounts)

            num_clusters = len(list_labelCounts)
            print("number of estimated clusters :" + str(num_clusters))         
            
            #print("list_custerCenters:")
            #print(list_custerCenters)
            
            df_output_raw_allCascades_range = df_input_allCascades_range.copy()
            
            df_output_raw_allCascades_range["range"] = str_range
            df_output_raw_allCascades_range["featureSet_name"] = str_featureSet
            df_output_raw_allCascades_range["featureSet_features"] = str(list_features_used)
            df_output_raw_allCascades_range["featureSet_size"] = str(len(list_features_used))
            df_output_raw_allCascades_range["standardizationMethod"] = str_standardizationMethod

            
            totalSize = len(list_training_input)
            df_output_raw_allCascades_range["totalSize"] = str(totalSize)

            
            for tuple_labelCount in list_labelCounts:
            
                
                #if tuple_labelCount[1] >= len(list_training_input)/5:
                #    print(tuple_labelCount)
                
                print("cluster:")
                print(tuple_labelCount)
                
                df_inCluster = df_input_allCascades_range.loc[df_input_allCascades_range["clusterLabel"]==tuple_labelCount[0],]
                print("len(df_inCluster)")
                print(len(df_inCluster))
                
                df_output_metrics_featureSet.loc[index_row, "range"] = str_range
                df_output_metrics_featureSet.loc[index_row, "featureSet_name"] = str_featureSet
                df_output_metrics_featureSet.loc[index_row, "featureSet_features"] = str(list_features_used)
                df_output_metrics_featureSet.loc[index_row, "standardizationMethod"] = str_standardizationMethod
                df_output_metrics_featureSet.loc[index_row, "featureSet_size"] = str(len(list_features_used))
                                
                totalSize = len(list_training_input)
                df_output_metrics_featureSet.loc[index_row, "totalSize"] = str(totalSize)
                            
                df_output_metrics_featureSet.loc[index_row, "clusterLabel"] = str(tuple_labelCount[0])
                df_output_metrics_featureSet.loc[index_row, "clusterSize"] = str(tuple_labelCount[1])            
                df_output_metrics_featureSet.loc[index_row, "clusterProportion"] = str(tuple_labelCount[1]/totalSize)
                
                df_output_raw_allCascades_range.loc[df_output_raw_allCascades_range["clusterLabel"]==tuple_labelCount[0], "clusterSize"] = str(tuple_labelCount[1])
                df_output_raw_allCascades_range.loc[df_output_raw_allCascades_range["clusterLabel"]==tuple_labelCount[0], "clusterProportion"] = str(tuple_labelCount[1]/totalSize)
                
                for feature in (list_features_used + list_features_output_eval):
                
                    print("feature:")
                    print(feature)
                
                    list_values = df_inCluster[feature].tolist()
                    
                    #list_values = list(map(float, list_values))
                    
                    if feature in (["val_bin"] + list_features_output_eval):
                        print("list_values[0:5]:")
                        print(list_values[0:5])
                        print(len(list_values))

                    metric_max = np.max(list_values)
                    metric_min = np.min(list_values)
                    metric_mean = np.mean(list_values)
                    metric_median = np.median(list_values)
                    metric_std = np.std(list_values)
                    metric_rsd = metric_std/metric_mean
                    
                    df_output_metrics_featureSet.loc[index_row, feature+"_max"] = str(metric_max)
                    df_output_metrics_featureSet.loc[index_row, feature+"_min"] = str(metric_min)
                    df_output_metrics_featureSet.loc[index_row, feature+"_mean"] = str(metric_mean)
                    df_output_metrics_featureSet.loc[index_row, feature+"_median"] = str(metric_median)
                    df_output_metrics_featureSet.loc[index_row, feature+"_std"] = str(metric_std)
                    df_output_metrics_featureSet.loc[index_row, feature+"_rsd"] = str(metric_rsd)
                    
                    
                    
                
                index_row += 1
                
                df_output_metrics_featureSet.to_csv(absFilename_output_metrics_featureSet, index=False)
    
            list_df_output_raw += [df_output_raw_allCascades_range.copy()]
        
        print("len(list_df_output_raw):")
        print(len(list_df_output_raw))
                
        df_output_raw_featureSet = pd.concat(list_df_output_raw)
        df_output_raw_featureSet = df_output_raw_featureSet.reset_index(drop=True)
        print("len(df_output_raw_featureSet):")
        print(len(df_output_raw_featureSet))
        
        print("absFilename_output_raw_featureSet:")
        print(absFilename_output_raw_featureSet)
        df_output_raw_featureSet.to_csv(absFilename_output_raw_featureSet, index=False)
        
        
        
if __name__ == "__main__":
    main(sys.argv[1:])