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
from sklearn.cluster import KMeans

def main(argv):
    random.seed(1113)

    # pd.set_option('display.height', 1000)
    pd.set_option('display.max_rows', 1000)
    pd.set_option('display.max_columns', 1000)
    pd.set_option('display.width', 300)

    absFilename_input = "C:\\vmwareSharedFolder\\TwitterDataAnalysis\\results\\preprocessedData.csv"
    absFilename_output = "C:\\vmwareSharedFolder\\TwitterDataAnalysis\\results\\preprocessedData_clusteredMeasures.csv"


    # str_measureName = "SpreaderPersona_retweeters_activeness_cascadeEnd"
    # # list_featuresForMeasure = ["CASCADEEND_RETWEETS_retweet_median_user_accountAge_day", "CASCADEEND_RETWEETS_retweet_median_user_followersCount", "CASCADEEND_RETWEETS_retweet_median_user_friendsCount", "CASCADEEND_RETWEETS_retweet_median_user_favouritesCount", "CASCADEEND_RETWEETS_retweet_median_user_statusesCount"]
    # # list_featuresForMeasure = ["CASCADEEND_RETWEETS_retweet_median_user_followersCount", "CASCADEEND_RETWEETS_retweet_median_user_friendsCount", "CASCADEEND_RETWEETS_retweet_median_user_favouritesCount", "CASCADEEND_RETWEETS_retweet_median_user_statusesCount"]
    # list_featuresForMeasure = ["CASCADEEND_RETWEETS_retweet_mean_user_followersCount", "CASCADEEND_RETWEETS_retweet_mean_user_friendsCount"]

    # # str_measureName = "SpreaderPersona_retweeters_sentEmo_cascadeEnd"
    # str_measureName = "SpreaderPersona_retweeters_test"
    # # list_featuresForMeasure = ["CASCADEEND_RETWEETS_retweet_median_user_description_sentiment_pos", "CASCADEEND_RETWEETS_retweet_median_user_description_subjectivity", "CASCADEEND_RETWEETS_retweet_median_user_description_emotion_trust", "CASCADEEND_RETWEETS_retweet_median_user_description_emotion_joy", "CASCADEEND_RETWEETS_retweet_median_user_description_sentiment_neg", "CASCADEEND_RETWEETS_retweet_median_user_description_emotion_fear", "CASCADEEND_RETWEETS_retweet_median_user_description_emotion_anger", "CASCADEEND_RETWEETS_retweet_median_user_description_emotion_sadness", "CASCADEEND_RETWEETS_retweet_median_user_description_emotion_surprise", "CASCADEEND_RETWEETS_retweet_median_user_description_emotion_disgust"]
    # # list_featuresForMeasure = ["CASCADEEND_RETWEETS_retweet_median_user_description_sentiment_pos", "CASCADEEND_RETWEETS_retweet_median_user_description_emotion_trust", "CASCADEEND_RETWEETS_retweet_median_user_description_emotion_joy", "CASCADEEND_RETWEETS_retweet_median_user_description_sentiment_neg", "CASCADEEND_RETWEETS_retweet_median_user_description_emotion_fear", "CASCADEEND_RETWEETS_retweet_median_user_description_emotion_anger", "CASCADEEND_RETWEETS_retweet_median_user_description_emotion_sadness", "CASCADEEND_RETWEETS_retweet_median_user_description_emotion_surprise", "CASCADEEND_RETWEETS_retweet_median_user_description_emotion_disgust"]
    # # list_featuresForMeasure = ["CASCADEEND_RETWEETS_retweet_median_user_description_sentiment_pos", "CASCADEEND_RETWEETS_retweet_median_user_description_sentiment_neg", "CASCADEEND_RETWEETS_retweet_median_user_description_emotion_anger"]
    # list_featuresForMeasure = ["CASCADEEND_RETWEETS_retweet_median_user_description_sentiment_pos", "CASCADEEND_RETWEETS_retweet_median_user_description_sentiment_neg"]

    str_measureName = "SpreaderPersona_retweeters_test"
    list_featuresForMeasure = ["CASCADEEND_RETWEETS_retweet_median_user_statusesCount", "TIMESERIES_RETWEETS_user_statusesCountDASHstl_linearity_allTimeStamps"]

    rootTweetGroup_feature = "ROOTTWEETS_veracityLabel_agg_misinformation"
    rootTweetGroup_value = "1"

    print("rootTweetGroup_feature:")
    print(rootTweetGroup_feature)
    print("rootTweetGroup_value:")
    print(rootTweetGroup_value)

    str_rootTweetGp = ""

    if rootTweetGroup_feature=="ROOTTWEETS_veracityLabel_agg_misinformation" and rootTweetGroup_value==1:
    	str_rootTweetGp = "misinformation"

    str_standardizationMethod = "MaxAbsScaler"
    # str_standardizationMethod = "EMPTY"

    df_input = pd.read_csv(absFilename_input, dtype=str)

    df_input["index"] = df_input.index

    print("len(df_input):")
    print(len(df_input))


    df_input_toCluster = df_input.loc[df_input[rootTweetGroup_feature]==rootTweetGroup_value, list_featuresForMeasure+["index"]].copy()

    print("df_input_toCluster.columns:")
    print(df_input_toCluster.columns)
    print("len(df_input_toCluster):")
    print(len(df_input_toCluster))

    df_input_toCluster[list_featuresForMeasure] = df_input_toCluster[list_featuresForMeasure].astype(float)
    df_input_toCluster = df_input_toCluster[~df_input_toCluster.isin([np.nan, np.inf, -np.inf]).any(1)]
    list_training_input = df_input_toCluster[list_featuresForMeasure].values.tolist()

    print("before normalization:")
    print("list_training_input[0:5]:")
    print(list_training_input[0:5])
    print("list_training_input[-5:]:")
    print(list_training_input[-5:])
    print("len(list_training_input):")
    print(len(list_training_input))

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

    print("after normalization:")
    print("list_training_input[0:5]:")
    print(list_training_input[0:5])
    print("list_training_input[-5:]:")
    print(list_training_input[-5:])
    print("len(list_training_input):")
    print(len(list_training_input))


    # bandwidth = estimate_bandwidth(list_training_input, n_jobs=5)
    # model = MeanShift(bandwidth=bandwidth)
    # print("fit model:")

    # model.fit(list_training_input)

    model = KMeans(n_clusters=2, init='k-means++', max_iter=500, n_init=10, random_state=0)
    model.fit_predict(list_training_input)

    print("model:")
    print(model)

    list_labels = model.labels_
    #list_custerCenters = model.cluster_centers_

    # print("list_labels:")
    # print(list_labels)
    print("Counter(list_labels):")
    print(Counter(list_labels))

    df_input_toCluster["measure_" + str_measureName] = list_labels

    # print("df_input_toCluster[[\"index\", \"clusterLabel\"]]:")
    # print(df_input_toCluster[["index", "clusterLabel"]])
    print("len(df_input_toCluster):")
    print(len(df_input_toCluster))

    df_output = df_input.copy()
    df_output = pd.merge(left=df_output, right=df_input_toCluster, how="left", on="index")

    print("len(df_output):")
    print(len(df_output))

    # print(df_output[["index", rootTweetGroup_feature, "measure_" + str_measureName]])


    print("absFilename_output:")
    print(absFilename_output)

    df_output.to_csv(absFilename_output, index=False)


if __name__ == "__main__":
    main(sys.argv[1:])