import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

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

import numpy as np
from collections import Counter
import numpy as np

from collections import OrderedDict
from operator import itemgetter

from matplotlib import pyplot as plt
from scipy.cluster.hierarchy import dendrogram

def main(argv):
    random.seed(1113)

    # pd.set_option('display.height', 1000)
    pd.set_option('display.max_rows', 1000)
    pd.set_option('display.max_columns', 1000)
    # pd.set_option('display.width', 170)
    
    list_ranges = ["falseCascades", "trueCascades", "allCascades"]
    list_featureSets = ["engagement", "emotion", "engagementAndEmotion"]

    list_features_engagement = ["account_age", "followers_count", "friends_count", "listed_count", "favourites_count", "statuses_count"]
            
    #list_features_emotion = ["fear", "anger", "anticip", "trust", "surprise", "positive", "negative", "sadness", "disgust", "joy", "neg", "neu", "pos", "compound"] 
    list_features_emotion = ["fear", "anger", "anticip", "trust", "surprise", "sadness", "disgust", "joy", "neg", "neu", "pos", "compound"] 
    
    list_rootTweetLabels = ["F1", "F2", "F3", "F4", "F5", "F6", "F7", "T1", "T2", "T3", "T4", "T5", "T6"]
    
    str_rootTweetLabel = ""
    
    for str_range in list_ranges:
        for str_featureSet in list_featureSets:  

            print("str_range:")
            print(str_range)
            print("str_featureSet:")
            print(str_featureSet)
            print("str_rootTweetLabel:")
            print(str_rootTweetLabel)        
             
            absFilename_input = "C:\\vmwareSharedFolder\\TwitterDataAnalysis\\results\\clustering\\clusteringResults_raw_featureSet=" + str_featureSet + ".csv"
            
            print("absFilename_input:")
            print(absFilename_input)
            
            df_input = pd.read_csv(absFilename_input)

            print("len(df_input):")
            print(len(df_input))
            
            #df_input["clusterProportion"] = df_input["clusterProportion"].astype(float).round(4) 
            df_input["clusterProportion"] = df_input["clusterProportion"].astype(float) 
            df_input["clusterProportion"] = df_input["clusterProportion"].apply(lambda x: round(x, 4))
            
            df_input_toVisualize = df_input[(df_input["range"]==str_range) & (df_input["clusterProportion"]>=0.15)]
            df_input_toVisualize = df_input_toVisualize.reset_index(drop=True)
            
            print("len(df_input_toVisualize):")
            print(len(df_input_toVisualize))
            
            print(Counter(df_input_toVisualize["range"].tolist()))
            print(Counter(df_input_toVisualize["clusterLabel"].tolist()))
            print(Counter(df_input_toVisualize["clusterProportion"].tolist()))
            print(Counter(df_input_toVisualize["rootTweetVeracity"].tolist()))
            print(Counter(df_input_toVisualize["rootTweetLabel"].tolist()))
           
            dict_clusterLabel2ClusterSize = dict(Counter(df_input_toVisualize["clusterLabel"].tolist()))
            print("dict_clusterLabel2ClusterSize:")
            print(dict_clusterLabel2ClusterSize)
            
            dict_clusterLabel2ClusterProportion = dict()
            
            list_clusterLabels = dict_clusterLabel2ClusterSize.keys()
            for str_clusterLabel in list_clusterLabels:
                float_proportion = df_input_toVisualize.loc[df_input_toVisualize["clusterLabel"]==str_clusterLabel, "clusterProportion"].reset_index(drop=True)[0]
                #print("float_proportion:")
                #print(float_proportion)
                dict_clusterLabel2ClusterProportion[str_clusterLabel] = float_proportion
            
            print("dict_clusterLabel2ClusterProportion:")
            print(dict_clusterLabel2ClusterProportion)
            
            
            for str_visFeatureSet in ["engagement", "emotion"]:
                """
                str_title = "data range: " + str_range    
                if str_rootTweetLabel != "":
                    str_title += ", root tweet label: " + str_rootTweetLabel
                str_title += ", training feature set: " + str_featureSet
                str_title += "\nvisualized clusters: " + str(Counter(df_input_toVisualize["clusterLabel"].tolist()))
                """
                str_title = ""
                str_title = str_range
                str_title += "\nClusters: "
                for label in sorted(dict_clusterLabel2ClusterSize.keys()):
                    str_title += str(label) + ": " + str(dict_clusterLabel2ClusterSize[label]) + " (" + str(round(dict_clusterLabel2ClusterProportion[label]*100, 2)) + "%) "
                              
                if str_visFeatureSet == "engagement":    
                    maxNum_rows = 2
                    maxNum_columns = 3
                    list_features_toVisualize = list_features_engagement
                elif str_visFeatureSet == "emotion":    
                    maxNum_rows = 3
                    maxNum_columns = 4
                    list_features_toVisualize = list_features_emotion
                    
                #sns.set_context("notebook", font_scale=0.8)
                sns.set_context("notebook")
                
                #f, axes = plt.subplots(maxNum_rows, maxNum_columns, figsize=(10, 10))
                f, axes = plt.subplots(maxNum_rows, maxNum_columns, sharex="all")
                sns.despine(left=True)
                
                
                
                coordinate_x = 0
                coordinate_y = 0
                
                f.suptitle(str_title, fontsize=12)
                    
                for feature_y in list_features_toVisualize:
                
                    #plt.figure(figsize=(0.1, 0.1))
                
                    # Draw a nested boxplot to show bills by day and time
                    sns_plot = sns.boxplot(x="clusterLabel", y=feature_y, data=df_input_toVisualize, showfliers=False, showmeans=True, ax=axes[coordinate_x, coordinate_y])
                    #sns.despine(offset=10, trim=True)  

                        
                    
                    if coordinate_x == 0:
                        sns_plot.set(xlabel=None)
                
                    if coordinate_y >= maxNum_columns-1:
                        coordinate_y = 0
                        if coordinate_x < maxNum_rows-1:
                            coordinate_x += 1
                    else:
                        coordinate_y += 1
                        
                #plt.setp(axes, yticks=[])
                f.tight_layout()
                f.subplots_adjust(top=0.85)

                
                absFilename_output_figure = "C:\\vmwareSharedFolder\\TwitterDataAnalysis\\results\\clustering\\figures\\boxplot" + "_range=" + str_range + "_cascadeLabel=" + str_rootTweetLabel + "_featureSet=" + str_featureSet + "_visFeaturesSet=" + str_visFeatureSet + ".png"
                        
                if not os.path.exists(os.path.dirname(absFilename_output_figure)):
                    os.makedirs(os.path.dirname(absFilename_output_figure))
                    
                print("absFilename_output_figure:")
                print(absFilename_output_figure)
                        
                f.savefig(absFilename_output_figure)
                f.clf()
                
                
    str_range = "singleCascade"
    
    for str_featureSet in list_featureSets:
        for str_rootTweetLabel in list_rootTweetLabels:
          

            print("str_range:")
            print(str_range)
            print("str_featureSet:")
            print(str_featureSet)
            print("str_rootTweetLabel:")
            print(str_rootTweetLabel)        
             
            absFilename_input = "C:\\vmwareSharedFolder\\TwitterDataAnalysis\\results\\clustering\\clusteringResults_raw_featureSet=" + str_featureSet + ".csv"
            
            print("absFilename_input:")
            print(absFilename_input)
            
            df_input = pd.read_csv(absFilename_input)

            print("len(df_input):")
            print(len(df_input))
            
            #df_input["clusterProportion"] = df_input["clusterProportion"].astype(float).round(4) 
            df_input["clusterProportion"] = df_input["clusterProportion"].astype(float) 
            df_input["clusterProportion"] = df_input["clusterProportion"].apply(lambda x: round(x, 4))
            
            df_input_toVisualize = df_input[(df_input["range"]==str_range) & (df_input["rootTweetLabel"]==str_rootTweetLabel) & (df_input["clusterProportion"]>=0.15)]
            
            print("len(df_input_toVisualize):")
            print(len(df_input_toVisualize))
            
            #df_input_toVisualize = df_input_toVisualize.assign(labelFreq=df_input_toVisualize.groupby("clusterLabel")["clusterLabel"].transform("count")).sort_values(by=["labelFreq", "clusterLabel"],ascending=[False,True])
            
            #print("len(df_input_toVisualize):")
            #print(len(df_input_toVisualize))
            
            print(Counter(df_input_toVisualize["range"].tolist()))
            print(Counter(df_input_toVisualize["clusterLabel"].tolist()))
            print(Counter(df_input_toVisualize["clusterProportion"].tolist()))
            print(Counter(df_input_toVisualize["rootTweetVeracity"].tolist()))
            print(Counter(df_input_toVisualize["rootTweetLabel"].tolist()))
            
            dict_clusterLabel2ClusterSize = dict(Counter(df_input_toVisualize["clusterLabel"].tolist()))
            print("dict_clusterLabel2ClusterSize:")
            print(dict_clusterLabel2ClusterSize)
            
            dict_clusterLabel2ClusterProportion = dict()
            
            list_clusterLabels = dict_clusterLabel2ClusterSize.keys()
            for str_clusterLabel in list_clusterLabels:
                float_proportion = df_input_toVisualize.loc[df_input_toVisualize["clusterLabel"]==str_clusterLabel, "clusterProportion"].reset_index(drop=True)[0]
                #print("float_proportion:")
                #print(float_proportion)
                dict_clusterLabel2ClusterProportion[str_clusterLabel] = float_proportion
            
            print("dict_clusterLabel2ClusterProportion:")
            print(dict_clusterLabel2ClusterProportion)
           
            
            for str_visFeatureSet in ["engagement", "emotion"]:
            
                """
                str_title = "data range: " + str_range    
                if str_rootTweetLabel != "":
                    str_title += ", root tweet label: " + str_rootTweetLabel
                str_title += ", training feature set: " + str_featureSet
                str_title += "\nvisualized clusters: " + str(Counter(df_input_toVisualize["clusterLabel"].tolist()))
                """
                str_title = ""
                str_title = str_rootTweetLabel
                str_title += "\nClusters: "
                #for label in sorted(dict_clusterLabel2ClusterSize.keys()):
                list_clusterLabels_ordered = []
                for label in OrderedDict(sorted(dict_clusterLabel2ClusterSize.items(), key = itemgetter(1), reverse = True)).keys():
                    str_title += str(label) + ": " + str(dict_clusterLabel2ClusterSize[label]) + " (" + str(round(dict_clusterLabel2ClusterProportion[label]*100, 2)) + "%) "
                    list_clusterLabels_ordered += [label]
                
                if str_visFeatureSet == "engagement":    
                    maxNum_rows = 2
                    maxNum_columns = 3
                    list_features_toVisualize = list_features_engagement
                elif str_visFeatureSet == "emotion":    
                    maxNum_rows = 3
                    maxNum_columns = 4
                    list_features_toVisualize = list_features_emotion
                    
                #sns.set_context("notebook", font_scale=0.8)
                sns.set_context("notebook")
                
                #f, axes = plt.subplots(maxNum_rows, maxNum_columns, figsize=(10, 10))
                f, axes = plt.subplots(maxNum_rows, maxNum_columns, sharex="all")
                sns.despine(left=True)
                
                
                
                coordinate_x = 0
                coordinate_y = 0
                
                f.suptitle(str_title, fontsize=12)
                                    
                for feature_y in list_features_toVisualize:
                
                    #plt.figure(figsize=(0.1, 0.1))
                
                    # Draw a nested boxplot to show bills by day and time
                    sns_plot = sns.boxplot(x="clusterLabel", y=feature_y, data=df_input_toVisualize, order=list_clusterLabels_ordered, showfliers=False, showmeans=True, ax=axes[coordinate_x, coordinate_y])
                    #sns.despine(offset=10, trim=True)  

                        
                    
                    if coordinate_x == 0:
                        sns_plot.set(xlabel=None)
                
                    if coordinate_y >= maxNum_columns-1:
                        coordinate_y = 0
                        if coordinate_x < maxNum_rows-1:
                            coordinate_x += 1
                    else:
                        coordinate_y += 1
                        
                #plt.setp(axes, yticks=[])
                f.tight_layout()
                f.subplots_adjust(top=0.85)
                                
                absFilename_output_figure = "C:\\vmwareSharedFolder\\TwitterDataAnalysis\\results\\clustering\\figures\\boxplot" + "_range=" + str_range + "_cascadeLabel=" + str_rootTweetLabel + "_featureSet=" + str_featureSet + "_visFeaturesSet=" + str_visFeatureSet + ".png"
                        
                if not os.path.exists(os.path.dirname(absFilename_output_figure)):
                    os.makedirs(os.path.dirname(absFilename_output_figure))
                    
                print("absFilename_output_figure:")
                print(absFilename_output_figure)
                        
                f.savefig(absFilename_output_figure)
                f.clf()
        
    
    
if __name__ == "__main__":
    main(sys.argv[1:])