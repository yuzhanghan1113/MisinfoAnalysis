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

from matplotlib import pyplot as plt
from scipy.cluster.hierarchy import dendrogram 

"""
num_rows = 20
years = list(range(1990, 1990 + num_rows))
df_input_toVisualize = pd.DataFrame({
    'Year': years, 
    'A': np.random.randn(num_rows).cumsum(),
    'B': np.random.randn(num_rows).cumsum(),
    'C': np.random.randn(num_rows).cumsum(),
    'D': np.random.randn(num_rows).cumsum()})
    
print("df_input_toVisualize:")
print(df_input_toVisualize)

df_input_toVisualize_melt = pd.melt(df_input_toVisualize, ['Year'])

print("df_input_toVisualize_melt:")
print(df_input_toVisualize_melt)
    
sns.lineplot(x='Year', y='value', hue='variable', data=df_input_toVisualize_melt)
plt.show()
""" 







def main(argv):
    random.seed(1113)

    # pd.set_option('display.height', 1000)
    pd.set_option('display.max_rows', 99999)
    pd.set_option('display.max_columns', 99999)
    # pd.set_option('display.width', 170)
    
    list_ranges = ["falseCascades", "trueCascades", "allCascades"]                
    #list_visMetricSets = ["cascade" ,"consumer"]
    list_visMetricSets = ["cascade"]
    
    list_rootTweetLabels_true = ["T1", "T2", "T3", "T4", "T5", "T6"]
    list_rootTweetLabels_false = ["F1", "F2", "F3", "F4", "F5", "F6", "F7"]
    list_metricsToVisualize_cascade = ["cascadeSize", "cascadeDepth", "cascadeVirality"]
    list_metricsToVisualize_consumer = ["followers", "followers", "followees", "followees", "account_age", "engagement"]
    
    
    list_timestamps_24hrs_per10min = sorted(list(set([n for n in range(10, 26*60+10, 10)])))
    list_timestamps_7days_perHr = sorted(list(set([n for n in range(60, 7*24*60+60, 60)])))
    list_timestamps_7days_combined = sorted(list(set(list_timestamps_24hrs_per10min + list_timestamps_7days_perHr)))
    
    list_timeScopes = ["24hrs-per10min", "7days-perHr", "7days-combined"]
        
    absFilename_input = "C:\\vmwareSharedFolder\\TwitterDataAnalysis\\programs\\FalseNews_Code\\FalseNews_Code\\data\\temporal\\temporalCharacteristics_merged.csv"
            
    print("absFilename_input:")
    print(absFilename_input)
    
    df_input = pd.read_csv(absFilename_input, dtype=str)

    print("len(df_input):")
    print(len(df_input))
    
    #df_input.replace("None", 0, inplace=True)
    df_input.replace("None", "", inplace=True)
        
    df_input = df_input.apply(pd.to_numeric, errors='ignore')
    
    
    
    
    for str_range in list_ranges:
    
        if str_range == "trueCascades":
            list_rootTweetLabels_toVisualize = list_rootTweetLabels_true
        elif str_range == "falseCascades":
            list_rootTweetLabels_toVisualize = list_rootTweetLabels_false
        elif str_range == "allCascades":
            list_rootTweetLabels_toVisualize = list_rootTweetLabels_true+list_rootTweetLabels_false
            
        for str_visMetricSet in list_visMetricSets:  

            print("str_range:")
            print(str_range)
            print("str_visMetricSet:")
            print(str_visMetricSet)               
            
            
            """
            str_title = "data range: " + str_range    
            if str_rootTweetLabel != "":
                str_title += ", root tweet label: " + str_rootTweetLabel
            str_title += ", training feature set: " + str_featureSet
            str_title += "\nvisualized clusters: " + str(Counter(df_input_toVisualize["clusterLabel"].tolist()))
            """
            """
            str_title = ""
            str_title = str_range
            str_title += " clusters: "
            for label in sorted(dict_clusterLabel2ClusterSize.keys()):
                str_title += str(label) + ": " + str(dict_clusterLabel2ClusterSize[label]) + " (" + str(dict_clusterLabel2ClusterProportion[label]*100) + "%) "
            """              
            if str_visMetricSet == "cascade":   
                list_metricsToVisualize = list_metricsToVisualize_cascade
                maxNum_rows = 1
                maxNum_columns = 3
                
            elif str_visMetricSet == "consumer":   
                list_metricsToVisualize = list_metricsToVisualize_consumer        
                maxNum_rows = 2
                maxNum_columns = 3
                
                
                        
            
            
            if str_visMetricSet == "cascade": 
            
                sns.set_context("notebook", font_scale=0.8)

                #f, axes = plt.subplots(maxNum_rows, maxNum_columns, figsize=(10, 10))
                #f, axes = plt.subplots(maxNum_rows, maxNum_columns, sharex=True)
                #sns.despine(left=True)  
                
                coordinate_x = 0
                coordinate_y = 0
                
                #f.suptitle(str_title, fontsize=8)
                
                for str_metricToVisualize in list_metricsToVisualize:   

                    str_title = str_metricToVisualize
                    
                    for str_timeScope in list_timeScopes:
                    
                        print("str_timeScope:")
                        print(str_timeScope)
                    
                        if str_timeScope == "24hrs-per10min":
                            list_timeSope_toVisualize = list_timestamps_24hrs_per10min
                        elif str_timeScope == "7days-perHr":
                            list_timeSope_toVisualize = list_timestamps_7days_perHr
                        elif str_timeScope == "7days-combined":
                            list_timeSope_toVisualize = list_timestamps_7days_combined
                                
                        list_columnsToVisualize = [l + "_" + str_metricToVisualize for l in list_rootTweetLabels_toVisualize]
                        
                        print("list_columnsToVisualize:")
                        print(list_columnsToVisualize)
                    
                        df_input_toVisualize = df_input[["cascadeAge_min"]+list_columnsToVisualize].copy()
                        df_input_toVisualize = df_input_toVisualize[df_input_toVisualize["cascadeAge_min"].isin(list_timeSope_toVisualize)]
                        df_input_toVisualize_melt = pd.melt(df_input_toVisualize, ["cascadeAge_min"])
                        df_input_toVisualize_melt = df_input_toVisualize_melt.sort_values(by=["variable", "cascadeAge_min"])
                        df_input_toVisualize_melt["rootTweetVeracity"] = df_input_toVisualize_melt["variable"].str.slice(0, 1)
                        df_input_toVisualize_melt["rootTweetNumber"] = df_input_toVisualize_melt["variable"].str.slice(1, 2)

                        print("df_input_toVisualize_melt:")
                        print(df_input_toVisualize_melt)
                                                
                        figure, axes = plt.subplots(figsize=(15, 6))
                        
                        
                            
                        #sns_plot = sns.lineplot(x="cascadeAge_min", y="value", data=df_input_toVisualize_melt, hue="variable", legend=False, ax=axes[coordinate_y])
                        sns_plot = sns.lineplot(x="cascadeAge_min", y="value", data=df_input_toVisualize_melt, hue="rootTweetNumber", style="rootTweetVeracity", legend="full", ax=axes, palette="Set1")
                        
                        axes.legend(bbox_to_anchor=(1.01, 1), loc=2, borderaxespad=0.)
                        
                        leg = axes.legend();
                        
                        sns_plot.set_title(str_title)
                         
                    
                        #plt.figure(figsize=(0.1, 0.1))
                    
                        # Draw a nested boxplot to show bills by day and time
                        #sns.despine(offset=10, trim=True)  
                        
                        absFilename_output_figure = "C:\\vmwareSharedFolder\\TwitterDataAnalysis\\programs\\FalseNews_Code\\FalseNews_Code\\figures\\temporal\\lineplot" + "_range=" + str_range + "_metric=" + str_metricToVisualize + "_statistics=" + "_timeScope=" + str_timeScope + ".png"
                            
                        if not os.path.exists(os.path.dirname(absFilename_output_figure)):
                            os.makedirs(os.path.dirname(absFilename_output_figure))
                            
                        print("absFilename_output_figure:")
                        print(absFilename_output_figure)
                                
                        #figure = sns_plot.get_figure()    
                        figure.savefig(absFilename_output_figure)
                        figure.clf()
                      
                        
                        if coordinate_x == 0:
                            sns_plot.set(xlabel=None)
                    
                        if coordinate_y >= maxNum_columns-1:
                            coordinate_y = 0
                            if coordinate_x < maxNum_rows-1:
                                coordinate_x += 1
                        else:
                            coordinate_y += 1
                        
                #plt.setp(axes, yticks=[])
                plt.tight_layout()
                """
                absFilename_output_figure = "C:\\vmwareSharedFolder\\TwitterDataAnalysis\\programs\\FalseNews_Code\\FalseNews_Code\\figures\\temporal\\lineplot" + "_range=" + str_range + "_metricSet=" + str_visMetricSet + "_statistics=.png"
                        
                if not os.path.exists(os.path.dirname(absFilename_output_figure)):
                    os.makedirs(os.path.dirname(absFilename_output_figure))
                    
                print("absFilename_output_figure:")
                print(absFilename_output_figure)
                        
                f.savefig(absFilename_output_figure)
                f.clf()
                """
                
            elif str_visMetricSet == "consumer":

                for str_statistics in ["mean", "median"]:
                                                    
                    print("str_statistics:")
                    print(str_statistics)
                    
                    #f, axes = plt.subplots(maxNum_rows, maxNum_columns, figsize=(10, 10))
                    #f, axes = plt.subplots(maxNum_rows, maxNum_columns, sharex=True)
                    #sns.despine(left=True)               
                    
                    coordinate_x = 0
                    coordinate_y = 0
                
                    for str_metricToVisualize in list_metricsToVisualize:         

                        str_title = str_metricToVisualize + " " + str_statistics
                        
                        for str_timeScope in list_timeScopes:
                        
                            print("str_timeScope:")
                            print(str_timeScope)
                        
                            if str_timeScope == "24hrs-per10min":
                                list_timeSope_toVisualize = list_timestamps_24hrs_per10min
                            elif str_timeScope == "7days-perHr":
                                list_timeSope_toVisualize = list_timestamps_7days_perHr
                            elif str_timeScope == "7days-combined":
                                list_timeSope_toVisualize = list_timestamps_7days_combined
                                    
                            list_columnsToVisualize = [l + "_" + str_statistics + "_" + str_metricToVisualize for l in list_rootTweetLabels_toVisualize]
                            
                            print("list_columnsToVisualize:")
                            print(list_columnsToVisualize)
                        
                            df_input_toVisualize = df_input[["cascadeAge_min"]+list_columnsToVisualize].copy()
                            df_input_toVisualize = df_input_toVisualize[df_input_toVisualize["cascadeAge_min"].isin(list_timeSope_toVisualize)]
                            df_input_toVisualize_melt = pd.melt(df_input_toVisualize, ["cascadeAge_min"])
                            df_input_toVisualize_melt = df_input_toVisualize_melt.sort_values(by=["variable", "cascadeAge_min"])
                            df_input_toVisualize_melt["veracity"] = df_input_toVisualize_melt["variable"].str.slice(0, 1)
                            
                            print("df_input_toVisualize_melt:")
                            print(df_input_toVisualize_melt)
                                
                            #sns_plot = sns.lineplot(x="cascadeAge_min", y="value", hue="variable", data=df_input_toVisualize_melt, legend=False, ax=axes[coordinate_x, coordinate_y])
                            
                            sns_plot = sns.lineplot(x="cascadeAge_min", y="value", hue="variable", data=df_input_toVisualize_melt, legend="full")
                            
                            plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
                             
                            sns_plot.set_title(str_title)
                            sns_plot.set(yscale="log")
                        
                            #plt.figure(figsize=(0.1, 0.1))
                        
                            # Draw a nested boxplot to show bills by day and time
                            #sns.despine(offset=10, trim=True)  
                            
                            absFilename_output_figure = "C:\\vmwareSharedFolder\\TwitterDataAnalysis\\programs\\FalseNews_Code\\FalseNews_Code\\figures\\temporal\\lineplot" + "_range=" + str_range + "_metric=" + str_metricToVisualize + "_statistics=" + str_statistics + "_timeScope=" + str_timeScope + ".png"
                                
                            if not os.path.exists(os.path.dirname(absFilename_output_figure)):
                                os.makedirs(os.path.dirname(absFilename_output_figure))
                                
                            print("absFilename_output_figure:")
                            print(absFilename_output_figure)
                            
                            figure = sns_plot.get_figure()    
                            figure.savefig(absFilename_output_figure)
                            figure.clf()
                                
                         
                      
                        
                        if coordinate_x == 0:
                            sns_plot.set(xlabel=None)
                    
                        if coordinate_y >= maxNum_columns-1:
                            coordinate_y = 0
                            if coordinate_x < maxNum_rows-1:
                                coordinate_x += 1
                        else:
                            coordinate_y += 1
                            
                    #plt.setp(axes, yticks=[])
                    plt.tight_layout()
                    
                    """
                    absFilename_output_figure = "C:\\vmwareSharedFolder\\TwitterDataAnalysis\\programs\\FalseNews_Code\\FalseNews_Code\\figures\\temporal\\lineplot" + "_range=" + str_range + "_metricSet=" + str_visMetricSet + "_statistics=" + str_statistics + ".png"
                            
                    if not os.path.exists(os.path.dirname(absFilename_output_figure)):
                        os.makedirs(os.path.dirname(absFilename_output_figure))
                        
                    print("absFilename_output_figure:")
                    print(absFilename_output_figure)
                            
                    f.savefig(absFilename_output_figure)
                    f.clf()    
                    """
        
    
    
if __name__ == "__main__":
    main(sys.argv[1:])