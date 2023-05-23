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
    list_visMetricSets = ["cascade" ,"consumer"]
    
    dict_rootTweetLabel2RootTweetID_true = {"T1":"1268269994074345473", "T2":"1269111354520211458", "T3":"1268155500753027073", "T4":"1269320689003216896", "T5":"1269077273770156032", "T6":"1268346742296252418"}
    dict_rootTweetLabel2RootTweetID_false = {"F1":"1246159197621727234", "F2":"1239336564334960642", "F3":"1240682713817976833", "F4":"1261326944492281859", "F5":"1262482651333738500", "F6":"1247287993095897088", "F7":"1256641587708342274"}
        
    list_metricsToVisualize_cascade = ["cascadeSize", "cascadeDepth", "cascadeVirality"]
    list_metricsToVisualize_consumer = ["mean_followers", "median_followers", "mean_followees", "median_followees", "mean_account_age", "median_account_age", "mean_engagement", "median_engagement"]
    
    dict_rootTweetNumber2Color = {"1":"red", "2":"blue", "3":"limegreen", "4":"black", "5":"pink", "6":"orange", "7":"magenta"}
    
    
    #list_timestamps_24hrs_per10min = sorted(list(set([n for n in range(10, 24*60+10, 10)])))
    #list_timestamps_7days_perHr = sorted(list(set([n for n in range(60, 7*24*60+60, 60)])))
    list_timestamps_24hrs_per10min = sorted(list(set([n for n in range(0, 24*60+10, 10)])))
    list_timestamps_7days_perHr = sorted(list(set([n for n in range(0, 7*24*60+60, 60)])))
    list_timestamps_7days_combined = sorted(list(set(list_timestamps_24hrs_per10min + list_timestamps_7days_perHr)))
    
    list_timeScopes = ["24hrs-per10min", "7days-perHr", "7days-combined"]
    #list_timeScopes = ["24hrs-per10min", "7days-combined"]
    
    list_staMethods = ["NONE", "producerEngagement", "producerFollowers"]
    
    
    list_rootTweetID2ProducerEngagement = {"1246159197621727234":23.599939885782987, "1239336564334960642":4.616879659211928, "1240682713817976833":3.471933471933472, "1261326944492281859":51.262068965517244, "1262482651333738500":11.207036193368767, "1247287993095897088":21.48663594470046, "1256641587708342274":13.501945525291829, "1268269994074345473":14.456136779971505, "1269111354520211458":42.177842565597665, "1268155500753027073":85.84689882546878, "1269320689003216896":64.38544474393531, "1269077273770156032":42.177842565597665, "1268346742296252418":85.84689882546878}
    
    list_rootTweetID2ProducerFollowers = {"1246159197621727234":1760303, "1239336564334960642":908162, "1240682713817976833":4148663, "1261326944492281859":338328, "1262482651333738500":326062, "1247287993095897088":2227254, "1256641587708342274":1521085, "1268269994074345473":57984210, "1269111354520211458":8216691, "1268155500753027073":46806130, "1269320689003216896":17832348, "1269077273770156032":8216691, "1268346742296252418":46806134} 
    
    list_rootTweetLabels_true = sorted(dict_rootTweetLabel2RootTweetID_true.keys())
    list_rootTweetLabels_false = sorted(dict_rootTweetLabel2RootTweetID_false.keys())
    
    print("list_rootTweetLabels_true:")
    print(list_rootTweetLabels_true)
    print("list_rootTweetLabels_false:")
    print(list_rootTweetLabels_false)
    
    
        
    absFilename_input = "C:\\vmwareSharedFolder\\TwitterDataAnalysis\\programs\\FalseNews_Code\\FalseNews_Code\\data\\temporal\\temporalCharacteristics_merged_producerRemoved.csv"
            
    print("absFilename_input:")
    print(absFilename_input)
    
    df_input = pd.read_csv(absFilename_input, dtype=str)

    print("len(df_input):")
    print(len(df_input))
    
    #df_input.replace("None", 0, inplace=True)
    df_input.replace("None", "", inplace=True)
        
    df_input = df_input.apply(pd.to_numeric, errors='ignore')
    
    
    sns.set_context("notebook", font_scale=1.5)               

    
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
            
            if str_visMetricSet == "cascade":   
                list_metricsToVisualize = list_metricsToVisualize_cascade
                                
            elif str_visMetricSet == "consumer":   
                list_metricsToVisualize = list_metricsToVisualize_consumer        
                                             
            for str_metricToVisualize in list_metricsToVisualize:   

                                
                for str_timeScope in list_timeScopes:
                                    
                    print("str_timeScope:")
                    print(str_timeScope)
                
                    if str_timeScope == "24hrs-per10min":
                        list_timeSope_toVisualize = list_timestamps_24hrs_per10min
                    elif str_timeScope == "7days-perHr":
                        list_timeSope_toVisualize = list_timestamps_7days_perHr
                    elif str_timeScope == "7days-combined":
                        list_timeSope_toVisualize = list_timestamps_7days_combined
                            
                    list_rootTweetLabelAndColumnsToVisualize = [(l, l + "_" + str_metricToVisualize) for l in list_rootTweetLabels_toVisualize]
                    
                    print("list_rootTweetLabelAndColumnsToVisualize:")
                    print(list_rootTweetLabelAndColumnsToVisualize)
                    
                    for str_staMethod in list_staMethods:
                    
                        figure, axes = plt.subplots(figsize=(15, 6))                               
                        
                        for str_rootTweetLabel, str_columnToVisualize in list_rootTweetLabelAndColumnsToVisualize:
                    
                            df_input_toVisualize = df_input[["cascadeAge_min"]+[str_columnToVisualize]].copy()
                            df_input_toVisualize = df_input_toVisualize.loc[df_input_toVisualize["cascadeAge_min"].isin(list_timeSope_toVisualize)]
                            
                            print("str_staMethod:")
                            print(str_staMethod)
                            
                            print("str_rootTweetLabel:")
                            print(str_rootTweetLabel)                    
                            
                            if str_rootTweetLabel.startswith("T"):
                                str_rootTweetID = dict_rootTweetLabel2RootTweetID_true[str_rootTweetLabel]
                            elif str_rootTweetLabel.startswith("F"):
                                str_rootTweetID = dict_rootTweetLabel2RootTweetID_false[str_rootTweetLabel]
                                
                            print("str_rootTweetID:")
                            print(str_rootTweetID)
                            
                            if str_staMethod == "NONE":
                                print("Do not standardize the metric values")
                                pass
                            elif str_staMethod == "producerEngagement":
                                print("Standardize metric values using producer engagement")
                                float_staBase = list_rootTweetID2ProducerEngagement[str_rootTweetID]
                                print("float_staBase:")
                                print(float_staBase)
                                
                                if float_staBase == -1:
                                    print("Do not standardize the metric values: base is not collected yet")
                                    pass
                                    
                                df_input_toVisualize[str_columnToVisualize] = df_input_toVisualize[str_columnToVisualize]/float_staBase
                            elif str_staMethod == "producerFollowers":
                                print("Standardize metric values using producer followers")
                                float_staBase = list_rootTweetID2ProducerFollowers[str_rootTweetID]
                                print("float_staBase:")
                                print(float_staBase)
                                
                                if float_staBase == -1:
                                    print("Do not standardize the metric values: base is not collected yet")
                                    pass
                                    
                                df_input_toVisualize[str_columnToVisualize] = df_input_toVisualize[str_columnToVisualize]/float_staBase
                            
                            #print("df_input_toVisualize:")
                            #print(df_input_toVisualize)     
                            print("len(df_input_toVisualize):")     
                            print(len(df_input_toVisualize))   
                                                                                  
                            sns_plot = sns.lineplot(x="cascadeAge_min", y=str_columnToVisualize, data=df_input_toVisualize, ax=axes, label=str_rootTweetLabel)                 
                        
                        
                        """                        
                        if str_timeScope == "24hrs-per10min":
                            list_xticks = [10] + [n for n in range(60, 25*60, 60)]
                            list_xtickLabels = ["0.1"] + [str(n) for n in range(1, 25, 1)]
                            str_xLabel = "(hr)"
                        elif str_timeScope in ["7days-perHr", "7days-combined"]:
                            list_xticks = [10] + [n for n in range(24*60, 8*24*60, 24*60)]
                            list_xtickLabels = ["10 min"] + [str(n) for n in range(1, 8, 1)]
                            str_xLabel = "(day)"  
                        """
                        
                        if str_timeScope == "24hrs-per10min":
                            list_xticks = [n for n in range(0, 25*60, 60)]
                            list_xtickLabels = [str(n) for n in range(0, 25, 1)]
                            str_xLabel = "(hr)"
                        elif str_timeScope in ["7days-perHr", "7days-combined"]:
                            list_xticks = [n for n in range(0, 8*24*60, 24*60)]
                            list_xtickLabels = [str(n) for n in range(0, 8, 1)]
                            str_xLabel = "(day)"
                            
                        sns_plot.set_xticks(list_xticks)
                        sns_plot.set_xticklabels(list_xtickLabels)
                        
                        #str_title = str_metricToVisualize
                        #figure.suptitle(str_title)
                        
                        str_yLabel = ""
                        
                        if str_staMethod != "NONE":
                            str_yLabel = "\n(standardized by " + str_staMethod + ")"
                        
                        sns_plot.set(xlabel="Time since cascade start " + str_xLabel, ylabel=str_metricToVisualize + str_yLabel)
                                                
                        
                        for line in axes.lines:
                        
                            print("line:")
                            print(line)
                            label = str(line.get_label())
                            line.set_color(dict_rootTweetNumber2Color[label[-1]])
                            if label.startswith("T"):
                                line.set_linestyle("--")
                            elif label.startswith("F"):
                                line.set_linestyle("-")                                
                                                        
                        axes.legend(frameon=False, loc="best", ncol=2)
                                                                
                        # Draw a nested boxplot to show bills by day and time
                        #sns.despine(offset=10, trim=True)  
                                              
                                               
                        absFilename_output_figure = "C:\\vmwareSharedFolder\\TwitterDataAnalysis\\programs\\FalseNews_Code\\FalseNews_Code\\figures\\temporal\\lineplot" + "_range=" + str_range + "_metric=" + str_metricToVisualize + "_timeScope=" + str_timeScope + "_standardizationBase=" + str_staMethod +".png"
                            
                        if not os.path.exists(os.path.dirname(absFilename_output_figure)):
                            os.makedirs(os.path.dirname(absFilename_output_figure))
                            
                        print("absFilename_output_figure:")
                        print(absFilename_output_figure)
                                
                        figure.savefig(absFilename_output_figure)
                        figure.clf()  
                        plt.close('all')
                
            
        
    
    
if __name__ == "__main__":
    main(sys.argv[1:])