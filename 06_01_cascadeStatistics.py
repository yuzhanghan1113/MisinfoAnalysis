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
    
    dict_rootTweetLabel2RootTweetID = {"T1":"1268269994074345473", "T2":"1269111354520211458", "T3":"1268155500753027073", "T4":"1269320689003216896", "T5":"1269077273770156032", "T6":"1268346742296252418", "F1":"1246159197621727234", "F2":"1239336564334960642", "F3":"1240682713817976833", "F4":"1261326944492281859", "F5":"1262482651333738500", "F6":"1247287993095897088", "F7":"1256641587708342274"}
        
    list_metricsToVisualize_cascade = ["cascadeSize", "cascadeDepth", "cascadeVirality"]
    list_metricsToVisualize_consumer = ["mean_followers", "median_followers", "mean_followees", "median_followees", "mean_account_age", "median_account_age", "mean_engagement", "median_engagement"]
    
    list_metricsToVisualize = list_metricsToVisualize_cascade + list_metricsToVisualize_consumer
    
    list_timeScopes = ["24hrs", "7days"]
    
        
    list_rootTweetLabels = sorted(dict_rootTweetLabel2RootTweetID.keys())
    
    print("list_rootTweetLabels:")
    print(list_rootTweetLabels)
    
    
        
    absFilename_input = "C:\\vmwareSharedFolder\\TwitterDataAnalysis\\programs\\FalseNews_Code\\FalseNews_Code\\data\\temporal\\temporalCharacteristics_merged_producerRemoved.csv"
            
    print("absFilename_input:")
    print(absFilename_input)
    
        
    df_input = pd.read_csv(absFilename_input, dtype=str)

    print("len(df_input):")
    print(len(df_input))
    
    #df_input.replace("None", 0, inplace=True)
    #df_input.replace("None", "", inplace=True)
        
    df_input = df_input.apply(pd.to_numeric, errors='coerce')
    #df_input["cascadeAge_min"] = df_input["cascadeAge_min"].apply(pd.to_numeric, errors='ignore')
    
    df_input = df_input.sort_values(by=["cascadeAge_min"])
    df_input = df_input.reset_index(drop=True)
    
    print("len(df_input):")
    print(len(df_input))
    
    df_output = pd.DataFrame()
    
    index_row = 0
                    
    for str_rootTweetLabel in list_rootTweetLabels:
    
        df_output.loc[index_row, "rootTweetLabel"] = str_rootTweetLabel
        df_output.loc[index_row, "rootTweetID"] = dict_rootTweetLabel2RootTweetID[str_rootTweetLabel]
        
        for str_timeScope in list_timeScopes:                                
                    
            if str_timeScope == "24hrs":
                int_cascadeAge_min_toMeasure = 60*24
            elif str_timeScope == "7days":
                int_cascadeAge_min_toMeasure = 60*24*7
                
            df_output.loc[index_row, "timeMeasuredAt_" + str_timeScope] = str(int_cascadeAge_min_toMeasure)
                       
            for str_metricToVisualize in list_metricsToVisualize:
                           
                str_metricToVisualize_rootTweetLabel = str_rootTweetLabel + "_" + str_metricToVisualize
                
                print("str_rootTweetLabel:")
                print(str_rootTweetLabel) 
                print("str_metricToVisualize:")
                print(str_metricToVisualize)
                print("str_timeScope:")
                print(str_timeScope)                
                print("str_timeScope:")
                print(str_timeScope)
                print("str_metricToVisualize_rootTweetLabel:")
                print(str_metricToVisualize_rootTweetLabel)
                
                value_temp = df_input.loc[df_input["cascadeAge_min"]==int_cascadeAge_min_toMeasure, str_metricToVisualize_rootTweetLabel].copy().reset_index(drop=True)[0]
                
                print("value_temp:")
                print(value_temp)                
                print(type(value_temp))
                
                if not np.isnan(value_temp):
                    if isinstance(value_temp, float):
                        value_temp = round(value_temp, 2)
                    str_measuredValue = str(value_temp)
                else:
                    print("No value available at timestamp " + str(int_cascadeAge_min_toMeasure))
                    print("the last available value is measured instead")
                    list_temp = df_input[str_metricToVisualize_rootTweetLabel].tolist()
                    list_temp = [x for x in list_temp if not np.isnan(x)]                    
                    str_measuredValue = str(list_temp[-1])                    
                    
                    str_measuredValue = "*** " + str_measuredValue
                
                print("str_measuredValue:")
                print(str_measuredValue)
                
                df_output.loc[index_row, str_metricToVisualize + "_" + str_timeScope] = str_measuredValue
                        
        index_row += 1
        
        
    absFilename_output = "C:\\vmwareSharedFolder\\TwitterDataAnalysis\\programs\\FalseNews_Code\\FalseNews_Code\\data\\cascadeStatistics.csv"
            
    print("absFilename_input:")
    print(absFilename_input)
    
    df_output = df_output.sort_values(by="rootTweetLabel")
    df_output.to_csv(absFilename_output, index=False)
                
            
        
    
    
if __name__ == "__main__":
    main(sys.argv[1:])