import json
import os
import getopt
import sys
import random
import pandas as pd
import traceback
import gc 
import time
from collections import Counter


def main(argv):
    random.seed(1113)

                
    list_rootTweetLabels = {"F1", "F2", "F3", "F4", "F5", "F6", "F7", "T1", "T2", "T3", "T4", "T5", "T6"} 
    
    
    list_metrics = ["mean_followers", "mean_followees", "mean_account_age", "mean_engagement"]  
    
    absFilename_input = "C:\\vmwareSharedFolder\\TwitterDataAnalysis\\programs\\FalseNews_Code\\FalseNews_Code\\data\\temporal\\temporalCharacteristics_merged.csv"
    
    absFilename_output = "C:\\vmwareSharedFolder\\TwitterDataAnalysis\\programs\\FalseNews_Code\\FalseNews_Code\\data\\temporal\\temporalCharacteristics_merged_producerRemoved.csv"
    

    print("absFilename_input:")
    print(absFilename_input)
    
    df_input = pd.read_csv(absFilename_input, dtype=str)
    df_output = df_input.copy()

    print("len(df_input):")
    print(len(df_input))
    print("len(df_output):")
    print(len(df_output))
    
    #df_output.replace("None", 0, inplace=True)
    df_output.replace("None", "", inplace=True)
        
    df_output = df_output.apply(pd.to_numeric, errors='ignore')
    
    list_timestamps = sorted(list(set(df_output["cascadeAge_min"].tolist())))
    print("list_timestamps:")
    print(list_timestamps)
    print(len(list_timestamps))
    
    
    
    for str_rootTweetLabel in list_rootTweetLabels:
        for str_metric in list_metrics:
        
            str_columnName = str_rootTweetLabel + "_" + str_metric
            print("str_columnName:")
            print(str_columnName)
            
            
            metricValue_producer = df_output.loc[df_output["cascadeAge_min"]==0, str_columnName].reset_index(drop=True)[0]
            print("metricValue_producer:")
            print(metricValue_producer)
            
            for timestamp in list_timestamps:
            
                int_cascadeSize = df_output.loc[df_output["cascadeAge_min"]==timestamp, str_rootTweetLabel + "_cascadeSize"].reset_index(drop=True)[0]
                print("int_cascadeSize:")
                print(int_cascadeSize)
            
                if timestamp == 0:
                    df_output.loc[df_output["cascadeAge_min"]==timestamp, str_columnName] = 0
                else:                  
                    metricValue_old = df_output.loc[df_output["cascadeAge_min"]==timestamp, str_columnName].reset_index(drop=True)[0]
                    metricValue_new = (metricValue_old*int_cascadeSize - metricValue_producer)/(int_cascadeSize-1)
                    
                    print("metricValue_old:")
                    print(metricValue_old)
                    print("metricValue_new:")
                    print(metricValue_new)
                
                    df_output.loc[df_output["cascadeAge_min"]==timestamp, str_columnName] = metricValue_new
                 
            df_output.loc[df_output["cascadeAge_min"]==0, str_columnName.replace("mean", "median")] = 0
            
            df_output.to_csv(absFilename_output, index=False)        
        
        for timestamp in list_timestamps:
            
            int_cascadeSize = df_output.loc[df_output["cascadeAge_min"]==timestamp, str_rootTweetLabel + "_cascadeSize"].reset_index(drop=True)[0]            
            df_output.loc[df_output["cascadeAge_min"]==timestamp, str_rootTweetLabel + "_cascadeSize"] = int_cascadeSize-1      
            
    print("absFilename_output:")
    print(absFilename_output)  
    df_output.to_csv(absFilename_output, index=False)
                
       
    
    
       
    
if __name__ == "__main__":
    main(sys.argv[1:])