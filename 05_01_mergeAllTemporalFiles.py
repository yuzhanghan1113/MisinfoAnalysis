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
import glob


def main(argv):
    random.seed(1113)

    # pd.set_option('display.height', 1000)
    pd.set_option('display.max_rows', 1000)
    pd.set_option('display.max_columns', 1000)
    # pd.set_option('display.width', 170)

    opts, args = getopt.getopt(argv, '', ["path_base=", "list_tweetIDs_toMerge="])

    print(opts)

    for opt, arg in opts:
        if opt == '--path_base':
            path_base = arg
        if opt == '--list_tweetIDs_toMerge':
            list_tweetIDs_toMerge = arg.split(" ")
            
    dict_mapping_tweetIDs2TypeNames = {"1246159197621727234":"F1", "1239336564334960642":"F2", "1240682713817976833":"F3", "1261326944492281859":"F4", "1262482651333738500":"F5", "1247287993095897088":"F6", "1256641587708342274":"F7", "1268269994074345473":"T1", "1269111354520211458":"T2", "1268155500753027073":"T3", "1269320689003216896":"T4", "1269077273770156032":"T5", "1268346742296252418":"T6"} 
    
    
    list_columnsToRename = ["cascadeSize", "cascadeDepth", "cascadeVirality", "mean_followers", "median_followers", "mean_followees", "median_followees", "mean_account_age", "median_account_age", "mean_engagement", "median_engagement"]         
    
                
       
    df_output = pd.DataFrame()
    
    for tweetIDs_toMerge in list_tweetIDs_toMerge:
    
        print("tweetIDs_toMerge:")
        print(tweetIDs_toMerge)
    
        list_absFilenames_input = []    

        absFilename_input = path_base + "temporalCharacteristics_rootTweetID=" + tweetIDs_toMerge + ".csv"
        list_absFilenames_input += [absFilename_input]
    
        filenamePattern = path_base + "temporalCharacteristics_rootTweetID=" + tweetIDs_toMerge + "_cascadeAgeStart=*.csv"    
        #print("filenamePattern:")
        #print(filenamePattern)        
        list_absFilenames_input += glob.glob(filenamePattern)
        
        print("list_absFilenames_input:")
        print(list_absFilenames_input)
        print(len(list_absFilenames_input))
        
        list_df_input = []
        
        for absFilename_input in list_absFilenames_input:
            df_input = pd.read_csv(absFilename_input, dtype=str)
            list_df_input += [df_input]
            
        df_input_cascade = pd.concat(list_df_input)
        
        print("len(df_input_cascade):")
        print(len(df_input_cascade))
        df_input_cascade = df_input_cascade.drop_duplicates()
        df_input_cascade = df_input_cascade.reset_index(drop=True)
        print("len(df_input_cascade):")
        print(len(df_input_cascade))
        
              
        
        typeName = dict_mapping_tweetIDs2TypeNames[tweetIDs_toMerge]
        dict_columnRenaming = {e: typeName + "_" + e for e in list_columnsToRename}
                
        print(dict_columnRenaming)        
        
        df_input_cascade = df_input_cascade.rename(columns=dict_columnRenaming)
        
        if len(df_output) == 0:
            df_output = df_input_cascade.copy()
        else:
            df_output = pd.merge(left=df_output, right=df_input_cascade, how="outer", on=["cascadeAge_min"])
            
    #print('df_output:')
    #print(df_output)
    print('len(df_output):')
    print(len(df_output))
    
    absFilename_output = path_base + "temporalCharacteristics_merged.csv"  
        
    print('absFilename_output:')
    print(absFilename_output)
    
    df_output.to_csv(path_or_buf=absFilename_output, index=False)

    
       
    
if __name__ == "__main__":
    main(sys.argv[1:])