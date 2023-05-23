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
import re
from nltk.tokenize import word_tokenize
from string import punctuation 
from nltk.corpus import stopwords 
from nltk.stem import PorterStemmer 
import os, glob

def main(argv):
    random.seed(1113)

    # pd.set_option('display.height', 1000)
    pd.set_option('display.max_rows', 1000)
    pd.set_option('display.max_columns', 1000)
    # pd.set_option('display.width', 170)

    opts, args = getopt.getopt(argv, '', ["path_input=", "absFilename_output="])

    print(opts)

    for opt, arg in opts:
        if opt == '--path_input':
            path_input = arg
        if opt == '--absFilename_output':
            absFilename_output = arg
            
          
                
            
    list_inputFileNames = []
    
    for dirpath,_,filenames in os.walk(path_input):
       for f in filenames:
           list_inputFileNames += [os.path.abspath(os.path.join(dirpath, f))]
           
    list_dfs = []
    
    for absFilename_input in list_inputFileNames:
        
        print('absFilename_input:')
        print(absFilename_input)
        df_input = pd.read_csv(absFilename_input, dtype=str)
        
        list_dfs += [df_input]
        
    print('len(list_dfs):')
    print(len(list_dfs))
    
    df_output = pd.concat(list_dfs)
    
    print('len(df_output):')
    print(len(df_output))
    
    print('absFilename_output:')
    print(absFilename_output)
    
    if not os.path.exists(os.path.dirname(absFilename_output)):
            os.makedirs(os.path.dirname(absFilename_output))
            
    df_output.to_csv(path_or_buf=absFilename_output, index=False)

   
       
    
if __name__ == "__main__":
    main(sys.argv[1:])