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



from TwitterAPI import TwitterAPI
from TwitterAPI import TwitterPager
from TwitterAPI import TwitterConnectionError, TwitterRequestError

list_symptoms = ['cough', 'breath', 'chill', 'muscl', 'pain', 'headach', 'sore', 'throat', 'chest', 'body temperature']
list_behaviors = ['mask', 'face cover', 'face covering', 'facial cover', 'facial covering', 'face-cover', 'face-covering', 'facial-cover', 'facial-covering', 'stay', 'home', 'distanc', 'remote', 'work from home', 'working from home', 'works from home', 'worked from home', 'work-from-home']



def main(argv):
    random.seed(1113)

    # pd.set_option('display.height', 1000)
    pd.set_option('display.max_rows', 1000)
    pd.set_option('display.max_columns', 1000)
    # pd.set_option('display.width', 170)

    opts, args = getopt.getopt(argv, '', ["absFilename_input=", "path_output="])

    print(opts)

    for opt, arg in opts:
        if opt == '--absFilename_input':
            absFilename_input = arg
        if opt == '--path_output':
            path_output = arg
   
    print('absFilename_input:')
    print(absFilename_input)
    #df_input = pd.read_csv(absFilename_input, dtype=str, quotechar='"', delimiter=',', escapechar='\\')
    df_input = pd.read_csv(absFilename_input, dtype=str)

    print(df_input)
    print(len(df_input))
    
    for index_row in range(0, len(df_input)):
    
        text = df_input.loc[index_row, 'text'].lower()
        text_pp = df_input.loc[index_row, 'text_pp'].lower()
       
        list_symptoms_text = []        
        list_symptoms_text_pp = []  
               
        for keyword in list_symptoms:
            if keyword in text:
                list_symptoms_text += [keyword]
            if keyword in text_pp:
                list_symptoms_text_pp += [keyword]
                
        df_input.loc[index_row, 'symptom_text'] = ','.join(list_symptoms_text)
        if len(list_symptoms_text) > 0:
            df_input.loc[index_row, 'symptomsMentioned_any_text'] = '1'
        else:
            df_input.loc[index_row, 'symptomsMentioned_any_text'] = '0'
        df_input.loc[index_row, 'symptomsMentioned_num_text'] = str(len(list_symptoms_text))
        
        df_input.loc[index_row, 'symptom_text_pp'] = ','.join(list_symptoms_text_pp)
        if len(list_symptoms_text_pp) > 0:
            df_input.loc[index_row, 'symptomsMentioned_any_text_pp'] = '1'
        else:
            df_input.loc[index_row, 'symptomsMentioned_any_text_pp'] = '0'
        df_input.loc[index_row, 'symptomsMentioned_num_text_pp'] = str(len(list_symptoms_text_pp))
        
        
        list_behaviors_text = []        
        list_behaviors_text_pp = []  
               
        for keyword in list_behaviors:
            if keyword in text:
                list_behaviors_text += [keyword]
            if keyword in text_pp:
                list_behaviors_text_pp += [keyword]
                
        df_input.loc[index_row, 'behaviors_text'] = ','.join(list_behaviors_text)
        if len(list_behaviors_text) > 0:
            df_input.loc[index_row, 'behaviorsMentioned_any_text'] = '1'
        else:
            df_input.loc[index_row, 'behaviorsMentioned_any_text'] = '0'
        df_input.loc[index_row, 'behaviorsMentioned_num_text'] = str(len(list_behaviors_text))
        
        df_input.loc[index_row, 'behaviors_text_pp'] = ','.join(list_behaviors_text_pp)
        if len(list_behaviors_text_pp) > 0:
            df_input.loc[index_row, 'behaviorsMentioned_any_text_pp'] = '1' 
        else:
            df_input.loc[index_row, 'behaviorsMentioned_any_text_pp'] = '0' 
        df_input.loc[index_row, 'behaviorsMentioned_num_text_pp'] = str(len(list_behaviors_text_pp))
            
        if index_row % 1000 == 0:
            print('processed rows: ' + str(index_row) + '/' + str(len(df_input)))
    
    print('processed rows: ' + str(index_row) + '/' + str(len(df_input)))

    absFilename_output = absFilename_input.replace('preprocessed_', 'keywordsDetected_')
    #absFilename_output = absFilename_output.replace('\\\\', '\\')
    print('absFilename_output:')
    print(absFilename_output)
    if not os.path.exists(os.path.dirname(absFilename_output)):
            os.makedirs(os.path.dirname(absFilename_output))
            
    df_input.to_csv(path_or_buf=absFilename_output, index=False)
    
    
if __name__ == "__main__":
    main(sys.argv[1:])