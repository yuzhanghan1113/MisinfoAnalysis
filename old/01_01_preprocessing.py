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

list_stateCodes = ["AL", "AK", "AZ", "AR", "CA", "CO", "CT", "DE", "DC", "FL", "GA", "HI", "ID", "IL", "IN", "IA", "KS", "KY", "LA", "ME", "MT", "NE", "NV", "NH", "NJ", "NM", "NY", "NC", "ND", "OH", "OK", "OR", "MD", "MA", "MI", "MN", "MS", "MO", "PA", "RI", "SC", "SD", "TN", "TX", "UT", "VT", "VA", "WA", "WV", "WI", "WY"]



def main(argv):
    random.seed(1113)

    # pd.set_option('display.height', 1000)
    pd.set_option('display.max_rows', 1000)
    pd.set_option('display.max_columns', 1000)
    # pd.set_option('display.width', 170)

    opts, args = getopt.getopt(argv, '', ["absFilename_input=", "path_output=", "usOnly="])

    print(opts)

    for opt, arg in opts:
        if opt == '--absFilename_input':
            absFilename_input = arg
        if opt == '--path_output':
            path_output = arg
        if opt == '--usOnly':
            usOnly = arg
   
    print('absFilename_input:')
    print(absFilename_input)
    #df_input = pd.read_csv(absFilename_input, dtype=str, quotechar='"', delimiter=',', escapechar='\\')
    df_input = pd.read_csv(absFilename_input, dtype=str)

    #print(df_input)
    print(len(df_input))
    
    
    
    #counter = Counter(df_input['place_full_name'].tolist())
    #print(counter)
    #counter = Counter(df_input['place_type'].tolist())
    #print(counter)
    #counter = Counter(df_input['country_code'].tolist())
    #print(counter)
    
    if usOnly == 'True':
        df_input = df_input[df_input['place_full_name'].notna()]  

        list_stateCodes_temp = [', ' + e for e in list_stateCodes]
        #print(list_stateCodes_temp)
        
        df_input = df_input[df_input['place_full_name'].str.slice(-4,).isin(list_stateCodes_temp)]
        
        df_input['state'] = df_input['place_full_name'].str.slice(-2,)
        df_input['place'] = df_input['place_full_name'].str.slice(0,-4)

        print(df_input[['status_id', 'place_full_name', 'state', 'place']])
    
        df_input = df_input.reset_index(drop=True)
    
    print(len(df_input))
    
    list_stopwords = set(stopwords.words('english') + list(punctuation) + ['AT_USER','URL'])
    
    ps = PorterStemmer()
    
    for index_row in range(0, len(df_input)):
    
        text = df_input.loc[index_row, 'text']
        #print(text)
        
        #text = re.findall(r"#(\w+)", s)
        list_hashtags = re.findall(r'#([^\s]+)', text)
        df_input.loc[index_row, 'hashtag'] = ' '.join(list_hashtags)
        
        text = text.lower() # convert text to lower-case
        text = re.sub('((www\.[^\s]+)|(https?://[^\s]+))', 'URL', text) # remove URLs
        text = re.sub('@[^\s]+', 'AT_USER', text) # remove usernames                        
        text = re.sub(r'#([^\s]+)', r'\1', text) # remove the # in #hashtag
        
        list_tokens = word_tokenize(text) # remove repeated characters (helloooooooo into hello) 
        list_tokens = [ps.stem(word) for word in list_tokens if word not in list_stopwords]
        
        text = ' '.join(list_tokens)
                
        df_input.loc[index_row, 'text_pp'] = text
        
        #print(df_input.loc[index_row, 'text'])
        #print(df_input.loc[index_row, 'text_pp'])
        #print(df_input.loc[index_row, 'hashtag'])
        
    print('processed rows: ' + str(index_row) + '/' + str(len(df_input)))

        
    str_temp = ''
    if usOnly=='True':
        str_temp = 'usOnly_'
        
    absFilename_output = path_output + os.path.sep + 'preprocessed_' + str_temp + absFilename_input.split(os.path.sep)[-1]
    #absFilename_output = absFilename_output.replace('\\\\', '\\')
    print('absFilename_output:')
    print(absFilename_output)
    if not os.path.exists(os.path.dirname(absFilename_output)):
            os.makedirs(os.path.dirname(absFilename_output))
            
    df_input.to_csv(path_or_buf=absFilename_output, index=False)
    
    
if __name__ == "__main__":
    main(sys.argv[1:])