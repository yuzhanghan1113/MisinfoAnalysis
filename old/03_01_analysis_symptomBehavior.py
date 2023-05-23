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

list_columns = ['state', 'place', 'hashtag', 'symptom_text', 'symptom_text_pp', 'behaviors_text', 'behaviors_text_pp', 'behaviorsMentioned_any_text', 'behaviorsMentioned_num_text', 'behaviorsMentioned_any_text_pp', 'behaviorsMentioned_num_text_pp', 'symptomsMentioned_any_text', 'symptomsMentioned_num_text', 'symptomsMentioned_any_text_pp', 'symptomsMentioned_num_text_pp']



def main(argv):
    random.seed(1113)

    # pd.set_option('display.height', 1000)
    pd.set_option('display.max_rows', 1000)
    pd.set_option('display.max_columns', 1000)
    # pd.set_option('display.width', 170)

    opts, args = getopt.getopt(argv, '', ["absFilename_input_data=", "absFilename_input_internetUsers=", "path_output="])

    print(opts)

    for opt, arg in opts:
        if opt == '--absFilename_input_data':
            absFilename_input_data = arg
        if opt == '--absFilename_input_internetUsers':
            absFilename_input_internetUsers = arg
        if opt == '--path_output':
            path_output = arg
   
    print('absFilename_input_data:')
    print(absFilename_input_data)
    #df_input = pd.read_csv(absFilename_input_data, dtype=str, quotechar='"', delimiter=',', escapechar='\\')
    df_input = pd.read_csv(absFilename_input_data, dtype=str, usecols=list_columns)
    
    print('absFilename_input_internetUsers:')
    print(absFilename_input_internetUsers)
    df_input_internetUsers = pd.read_csv(absFilename_input_internetUsers,  usecols=['state', 'num_internetUsers_M'])   
    
    
    df_input[['behaviorsMentioned_num_text', 'behaviorsMentioned_num_text_pp', 'symptomsMentioned_num_text', 'symptomsMentioned_num_text_pp']] = df_input[['behaviorsMentioned_num_text', 'behaviorsMentioned_num_text_pp', 'symptomsMentioned_num_text', 'symptomsMentioned_num_text_pp']].astype(int)

    print(df_input)
    print(len(df_input))
    
    print('number of tweets/state:')
    df_result_state_numTweets = df_input['state'].value_counts().rename_axis('state').reset_index(name='num_tweets')
    print(df_result_state_numTweets)
    
    print('number of tweets with any behaviors/state:')
    df_temp = df_input[df_input['behaviorsMentioned_any_text'] == '1']
    df_result_1 = df_temp['state'].value_counts().rename_axis('state').reset_index(name='num_tweets_anyBehaviors')
    print(df_result_1)
    
    print('number of tweets with any symptoms/state:')
    df_temp = df_input[df_input['symptomsMentioned_any_text'] == '1']
    df_result_2 = df_temp['state'].value_counts().rename_axis('state').reset_index(name='num_tweets_anySymptoms')
    print(df_result_2)
    
    print('number of tweets with behavior and symptom/state:')
    df_temp = df_input[(df_input['behaviorsMentioned_any_text'] == '1') & (df_input['symptomsMentioned_any_text'] == '1')]
    df_result_3 = df_temp['state'].value_counts().rename_axis('state').reset_index(name='num_tweets_behaviorAndSymptom')
    print(df_result_3)
    
    print('top 3 behaviors/state:')
    list_states_anyBehaviors = df_result_1['state'].tolist()
    
    df_result_4 = pd.DataFrame()
    
    n = 3
    
    for state in list_states_anyBehaviors:
            
        #print(state)
    
        df_temp = df_input[(df_input['state'] == state) & (df_input['behaviorsMentioned_any_text'] == "1")]
        #print(df_temp['behaviors_text'].tolist())
        str_temp = ','.join(df_temp['behaviors_text'].tolist())
        #print(str_temp.split(','))
        #print(len(str_temp.split(',')))
        list_top = Counter(str_temp.split(',')).most_common(n)
        #print(list_top)
        df_result_4.loc[state, 'state'] = state
        
        for index_column in range(0, min(n, len(list_top))):
            df_result_4.loc[state, 'topBehavior_' + str(index_column+1)] = list_top[index_column][0]
            df_result_4.loc[state, 'topBehavior_' + str(index_column+1) + '_num'] = str(list_top[index_column][1])
            
    df_result_4 = df_result_4.reset_index(drop=True)
            
    print(df_result_4)
        
    print('top 3 symptoms/state:')
    list_states_anySymptoms = df_result_2['state'].tolist()
    
    df_result_5 = pd.DataFrame()
    
    n = 3
    
    for state in list_states_anySymptoms:
            
        #print(state)
    
        df_temp = df_input[(df_input['state'] == state) & (df_input['symptomsMentioned_any_text'] == "1")]
        #print(df_temp['symptoms_text'].tolist())
        str_temp = ','.join(df_temp['symptom_text'].tolist())
        #print(str_temp.split(','))
        #print(len(str_temp.split(',')))
        list_top = Counter(str_temp.split(',')).most_common(n)
        #print(list_top)
        df_result_5.loc[state, 'state'] = state
        
        for index_column in range(0, min(n, len(list_top))):
            df_result_5.loc[state, 'topSymptom_' + str(index_column+1)] = list_top[index_column][0]
            df_result_5.loc[state, 'topSymptom_' + str(index_column+1) + '_num'] = str(list_top[index_column][1])
            
    df_result_5 = df_result_5.reset_index(drop=True)
            
    print(df_result_5)
    
    print('results:')
    
    df_output = pd.merge(df_result_state_numTweets, df_input_internetUsers, how='left', on=['state'])
    df_output = pd.merge(df_output, df_result_1, how='left', on=['state'])
    df_output = pd.merge(df_output, df_result_2, how='left', on=['state'])
    df_output = pd.merge(df_output, df_result_3, how='left', on=['state'])
    df_output = pd.merge(df_output, df_result_4, how='left', on=['state'])
    df_output = pd.merge(df_output, df_result_5, how='left', on=['state'])
    
    df_output['num_tweets_perMInetUsers'] = df_output['num_tweets']/df_output['num_internetUsers_M']
    df_output['num_tweets_anyBehaviors_perTweet'] = df_output['num_tweets_anyBehaviors']/df_output['num_tweets']
    df_output['num_tweets_anySymptoms_perTweet'] = df_output['num_tweets_anySymptoms']/df_output['num_tweets']
    df_output['num_tweets_behaviorAndSymptom_perTweet'] = df_output['num_tweets_behaviorAndSymptom']/df_output['num_tweets']
    
    print(df_output)
           
    absFilename_output = path_output + os.path.sep + 'results_' + absFilename_input_data.split(os.path.sep)[-1]
    print('absFilename_output:')
    print(absFilename_output)
    if not os.path.exists(os.path.dirname(absFilename_output)):
            os.makedirs(os.path.dirname(absFilename_output))
            
    df_output.to_csv(path_or_buf=absFilename_output, index=False)
    
    
    
    

    
    
if __name__ == "__main__":
    main(sys.argv[1:])