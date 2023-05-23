import json
import os
import getopt
import sys
import random
import pandas as pd
import numpy as np
import traceback
import gc 
import time
from datetime import datetime, timezone
from collections import Counter
import re
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from textblob import TextBlob
import preprocessor
import nltk
from nltk.tokenize import TweetTokenizer
from nltk import sent_tokenize, FreqDist
from nltk.corpus import stopwords
from nltk.stem import LancasterStemmer
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nrclex import NRCLex
import spacy
import textstat
import string
from nltk.tokenize import sent_tokenize
import csv
import glob



def main(argv):

    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)
    pd.set_option('display.max_colwidth', -1)
    
    random.seed(1113)

    WINDOW_HRS = 48
    WINDOW_MIN = WINDOW_HRS * 60
    WINDOW_SEC = WINDOW_MIN * 60

    
    """
    # preprocess cascade data (non-time series):

    

    # absFilename_input_rootTweets = "D:\\vmwareSharedFolder\\TwitterDataCollection\\data\\PFN\\snapshot\\dateRetrieval=20201111\\rootTweets_selected_factcheckArticleRep_20201111.csv"
    absFilename_input_rootTweets = "C:\\vmwareSharedFolder\\TwitterDataCollection\\data\\PFN\\snapshot\\dateRetrieval=20210209\\rootTweets_selected_factcheckArticleRep_annotated_20210209.csv"
    path_base_input = "C:\\vmwareSharedFolder\\TwitterDataAnalysis\\results\\results_20210209\\"
    absFilename_output = "C:\\vmwareSharedFolder\\TwitterDataAnalysis\\results\\cascadeResults_merged.csv"

    # absFilename_input_TS = "D:\\vmwareSharedFolder\\TwitterDataAnalysis\\results\\timeSeries_retweets.csv"
    # absFilename_input_replies_TS = "D:\\vmwareSharedFolder\\TwitterDataAnalysis\\results\\timeSeries_replies.csv"
    # absFilename_output_retweets_TS = "D:\\vmwareSharedFolder\\TwitterDataAnalysis\\results\\timeSeries_retweets_preprocessed.csv"
    # absFilename_output_replies_TS = "D:\\vmwareSharedFolder\\TwitterDataAnalysis\\results\\imeSeries_replies_prepreocessed.csv"

    list_filenames_input = ["cascadeResults_temporal_retweets", "cascadeResults_temporal_replies", "cascadeResults_producerTweets_timeWindow=7"]

    for filename_input in list_filenames_input:

        absFilename_output_merged = path_base_input + filename_input + ".csv"

        print("absFilename_output_merged:")
        print(absFilename_output_merged)

        if os.path.exists(absFilename_output_merged):

            print("Merged absFilename_output_merged exists. Load it directly.")
            df_merged = pd.read_csv(absFilename_output_merged, dtype=str)

        else:

            print("Merged absFilename_output_merged does not exist. Merge part files.")

            pattern_absFilename_input_part = path_base_input + filename_input + "_indexStart=*_indexEnd=*.csv"

            list_absFilenames_input_part = glob.glob(pattern_absFilename_input_part)

            print("list_absFilenames_input_part:")
            print(list_absFilenames_input_part)
            print("len(list_absFilenames_input_part):")
            print(len(list_absFilenames_input_part))

            df_merged = pd.DataFrame()
            
            for absFilename_input_part in list_absFilenames_input_part:

                df_input_part = pd.read_csv(absFilename_input_part, dtype=str)

                print("appending absFilename_input_part:")
                print(absFilename_input_part)

                print("Part file:")
                print("len(df_input_part):")
                print(len(df_input_part))
                print("len(list(set(df_input_part[\"rootTweetIdStr\"].tolist()))):")
                print(len(list(set(df_input_part["rootTweetIdStr"].tolist()))))

                if len(df_merged) <= 0:
                    df_merged = df_input_part.copy()
                else:
                    df_merged = df_merged.append(df_input_part)

            print("Write df_merged to output file:")
            print("absFilename_output_merged:")
            print(absFilename_output_merged)

            df_merged.to_csv(absFilename_output_merged, index=False, quoting=csv.QUOTE_ALL)

        print("Raw data:")
        print("len(df_merged):")
        print(len(df_merged))
        print("len(list(set(df_merged[\"rootTweetIdStr\"].tolist()))):")
        print(len(list(set(df_merged["rootTweetIdStr"].tolist()))))



        if filename_input == "cascadeResults_temporal_retweets":
            df_input_retweets = df_merged.copy()
        elif filename_input == "cascadeResults_temporal_replies":
            df_input_replies = df_merged.copy()
        elif filename_input == "cascadeResults_producerTweets_timeWindow=7":
            df_input_producerTweets = df_merged.copy()
        
    print("len(df_input_retweets):")
    print(len(df_input_retweets))
    print("len(list(set(df_input_retweets[\"rootTweetIdStr\"].tolist()))):")
    print(len(list(set(df_input_retweets["rootTweetIdStr"].tolist()))))

    print("len(df_input_replies):")
    print(len(df_input_replies))
    print("len(list(set(df_input_replies[\"rootTweetIdStr\"].tolist()))):")
    print(len(list(set(df_input_replies["rootTweetIdStr"].tolist()))))

    print("len(df_input_producerTweets):")
    print(len(df_input_producerTweets))
    print("len(list(set(df_input_producerTweets[\"rootTweetIdStr\"].tolist()))):")
    print(len(list(set(df_input_producerTweets["rootTweetIdStr"].tolist()))))
    print("len(list(set(df_input_producerTweets[\"producerScreenName\"].tolist()))):")
    print(len(list(set(df_input_producerTweets["producerScreenName"].tolist()))))

    df_input_rootTweets = pd.read_csv(absFilename_input_rootTweets, dtype=str, engine='python')

    df_input_rootTweets["id_rootTweet"] = df_input_rootTweets["tweet link"].apply(lambda x: x.split('/')[-1])
    
    print("len(df_input_rootTweets):")
    print(len(df_input_rootTweets))
    print("len(list(set(df_input_rootTweets[\"id_rootTweet\"].tolist()))):")
    print(len(list(set(df_input_rootTweets["id_rootTweet"].tolist()))))

    list_rootTweetIDs_raw = list(set(df_input_rootTweets["id_rootTweet"].tolist()))
    list_rootTweetIDs_temporal = list(set(df_input_retweets["rootTweetIdStr"].tolist()))

    list_rootTweetIDs_noTemporal = [e for e in list_rootTweetIDs_raw if e not in list_rootTweetIDs_temporal]

    print("list_rootTweetIDs_noTemporal:")
    print(list_rootTweetIDs_noTemporal)


    # return

    # df_input_rootTweets = pd.read_csv(absFilename_input_rootTweets, dtype=str, engine='python')
    # df_input_rootTweets["id_rootTweet"] = df_input_rootTweets["tweet link"].apply(lambda x: x.split('/')[-1])

    # df_input_retweets = pd.read_csv(absFilename_input_retweets, dtype=str)
    # df_input_replies = pd.read_csv(absFilename_input_replies, dtype=str)
    # df_input_producerTweets = pd.read_csv(absFilename_input_producerTweets, dtype=str)

    print("len(df_input_producerTweets):")
    print(len(df_input_producerTweets))

    df_input_producerTweets = df_input_producerTweets.drop_duplicates()
    df_input_producerTweets = df_input_producerTweets.sort_values(by=["producerScreenName", "rootTweetIdStr"], ascending=True)
    df_input_producerTweets = df_input_producerTweets.reset_index(drop=True)

    print("len(df_input_rootTweets):")
    print(len(df_input_rootTweets))
    print("len(df_input_retweets):")
    print(len(df_input_retweets))
    print("len(df_input_replies):")
    print(len(df_input_replies))
    print("len(df_input_producerTweets):")
    print(len(df_input_producerTweets))

    print("Get only the aggregate result at cascadeAge_min:")
    print(WINDOW_MIN)

    df_input_retweets["cascadeAge_min"] = df_input_retweets["cascadeAge_min"].astype(float).astype(int)
    df_input_replies["cascadeAge_min"] = df_input_replies["cascadeAge_min"].astype(float).astype(int)

    df_input_retweets = df_input_retweets[df_input_retweets["cascadeAge_min"]==WINDOW_MIN]
    df_input_replies = df_input_replies[df_input_replies["cascadeAge_min"]==WINDOW_MIN]

    df_input_retweets["cascadeAge_min"] = df_input_retweets["cascadeAge_min"].astype(str)
    df_input_replies["cascadeAge_min"] = df_input_replies["cascadeAge_min"].astype(str)

    # print(df_input_retweets["cascadeAge_min"].tolist())
    # print(df_input_replies["cascadeAge_min"].tolist())

    df_input_rootTweets.columns = ["ROOTTWEETS_" + c for c in df_input_rootTweets.columns]
    df_input_retweets.columns = ["RETWEETS_" + c for c in df_input_retweets.columns]
    df_input_replies.columns = ["REPLIES_" + c for c in df_input_replies.columns]
    df_input_producerTweets.columns = ["PRODUCERTWEETS_" + c for c in df_input_producerTweets.columns]

    print("len(list(set(df_input_rootTweets[\"ROOTTWEETS_id_rootTweet\"].tolist()))):")
    print(len(list(set(df_input_rootTweets["ROOTTWEETS_id_rootTweet"].tolist()))))
    print(len(list(df_input_rootTweets["ROOTTWEETS_id_rootTweet"].tolist())))

    print("len(list(set(df_input_retweets[\"RETWEETS_rootTweetIdStr\"].tolist()))):")
    print(len(list(set(df_input_retweets["RETWEETS_rootTweetIdStr"].tolist()))))
    print(len(list(df_input_retweets["RETWEETS_rootTweetIdStr"].tolist())))
    print("len(list(set(df_input_retweets[\"RETWEETS_cascadeAge_min\"].tolist()))):")
    print(len(list(set(df_input_retweets["RETWEETS_cascadeAge_min"].tolist()))))
    print(len(list(df_input_retweets["RETWEETS_cascadeAge_min"].tolist())))

    print("len(list(set(df_input_replies[\"REPLIES_rootTweetIdStr\"].tolist()))):")
    print(len(list(set(df_input_replies["REPLIES_rootTweetIdStr"].tolist()))))
    print(len(list(df_input_replies["REPLIES_rootTweetIdStr"].tolist())))
    print("len(list(set(df_input_replies[\"REPLIES_cascadeAge_min\"].tolist()))):")
    print(len(list(set(df_input_replies["REPLIES_cascadeAge_min"].tolist()))))
    print(len(list(df_input_replies["REPLIES_cascadeAge_min"].tolist())))

    print("len(list(set(df_input_producerTweets[\"PRODUCERTWEETS_rootTweetIdStr\"].tolist()))):")
    print(len(list(set(df_input_producerTweets["PRODUCERTWEETS_rootTweetIdStr"].tolist()))))
    print(len(list(df_input_producerTweets["PRODUCERTWEETS_rootTweetIdStr"].tolist())))

    df_result = pd.merge(df_input_rootTweets, df_input_retweets, how='left', left_on=["ROOTTWEETS_id_rootTweet"], right_on=["RETWEETS_rootTweetIdStr"])

    print("len(df_result):")
    print(len(df_result))

    df_result = pd.merge(df_result, df_input_replies, how='left', left_on=["ROOTTWEETS_id_rootTweet", "RETWEETS_cascadeAge_min"], right_on=["REPLIES_rootTweetIdStr", "REPLIES_cascadeAge_min"])

    print("len(df_result):")
    print(len(df_result))

    df_result = pd.merge(df_result, df_input_producerTweets, how='left', left_on=["ROOTTWEETS_id_rootTweet", "RETWEETS_rootTweet_user_screenName"], right_on=["PRODUCERTWEETS_rootTweetIdStr", "PRODUCERTWEETS_producerScreenName"])

    print("len(df_result):")
    print(len(df_result))

    print("list(df_result.columns):")
    print(list(df_result.columns))


    


    list_featuresToStandardize = ['ROOTTWEETS_num_retweets_total', 'ROOTTWEETS_num_retweets', 'RETWEETS_cascadeSize', 'RETWEETS_rootTweet_user_friendsCount', 'RETWEETS_rootTweet_user_listedCount', 'RETWEETS_rootTweet_user_favouritesCount', 'REPLIES_cascadeSize']
    feature_base = 'RETWEETS_rootTweet_user_followersCount'

    df_result[list_featuresToStandardize + [feature_base]] = df_result[list_featuresToStandardize + [feature_base]].astype(float)

    for f in list_featuresToStandardize:
        df_result[f + "_stadardized"] = df_result[f] / df_result[feature_base]



    list_columns = ["ROOTTWEETS_id_rootTweet", "RETWEETS_rootTweetIdStr", "REPLIES_rootTweetIdStr", "PRODUCERTWEETS_rootTweetIdStr"]
    for c in list_columns:
        df_result[c] = "id" + df_result[c].astype(str)

    print("absFilename_output:")
    print(absFilename_output)

    print("len(list(set(df_result[\"ROOTTWEETS_id_rootTweet\"].tolist()))):")
    print(len(list(set(df_result["ROOTTWEETS_id_rootTweet"].tolist()))))

    list_features_output = ["ROOTTWEETS_id_rootTweet", "RETWEETS_rootTweetIdStr", "RETWEETS_cascadeAge_min", "REPLIES_rootTweetIdStr", "REPLIES_cascadeAge_min", "PRODUCERTWEETS_rootTweetIdStr", "PRODUCERTWEETS_producerScreenName"]

    
    list_features_output += [feature_base]

    for f in list_featuresToStandardize:
        list_features_output += [f]
        list_features_output += [f + "_stadardized"]


    df_result = df_result[list_features_output]

    df_result.to_csv(absFilename_output, index=False, quoting=csv.QUOTE_ALL)


    print("len(df_result):")
    print(len(df_result))
    print("len(df_result[\"ROOTTWEETS_id_rootTweet\"]):")
    print(len(df_result["ROOTTWEETS_id_rootTweet"]))
    print("len(set(list(df_result[\"ROOTTWEETS_id_rootTweet\"]))):")
    print(len(set(list(df_result["ROOTTWEETS_id_rootTweet"]))))
    print("len(df_result[\"RETWEETS_rootTweetIdStr\"]):")
    print(len(df_result["RETWEETS_rootTweetIdStr"]))
    print("len(set(list(df_result[\"RETWEETS_rootTweetIdStr\"]))):")
    print(len(set(list(df_result["RETWEETS_rootTweetIdStr"]))))
    print("len(df_result[\"REPLIES_rootTweetIdStr\"]):")
    print(len(df_result["REPLIES_rootTweetIdStr"]))
    print("len(set(list(df_result[\"REPLIES_rootTweetIdStr\"]))):")
    print(len(set(list(df_result["REPLIES_rootTweetIdStr"]))))
    print("len(df_result[\"PRODUCERTWEETS_rootTweetIdStr\"]):")
    print(len(df_result["PRODUCERTWEETS_rootTweetIdStr"]))
    print("len(set(list(df_result[\"PRODUCERTWEETS_rootTweetIdStr\"]))):")
    print(len(set(list(df_result["PRODUCERTWEETS_rootTweetIdStr"]))))
    """

    

    
    # preprocess time series:

    # task = "retweets"
    task = "replies"

    list_featuresToStandardize = ["cascadeSize"]
    feature_base = "rootTweet_user_followersCount"

    
    if task == "retweets":
        # pattern_absFilename_input_TS = "D:\\vmwareSharedFolder\\TwitterDataAnalysis\\results\\timeSeries\\timeSeries_retweets_part*.csv"
        # pattern_absFilename_input_TS = "D:\\vmwareSharedFolder\\TwitterDataAnalysis\\results\\timeSeries\\timeSeries_retweets_part6.csv"
        pattern_absFilename_input_TS = "C:\\vmwareSharedFolder\\TwitterDataAnalysis\\results\\results_20210209\\timeSeries_retweets_indexStart=*_indexEnd=*.csv"

        absFilename_output_TS_allRootTweets_partialTimeStamps = "C:\\vmwareSharedFolder\\TwitterDataAnalysis\\results\\timeSeries\\preprocessed\\retweets\\timeSeries_retweets_preprocessed_allRootTweets_partialTimeStamps.csv"
        pattern_absFilename_output_TS_singleRootTweet = "C:\\vmwareSharedFolder\\TwitterDataAnalysis\\results\\timeSeries\\preprocessed\\retweets\\timeSeries_retweets_preprocessed_rootTweetID=ROOTTWEETID.csv"
    elif task == "replies":
        # pattern_absFilename_input_TS = "D:\\vmwareSharedFolder\\TwitterDataAnalysis\\results\\timeSeries\\timeSeries_replies.csv"
        pattern_absFilename_input_TS = "C:\\vmwareSharedFolder\\TwitterDataAnalysis\\results\\results_20210209\\timeSeries_replies_indexStart=*_indexEnd=*.csv"
        absFilename_output_TS_allRootTweets_partialTimeStamps = "C:\\vmwareSharedFolder\\TwitterDataAnalysis\\results\\timeSeries\\preprocessed\\replies\\timeSeries_replies_preprocessed_allRootTweets_partialTimeStamps.csv"
        pattern_absFilename_output_TS_singleRootTweet = "C:\\vmwareSharedFolder\\TwitterDataAnalysis\\results\\timeSeries\\preprocessed\\replies\\timeSeries_replies_preprocessed_rootTweetID=ROOTTWEETID.csv"
    
    # absFilename_input_retweets = "D:\\vmwareSharedFolder\\TwitterDataAnalysis\\results\\cascadeResults_temporal_retweets.csv"
    absFilename_input_retweets = "C:\\vmwareSharedFolder\\TwitterDataAnalysis\\results\\results_20210209\\cascadeResults_temporal_retweets.csv"

    list_absFilenames_input_TS = glob.glob(pattern_absFilename_input_TS)

    print("list_absFilenames_input_TS:")
    print(list_absFilenames_input_TS)
    print("len(list_absFilenames_input_TS):")
    print(len(list_absFilenames_input_TS))

    df_input_retweets = pd.read_csv(absFilename_input_retweets, dtype=str)

    print("len(df_input_retweets):")
    print(len(df_input_retweets))

    # df_temp1 = df_input_retweets[df_input_retweets["rootTweetIdStr"]==rootTweetIdStr, "rootTweet_user_followersCount"]
    print("print(len(df_input_retweets[\"rootTweet_user_followersCount\"])):")
    print(len(df_input_retweets["rootTweet_user_followersCount"]))

    



    df_TS = pd.DataFrame()

    for absFilename_input_TS in list_absFilenames_input_TS:

        df_input_TS = pd.read_csv(absFilename_input_TS, dtype=str)

        print("appending absFilename_input_TS:")
        print(absFilename_input_TS)

        if len(df_TS) <= 0:
            df_TS = df_input_TS.copy()
        else:
            df_TS = df_TS.append(df_input_TS)


    list_features_str = ["rootTweetIdStr", "idStr"]

    for feature in df_TS.columns:
        if feature in list_features_str:
            df_TS[feature] = df_TS[feature].astype(str)
        else:
            df_TS[feature] = df_TS[feature].astype(float)

    if task == "retweets":
        list_features_groupKey = ["rootTweetIdStr", "retweetAge_sec"]
    elif task == "replies":
        list_features_groupKey = ["rootTweetIdStr", "replyAge_sec"]

    df_TS = df_TS.drop_duplicates()
    df_TS = df_TS.sort_values(by=list_features_groupKey, ascending=[True, True])
    df_TS = df_TS.reset_index(drop=True)

    print("Raw data:")

    print("len(list(set(df_TS[\"rootTweetIdStr\"].tolist()))):")
    print(len(list(set(df_TS["rootTweetIdStr"].tolist()))))
    print("len(list(df_TS[\"idStr\"].tolist())):")
    print(len(list(df_TS["idStr"].tolist())))
    print("len(df_TS):")
    print(len(df_TS))

    if task == "retweets":
        df_TS = df_TS[df_TS["retweetAge_sec"] <= WINDOW_SEC]
    elif task == "replies":
        df_TS = df_TS[df_TS["replyAge_sec"] <= WINDOW_SEC]

    print("Raw data in time window:")

    print("len(list(set(df_TS[\"rootTweetIdStr\"].tolist()))):")
    print(len(list(set(df_TS["rootTweetIdStr"].tolist()))))
    print("len(list(df_TS[\"idStr\"].tolist())):")
    print(len(list(df_TS["idStr"].tolist())))
    print("len(df_TS):")
    print(len(df_TS))

    # print("df_TS.columns:")
    # print(df_TS.columns)
    # print(len(df_TS.columns))
    
    list_toGroup = list(set(df_TS.columns) - set(["idStr"]))

    # print("list_temp:")
    # print(list_temp)
    # print(len(list_temp))

    df_TS = df_TS[list_toGroup].groupby(list_features_groupKey).mean()
    df_TS = df_TS.reset_index()

    df_TS["rootTweetIdStr"] = "id" + df_TS["rootTweetIdStr"].astype(str)

    print("Aggregated data:")
    
    print("len(list(set(df_TS[\"rootTweetIdStr\"].tolist()))):")
    print(len(list(set(df_TS["rootTweetIdStr"].tolist()))))
    print("len(df_TS):")
    print(len(df_TS))

    print("absFilename_output_TS_allRootTweets_partialTimeStamps:")
    print(absFilename_output_TS_allRootTweets_partialTimeStamps)

    if not os.path.exists(os.path.dirname(absFilename_output_TS_allRootTweets_partialTimeStamps)):
        os.makedirs(os.path.dirname(absFilename_output_TS_allRootTweets_partialTimeStamps))
    
    df_TS.to_csv(absFilename_output_TS_allRootTweets_partialTimeStamps, index=False, quoting=csv.QUOTE_ALL)

    df_TS["hasData"] = 1
    list_rootTweetIdStr = list(set(df_TS["rootTweetIdStr"].tolist()))

    print("list_rootTweetIdStr:")
    print(list_rootTweetIdStr)
    print("len(list_rootTweetIdStr):")
    print(len(list_rootTweetIdStr))

    

    index_rootTweetIdStr = 0

    for rootTweetIdStr in list_rootTweetIdStr:

        print("index_rootTweetIdStr:")
        print(index_rootTweetIdStr)

        print("rootTweetIdStr:")
        print(rootTweetIdStr)

        df_output = pd.DataFrame()

        if task == "retweets":
            df_output["retweetAge_sec"] = range(0, WINDOW_SEC+1)
        elif task == "replies":
            df_output["replyAge_sec"] = range(0, WINDOW_SEC+1)

        df_output["rootTweetIdStr"] = rootTweetIdStr

        if task == "retweets":
            df_output = df_output[["rootTweetIdStr", "retweetAge_sec"]]
        elif task == "replies":
            df_output = df_output[["rootTweetIdStr", "replyAge_sec"]]

        df_temp = df_TS[df_TS["rootTweetIdStr"]==rootTweetIdStr]

        if task == "retweets":
            df_output = pd.merge(left=df_output, right=df_temp, how="left", on=["rootTweetIdStr", "retweetAge_sec"])
        elif task == "replies":
            df_output = pd.merge(left=df_output, right=df_temp, how="left", on=["rootTweetIdStr", "replyAge_sec"])

        df_output.loc[df_output["hasData"]!=1, "hasData"] = 0
        df_output["cascadeSize"] = df_output["hasData"].cumsum()
        df_output = df_output.drop(columns=["hasData"])

        df_temp1 = df_input_retweets[df_input_retweets["rootTweetIdStr"]==rootTweetIdStr.replace("id", "")]
        # print("len(df_temp1):")
        # print(len(df_temp1))
        # print(list(df_temp1.columns))
        # print(df_temp1["rootTweet_user_followersCount"].tolist())

        print("df_temp1.tail(1)[\"rootTweet_user_followersCount\"]:")
        print(df_temp1.tail(1)["rootTweet_user_followersCount"])

        if len(df_temp1) <= 0:
            print("rootTweetIdStr does not exist in df_input_retweets. Skip this rootTweetIdStr.")
            index_rootTweetIdStr += 1
            continue

        base = float(df_temp1.tail(1)["rootTweet_user_followersCount"])

        print("base:")
        print(base)

        # return

        df_output[list_featuresToStandardize] = df_output[list_featuresToStandardize].astype(float)

        for f in list_featuresToStandardize:
            df_output[f + "_stadardized"] = df_output[f] / base


        print("Final data:")
        
        print("len(list(set(df_output[\"rootTweetIdStr\"].tolist()))):")
        print(len(list(set(df_output["rootTweetIdStr"].tolist()))))
        if task == "retweets":
            print("len(list(set(df_output[\"retweetAge_sec\"].tolist()))):")
            print(len(list(set(df_output["retweetAge_sec"].tolist()))))    
        elif task == "replies":
            print("len(list(set(df_output[\"replyAge_sec\"].tolist()))):")
            print(len(list(set(df_output["replyAge_sec"].tolist()))))    
        print("len(df_output):")
        print(len(df_output))

        absFilename_output_TS_singleRootTweet = pattern_absFilename_output_TS_singleRootTweet.replace("ROOTTWEETID", rootTweetIdStr.replace("id", ""))
        print("absFilename_output_TS_singleRootTweet:")
        print(absFilename_output_TS_singleRootTweet)

        if not os.path.exists(os.path.dirname(absFilename_output_TS_singleRootTweet)):
            os.makedirs(os.path.dirname(absFilename_output_TS_singleRootTweet))

        df_output.to_csv(absFilename_output_TS_singleRootTweet, index=False, quoting=csv.QUOTE_ALL)

        index_rootTweetIdStr += 1

        

if __name__ == "__main__":
    main(sys.argv[1:])