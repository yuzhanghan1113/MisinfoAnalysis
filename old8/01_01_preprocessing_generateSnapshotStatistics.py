import json
import os
import getopt
import sys
import random
import pandas as pd
import traceback
import gc 
import time
import glob
import re
from datetime import datetime, timezone
import numpy as np
import shutil, errno
import filecmp


  
def main(argv):

    random.seed(1113)
    
    
    # absFilename_input_rootTweets = "D:\\vmwareSharedFolder\\TwitterDataCollection\\programs\\input\\rootTweets_dateRetrieval=20201008.txt"
    absFilename_input_rootTweets = "D:\\vmwareSharedFolder\\TwitterDataCollection\\programs\\input\\rootTweets_dateRetrieval=20201111.txt"
    # str_path_base = "D:\\vmwareSharedFolder\\TwitterDataCollection\\data\\PFN\\snapshot\\dateRetrieval=20201008" + os.path.sep
    str_path_base = "D:\\vmwareSharedFolder\\TwitterDataCollection\\data\\PFN\\snapshot\\dateRetrieval=20201111" + os.path.sep
    # absFilename_input_rootTweetList = "D:\\vmwareSharedFolder\\TwitterDataCollection\\programs\\input\\mapping_tweetID_searchString_20200912_allOldTweets_withScreenNameGreatedAt.csv"
    # absFilename_input_rootTweetList = "D:\\vmwareSharedFolder\\TwitterDataCollection\\programs\\input\\mapping_tweetID_searchString_20201008_allOldTweets_withScreenName.csv"
    absFilename_input_rootTweetList = "D:\\vmwareSharedFolder\\TwitterDataCollection\\programs\\input\\mapping_tweetID_searchString_20201107_allOldTweets_withScreenName.csv"
    # absFilename_output = str_path_base + "statistics_retweets_20201008.csv"
    absFilename_output = str_path_base + "statistics_retweets_20201111.csv"


    dict_rootTweetID2RetweetCount = {}
    dict_rootTweetID2CreatedAt = {}
    df_rootTweetID2CreatedAt = pd.DataFrame()

    file_input_rootTweets = open(absFilename_input_rootTweets, "r", encoding="utf-8")

    index_row = 0

    for line in file_input_rootTweets:
        json_tweet = json.loads(line)
        # print("json_tweet:")
        # print(json_tweet)
        # print("json_tweet['id_str']:")
        # print(json_tweet["id_str"])
        if "id_str" not in json_tweet:
            continue
            
        str_rootTweetID = str(json_tweet["id_str"])
        int_retweetCount = json_tweet["retweet_count"]
        dict_rootTweetID2RetweetCount[str_rootTweetID] = int_retweetCount
        dict_rootTweetID2CreatedAt[str_rootTweetID] = str_rootTweetID = str(json_tweet["created_at"])
        df_rootTweetID2CreatedAt.loc[index_row, "id_str"] = str(json_tweet["id_str"])
        df_rootTweetID2CreatedAt.loc[index_row, "created_at"] = str(json_tweet["created_at"])
        index_row += 1
    file_input_rootTweets.close()

    print("len(dict_rootTweetID2RetweetCount.keys())")
    print(len(dict_rootTweetID2RetweetCount.keys()))


    df_input_rootTweetList = pd.read_csv(absFilename_input_rootTweetList, dtype=str, quotechar='"', delimiter=',', escapechar='\\')

    print("len(df_input_rootTweetList):")
    print(len(df_input_rootTweetList))

    df_input_rootTweetList = pd.merge(left=df_input_rootTweetList, right=df_rootTweetID2CreatedAt, how="left", left_on=["tweetID"], right_on=["id_str"])

    print("len(df_input_rootTweetList):")
    print(len(df_input_rootTweetList))

    df_input_rootTweetList = df_input_rootTweetList.dropna(subset=["created_at"])
    df_input_rootTweetList = df_input_rootTweetList.reset_index(drop=True)

    print("len(df_input_rootTweetList):")
    print(len(df_input_rootTweetList))

    # return

    df_result = pd.DataFrame()

    index_row = 0

    for index_row in range(0, len(df_input_rootTweetList)):
        id_rootTweet = df_input_rootTweetList.loc[index_row, "tweetID"]
        str_createdAt_rootTweet = df_input_rootTweetList.loc[index_row, "created_at"]

        print("id_rootTweet:")
        print(id_rootTweet)

        print("str_createdAt_rootTweet:")
        print(str_createdAt_rootTweet)

        # Sun Sep 06 16:28:47 +0000 2020

        date_createdAt_rootTweet = datetime.strptime(str_createdAt_rootTweet, "%a %b %d %H:%M:%S %z %Y")

        print("date_createdAt_rootTweet:")
        print(date_createdAt_rootTweet)

        absFilename_input_retweets = str_path_base + "retweets" + os.path.sep + "rootTweetID=" + id_rootTweet + os.path.sep + "retweets_rootTweetID=" + id_rootTweet + ".txt"
        
        if not os.path.exists(absFilename_input_retweets):
            print("retweet file does not exist. skip.")
            continue

        file_input_retweets = open(absFilename_input_retweets, "r", encoding="utf-8")

        print("absFilename_input_retweets:")
        print(absFilename_input_retweets)

        num_retweets = sum(1 for line in file_input_retweets)
        print("num_retweets:")
        print(num_retweets)
        file_input_retweets.close()

        num_retweets_24hrs = 0
        num_retweets_48hrs = 0
        num_retweets_72hrs = 0

        file_input_retweets = open(absFilename_input_retweets, "r", encoding="utf-8")

        for line in file_input_retweets:
            json_tweet = json.loads(line)
            # "created_at": "Thu Mar 05 00:31:08 +0000 2020"
            str_createdAt_retweet = json_tweet["created_at"]
            date_createdAt_retweet = datetime.strptime(str_createdAt_retweet, "%a %b %d %H:%M:%S %z %Y")
            diff_hours = (date_createdAt_retweet - date_createdAt_rootTweet).total_seconds() / 3600

            # print(str_createdAt_rootTweet)
            # print(str_createdAt_retweet)
            # print(diff_hours)
            
            if diff_hours <= 24:
                num_retweets_24hrs += 1
            if diff_hours <= 48:
                num_retweets_48hrs += 1
            if diff_hours <= 72:
                num_retweets_72hrs += 1

        file_input_retweets.close()

        ptg_retweets_24hrs = -1
        ptg_retweets_48hrs = -1
        ptg_retweets_72hrs = -1

        if num_retweets > 0:
            ptg_retweets_24hrs = num_retweets_24hrs / num_retweets
            ptg_retweets_48hrs = num_retweets_48hrs / num_retweets
            ptg_retweets_72hrs = num_retweets_72hrs / num_retweets

        print("ptg_retweets_24hrs:")
        print(ptg_retweets_24hrs)
        print("ptg_retweets_48hrs:")
        print(ptg_retweets_48hrs)
        print("ptg_retweets_72hrs:")
        print(ptg_retweets_72hrs)

        num_retweets_total = -1
        if id_rootTweet in dict_rootTweetID2RetweetCount.keys():
            num_retweets_total = dict_rootTweetID2RetweetCount[id_rootTweet]

        print("num_retweets_total:")
        print(num_retweets_total)

        ptg_retweets_total_24hrs = -1
        ptg_retweets_total_48hrs = -1
        ptg_retweets_total_72hrs = -1

        if num_retweets_total > 0:
            ptg_retweets_total_24hrs = num_retweets_24hrs / num_retweets_total
            ptg_retweets_total_48hrs = num_retweets_48hrs / num_retweets_total
            ptg_retweets_total_72hrs = num_retweets_72hrs / num_retweets_total

        print("ptg_retweets_total_24hrs:")
        print(ptg_retweets_total_24hrs)
        print("ptg_retweets_total_48hrs:")
        print(ptg_retweets_total_48hrs)
        print("ptg_retweets_total_72hrs:")
        print(ptg_retweets_total_72hrs)


        df_result.loc[index_row, "id_rootTweet"] = str(id_rootTweet)
        df_result.loc[index_row, "str_createdAt_rootTweet"] = str(str_createdAt_rootTweet)
        df_result.loc[index_row, "num_retweets"] = str(num_retweets)
        df_result.loc[index_row, "num_retweets_24hrs"] = str(num_retweets_24hrs)
        df_result.loc[index_row, "num_retweets_48hrs"] = str(num_retweets_48hrs)
        df_result.loc[index_row, "num_retweets_72hrs"] = str(num_retweets_72hrs)
        df_result.loc[index_row, "ptg_retweets_24hrs"] = str(ptg_retweets_24hrs)
        df_result.loc[index_row, "ptg_retweets_48hrs"] = str(ptg_retweets_48hrs)
        df_result.loc[index_row, "ptg_retweets_72hrs"] = str(ptg_retweets_72hrs)
        df_result.loc[index_row, "num_retweets_total"] = str(num_retweets_total)
        df_result.loc[index_row, "ptg_retweets_total_24hrs"] = str(ptg_retweets_total_24hrs)
        df_result.loc[index_row, "ptg_retweets_total_48hrs"] = str(ptg_retweets_total_48hrs)
        df_result.loc[index_row, "ptg_retweets_total_72hrs"] = str(ptg_retweets_total_72hrs)

        index_row += 1

        df_result.to_csv(absFilename_output, index=False) 
    
    df_result.to_csv(absFilename_output, index=False)          


if __name__ == "__main__":
    main(sys.argv[1:])
