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




    # absFilename_input_rootTweets = "D:\\vmwareSharedFolder\\TwitterDataCollection\\data\\PFN\\snapshot\\dateRetrieval=20201008\\rootTweets_selected_factcheckArticleRep_20201010.csv"
    absFilename_input_rootTweets = "D:\\vmwareSharedFolder\\TwitterDataCollection\\data\\PFN\\snapshot\\dateRetrieval=20201008\\rootTweets_selected_factcheckArticleRep_20201010_test.csv"
    path_input_rootTweetRetweets = "D:\\vmwareSharedFolder\\TwitterDataCollection\\data\\PFN\\snapshot\\dateRetrieval=20201008\\retweets\\" 

    # df_input_rootTweets = pd.read_csv(absFilename_input_rootTweets, dtype=str)

    df_input_rootTweets = df_input_rootTweets.sort_values(by=["id_rootTweet"], ascending=True)
    df_input_rootTweets = df_input_rootTweets.reset_index(drop=True)

    df_output = pd.DataFrame()


    for index_row_rootTweet in range(0, len(c)):

        id_rootTweet = index_row.loc[index_row, "id_rootTweet"]

        absFilename_input_retweets = path_input_rootTweetRetweets + "rootTweetID=" + id_rootTweet + os.path.sep + "retweets_rootTweetID=" + id_rootTweet + ".txt"

        df_retweets = pd.DataFrame()

        file_input_retweets = open(absFilename_input_retweets, "r", encoding="utf-8")        

        index_row_retweet = 0

        for line in file_input_retweets:

            dict_retweet = json.loads(line)
            df_retweets.loc[index_row_retweet, "createdAt"] = str(dict_retweet["created_at"])
            df_retweets.loc[index_row_retweet, "fullText"] = str(dict_retweet["full_text"])
            df_retweets.loc[index_row_retweet, "fullText"] = str(dict_retweet["full_text"])
            df_retweets.loc[index_row_retweet, "fullText"] = str(dict_retweet["full_text"])
            df_retweets.loc[index_row_retweet, "entities_hashtags"] = str(dict_retweet["entities"]["hashtags"])
            df_retweets.loc[index_row_retweet, "entities_symbols"] = str(dict_retweet["entities"]["symbols"])
            df_retweets.loc[index_row_retweet, "count_entities_userMentions"] = len(dict_retweet["entities"]["user_mentions"])
            df_retweets.loc[index_row_retweet, "user_description"] = str(dict_retweet["user"]["description"])
            df_retweets.loc[index_row_retweet, "user_protected"] = str(dict_retweet["user"]["protected"])
            df_retweets.loc[index_row_retweet, "user_followersCount"] = dict_retweet["user"]["followers_count"]
            df_retweets.loc[index_row_retweet, "user_friendsCount"] = dict_retweet["user"]["friends_count"]
            df_retweets.loc[index_row_retweet, "user_listedCount"] = dict_retweet["user"]["listed_count"]
            df_retweets.loc[index_row_retweet, "user_createdAt"] = str(dict_retweet["user"]["created_at"])
            df_retweets.loc[index_row_retweet, "user_favouritesCount"] = dict_retweet["user"]["favourites_count"]
            df_retweets.loc[index_row_retweet, "user_geoEnabled"] = str(dict_retweet["user"]["geo_enabled"])
            df_retweets.loc[index_row_retweet, "user_verified"] = str(dict_retweet["user"]["verified"])
            df_retweets.loc[index_row_retweet, "user_statusesCount"] = dict_retweet["user"]["statuses_count"]
            df_retweets.loc[index_row_retweet, "user_lang"] = str(dict_retweet["user"]["lang"]
            df_retweets.loc[index_row_retweet, "user_profileBackgroundColor"] = str(dict_retweet["user"]["profile_background_color"])
            df_retweets.loc[index_row_retweet, "user_profileTextColor"] = str(dict_retweet["user"]["profile_text_color"])
            df_retweets.loc[index_row_retweet, "user_profileUseBackgroundImage"] = str(dict_retweet["user"]["profile_use_background_image"])
            df_retweets.loc[index_row_retweet, "user_defaultProfile"] = str(dict_retweet["user"]["default_profile"])
            df_retweets.loc[index_row_retweet, "user_following"] = str(dict_retweet["user"]["following"])
            df_retweets.loc[index_row_retweet, "geo"] = str(dict_retweet["geo"])
            df_retweets.loc[index_row_retweet, "coordinates"] = str(dict_retweet["coordinates"])
            df_retweets.loc[index_row_retweet, "place"] = str(dict_retweet["place"])
            df_retweets.loc[index_row_retweet, "retweetedStatus_retweetCount"] = dict_retweet["retweeted_status"]["retweet_count"]
            df_retweets.loc[index_row_retweet, "retweetedStatus_favoriteCount"] = dict_retweet["retweeted_status"]["favorite_count"]

            index_row_retweet += 1

        print(dict_retweet)
	

if __name__ == "__main__":
    main(sys.argv[1:])
