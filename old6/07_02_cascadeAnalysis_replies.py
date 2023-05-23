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



def preprocessText(text, nlp, list_stopWords):

    text = re.sub(r"#", "", text)
    # print(text)
    text = preprocessor.clean(text)
    text = text.lower()
    text = re.sub(r"\d+", "", text)
    
    list_words = nlp(text)
    list_words = [token.lemma_ for token in list_words]
    list_words = [w for w in list_words if w != "-PRON-"]
    list_words = [w for w in list_words if w not in list_stopWords]
    text = " ".join(list_words)
    
    return text

def textStats(text:str):

    text_org = str(text)

    dict_result = {}
    text = text.lower()
    text = re.sub(r"#", "", text)
    text = preprocessor.clean(text)
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"[0-9]", " ", text)
    list_words = re.split(" +|\t+|\n+", text)
    list_words = [w for w in list_words if w != ""]
    temp_text = re.sub(r" +|\t+|\n+", "", text)

    # print("text_org:")
    # print(text_org)
    # print("text:")
    # print(text)
    # print("temp_text:")
    # print(temp_text)

    dict_result["charCount"] = len(temp_text)
    dict_result["wordCount"] = len(list_words)
    dict_result["uqWordCount"] = len(list(set(list_words)))

    temp_sentences = [s for s in re.split(r"[.!?\n]+", text_org) if s != ""]
    # print(temp_sentences)
    dict_result["sentenceCount"] = len(temp_sentences)

    if dict_result["wordCount"] <= 0:
        dict_result["charsPerWord"] = np.nan
    else:
        dict_result["charsPerWord"] = dict_result["charCount"]/dict_result["wordCount"]

    if dict_result["sentenceCount"] <= 0:
        dict_result["wordsPerSentence"] = np.nan
    else:
        dict_result["wordsPerSentence"] = dict_result["wordCount"]/dict_result["sentenceCount"]

    dict_result["readability"] = textstat.flesch_reading_ease(text_org)

    return dict_result


def main(argv):

    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)
    pd.set_option('display.max_colwidth', -1)
    
    random.seed(1113)

    opts, args = getopt.getopt(argv, '', ["absFilename_input_rootTweets=", "path_input_rootTweetReplies=", "absFilename_output_temporal=", "absFilename_log="])

    print(opts)

    for opt, arg in opts:
        if opt == '--absFilename_input_rootTweets':
            absFilename_input_rootTweets = arg
        if opt == '--path_input_rootTweetReplies':
            path_input_rootTweetReplies = arg
        if opt == '--absFilename_output_temporal':
            absFilename_output_temporal = arg
        elif opt == '--absFilename_log':
            absFilename_log = arg

    # text = "We went from 4 years of R\n\n!??..\nusia rigged the election \nto elections can’t be rigged really fast didn’t we?!"
    # text = "We went from 4 years of R   really fast didn’t we?!"
    # text = "President @RealDonaldTrump should use every legal and constitutional remedy to restore Americans’ faith in our elections. This fight is about the fundamental fairness and integrity of our election system."
    # dict_result = textStats(text)

    # print(dict_result["num_chars"])
    # print(dict_result["num_words"])
    # print(dict_result["num_uqWords"])
    # print(dict_result["num_sentences"])
    # print(dict_result["ratio_charsPerWord"])
    # print(dict_result["ratio_wordsPerSentences"])
    # print(dict_result["readability"])

    # return

    list_timestamp = [t for t in range(0, 1440+10, 10)]
    list_timestamp += [t for t in range(0, 4320+60, 60)]
    list_timestamp += [t for t in range(0, 10080+1440, 1440)]

    list_timestamp = sorted(list(set(list_timestamp)))


    print("list_timestamp:")
    print(list_timestamp)

    str_date_current = "Thu Nov 12 00:00:00 +0000 2020"
    date_current = datetime.strptime(str_date_current, "%a %b %d %H:%M:%S %z %Y")

    # return


    # absFilename_input_rootTweets = "D:\\vmwareSharedFolder\\TwitterDataCollection\\data\\PFN\\snapshot\\dateRetrieval=20201008\\rootTweets_selected_factcheckArticleRep_20201010.csv"
    # # absFilename_input_rootTweets = "D:\\vmwareSharedFolder\\TwitterDataCollection\\data\\PFN\\snapshot\\dateRetrieval=20201008\\rootTweets_selected_factcheckArticleRep_20201010_test.csv"
    # path_input_rootTweetReplies = "D:\\vmwareSharedFolder\\TwitterDataCollection\\data\\PFN\\snapshot\\dateRetrieval=20201008\\replies\\" 
    # absFilename_output_temporal = "D:\\vmwareSharedFolder\\TwitterDataAnalysis\\results\\cascadeResults_temporal_replies.csv"
    # absFilename_log = "D:\\vmwareSharedFolder\\TwitterDataAnalysis\\programs\\logs\\07_02_cascadeAnalysis_replies.log"

    if not os.path.exists(os.path.dirname(absFilename_log)):
        os.makedirs(os.path.dirname(absFilename_log))
    file_log = open(absFilename_log, "w+")
    file_log.close()
    file_log = open(absFilename_log, "a")

    if os.path.exists(absFilename_output_temporal):
        df_output = pd.read_csv(absFilename_output_temporal, dtype=str)
        print("Output file exists. Load and continue with this file.")
    else:
        df_output = pd.DataFrame()
        print("Output file does not exists. Start from scratch.")
    
    print("len(df_output):")
    print(len(df_output))



    df_input_rootTweets = pd.read_csv(absFilename_input_rootTweets, dtype=str)

    print("len(df_input_rootTweets):")
    print(len(df_input_rootTweets))

    df_input_rootTweets = df_input_rootTweets.sort_values(by=["id_rootTweet"], ascending=True)
    df_input_rootTweets = df_input_rootTweets.reset_index(drop=True)

    print("len(df_input_rootTweets):")
    print(len(df_input_rootTweets))

    

    analyzer = SentimentIntensityAnalyzer()
    nlp = spacy.load('en', disable=['parser', 'ner'])
    list_stopWords = list(set(stopwords.words('english')))


    list_affects = ["fear", "anger", "anticip", "trust", "surprise", "positive", "negative", "sadness", "disgust", "joy"]
    list_textStats = ["charCount", "wordCount", "uqWordCount", "sentenceCount", "charsPerWord", "wordsPerSentence", "readability"]

    index_row_rootTweet = "EMPTY"
    id_rootTweet = "EMPTY"
    index_row_reply = "EMPTY"
    id_reply = "EMPTY"
    timestamp = "EMPTY"
    line = "EMPTY"

    index_row_output = len(df_output)

    for index_row_rootTweet in range(0, len(df_input_rootTweets)):

        id_rootTweet = df_input_rootTweets.loc[index_row_rootTweet, "id_rootTweet"]

        print("id_rootTweet:")
        print(id_rootTweet)

        try:

            if (len(df_output) > 0) and (id_rootTweet in df_output["rootTweetIdStr"].tolist()):
                print("id_rootTweet already processed. Skip.")
                continue


            date_createdAt_rootTweet = datetime.strptime(df_input_rootTweets.loc[index_row_rootTweet, "str_createdAt_rootTweet"], "%a %b %d %H:%M:%S %z %Y")

            print("date_createdAt_rootTweet:")
            print(date_createdAt_rootTweet)

            absFilename_input_replies = path_input_rootTweetReplies + "rootTweetID=" + id_rootTweet + os.path.sep + "replies_rootTweetID=" + id_rootTweet + ".txt"

            if not os.path.exists(absFilename_input_replies):
                print("Reply file does not exist:")
                print("absFilename_input_replies:")
                print(absFilename_input_replies)
                print("Skip this root tweet.")
                continue

            num_lines = 0
            file_input_replies = open(absFilename_input_replies, "r", encoding="utf-8")    
            for line in file_input_replies:    
                num_lines += 1
            file_input_replies.close()
            print("num_lines:")
            print(num_lines)


            df_replies = pd.DataFrame()

            

            file_input_replies = open(absFilename_input_replies, "r", encoding="utf-8")        

            index_row_reply = 0

            for line in file_input_replies:

                dict_reply = json.loads(line)

                # print("dict_reply:")
                # print(dict_reply)

                # if "id_str" not in dict_reply:
                #     continue
                if "id_str" not in dict_reply:
                    print("index_row_rootTweet:")
                    print(index_row_rootTweet)
                    print("id_rootTweet:")
                    print(id_rootTweet)
                    print("index_row_reply:")
                    print(index_row_reply)
                    print("dict_reply[\"id_str\"]:")
                    print(dict_reply["id_str"])
                    print("line:")
                    print(line)
                    print("id_str is not in the line. Skip.")
                    continue

                id_reply = str(dict_reply["id_str"])

                if dict_reply["id_str"] == id_rootTweet:
                    print("index_row_rootTweet:")
                    print(index_row_rootTweet)
                    print("id_rootTweet:")
                    print(id_rootTweet)
                    print("index_row_reply:")
                    print(index_row_reply)
                    print("id_reply:")
                    print(id_reply)
                    print("line:")
                    print(line)
                    print("This is the root tweet, not a reply to the root tweet. Skip.")
                    continue
                if ("in_reply_to_status_id_str" not in dict_reply) or (dict_reply["in_reply_to_status_id_str"]==None) or (str(dict_reply["in_reply_to_status_id_str"])=="None"):
                    print("index_row_rootTweet:")
                    print(index_row_rootTweet)
                    print("id_rootTweet:")
                    print(id_rootTweet)
                    print("index_row_reply:")
                    print(index_row_reply)
                    print("id_reply:")
                    print(id_reply)
                    print("line:")
                    print(line)
                    print("This is not a reply. Skip.")
                    continue
                if dict_reply["in_reply_to_status_id_str"] != id_rootTweet:
                    print("index_row_rootTweet:")
                    print(index_row_rootTweet)
                    print("id_rootTweet:")
                    print(id_rootTweet)
                    print("index_row_reply:")
                    print(index_row_reply)
                    print("id_reply:")
                    print(id_reply)
                    print("line:")
                    print(line)
                    print("This is a reply, but not made to the root tweet. Skip.")
                    continue


                df_replies.loc[index_row_reply, "idStr"] = str(dict_reply["id_str"])
                df_replies.loc[index_row_reply, "createdAt"] = str(dict_reply["created_at"])

                if "full_text" in dict_reply:
                    df_replies.loc[index_row_reply, "fullText"] = str(dict_reply["full_text"])
                elif "text" in dict_retweet:
                    df_replies.loc[index_row_reply, "fullText"] = str(dict_reply["text"])
                

                df_replies.loc[index_row_reply, "count_entities_hashtags"] = len([t["text"] for t in dict_reply["entities"]["hashtags"]])
                df_replies.loc[index_row_reply, "entities_hashtags"] = str(dict_reply["entities"]["hashtags"])
                df_replies.loc[index_row_reply, "entities_symbols"] = str(dict_reply["entities"]["symbols"])
                df_replies.loc[index_row_reply, "count_entities_userMentions"] = len(dict_reply["entities"]["user_mentions"])
                df_replies.loc[index_row_reply, "user_description"] = str(dict_reply["user"]["description"])
                df_replies.loc[index_row_reply, "user_protected"] = str(dict_reply["user"]["protected"])
                df_replies.loc[index_row_reply, "user_followersCount"] = dict_reply["user"]["followers_count"]
                df_replies.loc[index_row_reply, "user_friendsCount"] = dict_reply["user"]["friends_count"]
                df_replies.loc[index_row_reply, "user_listedCount"] = dict_reply["user"]["listed_count"]
                df_replies.loc[index_row_reply, "user_createdAt"] = str(dict_reply["user"]["created_at"])
                df_replies.loc[index_row_reply, "user_favouritesCount"] = dict_reply["user"]["favourites_count"]
                df_replies.loc[index_row_reply, "user_geoEnabled"] = str(dict_reply["user"]["geo_enabled"])
                df_replies.loc[index_row_reply, "user_verified"] = str(dict_reply["user"]["verified"])
                df_replies.loc[index_row_reply, "user_statusesCount"] = dict_reply["user"]["statuses_count"]
                df_replies.loc[index_row_reply, "user_lang"] = str(dict_reply["user"]["lang"])
                df_replies.loc[index_row_reply, "user_profileBackgroundColor"] = str(dict_reply["user"]["profile_background_color"])
                df_replies.loc[index_row_reply, "user_profileTextColor"] = str(dict_reply["user"]["profile_text_color"])
                df_replies.loc[index_row_reply, "user_profileUseBackgroundImage"] = str(dict_reply["user"]["profile_use_background_image"])
                df_replies.loc[index_row_reply, "user_defaultProfile"] = str(dict_reply["user"]["default_profile"])
                df_replies.loc[index_row_reply, "user_following"] = str(dict_reply["user"]["following"])
                df_replies.loc[index_row_reply, "geo"] = str(dict_reply["geo"])
                df_replies.loc[index_row_reply, "coordinates"] = str(dict_reply["coordinates"])
                df_replies.loc[index_row_reply, "place"] = str(dict_reply["place"])
                if "possibly_sensitive" in dict_reply:
                    df_replies.loc[index_row_reply, "possiblySensitive"] = str(dict_reply["possibly_sensitive"])


                date_createdAt_reply = datetime.strptime(df_replies.loc[index_row_reply, "createdAt"], "%a %b %d %H:%M:%S %z %Y")
                df_replies.loc[index_row_reply, "replyAge_min"] = (date_createdAt_reply - date_createdAt_rootTweet).total_seconds()/60

                date_user_createdAt_reply = datetime.strptime(df_replies.loc[index_row_reply, "user_createdAt"], "%a %b %d %H:%M:%S %z %Y")
                df_replies.loc[index_row_reply, "user_accountAge_day"] = (date_current - date_user_createdAt_reply).total_seconds()/(60*60*24)



                dict_sentimentScores = analyzer.polarity_scores(df_replies.loc[index_row_reply, "fullText"])
                df_replies.loc[index_row_reply, "fullText_sentiment_neg"] = dict_sentimentScores["neg"]
                df_replies.loc[index_row_reply, "fullText_sentiment_neu"] = dict_sentimentScores["neu"]
                df_replies.loc[index_row_reply, "fullText_sentiment_pos"] = dict_sentimentScores["pos"]
                df_replies.loc[index_row_reply, "fullText_sentiment_compound"] = dict_sentimentScores["compound"]

                df_replies.loc[index_row_reply, "fullText_subjectivity"] = TextBlob(df_replies.loc[index_row_reply, "fullText"]).sentiment.subjectivity

                str_text_preprocessed = preprocessText(df_replies.loc[index_row_reply, "fullText"], nlp, list_stopWords)
                results_emotion = NRCLex(str_text_preprocessed)
                for affect in list_affects:
                    df_replies.loc[index_row_reply, "fullText_emotion_" + affect] = results_emotion.affect_frequencies[affect]
                    # print(affect)
                    # print(df_replies.loc[index_row_reply, "fullText_emotion_" + affect])

                dict_textStats = textStats(df_replies.loc[index_row_reply, "fullText"])
                for metric in list_textStats:
                    df_replies.loc[index_row_reply, "fullText_textStats_" + metric] = dict_textStats[metric]



                dict_sentimentScores = analyzer.polarity_scores(df_replies.loc[index_row_reply, "user_description"])
                df_replies.loc[index_row_reply, "user_description_sentiment_neg"] = dict_sentimentScores["neg"]
                df_replies.loc[index_row_reply, "user_description_sentiment_neu"] = dict_sentimentScores["neu"]
                df_replies.loc[index_row_reply, "user_description_sentiment_pos"] = dict_sentimentScores["pos"]
                df_replies.loc[index_row_reply, "user_description_sentiment_compound"] = dict_sentimentScores["compound"]

                df_replies.loc[index_row_reply, "user_description_subjectivity"] = TextBlob(df_replies.loc[index_row_reply, "user_description"]).sentiment.subjectivity

                str_text_preprocessed = preprocessText(df_replies.loc[index_row_reply, "user_description"], nlp, list_stopWords)
                results_emotion = NRCLex(str_text_preprocessed)
                for affect in list_affects:
                    df_replies.loc[index_row_reply, "user_description_emotion_" + affect] = results_emotion.affect_frequencies[affect]
                    # print(affect)
                    # print(df_replies.loc[index_row_reply, "user_description_emotion_" + affect])

                dict_textStats = textStats(df_replies.loc[index_row_reply, "user_description"])
                for metric in list_textStats:
                    df_replies.loc[index_row_reply, "user_description_textStats_" + metric] = dict_textStats[metric]

                if "retweeted_status" in dict_reply:
                    df_replies.loc[index_row_reply, "retweetedStatus_idStr"] = str(dict_reply["retweeted_status"]["id_str"])
                    df_replies.loc[index_row_reply, "retweetedStatus_createdAt"] = str(dict_reply["retweeted_status"]["created_at"])
                    
                    if "full_text" in dict_reply["retweeted_status"]:
                        df_replies.loc[index_row_reply, "retweetedStatus_fullText"] = str(dict_reply["retweeted_status"]["full_text"])
                    elif "text" in dict_reply["retweeted_status"]:
                        df_replies.loc[index_row_reply, "retweetedStatus_fullText"] = str(dict_reply["retweeted_status"]["text"])
                    
                    # df_replies.loc[index_row_reply, "retweetedStatus_entities_hashtags"] = str([t["text"] for t in dict_reply["retweeted_status"]["entities"]["hashtags"]])
                    df_replies.loc[index_row_reply, "retweetedStatus_entities_hashtags"] = str(dict_reply["retweeted_status"]["entities"]["hashtags"])
                    df_replies.loc[index_row_reply, "retweetedStatus_entities_symbols"] = str(dict_reply["retweeted_status"]["entities"]["symbols"])
                    df_replies.loc[index_row_reply, "retweetedStatus_entities_userMentions_screenName"] = str([u["screen_name"] for u in dict_reply["retweeted_status"]["entities"]["user_mentions"]])
                    df_replies.loc[index_row_reply, "count_retweetedStatus_entities_userMentions"] = len(dict_reply["retweeted_status"]["entities"]["user_mentions"])
                    df_replies.loc[index_row_reply, "count_retweetedStatus_entities_urls"] = len(dict_reply["retweeted_status"]["entities"]["urls"])
                    df_replies.loc[index_row_reply, "retweetedStatu_entitiess_urls"] = str([u for u in dict_reply["retweeted_status"]["entities"]["urls"]])
                    df_replies.loc[index_row_reply, "retweetedStatus_inReplyToStatusIdStr"] = str(dict_reply["retweeted_status"]["in_reply_to_status_id_str"])
                    df_replies.loc[index_row_reply, "retweetedStatus_inReplyToScreenName"] = str(dict_reply["retweeted_status"]["in_reply_to_screen_name"])
                    df_replies.loc[index_row_reply, "retweetedStatus_user_idStr"] = str(dict_reply["retweeted_status"]["user"]["id_str"])
                    df_replies.loc[index_row_reply, "retweetedStatus_user_screenName"] = str(dict_reply["retweeted_status"]["user"]["screen_name"])
                    df_replies.loc[index_row_reply, "retweetedStatus_user_location"] = str(dict_reply["retweeted_status"]["user"]["location"])
                    df_replies.loc[index_row_reply, "retweetedStatus_user_description"] = str(dict_reply["retweeted_status"]["user"]["description"])
                    df_replies.loc[index_row_reply, "retweetedStatus_user_url"] = str(dict_reply["retweeted_status"]["user"]["url"])
                    df_replies.loc[index_row_reply, "retweetedStatus_user_entities"] = str(dict_reply["retweeted_status"]["user"]["entities"])
                    df_replies.loc[index_row_reply, "retweetedStatus_user_protected"] = str(dict_reply["retweeted_status"]["user"]["protected"])
                    df_replies.loc[index_row_reply, "retweetedStatus_user_followersCount"] = dict_reply["retweeted_status"]["user"]["followers_count"]
                    df_replies.loc[index_row_reply, "retweetedStatus_user_friendsCount"] = dict_reply["retweeted_status"]["user"]["friends_count"]
                    df_replies.loc[index_row_reply, "retweetedStatus_user_listedCount"] = dict_reply["retweeted_status"]["user"]["listed_count"]
                    df_replies.loc[index_row_reply, "retweetedStatus_user_createdAt"] = str(dict_reply["retweeted_status"]["user"]["created_at"])
                    df_replies.loc[index_row_reply, "retweetedStatus_user_favouritesCount"] = dict_reply["retweeted_status"]["user"]["favourites_count"]
                    df_replies.loc[index_row_reply, "retweetedStatus_user_geoEnabled"] = str(dict_reply["retweeted_status"]["user"]["geo_enabled"])
                    df_replies.loc[index_row_reply, "retweetedStatus_user_verified"] = str(dict_reply["retweeted_status"]["user"]["verified"])
                    df_replies.loc[index_row_reply, "retweetedStatus_user_statusesCount"] = dict_reply["retweeted_status"]["user"]["statuses_count"]
                    df_replies.loc[index_row_reply, "retweetedStatus_user_lang"] = str(dict_reply["retweeted_status"]["user"]["lang"])
                    df_replies.loc[index_row_reply, "retweetedStatus_user_contributorsEnabled"] = str(dict_reply["retweeted_status"]["user"]["contributors_enabled"])
                    df_replies.loc[index_row_reply, "retweetedStatus_user_isTranslator"] = str(dict_reply["retweeted_status"]["user"]["is_translator"])
                    df_replies.loc[index_row_reply, "retweetedStatus_user_isTranslationEnabled"] = str(dict_reply["retweeted_status"]["user"]["is_translation_enabled"])
                    df_replies.loc[index_row_reply, "retweetedStatus_user_profileBackgroundColor"] = str(dict_reply["retweeted_status"]["user"]["profile_background_color"])
                    df_replies.loc[index_row_reply, "retweetedStatus_user_profileTextColor"] = str(dict_reply["retweeted_status"]["user"]["profile_text_color"])
                    df_replies.loc[index_row_reply, "retweetedStatus_user_profileUseBackgroundImage"] = str(dict_reply["retweeted_status"]["user"]["profile_use_background_image"])
                    df_replies.loc[index_row_reply, "retweetedStatus_user_defaultProfile"] = str(dict_reply["retweeted_status"]["user"]["default_profile"])
                    df_replies.loc[index_row_reply, "retweetedStatus_user_following"] = str(dict_reply["retweeted_status"]["user"]["following"])
                    df_replies.loc[index_row_reply, "retweetedStatus_user_translatorType"] = str(dict_reply["retweeted_status"]["user"]["translator_type"])
                    df_replies.loc[index_row_reply, "retweetedStatus_geo"] = str(dict_reply["retweeted_status"]["geo"])
                    df_replies.loc[index_row_reply, "retweetedStatus_coordinates"] = str(dict_reply["retweeted_status"]["coordinates"])
                    df_replies.loc[index_row_reply, "retweetedStatus_place"] = str(dict_reply["retweeted_status"]["place"])
                    df_replies.loc[index_row_reply, "retweetedStatus_contributors"] = str(dict_reply["retweeted_status"]["contributors"])
                    df_replies.loc[index_row_reply, "retweetedStatus_isQuoteStatus"] = str(dict_reply["retweeted_status"]["is_quote_status"])
                    df_replies.loc[index_row_reply, "retweetedStatus_retweetCount"] = dict_reply["retweeted_status"]["retweet_count"]
                    df_replies.loc[index_row_reply, "retweetedStatus_favoriteCount"] = dict_reply["retweeted_status"]["favorite_count"]
                    df_replies.loc[index_row_reply, "retweetedStatus_lang"] = dict_reply["retweeted_status"]["lang"]
                    if "possibly_sensitive" in dict_reply["retweeted_status"]:
                        df_replies.loc[index_row_reply, "retweetedStatus_possiblySensitive"] = dict_reply["retweeted_status"]["possibly_sensitive"]

                index_row_reply += 1

                if(index_row_reply % 50 == 0):
                    print("index_row_reply = " + str(index_row_reply))

            file_input_replies.close()

            if len(df_replies) > 0:
                df_replies = df_replies.sort_values(by=["idStr"], ascending=True)
                df_replies = df_replies.reset_index(drop=True)

            # print("df_replies:")
            # print(df_replies[["idStr", "createdAt"]])
            # print(df_replies[["idStr", "fullText"]])
            # print(df_replies[["idStr", "user_friendsCount"]])
            # print(df_replies[["idStr", "createdAt", "replyAge_min"]])
            # print(df_replies[["idStr", "geo", "coordinates", "place"]])
            # print(df_replies[["idStr", "replyAge_min", "user_description", "user_followersCount"]])
            # print(df_replies[["idStr", "replyAge_min"]])

            # print("len(df_replies):")
            # print(len(df_replies))

            # print(df_replies[["idStr", "retweetedStatus_fullText"]])
            # print(len(df_replies))
            if len(df_replies) > 0:
                for timestamp in list_timestamp:

                    df_cascade_current = df_replies[df_replies["replyAge_min"] <= timestamp]

                    df_output.loc[index_row_output, "cascadeAge_min"] = timestamp

                    df_output.loc[index_row_output, "rootTweetIdStr"] = id_rootTweet

                    df_output.loc[index_row_output, "cascadeSize"] = len(df_cascade_current)

                    if len(df_cascade_current) <= 0:
                        continue


                    df_output.loc[index_row_output, "reply_ptg_user_protected"] = len(df_cascade_current[df_cascade_current["user_protected"]=="True"])/len(df_cascade_current)
                    df_output.loc[index_row_output, "reply_mean_user_followersCount"] = np.nanmean(df_cascade_current["user_followersCount"])
                    df_output.loc[index_row_output, "reply_median_user_followersCount"] = np.nanmedian(df_cascade_current["user_followersCount"])
                    df_output.loc[index_row_output, "reply_mean_user_friendsCount"] = np.nanmean(df_cascade_current["user_friendsCount"])
                    df_output.loc[index_row_output, "reply_median_user_friendsCount"] = np.nanmedian(df_cascade_current["user_friendsCount"])
                    df_output.loc[index_row_output, "reply_mean_user_listedCount"] = np.nanmean(df_cascade_current["user_listedCount"])
                    df_output.loc[index_row_output, "reply_median_user_listedCount"] = np.nanmedian(df_cascade_current["user_listedCount"])
                    df_output.loc[index_row_output, "reply_mean_user_listedCount"] = np.nanmean(df_cascade_current["user_listedCount"])
                    df_output.loc[index_row_output, "reply_median_user_listedCount"] = np.nanmedian(df_cascade_current["user_listedCount"])
                    df_output.loc[index_row_output, "reply_mean_user_accountAge_day"] = np.nanmean(df_cascade_current["user_accountAge_day"])
                    df_output.loc[index_row_output, "reply_median_user_accountAge_day"] = np.nanmedian(df_cascade_current["user_accountAge_day"])
                    df_output.loc[index_row_output, "reply_mean_user_favouritesCount"] = np.nanmean(df_cascade_current["user_favouritesCount"])
                    df_output.loc[index_row_output, "reply_median_user_favouritesCount"] = np.nanmedian(df_cascade_current["user_favouritesCount"])
                    df_output.loc[index_row_output, "reply_ptg_user_geoEnabled"] = len(df_cascade_current[df_cascade_current["user_geoEnabled"]=="True"])/len(df_cascade_current)
                    df_output.loc[index_row_output, "reply_ptg_user_verified"] = len(df_cascade_current[df_cascade_current["user_verified"]=="True"])/len(df_cascade_current)
                    df_output.loc[index_row_output, "reply_mean_user_statusesCount"] = np.nanmean(df_cascade_current["user_statusesCount"])
                    df_output.loc[index_row_output, "reply_median_user_statusesCount"] = np.nanmedian(df_cascade_current["user_statusesCount"])

                    df_output.loc[index_row_output, "reply_ptg_user_protected"] = len(df_cascade_current[df_cascade_current["user_protected"]=="True"])/len(df_cascade_current)

                    df_output.loc[index_row_output, "reply_ptg_user_profileUseBackgroundImage"] = len(df_cascade_current[df_cascade_current["user_profileUseBackgroundImage"]=="True"])/len(df_cascade_current)
                    df_output.loc[index_row_output, "reply_ptg_user_defaultProfile"] = len(df_cascade_current[df_cascade_current["user_defaultProfile"]=="True"])/len(df_cascade_current)
                    df_output.loc[index_row_output, "reply_ptg_geo"] = 1-(len(df_cascade_current[df_cascade_current["geo"]=="None"])/len(df_cascade_current))
                    df_output.loc[index_row_output, "reply_ptg_coordinates"] = 1-(len(df_cascade_current[df_cascade_current["coordinates"]=="None"])/len(df_cascade_current))
                    if "possiblySensitive" in df_cascade_current.columns:
                        df_output.loc[index_row_output, "reply_ptg_possiblySensitive"] = len(df_cascade_current[df_cascade_current["possiblySensitive"]=="True"])/len(df_cascade_current)


                    if "retweetedStatus_retweetCount" in df_cascade_current:
                        df_output.loc[index_row_output, "reply_latest_retweetedStatus_retweetCount"] = df_cascade_current.loc[len(df_cascade_current)-1, "retweetedStatus_retweetCount"]
                    if "retweetedStatus_favoriteCount" in df_cascade_current:                                  
                        df_output.loc[index_row_output, "reply_latest_retweetedStatus_favoriteCount"] = df_cascade_current.loc[len(df_cascade_current)-1, "retweetedStatus_favoriteCount"]

                    df_output.loc[index_row_output, "reply_mean_user_accountAge_day"] = np.nanmean(df_cascade_current["user_accountAge_day"])
                    df_output.loc[index_row_output, "reply_median_user_accountAge_day"] = np.nanmedian(df_cascade_current["user_accountAge_day"])





                    df_output.loc[index_row_output, "reply_mean_fullText_sentiment_neg"] = np.nanmean(df_cascade_current["fullText_sentiment_neg"])
                    df_output.loc[index_row_output, "reply_median_fullText_sentiment_neg"] = np.nanmedian(df_cascade_current["fullText_sentiment_neg"])
                    df_output.loc[index_row_output, "reply_mean_fullText_sentiment_neu"] = np.nanmean(df_cascade_current["fullText_sentiment_neu"])
                    df_output.loc[index_row_output, "reply_median_fullText_sentiment_neu"] = np.nanmedian(df_cascade_current["fullText_sentiment_neu"])
                    df_output.loc[index_row_output, "reply_mean_fullText_sentiment_pos"] = np.nanmean(df_cascade_current["fullText_sentiment_pos"])
                    df_output.loc[index_row_output, "reply_median_fullText_sentiment_pos"] = np.nanmedian(df_cascade_current["fullText_sentiment_pos"])
                    df_output.loc[index_row_output, "reply_mean_fullText_sentiment_compound"] = np.nanmean(df_cascade_current["fullText_sentiment_compound"])
                    df_output.loc[index_row_output, "reply_median_fullText_sentiment_compound"] = np.nanmedian(df_cascade_current["fullText_sentiment_compound"])
                    
                    df_output.loc[index_row_output, "reply_mean_fullText_subjectivity"] = np.nanmean(df_cascade_current["fullText_subjectivity"])
                    df_output.loc[index_row_output, "reply_median_fullText_subjectivity"] = np.nanmedian(df_cascade_current["fullText_subjectivity"])
                    
                    for affect in list_affects:
                        df_output.loc[index_row_output, "reply_mean_fullText_emotion_" + affect] = np.nanmean(df_cascade_current["fullText_emotion_" + affect])
                        df_output.loc[index_row_output, "reply_median_fullText_emotion_" + affect] = np.nanmedian(df_cascade_current["fullText_emotion_" + affect])

                    for metric in list_textStats:
                        df_output.loc[index_row_output, "reply_mean_fullText_textStats_" + metric] = np.nanmean(df_cascade_current["fullText_textStats_" + metric])
                        df_output.loc[index_row_output, "reply_median_fullText_textStats_" + metric] = np.nanmedian(df_cascade_current["fullText_textStats_" + metric])




                    df_output.loc[index_row_output, "reply_mean_user_description_sentiment_neg"] = np.nanmean(df_cascade_current["user_description_sentiment_neg"])
                    df_output.loc[index_row_output, "reply_median_user_description_sentiment_neg"] = np.nanmedian(df_cascade_current["user_description_sentiment_neg"])
                    df_output.loc[index_row_output, "reply_mean_user_description_sentiment_neu"] = np.nanmean(df_cascade_current["user_description_sentiment_neu"])
                    df_output.loc[index_row_output, "reply_median_user_description_sentiment_neu"] = np.nanmedian(df_cascade_current["user_description_sentiment_neu"])
                    df_output.loc[index_row_output, "reply_mean_user_description_sentiment_pos"] = np.nanmean(df_cascade_current["user_description_sentiment_pos"])
                    df_output.loc[index_row_output, "reply_median_user_description_sentiment_pos"] = np.nanmedian(df_cascade_current["user_description_sentiment_pos"])
                    df_output.loc[index_row_output, "reply_mean_user_description_sentiment_compound"] = np.nanmean(df_cascade_current["user_description_sentiment_compound"])
                    df_output.loc[index_row_output, "reply_median_user_description_sentiment_compound"] = np.nanmedian(df_cascade_current["user_description_sentiment_compound"])

                    df_output.loc[index_row_output, "reply_mean_user_description_subjectivity"] = np.nanmean(df_cascade_current["user_description_subjectivity"])
                    df_output.loc[index_row_output, "reply_median_user_description_subjectivity"] = np.nanmedian(df_cascade_current["user_description_subjectivity"])
                    
                    for affect in list_affects:
                        df_output.loc[index_row_output, "reply_mean_user_description_emotion_" + affect] = np.nanmean(df_cascade_current["user_description_emotion_" + affect])
                        df_output.loc[index_row_output, "reply_median_user_description_emotion_" + affect] = np.nanmedian(df_cascade_current["user_description_emotion_" + affect])

                    for metric in list_textStats:
                        df_output.loc[index_row_output, "reply_mean_user_description_textStats_" + metric] = np.nanmean(df_cascade_current["user_description_textStats_" + metric])
                        df_output.loc[index_row_output, "reply_median_user_description_textStats_" + metric] = np.nanmedian(df_cascade_current["user_description_textStats_" + metric])

                    """
                    df_output.loc[index_row_output, "rootTweet_idStr"] = df_replies.loc[len(df_replies)-1, "retweetedStatus_idStr"]
                    df_output.loc[index_row_output, "rootTweet_createdAt"] = df_replies.loc[len(df_replies)-1, "retweetedStatus_createdAt"]
                    df_output.loc[index_row_output, "rootTweet_fullText"] = df_replies.loc[len(df_replies)-1, "retweetedStatus_fullText"]
                    df_output.loc[index_row_output, "rootTweet_entities_hashtags"] = df_replies.loc[len(df_replies)-1, "retweetedStatus_entities_hashtags"]
                    df_output.loc[index_row_output, "rootTweet_entities_symbols"] = df_replies.loc[len(df_replies)-1, "retweetedStatus_entities_symbols"]
                    df_output.loc[index_row_output, "rootTweet_entities_userMentions_screenName"] = df_replies.loc[len(df_replies)-1, "retweetedStatus_entities_userMentions_screenName"]
                    df_output.loc[index_row_output, "rootTweet_count_entities_userMentions"] = df_replies.loc[len(df_replies)-1, "count_retweetedStatus_entities_userMentions"]
                    df_output.loc[index_row_output, "rootTweet_count_entities_urls"] = df_replies.loc[len(df_replies)-1, "count_retweetedStatus_entities_urls"]
                    df_output.loc[index_row_output, "rootTweet_inReplyToStatusIdStr"] = df_replies.loc[len(df_replies)-1, "retweetedStatus_inReplyToStatusIdStr"]
                    df_output.loc[index_row_output, "rootTweet_inReplyToScreenName"] = df_replies.loc[len(df_replies)-1, "retweetedStatus_inReplyToScreenName"]
                    df_output.loc[index_row_output, "rootTweet_user_idStr"] = df_replies.loc[len(df_replies)-1, "retweetedStatus_user_idStr"]
                    df_output.loc[index_row_output, "rootTweet_user_screenName"] = df_replies.loc[len(df_replies)-1, "retweetedStatus_user_screenName"]
                    df_output.loc[index_row_output, "rootTweet_user_location"] = df_replies.loc[len(df_replies)-1, "retweetedStatus_user_location"]
                    df_output.loc[index_row_output, "rootTweet_user_description"] = df_replies.loc[len(df_replies)-1, "retweetedStatus_user_description"]
                    df_output.loc[index_row_output, "rootTweet_user_url"] = df_replies.loc[len(df_replies)-1, "retweetedStatus_user_url"]
                    df_output.loc[index_row_output, "rootTweet_user_protected"] = df_replies.loc[len(df_replies)-1, "retweetedStatus_user_protected"]
                    df_output.loc[index_row_output, "rootTweet_user_followersCount"] = df_replies.loc[len(df_replies)-1, "retweetedStatus_user_followersCount"]
                    df_output.loc[index_row_output, "rootTweet_user_friendsCount"] = df_replies.loc[len(df_replies)-1, "retweetedStatus_user_friendsCount"]
                    df_output.loc[index_row_output, "rootTweet_user_listedCount"] = df_replies.loc[len(df_replies)-1, "retweetedStatus_user_listedCount"]
                    df_output.loc[index_row_output, "rootTweet_user_createdAt"] = df_replies.loc[len(df_replies)-1, "retweetedStatus_user_createdAt"]
                    df_output.loc[index_row_output, "rootTweet_user_favouritesCount"] = df_replies.loc[len(df_replies)-1, "retweetedStatus_user_favouritesCount"]
                    df_output.loc[index_row_output, "rootTweet_user_geoEnabled"] = df_replies.loc[len(df_replies)-1, "retweetedStatus_user_geoEnabled"]
                    df_output.loc[index_row_output, "rootTweet_user_verified"] = df_replies.loc[len(df_replies)-1, "retweetedStatus_user_verified"]
                    df_output.loc[index_row_output, "rootTweet_user_statusesCount"] = df_replies.loc[len(df_replies)-1, "retweetedStatus_user_statusesCount"]
                    df_output.loc[index_row_output, "rootTweet_user_lang"] = df_replies.loc[len(df_replies)-1, "retweetedStatus_user_lang"]
                    df_output.loc[index_row_output, "rootTweet_user_contributorsEnabled"] = df_replies.loc[len(df_replies)-1, "retweetedStatus_user_contributorsEnabled"]
                    df_output.loc[index_row_output, "rootTweet_user_isTranslator"] = df_replies.loc[len(df_replies)-1, "retweetedStatus_user_isTranslator"]
                    df_output.loc[index_row_output, "rootTweet_user_isTranslationEnabled"] = df_replies.loc[len(df_replies)-1, "retweetedStatus_user_isTranslationEnabled"]
                    df_output.loc[index_row_output, "rootTweet_user_profileBackgroundColor"] = df_replies.loc[len(df_replies)-1, "retweetedStatus_user_profileBackgroundColor"]
                    df_output.loc[index_row_output, "rootTweet_user_profileTextColor"] = df_replies.loc[len(df_replies)-1, "retweetedStatus_user_profileTextColor"]
                    df_output.loc[index_row_output, "rootTweet_user_profileUseBackgroundImage"] = df_replies.loc[len(df_replies)-1, "retweetedStatus_user_profileUseBackgroundImage"]
                    df_output.loc[index_row_output, "rootTweet_user_defaultProfile"] = df_replies.loc[len(df_replies)-1, "retweetedStatus_user_defaultProfile"]
                    df_output.loc[index_row_output, "rootTweet_user_following"] = df_replies.loc[len(df_replies)-1, "retweetedStatus_user_following"]
                    df_output.loc[index_row_output, "rootTweet_user_translatorType"] = df_replies.loc[len(df_replies)-1, "retweetedStatus_user_translatorType"]
                    df_output.loc[index_row_output, "rootTweet_geo"] = df_replies.loc[len(df_replies)-1, "retweetedStatus_geo"]
                    df_output.loc[index_row_output, "rootTweet_coordinates"] = df_replies.loc[len(df_replies)-1, "retweetedStatus_coordinates"]
                    df_output.loc[index_row_output, "rootTweet_place"] = df_replies.loc[len(df_replies)-1, "retweetedStatus_place"]
                    df_output.loc[index_row_output, "rootTweet_contributors"] = df_replies.loc[len(df_replies)-1, "retweetedStatus_contributors"]
                    df_output.loc[index_row_output, "rootTweet_isQuoteStatus"] = df_replies.loc[len(df_replies)-1, "retweetedStatus_isQuoteStatus"]
                    df_output.loc[index_row_output, "rootTweet_retweetCount"] = df_replies.loc[len(df_replies)-1, "retweetedStatus_retweetCount"]
                    df_output.loc[index_row_output, "rootTweet_favoriteCount"] = df_replies.loc[len(df_replies)-1, "retweetedStatus_favoriteCount"]
                    df_output.loc[index_row_output, "rootTweet_lang"] = df_replies.loc[len(df_replies)-1, "retweetedStatus_lang"]
                    """
                    

                    index_row_output += 1

                # print(df_replies[["idStr", "retweetedStatus_fullText"]])
                # print(len(df_replies))
                if "retweetedStatus_idStr" in df_replies:
                    df_output.loc[df_output["rootTweetIdStr"]==id_rootTweet, "rootTweet_idStr"] = df_replies.loc[len(df_replies)-1, "retweetedStatus_idStr"]
                    df_output.loc[df_output["rootTweetIdStr"]==id_rootTweet, "rootTweet_createdAt"] = df_replies.loc[len(df_replies)-1, "retweetedStatus_createdAt"]
                    df_output.loc[df_output["rootTweetIdStr"]==id_rootTweet, "rootTweet_fullText"] = df_replies.loc[len(df_replies)-1, "retweetedStatus_fullText"]
                    df_output.loc[df_output["rootTweetIdStr"]==id_rootTweet, "rootTweet_entities_hashtags"] = df_replies.loc[len(df_replies)-1, "retweetedStatus_entities_hashtags"]
                    df_output.loc[df_output["rootTweetIdStr"]==id_rootTweet, "rootTweet_entities_symbols"] = df_replies.loc[len(df_replies)-1, "retweetedStatus_entities_symbols"]
                    df_output.loc[df_output["rootTweetIdStr"]==id_rootTweet, "rootTweet_entities_userMentions_screenName"] = df_replies.loc[len(df_replies)-1, "retweetedStatus_entities_userMentions_screenName"]
                    df_output.loc[df_output["rootTweetIdStr"]==id_rootTweet, "rootTweet_count_entities_userMentions"] = df_replies.loc[len(df_replies)-1, "count_retweetedStatus_entities_userMentions"]
                    df_output.loc[df_output["rootTweetIdStr"]==id_rootTweet, "rootTweet_count_entities_urls"] = df_replies.loc[len(df_replies)-1, "count_retweetedStatus_entities_urls"]
                    df_output.loc[df_output["rootTweetIdStr"]==id_rootTweet, "rootTweet_inReplyToStatusIdStr"] = df_replies.loc[len(df_replies)-1, "retweetedStatus_inReplyToStatusIdStr"]
                    df_output.loc[df_output["rootTweetIdStr"]==id_rootTweet, "rootTweet_inReplyToScreenName"] = df_replies.loc[len(df_replies)-1, "retweetedStatus_inReplyToScreenName"]
                    df_output.loc[df_output["rootTweetIdStr"]==id_rootTweet, "rootTweet_user_idStr"] = df_replies.loc[len(df_replies)-1, "retweetedStatus_user_idStr"]
                    df_output.loc[df_output["rootTweetIdStr"]==id_rootTweet, "rootTweet_user_screenName"] = df_replies.loc[len(df_replies)-1, "retweetedStatus_user_screenName"]
                    df_output.loc[df_output["rootTweetIdStr"]==id_rootTweet, "rootTweet_user_location"] = df_replies.loc[len(df_replies)-1, "retweetedStatus_user_location"]
                    df_output.loc[df_output["rootTweetIdStr"]==id_rootTweet, "rootTweet_user_description"] = df_replies.loc[len(df_replies)-1, "retweetedStatus_user_description"]
                    df_output.loc[df_output["rootTweetIdStr"]==id_rootTweet, "rootTweet_user_url"] = df_replies.loc[len(df_replies)-1, "retweetedStatus_user_url"]
                    df_output.loc[df_output["rootTweetIdStr"]==id_rootTweet, "rootTweet_user_protected"] = df_replies.loc[len(df_replies)-1, "retweetedStatus_user_protected"]
                    df_output.loc[df_output["rootTweetIdStr"]==id_rootTweet, "rootTweet_user_followersCount"] = df_replies.loc[len(df_replies)-1, "retweetedStatus_user_followersCount"]
                    df_output.loc[df_output["rootTweetIdStr"]==id_rootTweet, "rootTweet_user_friendsCount"] = df_replies.loc[len(df_replies)-1, "retweetedStatus_user_friendsCount"]
                    df_output.loc[df_output["rootTweetIdStr"]==id_rootTweet, "rootTweet_user_listedCount"] = df_replies.loc[len(df_replies)-1, "retweetedStatus_user_listedCount"]
                    df_output.loc[df_output["rootTweetIdStr"]==id_rootTweet, "rootTweet_user_createdAt"] = df_replies.loc[len(df_replies)-1, "retweetedStatus_user_createdAt"]
                    df_output.loc[df_output["rootTweetIdStr"]==id_rootTweet, "rootTweet_user_favouritesCount"] = df_replies.loc[len(df_replies)-1, "retweetedStatus_user_favouritesCount"]
                    df_output.loc[df_output["rootTweetIdStr"]==id_rootTweet, "rootTweet_user_geoEnabled"] = df_replies.loc[len(df_replies)-1, "retweetedStatus_user_geoEnabled"]
                    df_output.loc[df_output["rootTweetIdStr"]==id_rootTweet, "rootTweet_user_verified"] = df_replies.loc[len(df_replies)-1, "retweetedStatus_user_verified"]
                    df_output.loc[df_output["rootTweetIdStr"]==id_rootTweet, "rootTweet_user_statusesCount"] = df_replies.loc[len(df_replies)-1, "retweetedStatus_user_statusesCount"]
                    df_output.loc[df_output["rootTweetIdStr"]==id_rootTweet, "rootTweet_user_lang"] = df_replies.loc[len(df_replies)-1, "retweetedStatus_user_lang"]
                    df_output.loc[df_output["rootTweetIdStr"]==id_rootTweet, "rootTweet_user_contributorsEnabled"] = df_replies.loc[len(df_replies)-1, "retweetedStatus_user_contributorsEnabled"]
                    df_output.loc[df_output["rootTweetIdStr"]==id_rootTweet, "rootTweet_user_isTranslator"] = df_replies.loc[len(df_replies)-1, "retweetedStatus_user_isTranslator"]
                    df_output.loc[df_output["rootTweetIdStr"]==id_rootTweet, "rootTweet_user_isTranslationEnabled"] = df_replies.loc[len(df_replies)-1, "retweetedStatus_user_isTranslationEnabled"]
                    df_output.loc[df_output["rootTweetIdStr"]==id_rootTweet, "rootTweet_user_profileBackgroundColor"] = df_replies.loc[len(df_replies)-1, "retweetedStatus_user_profileBackgroundColor"]
                    df_output.loc[df_output["rootTweetIdStr"]==id_rootTweet, "rootTweet_user_profileTextColor"] = df_replies.loc[len(df_replies)-1, "retweetedStatus_user_profileTextColor"]
                    df_output.loc[df_output["rootTweetIdStr"]==id_rootTweet, "rootTweet_user_profileUseBackgroundImage"] = df_replies.loc[len(df_replies)-1, "retweetedStatus_user_profileUseBackgroundImage"]
                    df_output.loc[df_output["rootTweetIdStr"]==id_rootTweet, "rootTweet_user_defaultProfile"] = df_replies.loc[len(df_replies)-1, "retweetedStatus_user_defaultProfile"]
                    df_output.loc[df_output["rootTweetIdStr"]==id_rootTweet, "rootTweet_user_following"] = df_replies.loc[len(df_replies)-1, "retweetedStatus_user_following"]
                    df_output.loc[df_output["rootTweetIdStr"]==id_rootTweet, "rootTweet_user_translatorType"] = df_replies.loc[len(df_replies)-1, "retweetedStatus_user_translatorType"]
                    df_output.loc[df_output["rootTweetIdStr"]==id_rootTweet, "rootTweet_geo"] = df_replies.loc[len(df_replies)-1, "retweetedStatus_geo"]
                    df_output.loc[df_output["rootTweetIdStr"]==id_rootTweet, "rootTweet_coordinates"] = df_replies.loc[len(df_replies)-1, "retweetedStatus_coordinates"]
                    df_output.loc[df_output["rootTweetIdStr"]==id_rootTweet, "rootTweet_place"] = df_replies.loc[len(df_replies)-1, "retweetedStatus_place"]
                    df_output.loc[df_output["rootTweetIdStr"]==id_rootTweet, "rootTweet_contributors"] = df_replies.loc[len(df_replies)-1, "retweetedStatus_contributors"]
                    df_output.loc[df_output["rootTweetIdStr"]==id_rootTweet, "rootTweet_isQuoteStatus"] = df_replies.loc[len(df_replies)-1, "retweetedStatus_isQuoteStatus"]
                    df_output.loc[df_output["rootTweetIdStr"]==id_rootTweet, "rootTweet_retweetCount"] = df_replies.loc[len(df_replies)-1, "retweetedStatus_retweetCount"]
                    df_output.loc[df_output["rootTweetIdStr"]==id_rootTweet, "rootTweet_favoriteCount"] = df_replies.loc[len(df_replies)-1, "retweetedStatus_favoriteCount"]
                    df_output.loc[df_output["rootTweetIdStr"]==id_rootTweet, "rootTweet_lang"] = df_replies.loc[len(df_replies)-1, "retweetedStatus_lang"]
                    if "retweetedStatus_possiblySensitive" in df_replies.columns:
                        df_output.loc[df_output["rootTweetIdStr"]==id_rootTweet, "rootTweet_possiblySensitive"] = df_replies.loc[len(df_replies)-1, "retweetedStatus_possiblySensitive"]

                    # print(df_replies.loc[len(df_replies)-1, "retweetedStatus_fullText"])

                    # df_replies.loc[len(df_replies)-1, "retweetedStatus_fullText"]

                    # print(df_replies[["idStr", "retweetedStatus_fullText"]])

                    dict_sentimentScores = analyzer.polarity_scores(df_replies.loc[len(df_replies)-1, "retweetedStatus_fullText"])
                    df_output.loc[df_output["rootTweetIdStr"]==id_rootTweet, "rootTweet_fullText_sentiment_neg"] = dict_sentimentScores["neg"]
                    df_output.loc[df_output["rootTweetIdStr"]==id_rootTweet, "rootTweet_fullText_sentiment_neu"] = dict_sentimentScores["neu"]
                    df_output.loc[df_output["rootTweetIdStr"]==id_rootTweet, "rootTweet_fullText_sentiment_pos"] = dict_sentimentScores["pos"]
                    df_output.loc[df_output["rootTweetIdStr"]==id_rootTweet, "rootTweet_fullText_sentiment_compound"] = dict_sentimentScores["compound"]

                    df_output.loc[df_output["rootTweetIdStr"]==id_rootTweet, "rootTweet_fullText_subjectivity"] = TextBlob(df_replies.loc[len(df_replies)-1, "retweetedStatus_fullText"]).sentiment.subjectivity

                    dict_sentimentScores = analyzer.polarity_scores(df_replies.loc[len(df_replies)-1, "retweetedStatus_user_description"])
                    df_output.loc[df_output["rootTweetIdStr"]==id_rootTweet, "rootTweet_user_description_sentiment_neg"] = dict_sentimentScores["neg"]
                    df_output.loc[df_output["rootTweetIdStr"]==id_rootTweet, "rootTweet_user_description_sentiment_neu"] = dict_sentimentScores["neu"]
                    df_output.loc[df_output["rootTweetIdStr"]==id_rootTweet, "rootTweet_user_description_sentiment_pos"] = dict_sentimentScores["pos"]
                    df_output.loc[df_output["rootTweetIdStr"]==id_rootTweet, "rootTweet_user_description_sentiment_compound"] = dict_sentimentScores["compound"]

                    df_output.loc[df_output["rootTweetIdStr"]==id_rootTweet, "rootTweet_user_description_subjectivity"] = TextBlob(df_replies.loc[len(df_replies)-1, "retweetedStatus_user_description"]).sentiment.subjectivity

                    str_text_preprocessed = preprocessText(df_replies.loc[len(df_replies)-1, "retweetedStatus_fullText"], nlp, list_stopWords)
                    results_emotion = NRCLex(str_text_preprocessed)
                    for affect in list_affects:
                        df_output.loc[df_output["rootTweetIdStr"]==id_rootTweet, "rootTweet_fullText_emotion_" + affect] = results_emotion.affect_frequencies[affect]

                    str_text_preprocessed = preprocessText(df_replies.loc[len(df_replies)-1, "retweetedStatus_user_description"], nlp, list_stopWords)
                    results_emotion = NRCLex(str_text_preprocessed)
                    for affect in list_affects:
                        df_output.loc[df_output["rootTweetIdStr"]==id_rootTweet, "rootTweet_user_description_emotion_" + affect] = results_emotion.affect_frequencies[affect]

                    dict_textStats = textStats(df_replies.loc[len(df_replies)-1, "retweetedStatus_fullText"])
                    for metric in list_textStats:
                        df_output.loc[df_output["rootTweetIdStr"]==id_rootTweet, "rootTweet_fullText_textStats_" + metric] = dict_textStats[metric]

                    dict_textStats = textStats(df_replies.loc[len(df_replies)-1, "retweetedStatus_user_description"])
                    for metric in list_textStats:
                        df_output.loc[df_output["rootTweetIdStr"]==id_rootTweet, "rootTweet_user_description_textStats_" + metric] = dict_textStats[metric]

                df_output_str = df_output.astype(str).copy()

                print("len(df_output_str):")
                print(len(df_output_str))

                print("absFilename_output_temporal:")
                print(absFilename_output_temporal)

                df_output_str.to_csv(absFilename_output_temporal, index=False, quoting=csv.QUOTE_ALL)
        
        except Exception as e:

            track = traceback.format_exc()
            print(track + "\n")

            print("index_row_rootTweet:")
            print(str(index_row_rootTweet))
            print("id_rootTweet:")
            print(str(id_rootTweet))
            print("index_row_reply:")
            print(str(index_row_reply))
            print("id_reply:")
            print(id_reply)
            print("timestamp:")
            print(timestamp)
            print("line:")
            print(line)


            file_log.write("Exception:\n")
            file_log.write(str(datetime.now()) + "\n")
            file_log.write("track:\n")
            file_log.write(track + "\n")
            file_log.write("index_row_rootTweet:\n")
            file_log.write(str(index_row_rootTweet) + "\n")
            file_log.write("id_rootTweet:\n")
            file_log.write(str(id_rootTweet) + "\n")
            file_log.write("index_row_reply:\n")
            file_log.write(str(index_row_reply) + "\n")
            file_log.write("id_reply:\n")
            file_log.write(id_reply + "\n")
            file_log.write("timestamp:\n")
            file_log.write(str(timestamp) + "\n")
            file_log.write("line:\n")
            file_log.write(str(line) + "\n")
            file_log.flush()

            continue

if __name__ == "__main__":
    main(sys.argv[1:])
