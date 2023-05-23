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

    opts, args = getopt.getopt(argv, '', ["absFilename_input_rootTweets=", "path_input_producerTweets=", "absFilename_output_temporal=", "absFilename_log=", "timeWindow_days="])

    print(opts)

    for opt, arg in opts:
        if opt == '--absFilename_input_rootTweets':
            absFilename_input_rootTweets = arg
        if opt == '--path_input_producerTweets':
            path_input_producerTweets = arg
        if opt == '--absFilename_output_temporal':
            absFilename_output_temporal = arg
        elif opt == '--absFilename_log':
            absFilename_log = arg
        elif opt == '--timeWindow_days':
            timeWindow_days = arg


    timeWindow_days = int(timeWindow_days)

    str_date_current = "Thu Nov 12 00:00:00 +0000 2020"
    date_current = datetime.strptime(str_date_current, "%a %b %d %H:%M:%S %z %Y")

        
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
    index_row_producerTweet = "EMPTY"
    url_rootTweet = "EMPTY"
    timestamp = "EMPTY"
    line = "EMPTY"

    index_row_output = len(df_output)

    for index_row_rootTweet in range(0, len(df_input_rootTweets)):

        id_rootTweet = df_input_rootTweets.loc[index_row_rootTweet, "id_rootTweet"]

        print("id_rootTweet:")
        print(id_rootTweet)

        # https://twitter.com/nzherald/status/1291593626472988673

        url_rootTweet = df_input_rootTweets.loc[index_row_rootTweet, "tweet link"]

        print("url_rootTweet:")
        print(url_rootTweet)

        str_start = "https://twitter.com/"
        str_end = "/status"
        start = url_rootTweet.find(str_start) + len(str_start)
        end = url_rootTweet.find(str_end)
        screenName_producer = url_rootTweet[start:end]

        print("screenName_producer:")
        print(screenName_producer)



        try:

            if (len(df_output) > 0) and (screenName_producer in df_output["producerScreenName"].tolist()):
                print("screenName_producer already processed. Skip.")
                continue


            date_createdAt_rootTweet = datetime.strptime(df_input_rootTweets.loc[index_row_rootTweet, "str_createdAt_rootTweet"], "%a %b %d %H:%M:%S %z %Y")

            print("date_createdAt_rootTweet:")
            print(date_createdAt_rootTweet)

            absFilename_input_producerTweets = path_input_producerTweets + "producerScreenName=" + screenName_producer + os.path.sep + "producersTweets_producerScreenName=" + screenName_producer + ".txt"

            if not os.path.exists(absFilename_input_producerTweets):
                print("Producer tweets file does not exist:")
                print("absFilename_input_producerTweets:")
                print(absFilename_input_producerTweets)
                print("Skip this root tweet.")
                continue

            num_lines = 0
            file_input_producerTweets = open(absFilename_input_producerTweets, "r", encoding="utf-8")    
            for line in file_input_producerTweets:    
                num_lines += 1
            file_input_producerTweets.close()
            print("num_lines:")
            print(num_lines)


            df_producerTweets = pd.DataFrame()

            

            file_input_producerTweets = open(absFilename_input_producerTweets, "r", encoding="utf-8")        

            index_row_producerTweet = 0

            date_producerTweet_first = None

            for line in file_input_producerTweets:

                dict_producerTweet = json.loads(line)

                # print("dict_producerTweet:")
                # print(dict_producerTweet)

                # if "id_str" not in dict_producerTweet:
                #     continue
                if "id_str" not in dict_producerTweet:
                    print("index_row_rootTweet:")
                    print(index_row_rootTweet)
                    print("id_rootTweet:")
                    print(id_rootTweet)
                    print("index_row_producerTweet:")
                    print(index_row_producerTweet)
                    print("dict_producerTweet[\"id_str\"]:")
                    print(dict_producerTweet["id_str"])
                    print("line:")
                    print(line)
                    print("id_str is not in the line. Skip.")
                    continue

                if dict_producerTweet["user"]["screen_name"] != screenName_producer:
                    print("index_row_rootTweet:")
                    print(index_row_rootTweet)
                    print("id_rootTweet:")
                    print(id_rootTweet)
                    print("index_row_producerTweet:")
                    print(index_row_producerTweet)
                    print("url_rootTweet:")
                    print(url_rootTweet)
                    print("line:")
                    print(line)
                    print("This is good tweet, but not made made by the producer. Skip.")
                    continue


                
                if date_producerTweet_first == None:
                    date_producerTweet_first = datetime.strptime(str(dict_producerTweet["created_at"]), "%a %b %d %H:%M:%S %z %Y")
                else:
                    date_producerTweet_current = datetime.strptime(str(dict_producerTweet["created_at"]), "%a %b %d %H:%M:%S %z %Y")
                    dateDifference_days = (date_producerTweet_current - date_producerTweet_first).total_seconds()/(60*60*24)
                    # print("date_producerTweet_first:")
                    # print(date_producerTweet_first)
                    # print("date_producerTweet_current:")
                    # print(date_producerTweet_current)
                    # print("dateDifference_days:")
                    # print(dateDifference_days)
                    if dateDifference_days >= timeWindow_days:
                        # print("Current producer tweet is more than " + str(timeWindow_days) + " days of the first collected tweet. Skip.")
                        continue


                df_producerTweets.loc[index_row_producerTweet, "idStr"] = str(dict_producerTweet["id_str"])
                df_producerTweets.loc[index_row_producerTweet, "createdAt"] = str(dict_producerTweet["created_at"])
                    
                if "full_text" in dict_producerTweet:
                    df_producerTweets.loc[index_row_producerTweet, "fullText"] = str(dict_producerTweet["full_text"])
                elif "text" in dict_producerTweet:
                    df_producerTweets.loc[index_row_producerTweet, "fullText"] = str(dict_producerTweet["text"])


                df_producerTweets.loc[index_row_producerTweet, "count_entities_hashtags"] = len([t["text"] for t in dict_producerTweet["entities"]["hashtags"]])
                df_producerTweets.loc[index_row_producerTweet, "entities_hashtags"] = str(dict_producerTweet["entities"]["hashtags"])
                df_producerTweets.loc[index_row_producerTweet, "entities_symbols"] = str(dict_producerTweet["entities"]["symbols"])
                df_producerTweets.loc[index_row_producerTweet, "count_entities_userMentions"] = len(dict_producerTweet["entities"]["user_mentions"])
                df_producerTweets.loc[index_row_producerTweet, "user_description"] = str(dict_producerTweet["user"]["description"])
                df_producerTweets.loc[index_row_producerTweet, "user_protected"] = str(dict_producerTweet["user"]["protected"])
                df_producerTweets.loc[index_row_producerTweet, "user_followersCount"] = dict_producerTweet["user"]["followers_count"]
                df_producerTweets.loc[index_row_producerTweet, "user_friendsCount"] = dict_producerTweet["user"]["friends_count"]
                df_producerTweets.loc[index_row_producerTweet, "user_listedCount"] = dict_producerTweet["user"]["listed_count"]
                df_producerTweets.loc[index_row_producerTweet, "user_createdAt"] = str(dict_producerTweet["user"]["created_at"])
                df_producerTweets.loc[index_row_producerTweet, "user_favouritesCount"] = dict_producerTweet["user"]["favourites_count"]
                df_producerTweets.loc[index_row_producerTweet, "user_geoEnabled"] = str(dict_producerTweet["user"]["geo_enabled"])
                df_producerTweets.loc[index_row_producerTweet, "user_verified"] = str(dict_producerTweet["user"]["verified"])
                df_producerTweets.loc[index_row_producerTweet, "user_statusesCount"] = dict_producerTweet["user"]["statuses_count"]
                df_producerTweets.loc[index_row_producerTweet, "user_lang"] = str(dict_producerTweet["user"]["lang"])
                df_producerTweets.loc[index_row_producerTweet, "user_profileBackgroundColor"] = str(dict_producerTweet["user"]["profile_background_color"])
                df_producerTweets.loc[index_row_producerTweet, "user_profileTextColor"] = str(dict_producerTweet["user"]["profile_text_color"])
                df_producerTweets.loc[index_row_producerTweet, "user_profileUseBackgroundImage"] = str(dict_producerTweet["user"]["profile_use_background_image"])
                df_producerTweets.loc[index_row_producerTweet, "user_defaultProfile"] = str(dict_producerTweet["user"]["default_profile"])
                df_producerTweets.loc[index_row_producerTweet, "user_following"] = str(dict_producerTweet["user"]["following"])
                df_producerTweets.loc[index_row_producerTweet, "geo"] = str(dict_producerTweet["geo"])
                df_producerTweets.loc[index_row_producerTweet, "coordinates"] = str(dict_producerTweet["coordinates"])
                df_producerTweets.loc[index_row_producerTweet, "place"] = str(dict_producerTweet["place"])
                if "possibly_sensitive" in dict_producerTweet:
                    df_producerTweets.loc[index_row_producerTweet, "possiblySensitive"] = str(dict_producerTweet["possibly_sensitive"])


                date_createdAt_producerTweet = datetime.strptime(df_producerTweets.loc[index_row_producerTweet, "createdAt"], "%a %b %d %H:%M:%S %z %Y")
                df_producerTweets.loc[index_row_producerTweet, "producerTweetAge_min"] = (date_createdAt_producerTweet - date_createdAt_rootTweet).total_seconds()/60

                date_user_createdAt_producerTweet = datetime.strptime(df_producerTweets.loc[index_row_producerTweet, "user_createdAt"], "%a %b %d %H:%M:%S %z %Y")
                df_producerTweets.loc[index_row_producerTweet, "user_accountAge_day"] = (date_current - date_user_createdAt_producerTweet).total_seconds()/(60*60*24)



                dict_sentimentScores = analyzer.polarity_scores(df_producerTweets.loc[index_row_producerTweet, "fullText"])
                df_producerTweets.loc[index_row_producerTweet, "fullText_sentiment_neg"] = dict_sentimentScores["neg"]
                df_producerTweets.loc[index_row_producerTweet, "fullText_sentiment_neu"] = dict_sentimentScores["neu"]
                df_producerTweets.loc[index_row_producerTweet, "fullText_sentiment_pos"] = dict_sentimentScores["pos"]
                df_producerTweets.loc[index_row_producerTweet, "fullText_sentiment_compound"] = dict_sentimentScores["compound"]

                df_producerTweets.loc[index_row_producerTweet, "fullText_subjectivity"] = TextBlob(df_producerTweets.loc[index_row_producerTweet, "fullText"]).sentiment.subjectivity

                str_text_preprocessed = preprocessText(df_producerTweets.loc[index_row_producerTweet, "fullText"], nlp, list_stopWords)
                results_emotion = NRCLex(str_text_preprocessed)
                for affect in list_affects:
                    df_producerTweets.loc[index_row_producerTweet, "fullText_emotion_" + affect] = results_emotion.affect_frequencies[affect]
                    # print(affect)
                    # print(df_producerTweets.loc[index_row_producerTweet, "fullText_emotion_" + affect])

                dict_textStats = textStats(df_producerTweets.loc[index_row_producerTweet, "fullText"])
                for metric in list_textStats:
                    df_producerTweets.loc[index_row_producerTweet, "fullText_textStats_" + metric] = dict_textStats[metric]



                dict_sentimentScores = analyzer.polarity_scores(df_producerTweets.loc[index_row_producerTweet, "user_description"])
                df_producerTweets.loc[index_row_producerTweet, "user_description_sentiment_neg"] = dict_sentimentScores["neg"]
                df_producerTweets.loc[index_row_producerTweet, "user_description_sentiment_neu"] = dict_sentimentScores["neu"]
                df_producerTweets.loc[index_row_producerTweet, "user_description_sentiment_pos"] = dict_sentimentScores["pos"]
                df_producerTweets.loc[index_row_producerTweet, "user_description_sentiment_compound"] = dict_sentimentScores["compound"]

                df_producerTweets.loc[index_row_producerTweet, "user_description_subjectivity"] = TextBlob(df_producerTweets.loc[index_row_producerTweet, "user_description"]).sentiment.subjectivity

                str_text_preprocessed = preprocessText(df_producerTweets.loc[index_row_producerTweet, "user_description"], nlp, list_stopWords)
                results_emotion = NRCLex(str_text_preprocessed)
                for affect in list_affects:
                    df_producerTweets.loc[index_row_producerTweet, "user_description_emotion_" + affect] = results_emotion.affect_frequencies[affect]
                    # print(affect)
                    # print(df_producerTweets.loc[index_row_producerTweet, "user_description_emotion_" + affect])

                dict_textStats = textStats(df_producerTweets.loc[index_row_producerTweet, "user_description"])
                for metric in list_textStats:
                    df_producerTweets.loc[index_row_producerTweet, "user_description_textStats_" + metric] = dict_textStats[metric]

                if "in_reply_to_status_id_str" in dict_producerTweet:
                    df_producerTweets.loc[index_row_producerTweet, "isReply"] = "True"
                else:
                    df_producerTweets.loc[index_row_producerTweet, "isReply"] = "False"

                if "retweeted_status" in dict_producerTweet:
                    df_producerTweets.loc[index_row_producerTweet, "isRetweet"] = "True"
                    df_producerTweets.loc[index_row_producerTweet, "retweetedStatus_idStr"] = str(dict_producerTweet["retweeted_status"]["id_str"])
                    df_producerTweets.loc[index_row_producerTweet, "retweetedStatus_createdAt"] = str(dict_producerTweet["retweeted_status"]["created_at"])
                    
                    if "full_text" in dict_producerTweet["retweeted_status"]:
                        df_producerTweets.loc[index_row_producerTweet, "retweetedStatus_fullText"] = str(dict_producerTweet["retweeted_status"]["full_text"])
                    elif "text" in dict_producerTweet["retweeted_status"]:
                        df_producerTweets.loc[index_row_producerTweet, "retweetedStatus_fullText"] = str(dict_producerTweet["retweeted_status"]["text"])

                    # df_producerTweets.loc[index_row_producerTweet, "retweetedStatus_entities_hashtags"] = str([t["text"] for t in dict_producerTweet["retweeted_status"]["entities"]["hashtags"]])
                    df_producerTweets.loc[index_row_producerTweet, "retweetedStatus_entities_hashtags"] = str(dict_producerTweet["retweeted_status"]["entities"]["hashtags"])
                    df_producerTweets.loc[index_row_producerTweet, "retweetedStatus_entities_symbols"] = str(dict_producerTweet["retweeted_status"]["entities"]["symbols"])
                    df_producerTweets.loc[index_row_producerTweet, "retweetedStatus_entities_userMentions_screenName"] = str([u["screen_name"] for u in dict_producerTweet["retweeted_status"]["entities"]["user_mentions"]])
                    df_producerTweets.loc[index_row_producerTweet, "count_retweetedStatus_entities_userMentions"] = len(dict_producerTweet["retweeted_status"]["entities"]["user_mentions"])
                    df_producerTweets.loc[index_row_producerTweet, "count_retweetedStatus_entities_urls"] = len(dict_producerTweet["retweeted_status"]["entities"]["urls"])
                    df_producerTweets.loc[index_row_producerTweet, "retweetedStatu_entitiess_urls"] = str([u for u in dict_producerTweet["retweeted_status"]["entities"]["urls"]])
                    if "in_reply_to_status_id_str" in dict_producerTweet["retweeted_status"]:
                        df_producerTweets.loc[index_row_producerTweet, "retweetedStatus_inReplyToStatusIdStr"] = str(dict_producerTweet["retweeted_status"]["in_reply_to_status_id_str"])
                        df_producerTweets.loc[index_row_producerTweet, "retweetedStatus_inReplyToScreenName"] = str(dict_producerTweet["retweeted_status"]["in_reply_to_screen_name"])
                    df_producerTweets.loc[index_row_producerTweet, "retweetedStatus_user_idStr"] = str(dict_producerTweet["retweeted_status"]["user"]["id_str"])
                    df_producerTweets.loc[index_row_producerTweet, "retweetedStatus_user_screenName"] = str(dict_producerTweet["retweeted_status"]["user"]["screen_name"])
                    df_producerTweets.loc[index_row_producerTweet, "retweetedStatus_user_location"] = str(dict_producerTweet["retweeted_status"]["user"]["location"])
                    df_producerTweets.loc[index_row_producerTweet, "retweetedStatus_user_description"] = str(dict_producerTweet["retweeted_status"]["user"]["description"])
                    df_producerTweets.loc[index_row_producerTweet, "retweetedStatus_user_url"] = str(dict_producerTweet["retweeted_status"]["user"]["url"])
                    df_producerTweets.loc[index_row_producerTweet, "retweetedStatus_user_entities"] = str(dict_producerTweet["retweeted_status"]["user"]["entities"])
                    df_producerTweets.loc[index_row_producerTweet, "retweetedStatus_user_protected"] = str(dict_producerTweet["retweeted_status"]["user"]["protected"])
                    df_producerTweets.loc[index_row_producerTweet, "retweetedStatus_user_followersCount"] = dict_producerTweet["retweeted_status"]["user"]["followers_count"]
                    df_producerTweets.loc[index_row_producerTweet, "retweetedStatus_user_friendsCount"] = dict_producerTweet["retweeted_status"]["user"]["friends_count"]
                    df_producerTweets.loc[index_row_producerTweet, "retweetedStatus_user_listedCount"] = dict_producerTweet["retweeted_status"]["user"]["listed_count"]
                    df_producerTweets.loc[index_row_producerTweet, "retweetedStatus_user_createdAt"] = str(dict_producerTweet["retweeted_status"]["user"]["created_at"])
                    df_producerTweets.loc[index_row_producerTweet, "retweetedStatus_user_favouritesCount"] = dict_producerTweet["retweeted_status"]["user"]["favourites_count"]
                    df_producerTweets.loc[index_row_producerTweet, "retweetedStatus_user_geoEnabled"] = str(dict_producerTweet["retweeted_status"]["user"]["geo_enabled"])
                    df_producerTweets.loc[index_row_producerTweet, "retweetedStatus_user_verified"] = str(dict_producerTweet["retweeted_status"]["user"]["verified"])
                    df_producerTweets.loc[index_row_producerTweet, "retweetedStatus_user_statusesCount"] = dict_producerTweet["retweeted_status"]["user"]["statuses_count"]
                    df_producerTweets.loc[index_row_producerTweet, "retweetedStatus_user_lang"] = str(dict_producerTweet["retweeted_status"]["user"]["lang"])
                    df_producerTweets.loc[index_row_producerTweet, "retweetedStatus_user_contributorsEnabled"] = str(dict_producerTweet["retweeted_status"]["user"]["contributors_enabled"])
                    df_producerTweets.loc[index_row_producerTweet, "retweetedStatus_user_isTranslator"] = str(dict_producerTweet["retweeted_status"]["user"]["is_translator"])
                    df_producerTweets.loc[index_row_producerTweet, "retweetedStatus_user_isTranslationEnabled"] = str(dict_producerTweet["retweeted_status"]["user"]["is_translation_enabled"])
                    df_producerTweets.loc[index_row_producerTweet, "retweetedStatus_user_profileBackgroundColor"] = str(dict_producerTweet["retweeted_status"]["user"]["profile_background_color"])
                    df_producerTweets.loc[index_row_producerTweet, "retweetedStatus_user_profileTextColor"] = str(dict_producerTweet["retweeted_status"]["user"]["profile_text_color"])
                    df_producerTweets.loc[index_row_producerTweet, "retweetedStatus_user_profileUseBackgroundImage"] = str(dict_producerTweet["retweeted_status"]["user"]["profile_use_background_image"])
                    df_producerTweets.loc[index_row_producerTweet, "retweetedStatus_user_defaultProfile"] = str(dict_producerTweet["retweeted_status"]["user"]["default_profile"])
                    df_producerTweets.loc[index_row_producerTweet, "retweetedStatus_user_following"] = str(dict_producerTweet["retweeted_status"]["user"]["following"])
                    df_producerTweets.loc[index_row_producerTweet, "retweetedStatus_user_translatorType"] = str(dict_producerTweet["retweeted_status"]["user"]["translator_type"])
                    df_producerTweets.loc[index_row_producerTweet, "retweetedStatus_geo"] = str(dict_producerTweet["retweeted_status"]["geo"])
                    df_producerTweets.loc[index_row_producerTweet, "retweetedStatus_coordinates"] = str(dict_producerTweet["retweeted_status"]["coordinates"])
                    df_producerTweets.loc[index_row_producerTweet, "retweetedStatus_place"] = str(dict_producerTweet["retweeted_status"]["place"])
                    df_producerTweets.loc[index_row_producerTweet, "retweetedStatus_contributors"] = str(dict_producerTweet["retweeted_status"]["contributors"])
                    df_producerTweets.loc[index_row_producerTweet, "retweetedStatus_isQuoteStatus"] = str(dict_producerTweet["retweeted_status"]["is_quote_status"])
                    df_producerTweets.loc[index_row_producerTweet, "retweetedStatus_retweetCount"] = dict_producerTweet["retweeted_status"]["retweet_count"]
                    df_producerTweets.loc[index_row_producerTweet, "retweetedStatus_favoriteCount"] = dict_producerTweet["retweeted_status"]["favorite_count"]
                    df_producerTweets.loc[index_row_producerTweet, "retweetedStatus_lang"] = dict_producerTweet["retweeted_status"]["lang"]
                    if "possibly_sensitive" in dict_producerTweet["retweeted_status"]:
                        df_producerTweets.loc[index_row_producerTweet, "retweetedStatus_possiblySensitive"] = dict_producerTweet["retweeted_status"]["possibly_sensitive"]


                    date_createdAt_producerTweet = datetime.strptime(df_producerTweets.loc[index_row_producerTweet, "createdAt"], "%a %b %d %H:%M:%S %z %Y")
                    date_createdAt_retweetedStatus = datetime.strptime(df_producerTweets.loc[index_row_producerTweet, "retweetedStatus_createdAt"], "%a %b %d %H:%M:%S %z %Y")
                    df_producerTweets.loc[index_row_producerTweet, "producerTweet_retweetAge_min"] = (date_createdAt_producerTweet - date_createdAt_retweetedStatus).total_seconds()/60

                    date_user_createdAt_retweetedStatus = datetime.strptime(df_producerTweets.loc[index_row_producerTweet, "retweetedStatus_user_createdAt"], "%a %b %d %H:%M:%S %z %Y")
                    df_producerTweets.loc[index_row_producerTweet, "retweetedStatus_user_accountAge_day"] = (date_current - date_user_createdAt_retweetedStatus).total_seconds()/(60*60*24)



                    dict_sentimentScores = analyzer.polarity_scores(df_producerTweets.loc[index_row_producerTweet, "retweetedStatus_fullText"])
                    df_producerTweets.loc[index_row_producerTweet, "retweetedStatus_fullText_sentiment_neg"] = dict_sentimentScores["neg"]
                    df_producerTweets.loc[index_row_producerTweet, "retweetedStatus_fullText_sentiment_neu"] = dict_sentimentScores["neu"]
                    df_producerTweets.loc[index_row_producerTweet, "retweetedStatus_fullText_sentiment_pos"] = dict_sentimentScores["pos"]
                    df_producerTweets.loc[index_row_producerTweet, "retweetedStatus_fullText_sentiment_compound"] = dict_sentimentScores["compound"]

                    df_producerTweets.loc[index_row_producerTweet, "retweetedStatus_fullText_subjectivity"] = TextBlob(df_producerTweets.loc[index_row_producerTweet, "retweetedStatus_fullText"]).sentiment.subjectivity

                    str_text_preprocessed = preprocessText(df_producerTweets.loc[index_row_producerTweet, "retweetedStatus_fullText"], nlp, list_stopWords)
                    results_emotion = NRCLex(str_text_preprocessed)
                    for affect in list_affects:
                        df_producerTweets.loc[index_row_producerTweet, "retweetedStatus_fullText_emotion_" + affect] = results_emotion.affect_frequencies[affect]
                        # print(affect)
                        # print(df_producerTweets.loc[index_row_producerTweet, "retweetedStatus_fullText_emotion_" + affect])

                    dict_textStats = textStats(df_producerTweets.loc[index_row_producerTweet, "retweetedStatus_fullText"])
                    for metric in list_textStats:
                        df_producerTweets.loc[index_row_producerTweet, "retweetedStatus_fullText_textStats_" + metric] = dict_textStats[metric]



                    dict_sentimentScores = analyzer.polarity_scores(df_producerTweets.loc[index_row_producerTweet, "retweetedStatus_user_description"])
                    df_producerTweets.loc[index_row_producerTweet, "retweetedStatus_user_description_sentiment_neg"] = dict_sentimentScores["neg"]
                    df_producerTweets.loc[index_row_producerTweet, "retweetedStatus_user_description_sentiment_neu"] = dict_sentimentScores["neu"]
                    df_producerTweets.loc[index_row_producerTweet, "retweetedStatus_user_description_sentiment_pos"] = dict_sentimentScores["pos"]
                    df_producerTweets.loc[index_row_producerTweet, "retweetedStatus_user_description_sentiment_compound"] = dict_sentimentScores["compound"]

                    df_producerTweets.loc[index_row_producerTweet, "retweetedStatus_user_description_subjectivity"] = TextBlob(df_producerTweets.loc[index_row_producerTweet, "retweetedStatus_user_description"]).sentiment.subjectivity

                    str_text_preprocessed = preprocessText(df_producerTweets.loc[index_row_producerTweet, "retweetedStatus_user_description"], nlp, list_stopWords)
                    results_emotion = NRCLex(str_text_preprocessed)
                    for affect in list_affects:
                        df_producerTweets.loc[index_row_producerTweet, "retweetedStatus_user_description_emotion_" + affect] = results_emotion.affect_frequencies[affect]
                        # print(affect)
                        # print(df_producerTweets.loc[index_row_producerTweet, "retweetedStatus_user_description_emotion_" + affect])

                    dict_textStats = textStats(df_producerTweets.loc[index_row_producerTweet, "retweetedStatus_user_description"])
                    for metric in list_textStats:
                        df_producerTweets.loc[index_row_producerTweet, "retweetedStatus_user_description_textStats_" + metric] = dict_textStats[metric]
                
                else:
                    df_producerTweets.loc[index_row_producerTweet, "isRetweet"] = "False"




                index_row_producerTweet += 1

                if(index_row_producerTweet % 50 == 0):
                    print("index_row_producerTweet = " + str(index_row_producerTweet))

            file_input_producerTweets.close()

            if len(df_producerTweets) > 0:

                df_producerTweets = df_producerTweets.sort_values(by=["idStr"], ascending=True)
                df_producerTweets = df_producerTweets.reset_index(drop=True)

                # print("df_producerTweets:")
                # print(df_producerTweets[["idStr", "createdAt"]])
                # print(df_producerTweets[["idStr", "fullText"]])
                # print(df_producerTweets[["idStr", "user_friendsCount"]])
                # print(df_producerTweets[["idStr", "createdAt", "replyAge_min"]])
                # print(df_producerTweets[["idStr", "geo", "coordinates", "place"]])
                # print(df_producerTweets[["idStr", "replyAge_min", "user_description", "user_followersCount"]])
                # print(df_producerTweets[["idStr", "replyAge_min"]])

                # print("len(df_producerTweets):")
                # print(len(df_producerTweets))

                # print(df_producerTweets[["idStr", "retweetedStatus_fullText"]])
                # print(len(df_producerTweets))
                # if len(df_producerTweets) > 0:

                df_output.loc[index_row_output, "producerScreenName"] = screenName_producer
                df_output.loc[index_row_output, "rootTweetIdStr"] = id_rootTweet

                df_output.loc[index_row_output, "numProduerTweets"] = len(df_producerTweets)

                if len(df_producerTweets) <= 0:
                    continue


                df_output.loc[index_row_output, "producerTweet_ptg_user_protected"] = len(df_producerTweets[df_producerTweets["user_protected"]=="True"])/len(df_producerTweets)
                df_output.loc[index_row_output, "producerTweet_mean_user_followersCount"] = np.nanmean(df_producerTweets["user_followersCount"])
                df_output.loc[index_row_output, "producerTweet_median_user_followersCount"] = np.nanmedian(df_producerTweets["user_followersCount"])
                df_output.loc[index_row_output, "producerTweet_mean_user_friendsCount"] = np.nanmean(df_producerTweets["user_friendsCount"])
                df_output.loc[index_row_output, "producerTweet_median_user_friendsCount"] = np.nanmedian(df_producerTweets["user_friendsCount"])
                df_output.loc[index_row_output, "producerTweet_mean_user_listedCount"] = np.nanmean(df_producerTweets["user_listedCount"])
                df_output.loc[index_row_output, "producerTweet_median_user_listedCount"] = np.nanmedian(df_producerTweets["user_listedCount"])
                df_output.loc[index_row_output, "producerTweet_mean_user_listedCount"] = np.nanmean(df_producerTweets["user_listedCount"])
                df_output.loc[index_row_output, "producerTweet_median_user_listedCount"] = np.nanmedian(df_producerTweets["user_listedCount"])
                df_output.loc[index_row_output, "producerTweet_mean_user_accountAge_day"] = np.nanmean(df_producerTweets["user_accountAge_day"])
                df_output.loc[index_row_output, "producerTweet_median_user_accountAge_day"] = np.nanmedian(df_producerTweets["user_accountAge_day"])
                df_output.loc[index_row_output, "producerTweet_mean_user_favouritesCount"] = np.nanmean(df_producerTweets["user_favouritesCount"])
                df_output.loc[index_row_output, "producerTweet_median_user_favouritesCount"] = np.nanmedian(df_producerTweets["user_favouritesCount"])
                df_output.loc[index_row_output, "producerTweet_ptg_user_geoEnabled"] = len(df_producerTweets[df_producerTweets["user_geoEnabled"]=="True"])/len(df_producerTweets)
                df_output.loc[index_row_output, "producerTweet_ptg_user_verified"] = len(df_producerTweets[df_producerTweets["user_verified"]=="True"])/len(df_producerTweets)
                df_output.loc[index_row_output, "producerTweet_mean_user_statusesCount"] = np.nanmean(df_producerTweets["user_statusesCount"])
                df_output.loc[index_row_output, "producerTweet_median_user_statusesCount"] = np.nanmedian(df_producerTweets["user_statusesCount"])

                df_output.loc[index_row_output, "producerTweet_ptg_user_protected"] = len(df_producerTweets[df_producerTweets["user_protected"]=="True"])/len(df_producerTweets)

                df_output.loc[index_row_output, "producerTweet_ptg_user_profileUseBackgroundImage"] = len(df_producerTweets[df_producerTweets["user_profileUseBackgroundImage"]=="True"])/len(df_producerTweets)
                df_output.loc[index_row_output, "producerTweet_ptg_user_defaultProfile"] = len(df_producerTweets[df_producerTweets["user_defaultProfile"]=="True"])/len(df_producerTweets)
                df_output.loc[index_row_output, "producerTweet_ptg_geo"] = 1-(len(df_producerTweets[df_producerTweets["geo"]=="None"])/len(df_producerTweets))
                df_output.loc[index_row_output, "producerTweet_ptg_coordinates"] = 1-(len(df_producerTweets[df_producerTweets["coordinates"]=="None"])/len(df_producerTweets))
                if "possiblySensitive" in df_producerTweets.columns:
                    df_output.loc[index_row_output, "producerTweet_ptg_possiblySensitive"] = len(df_producerTweets[df_producerTweets["possiblySensitive"]=="True"])/len(df_producerTweets)


                if "retweetedStatus_retweetCount" in df_producerTweets:
                    df_output.loc[index_row_output, "producerTweet_latest_retweetedStatus_retweetCount"] = df_producerTweets.loc[len(df_producerTweets)-1, "retweetedStatus_retweetCount"]
                if "retweetedStatus_favoriteCount" in df_producerTweets:                                  
                    df_output.loc[index_row_output, "producerTweet_latest_retweetedStatus_favoriteCount"] = df_producerTweets.loc[len(df_producerTweets)-1, "retweetedStatus_favoriteCount"]

                df_output.loc[index_row_output, "producerTweet_mean_user_accountAge_day"] = np.nanmean(df_producerTweets["user_accountAge_day"])
                df_output.loc[index_row_output, "producerTweet_median_user_accountAge_day"] = np.nanmedian(df_producerTweets["user_accountAge_day"])

                df_output.loc[index_row_output, "producerTweet_ptg_isRetweet"] = len(df_producerTweets[df_producerTweets["isRetweet"]=="True"])/len(df_producerTweets)
                df_output.loc[index_row_output, "producerTweet_ptg_isReply"] = len(df_producerTweets[df_producerTweets["isReply"]=="True"])/len(df_producerTweets)




                df_output.loc[index_row_output, "producerTweet_mean_fullText_sentiment_neg"] = np.nanmean(df_producerTweets["fullText_sentiment_neg"])
                df_output.loc[index_row_output, "producerTweet_median_fullText_sentiment_neg"] = np.nanmedian(df_producerTweets["fullText_sentiment_neg"])
                df_output.loc[index_row_output, "producerTweet_mean_fullText_sentiment_neu"] = np.nanmean(df_producerTweets["fullText_sentiment_neu"])
                df_output.loc[index_row_output, "producerTweet_median_fullText_sentiment_neu"] = np.nanmedian(df_producerTweets["fullText_sentiment_neu"])
                df_output.loc[index_row_output, "producerTweet_mean_fullText_sentiment_pos"] = np.nanmean(df_producerTweets["fullText_sentiment_pos"])
                df_output.loc[index_row_output, "producerTweet_median_fullText_sentiment_pos"] = np.nanmedian(df_producerTweets["fullText_sentiment_pos"])
                df_output.loc[index_row_output, "producerTweet_mean_fullText_sentiment_compound"] = np.nanmean(df_producerTweets["fullText_sentiment_compound"])
                df_output.loc[index_row_output, "producerTweet_median_fullText_sentiment_compound"] = np.nanmedian(df_producerTweets["fullText_sentiment_compound"])
                
                df_output.loc[index_row_output, "producerTweet_mean_fullText_subjectivity"] = np.nanmean(df_producerTweets["fullText_subjectivity"])
                df_output.loc[index_row_output, "producerTweet_median_fullText_subjectivity"] = np.nanmedian(df_producerTweets["fullText_subjectivity"])
                
                for affect in list_affects:
                    df_output.loc[index_row_output, "producerTweet_mean_fullText_emotion_" + affect] = np.nanmean(df_producerTweets["fullText_emotion_" + affect])
                    df_output.loc[index_row_output, "producerTweet_median_fullText_emotion_" + affect] = np.nanmedian(df_producerTweets["fullText_emotion_" + affect])

                for metric in list_textStats:
                    df_output.loc[index_row_output, "producerTweet_mean_fullText_textStats_" + metric] = np.nanmean(df_producerTweets["fullText_textStats_" + metric])
                    df_output.loc[index_row_output, "producerTweet_median_fullText_textStats_" + metric] = np.nanmedian(df_producerTweets["fullText_textStats_" + metric])




                df_output.loc[index_row_output, "producerTweet_mean_user_description_sentiment_neg"] = np.nanmean(df_producerTweets["user_description_sentiment_neg"])
                df_output.loc[index_row_output, "producerTweet_median_user_description_sentiment_neg"] = np.nanmedian(df_producerTweets["user_description_sentiment_neg"])
                df_output.loc[index_row_output, "producerTweet_mean_user_description_sentiment_neu"] = np.nanmean(df_producerTweets["user_description_sentiment_neu"])
                df_output.loc[index_row_output, "producerTweet_median_user_description_sentiment_neu"] = np.nanmedian(df_producerTweets["user_description_sentiment_neu"])
                df_output.loc[index_row_output, "producerTweet_mean_user_description_sentiment_pos"] = np.nanmean(df_producerTweets["user_description_sentiment_pos"])
                df_output.loc[index_row_output, "producerTweet_median_user_description_sentiment_pos"] = np.nanmedian(df_producerTweets["user_description_sentiment_pos"])
                df_output.loc[index_row_output, "producerTweet_mean_user_description_sentiment_compound"] = np.nanmean(df_producerTweets["user_description_sentiment_compound"])
                df_output.loc[index_row_output, "producerTweet_median_user_description_sentiment_compound"] = np.nanmedian(df_producerTweets["user_description_sentiment_compound"])

                df_output.loc[index_row_output, "producerTweet_mean_user_description_subjectivity"] = np.nanmean(df_producerTweets["user_description_subjectivity"])
                df_output.loc[index_row_output, "producerTweet_median_user_description_subjectivity"] = np.nanmedian(df_producerTweets["user_description_subjectivity"])
                
                for affect in list_affects:
                    df_output.loc[index_row_output, "producerTweet_mean_user_description_emotion_" + affect] = np.nanmean(df_producerTweets["user_description_emotion_" + affect])
                    df_output.loc[index_row_output, "producerTweet_median_user_description_emotion_" + affect] = np.nanmedian(df_producerTweets["user_description_emotion_" + affect])

                for metric in list_textStats:
                    df_output.loc[index_row_output, "producerTweet_mean_user_description_textStats_" + metric] = np.nanmean(df_producerTweets["user_description_textStats_" + metric])
                    df_output.loc[index_row_output, "producerTweet_median_user_description_textStats_" + metric] = np.nanmedian(df_producerTweets["user_description_textStats_" + metric])

                """
                df_output.loc[index_row_output, "rootTweet_idStr"] = df_producerTweets.loc[len(df_producerTweets)-1, "retweetedStatus_idStr"]
                df_output.loc[index_row_output, "rootTweet_createdAt"] = df_producerTweets.loc[len(df_producerTweets)-1, "retweetedStatus_createdAt"]
                df_output.loc[index_row_output, "rootTweet_fullText"] = df_producerTweets.loc[len(df_producerTweets)-1, "retweetedStatus_fullText"]
                df_output.loc[index_row_output, "rootTweet_entities_hashtags"] = df_producerTweets.loc[len(df_producerTweets)-1, "retweetedStatus_entities_hashtags"]
                df_output.loc[index_row_output, "rootTweet_entities_symbols"] = df_producerTweets.loc[len(df_producerTweets)-1, "retweetedStatus_entities_symbols"]
                df_output.loc[index_row_output, "rootTweet_entities_userMentions_screenName"] = df_producerTweets.loc[len(df_producerTweets)-1, "retweetedStatus_entities_userMentions_screenName"]
                df_output.loc[index_row_output, "rootTweet_count_entities_userMentions"] = df_producerTweets.loc[len(df_producerTweets)-1, "count_retweetedStatus_entities_userMentions"]
                df_output.loc[index_row_output, "rootTweet_count_entities_urls"] = df_producerTweets.loc[len(df_producerTweets)-1, "count_retweetedStatus_entities_urls"]
                df_output.loc[index_row_output, "rootTweet_inReplyToStatusIdStr"] = df_producerTweets.loc[len(df_producerTweets)-1, "retweetedStatus_inReplyToStatusIdStr"]
                df_output.loc[index_row_output, "rootTweet_inReplyToScreenName"] = df_producerTweets.loc[len(df_producerTweets)-1, "retweetedStatus_inReplyToScreenName"]
                df_output.loc[index_row_output, "rootTweet_user_idStr"] = df_producerTweets.loc[len(df_producerTweets)-1, "retweetedStatus_user_idStr"]
                df_output.loc[index_row_output, "rootTweet_user_screenName"] = df_producerTweets.loc[len(df_producerTweets)-1, "retweetedStatus_user_screenName"]
                df_output.loc[index_row_output, "rootTweet_user_location"] = df_producerTweets.loc[len(df_producerTweets)-1, "retweetedStatus_user_location"]
                df_output.loc[index_row_output, "rootTweet_user_description"] = df_producerTweets.loc[len(df_producerTweets)-1, "retweetedStatus_user_description"]
                df_output.loc[index_row_output, "rootTweet_user_url"] = df_producerTweets.loc[len(df_producerTweets)-1, "retweetedStatus_user_url"]
                df_output.loc[index_row_output, "rootTweet_user_protected"] = df_producerTweets.loc[len(df_producerTweets)-1, "retweetedStatus_user_protected"]
                df_output.loc[index_row_output, "rootTweet_user_followersCount"] = df_producerTweets.loc[len(df_producerTweets)-1, "retweetedStatus_user_followersCount"]
                df_output.loc[index_row_output, "rootTweet_user_friendsCount"] = df_producerTweets.loc[len(df_producerTweets)-1, "retweetedStatus_user_friendsCount"]
                df_output.loc[index_row_output, "rootTweet_user_listedCount"] = df_producerTweets.loc[len(df_producerTweets)-1, "retweetedStatus_user_listedCount"]
                df_output.loc[index_row_output, "rootTweet_user_createdAt"] = df_producerTweets.loc[len(df_producerTweets)-1, "retweetedStatus_user_createdAt"]
                df_output.loc[index_row_output, "rootTweet_user_favouritesCount"] = df_producerTweets.loc[len(df_producerTweets)-1, "retweetedStatus_user_favouritesCount"]
                df_output.loc[index_row_output, "rootTweet_user_geoEnabled"] = df_producerTweets.loc[len(df_producerTweets)-1, "retweetedStatus_user_geoEnabled"]
                df_output.loc[index_row_output, "rootTweet_user_verified"] = df_producerTweets.loc[len(df_producerTweets)-1, "retweetedStatus_user_verified"]
                df_output.loc[index_row_output, "rootTweet_user_statusesCount"] = df_producerTweets.loc[len(df_producerTweets)-1, "retweetedStatus_user_statusesCount"]
                df_output.loc[index_row_output, "rootTweet_user_lang"] = df_producerTweets.loc[len(df_producerTweets)-1, "retweetedStatus_user_lang"]
                df_output.loc[index_row_output, "rootTweet_user_contributorsEnabled"] = df_producerTweets.loc[len(df_producerTweets)-1, "retweetedStatus_user_contributorsEnabled"]
                df_output.loc[index_row_output, "rootTweet_user_isTranslator"] = df_producerTweets.loc[len(df_producerTweets)-1, "retweetedStatus_user_isTranslator"]
                df_output.loc[index_row_output, "rootTweet_user_isTranslationEnabled"] = df_producerTweets.loc[len(df_producerTweets)-1, "retweetedStatus_user_isTranslationEnabled"]
                df_output.loc[index_row_output, "rootTweet_user_profileBackgroundColor"] = df_producerTweets.loc[len(df_producerTweets)-1, "retweetedStatus_user_profileBackgroundColor"]
                df_output.loc[index_row_output, "rootTweet_user_profileTextColor"] = df_producerTweets.loc[len(df_producerTweets)-1, "retweetedStatus_user_profileTextColor"]
                df_output.loc[index_row_output, "rootTweet_user_profileUseBackgroundImage"] = df_producerTweets.loc[len(df_producerTweets)-1, "retweetedStatus_user_profileUseBackgroundImage"]
                df_output.loc[index_row_output, "rootTweet_user_defaultProfile"] = df_producerTweets.loc[len(df_producerTweets)-1, "retweetedStatus_user_defaultProfile"]
                df_output.loc[index_row_output, "rootTweet_user_following"] = df_producerTweets.loc[len(df_producerTweets)-1, "retweetedStatus_user_following"]
                df_output.loc[index_row_output, "rootTweet_user_translatorType"] = df_producerTweets.loc[len(df_producerTweets)-1, "retweetedStatus_user_translatorType"]
                df_output.loc[index_row_output, "rootTweet_geo"] = df_producerTweets.loc[len(df_producerTweets)-1, "retweetedStatus_geo"]
                df_output.loc[index_row_output, "rootTweet_coordinates"] = df_producerTweets.loc[len(df_producerTweets)-1, "retweetedStatus_coordinates"]
                df_output.loc[index_row_output, "rootTweet_place"] = df_producerTweets.loc[len(df_producerTweets)-1, "retweetedStatus_place"]
                df_output.loc[index_row_output, "rootTweet_contributors"] = df_producerTweets.loc[len(df_producerTweets)-1, "retweetedStatus_contributors"]
                df_output.loc[index_row_output, "rootTweet_isQuoteStatus"] = df_producerTweets.loc[len(df_producerTweets)-1, "retweetedStatus_isQuoteStatus"]
                df_output.loc[index_row_output, "rootTweet_retweetCount"] = df_producerTweets.loc[len(df_producerTweets)-1, "retweetedStatus_retweetCount"]
                df_output.loc[index_row_output, "rootTweet_favoriteCount"] = df_producerTweets.loc[len(df_producerTweets)-1, "retweetedStatus_favoriteCount"]
                df_output.loc[index_row_output, "rootTweet_lang"] = df_producerTweets.loc[len(df_producerTweets)-1, "retweetedStatus_lang"]
                """
                

                

                # print(df_producerTweets[["idStr", "retweetedStatus_fullText"]])
                # print(len(df_producerTweets))
                if "retweetedStatus_idStr" in df_producerTweets:

                    df_output.loc[index_row_output, "retweetedStatus_ptg_user_protected"] = len(df_producerTweets[df_producerTweets["retweetedStatus_user_protected"]=="True"])/len(df_producerTweets)
                    df_output.loc[index_row_output, "retweetedStatus_mean_user_followersCount"] = np.nanmean(df_producerTweets["retweetedStatus_user_followersCount"])
                    df_output.loc[index_row_output, "retweetedStatus_median_user_followersCount"] = np.nanmedian(df_producerTweets["retweetedStatus_user_followersCount"])
                    df_output.loc[index_row_output, "retweetedStatus_mean_user_friendsCount"] = np.nanmean(df_producerTweets["retweetedStatus_user_friendsCount"])
                    df_output.loc[index_row_output, "retweetedStatus_median_user_friendsCount"] = np.nanmedian(df_producerTweets["retweetedStatus_user_friendsCount"])
                    df_output.loc[index_row_output, "retweetedStatus_mean_user_listedCount"] = np.nanmean(df_producerTweets["retweetedStatus_user_listedCount"])
                    df_output.loc[index_row_output, "retweetedStatus_median_user_listedCount"] = np.nanmedian(df_producerTweets["retweetedStatus_user_listedCount"])
                    df_output.loc[index_row_output, "retweetedStatus_mean_user_listedCount"] = np.nanmean(df_producerTweets["retweetedStatus_user_listedCount"])
                    df_output.loc[index_row_output, "retweetedStatus_median_user_listedCount"] = np.nanmedian(df_producerTweets["retweetedStatus_user_listedCount"])
                    # df_output.loc[index_row_output, "retweetedStatus_mean_user_accountAge_day"] = np.nanmean(df_producerTweets["retweetedStatus_user_accountAge_day"])
                    # df_output.loc[index_row_output, "retweetedStatus_median_user_accountAge_day"] = np.nanmedian(df_producerTweets["retweetedStatus_user_accountAge_day"])
                    df_output.loc[index_row_output, "retweetedStatus_mean_user_favouritesCount"] = np.nanmean(df_producerTweets["retweetedStatus_user_favouritesCount"])
                    df_output.loc[index_row_output, "retweetedStatus_median_user_favouritesCount"] = np.nanmedian(df_producerTweets["retweetedStatus_user_favouritesCount"])
                    df_output.loc[index_row_output, "retweetedStatus_ptg_user_geoEnabled"] = len(df_producerTweets[df_producerTweets["retweetedStatus_user_geoEnabled"]=="True"])/len(df_producerTweets)
                    df_output.loc[index_row_output, "retweetedStatus_ptg_user_verified"] = len(df_producerTweets[df_producerTweets["retweetedStatus_user_verified"]=="True"])/len(df_producerTweets)
                    df_output.loc[index_row_output, "retweetedStatus_mean_user_statusesCount"] = np.nanmean(df_producerTweets["retweetedStatus_user_statusesCount"])
                    df_output.loc[index_row_output, "retweetedStatus_median_user_statusesCount"] = np.nanmedian(df_producerTweets["retweetedStatus_user_statusesCount"])

                    df_output.loc[index_row_output, "retweetedStatus_ptg_user_protected"] = len(df_producerTweets[df_producerTweets["retweetedStatus_user_protected"]=="True"])/len(df_producerTweets)

                    df_output.loc[index_row_output, "retweetedStatus_ptg_user_profileUseBackgroundImage"] = len(df_producerTweets[df_producerTweets["retweetedStatus_user_profileUseBackgroundImage"]=="True"])/len(df_producerTweets)
                    df_output.loc[index_row_output, "retweetedStatus_ptg_user_defaultProfile"] = len(df_producerTweets[df_producerTweets["retweetedStatus_user_defaultProfile"]=="True"])/len(df_producerTweets)
                    df_output.loc[index_row_output, "retweetedStatus_ptg_geo"] = 1-(len(df_producerTweets[df_producerTweets["retweetedStatus_geo"]=="None"])/len(df_producerTweets))
                    df_output.loc[index_row_output, "retweetedStatus_ptg_coordinates"] = 1-(len(df_producerTweets[df_producerTweets["retweetedStatus_coordinates"]=="None"])/len(df_producerTweets))
                    if "retweetedStatus_possiblySensitive" in df_producerTweets.columns:
                        df_output.loc[index_row_output, "retweetedStatus_ptg_possiblySensitive"] = len(df_producerTweets[df_producerTweets["retweetedStatus_possiblySensitive"]=="True"])/len(df_producerTweets)


                    if "retweetedStatus_retweetCount" in df_producerTweets:
                        df_output.loc[index_row_output, "retweetedStatus_latest_retweetedStatus_retweetCount"] = df_producerTweets.loc[len(df_producerTweets)-1, "retweetedStatus_retweetCount"]
                    if "retweetedStatus_favoriteCount" in df_producerTweets:                                  
                        df_output.loc[index_row_output, "retweetedStatus_latest_retweetedStatus_favoriteCount"] = df_producerTweets.loc[len(df_producerTweets)-1, "retweetedStatus_favoriteCount"]

                    df_output.loc[index_row_output, "producerTweet_mean_retweetAge_min"] = np.nanmean(df_producerTweets["producerTweet_retweetAge_min"])
                    df_output.loc[index_row_output, "retweetedStatus_median_retweetAge_min"] = np.nanmedian(df_producerTweets["producerTweet_retweetAge_min"])

                    df_output.loc[index_row_output, "retweetedStatus_mean_user_accountAge_day"] = np.nanmean(df_producerTweets["retweetedStatus_user_accountAge_day"])
                    df_output.loc[index_row_output, "retweetedStatus_median_user_accountAge_day"] = np.nanmedian(df_producerTweets["retweetedStatus_user_accountAge_day"])





                    df_output.loc[index_row_output, "retweetedStatus_mean_fullText_sentiment_neg"] = np.nanmean(df_producerTweets["retweetedStatus_fullText_sentiment_neg"])
                    df_output.loc[index_row_output, "retweetedStatus_median_fullText_sentiment_neg"] = np.nanmedian(df_producerTweets["retweetedStatus_fullText_sentiment_neg"])
                    df_output.loc[index_row_output, "retweetedStatus_mean_fullText_sentiment_neu"] = np.nanmean(df_producerTweets["retweetedStatus_fullText_sentiment_neu"])
                    df_output.loc[index_row_output, "retweetedStatus_median_fullText_sentiment_neu"] = np.nanmedian(df_producerTweets["retweetedStatus_fullText_sentiment_neu"])
                    df_output.loc[index_row_output, "retweetedStatus_mean_fullText_sentiment_pos"] = np.nanmean(df_producerTweets["retweetedStatus_fullText_sentiment_pos"])
                    df_output.loc[index_row_output, "retweetedStatus_median_fullText_sentiment_pos"] = np.nanmedian(df_producerTweets["retweetedStatus_fullText_sentiment_pos"])
                    df_output.loc[index_row_output, "retweetedStatus_mean_fullText_sentiment_compound"] = np.nanmean(df_producerTweets["retweetedStatus_fullText_sentiment_compound"])
                    df_output.loc[index_row_output, "retweetedStatus_median_fullText_sentiment_compound"] = np.nanmedian(df_producerTweets["retweetedStatus_fullText_sentiment_compound"])
                    
                    df_output.loc[index_row_output, "retweetedStatus_mean_fullText_subjectivity"] = np.nanmean(df_producerTweets["retweetedStatus_fullText_subjectivity"])
                    df_output.loc[index_row_output, "retweetedStatus_median_fullText_subjectivity"] = np.nanmedian(df_producerTweets["retweetedStatus_fullText_subjectivity"])
                    
                    for affect in list_affects:
                        df_output.loc[index_row_output, "retweetedStatus_mean_fullText_emotion_" + affect] = np.nanmean(df_producerTweets["retweetedStatus_fullText_emotion_" + affect])
                        df_output.loc[index_row_output, "retweetedStatus_median_fullText_emotion_" + affect] = np.nanmedian(df_producerTweets["retweetedStatus_fullText_emotion_" + affect])

                    for metric in list_textStats:
                        df_output.loc[index_row_output, "retweetedStatus_mean_fullText_textStats_" + metric] = np.nanmean(df_producerTweets["retweetedStatus_fullText_textStats_" + metric])
                        df_output.loc[index_row_output, "retweetedStatus_median_fullText_textStats_" + metric] = np.nanmedian(df_producerTweets["retweetedStatus_fullText_textStats_" + metric])




                    df_output.loc[index_row_output, "retweetedStatus_mean_user_description_sentiment_neg"] = np.nanmean(df_producerTweets["retweetedStatus_user_description_sentiment_neg"])
                    df_output.loc[index_row_output, "retweetedStatus_median_user_description_sentiment_neg"] = np.nanmedian(df_producerTweets["retweetedStatus_user_description_sentiment_neg"])
                    df_output.loc[index_row_output, "retweetedStatus_mean_user_description_sentiment_neu"] = np.nanmean(df_producerTweets["retweetedStatus_user_description_sentiment_neu"])
                    df_output.loc[index_row_output, "retweetedStatus_median_user_description_sentiment_neu"] = np.nanmedian(df_producerTweets["retweetedStatus_user_description_sentiment_neu"])
                    df_output.loc[index_row_output, "retweetedStatus_mean_user_description_sentiment_pos"] = np.nanmean(df_producerTweets["retweetedStatus_user_description_sentiment_pos"])
                    df_output.loc[index_row_output, "retweetedStatus_median_user_description_sentiment_pos"] = np.nanmedian(df_producerTweets["retweetedStatus_user_description_sentiment_pos"])
                    df_output.loc[index_row_output, "retweetedStatus_mean_user_description_sentiment_compound"] = np.nanmean(df_producerTweets["retweetedStatus_user_description_sentiment_compound"])
                    df_output.loc[index_row_output, "retweetedStatus_median_user_description_sentiment_compound"] = np.nanmedian(df_producerTweets["retweetedStatus_user_description_sentiment_compound"])

                    df_output.loc[index_row_output, "retweetedStatus_mean_user_description_subjectivity"] = np.nanmean(df_producerTweets["retweetedStatus_user_description_subjectivity"])
                    df_output.loc[index_row_output, "retweetedStatus_median_user_description_subjectivity"] = np.nanmedian(df_producerTweets["retweetedStatus_user_description_subjectivity"])
                    
                    for affect in list_affects:
                        df_output.loc[index_row_output, "retweetedStatus_mean_user_description_emotion_" + affect] = np.nanmean(df_producerTweets["retweetedStatus_user_description_emotion_" + affect])
                        df_output.loc[index_row_output, "retweetedStatus_median_user_description_emotion_" + affect] = np.nanmedian(df_producerTweets["retweetedStatus_user_description_emotion_" + affect])

                    for metric in list_textStats:
                        df_output.loc[index_row_output, "retweetedStatus_mean_user_description_textStats_" + metric] = np.nanmean(df_producerTweets["retweetedStatus_user_description_textStats_" + metric])
                        df_output.loc[index_row_output, "retweetedStatus_median_user_description_textStats_" + metric] = np.nanmedian(df_producerTweets["retweetedStatus_user_description_textStats_" + metric])




                    """
                    df_output.loc[index_row_output, "rootTweet_idStr"] = df_producerTweets.loc[len(df_producerTweets)-1, "retweetedStatus_idStr"]
                    df_output.loc[index_row_output, "rootTweet_createdAt"] = df_producerTweets.loc[len(df_producerTweets)-1, "retweetedStatus_createdAt"]
                    df_output.loc[index_row_output, "rootTweet_fullText"] = df_producerTweets.loc[len(df_producerTweets)-1, "retweetedStatus_fullText"]
                    df_output.loc[index_row_output, "rootTweet_entities_hashtags"] = df_producerTweets.loc[len(df_producerTweets)-1, "retweetedStatus_entities_hashtags"]
                    df_output.loc[index_row_output, "rootTweet_entities_symbols"] = df_producerTweets.loc[len(df_producerTweets)-1, "retweetedStatus_entities_symbols"]
                    df_output.loc[index_row_output, "rootTweet_entities_userMentions_screenName"] = df_producerTweets.loc[len(df_producerTweets)-1, "retweetedStatus_entities_userMentions_screenName"]
                    df_output.loc[index_row_output, "rootTweet_count_entities_userMentions"] = df_producerTweets.loc[len(df_producerTweets)-1, "count_retweetedStatus_entities_userMentions"]
                    df_output.loc[index_row_output, "rootTweet_count_entities_urls"] = df_producerTweets.loc[len(df_producerTweets)-1, "count_retweetedStatus_entities_urls"]
                    df_output.loc[index_row_output, "rootTweet_inReplyToStatusIdStr"] = df_producerTweets.loc[len(df_producerTweets)-1, "retweetedStatus_inReplyToStatusIdStr"]
                    df_output.loc[index_row_output, "rootTweet_inReplyToScreenName"] = df_producerTweets.loc[len(df_producerTweets)-1, "retweetedStatus_inReplyToScreenName"]
                    df_output.loc[index_row_output, "rootTweet_user_idStr"] = df_producerTweets.loc[len(df_producerTweets)-1, "retweetedStatus_user_idStr"]
                    df_output.loc[index_row_output, "rootTweet_user_screenName"] = df_producerTweets.loc[len(df_producerTweets)-1, "retweetedStatus_user_screenName"]
                    df_output.loc[index_row_output, "rootTweet_user_location"] = df_producerTweets.loc[len(df_producerTweets)-1, "retweetedStatus_user_location"]
                    df_output.loc[index_row_output, "rootTweet_user_description"] = df_producerTweets.loc[len(df_producerTweets)-1, "retweetedStatus_user_description"]
                    df_output.loc[index_row_output, "rootTweet_user_url"] = df_producerTweets.loc[len(df_producerTweets)-1, "retweetedStatus_user_url"]
                    df_output.loc[index_row_output, "rootTweet_user_protected"] = df_producerTweets.loc[len(df_producerTweets)-1, "retweetedStatus_user_protected"]
                    df_output.loc[index_row_output, "rootTweet_user_followersCount"] = df_producerTweets.loc[len(df_producerTweets)-1, "retweetedStatus_user_followersCount"]
                    df_output.loc[index_row_output, "rootTweet_user_friendsCount"] = df_producerTweets.loc[len(df_producerTweets)-1, "retweetedStatus_user_friendsCount"]
                    df_output.loc[index_row_output, "rootTweet_user_listedCount"] = df_producerTweets.loc[len(df_producerTweets)-1, "retweetedStatus_user_listedCount"]
                    df_output.loc[index_row_output, "rootTweet_user_createdAt"] = df_producerTweets.loc[len(df_producerTweets)-1, "retweetedStatus_user_createdAt"]
                    df_output.loc[index_row_output, "rootTweet_user_favouritesCount"] = df_producerTweets.loc[len(df_producerTweets)-1, "retweetedStatus_user_favouritesCount"]
                    df_output.loc[index_row_output, "rootTweet_user_geoEnabled"] = df_producerTweets.loc[len(df_producerTweets)-1, "retweetedStatus_user_geoEnabled"]
                    df_output.loc[index_row_output, "rootTweet_user_verified"] = df_producerTweets.loc[len(df_producerTweets)-1, "retweetedStatus_user_verified"]
                    df_output.loc[index_row_output, "rootTweet_user_statusesCount"] = df_producerTweets.loc[len(df_producerTweets)-1, "retweetedStatus_user_statusesCount"]
                    df_output.loc[index_row_output, "rootTweet_user_lang"] = df_producerTweets.loc[len(df_producerTweets)-1, "retweetedStatus_user_lang"]
                    df_output.loc[index_row_output, "rootTweet_user_contributorsEnabled"] = df_producerTweets.loc[len(df_producerTweets)-1, "retweetedStatus_user_contributorsEnabled"]
                    df_output.loc[index_row_output, "rootTweet_user_isTranslator"] = df_producerTweets.loc[len(df_producerTweets)-1, "retweetedStatus_user_isTranslator"]
                    df_output.loc[index_row_output, "rootTweet_user_isTranslationEnabled"] = df_producerTweets.loc[len(df_producerTweets)-1, "retweetedStatus_user_isTranslationEnabled"]
                    df_output.loc[index_row_output, "rootTweet_user_profileBackgroundColor"] = df_producerTweets.loc[len(df_producerTweets)-1, "retweetedStatus_user_profileBackgroundColor"]
                    df_output.loc[index_row_output, "rootTweet_user_profileTextColor"] = df_producerTweets.loc[len(df_producerTweets)-1, "retweetedStatus_user_profileTextColor"]
                    df_output.loc[index_row_output, "rootTweet_user_profileUseBackgroundImage"] = df_producerTweets.loc[len(df_producerTweets)-1, "retweetedStatus_user_profileUseBackgroundImage"]
                    df_output.loc[index_row_output, "rootTweet_user_defaultProfile"] = df_producerTweets.loc[len(df_producerTweets)-1, "retweetedStatus_user_defaultProfile"]
                    df_output.loc[index_row_output, "rootTweet_user_following"] = df_producerTweets.loc[len(df_producerTweets)-1, "retweetedStatus_user_following"]
                    df_output.loc[index_row_output, "rootTweet_user_translatorType"] = df_producerTweets.loc[len(df_producerTweets)-1, "retweetedStatus_user_translatorType"]
                    df_output.loc[index_row_output, "rootTweet_geo"] = df_producerTweets.loc[len(df_producerTweets)-1, "retweetedStatus_geo"]
                    df_output.loc[index_row_output, "rootTweet_coordinates"] = df_producerTweets.loc[len(df_producerTweets)-1, "retweetedStatus_coordinates"]
                    df_output.loc[index_row_output, "rootTweet_place"] = df_producerTweets.loc[len(df_producerTweets)-1, "retweetedStatus_place"]
                    df_output.loc[index_row_output, "rootTweet_contributors"] = df_producerTweets.loc[len(df_producerTweets)-1, "retweetedStatus_contributors"]
                    df_output.loc[index_row_output, "rootTweet_isQuoteStatus"] = df_producerTweets.loc[len(df_producerTweets)-1, "retweetedStatus_isQuoteStatus"]
                    df_output.loc[index_row_output, "rootTweet_retweetCount"] = df_producerTweets.loc[len(df_producerTweets)-1, "retweetedStatus_retweetCount"]
                    df_output.loc[index_row_output, "rootTweet_favoriteCount"] = df_producerTweets.loc[len(df_producerTweets)-1, "retweetedStatus_favoriteCount"]
                    df_output.loc[index_row_output, "rootTweet_lang"] = df_producerTweets.loc[len(df_producerTweets)-1, "retweetedStatus_lang"]
                    if "retweetedStatus_possiblySensitive" in df_producerTweets.columns:
                        df_output.loc[index_row_output, "rootTweet_possiblySensitive"] = df_producerTweets.loc[len(df_producerTweets)-1, "retweetedStatus_possiblySensitive"]

                    # print(df_producerTweets.loc[len(df_producerTweets)-1, "retweetedStatus_fullText"])

                    # df_producerTweets.loc[len(df_producerTweets)-1, "retweetedStatus_fullText"]

                    # print(df_producerTweets[["idStr", "retweetedStatus_fullText"]])

                    dict_sentimentScores = analyzer.polarity_scores(df_producerTweets.loc[len(df_producerTweets)-1, "retweetedStatus_fullText"])
                    df_output.loc[index_row_output, "rootTweet_fullText_sentiment_neg"] = dict_sentimentScores["neg"]
                    df_output.loc[index_row_output, "rootTweet_fullText_sentiment_neu"] = dict_sentimentScores["neu"]
                    df_output.loc[index_row_output, "rootTweet_fullText_sentiment_pos"] = dict_sentimentScores["pos"]
                    df_output.loc[index_row_output, "rootTweet_fullText_sentiment_compound"] = dict_sentimentScores["compound"]

                    df_output.loc[index_row_output, "rootTweet_fullText_subjectivity"] = TextBlob(df_producerTweets.loc[len(df_producerTweets)-1, "retweetedStatus_fullText"]).sentiment.subjectivity

                    dict_sentimentScores = analyzer.polarity_scores(df_producerTweets.loc[len(df_producerTweets)-1, "retweetedStatus_user_description"])
                    df_output.loc[index_row_output, "rootTweet_user_description_sentiment_neg"] = dict_sentimentScores["neg"]
                    df_output.loc[index_row_output, "rootTweet_user_description_sentiment_neu"] = dict_sentimentScores["neu"]
                    df_output.loc[index_row_output, "rootTweet_user_description_sentiment_pos"] = dict_sentimentScores["pos"]
                    df_output.loc[index_row_output, "rootTweet_user_description_sentiment_compound"] = dict_sentimentScores["compound"]

                    df_output.loc[index_row_output, "rootTweet_user_description_subjectivity"] = TextBlob(df_producerTweets.loc[len(df_producerTweets)-1, "retweetedStatus_user_description"]).sentiment.subjectivity

                    str_text_preprocessed = preprocessText(df_producerTweets.loc[len(df_producerTweets)-1, "retweetedStatus_fullText"], nlp, list_stopWords)
                    results_emotion = NRCLex(str_text_preprocessed)
                    for affect in list_affects:
                        df_output.loc[index_row_output, "rootTweet_fullText_emotion_" + affect] = results_emotion.affect_frequencies[affect]

                    str_text_preprocessed = preprocessText(df_producerTweets.loc[len(df_producerTweets)-1, "retweetedStatus_user_description"], nlp, list_stopWords)
                    results_emotion = NRCLex(str_text_preprocessed)
                    for affect in list_affects:
                        df_output.loc[index_row_output, "rootTweet_user_description_emotion_" + affect] = results_emotion.affect_frequencies[affect]

                    dict_textStats = textStats(df_producerTweets.loc[len(df_producerTweets)-1, "retweetedStatus_fullText"])
                    for metric in list_textStats:
                        df_output.loc[index_row_output, "rootTweet_fullText_textStats_" + metric] = dict_textStats[metric]

                    dict_textStats = textStats(df_producerTweets.loc[len(df_producerTweets)-1, "retweetedStatus_user_description"])
                    for metric in list_textStats:
                        df_output.loc[index_row_output, "rootTweet_user_description_textStats_" + metric] = dict_textStats[metric]
                    """

                index_row_output += 1

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
            print("index_row_producerTweet:")
            print(str(index_row_producerTweet))
            print("url_rootTweet:")
            print(url_rootTweet)
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
            file_log.write("index_row_producerTweet:\n")
            file_log.write(str(index_row_producerTweet) + "\n")
            file_log.write("url_rootTweet:\n")
            file_log.write(url_rootTweet + "\n")
            file_log.write("timestamp:\n")
            file_log.write(str(timestamp) + "\n")
            file_log.write("line:\n")
            file_log.write(str(line) + "\n")
            file_log.flush()

            continue

if __name__ == "__main__":
    main(sys.argv[1:])
