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

    opts, args = getopt.getopt(argv, '', ["absFilename_input_rootTweets=", "path_input_rootTweetRetweets=", "absFilenamePattern_output_timeSeries=", "index_inputDF_start=", "index_inputDF_end=", "absFilenamePattern_log="])

    print(opts)

    for opt, arg in opts:
        if opt == '--absFilename_input_rootTweets':
            absFilename_input_rootTweets = arg
        elif opt == '--path_input_rootTweetRetweets':
            path_input_rootTweetRetweets = arg
        elif opt == '--absFilenamePattern_output_timeSeries':
            absFilenamePattern_output_timeSeries = arg
        elif opt == '--index_inputDF_start':
            index_inputDF_start = int(arg)
        elif opt == '--index_inputDF_end':
            index_inputDF_end = int(arg)
        elif opt == '--absFilenamePattern_log':
            absFilenamePattern_log = arg

    absFilename_output_timeSeries = absFilenamePattern_output_timeSeries.replace("INDEXSTART", str(index_inputDF_start)).replace("INDEXEND", str(index_inputDF_end))
    absFilename_log = absFilenamePattern_log.replace("INDEXSTART", str(index_inputDF_start)).replace("INDEXEND", str(index_inputDF_end))

    print("absFilename_output_timeSeries:")
    print(absFilename_output_timeSeries)
    print("absFilename_log:")
    print(absFilename_log)

       
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
    # path_input_rootTweetRetweets = "D:\\vmwareSharedFolder\\TwitterDataCollection\\data\\PFN\\snapshot\\dateRetrieval=20201008\\retweets\\" 
    # absFilename_output_temporal = "D:\\vmwareSharedFolder\\TwitterDataAnalysis\\results\\cascadeResults_temporal_retweets.csv"
    # absFilename_log = "D:\\vmwareSharedFolder\\TwitterDataAnalysis\\programs\\logs\\07_01_cascadeAnalysis_retweets.log"

    if not os.path.exists(os.path.dirname(absFilename_log)):
        os.makedirs(os.path.dirname(absFilename_log))
    file_log = open(absFilename_log, "w+")
    file_log.close()
    file_log = open(absFilename_log, "a")

    if os.path.exists(absFilename_output_timeSeries):
        df_retweets = pd.read_csv(absFilename_output_timeSeries, dtype=str)
        print("Output file exists. Load and continue with this file.")
    else:
        df_retweets = pd.DataFrame()
        print("Output file does not exists. Start from scratch.")
    
    print("len(df_retweets):")
    print(len(df_retweets))

    # return



    df_input_rootTweets = pd.read_csv(absFilename_input_rootTweets, dtype=str)

    df_input_rootTweets["id_rootTweet"] = df_input_rootTweets["tweet link"].apply(lambda x: x.split('/')[-1])


    print("len(df_input_rootTweets):")
    print(len(df_input_rootTweets))

    # df_input_rootTweets = df_input_rootTweets.sort_values(by=["id_rootTweet"], ascending=True)
    # df_input_rootTweets = df_input_rootTweets.reset_index(drop=True)

    print("len(df_input_rootTweets):")
    print(len(df_input_rootTweets))

    

    analyzer = SentimentIntensityAnalyzer()
    nlp = spacy.load('en_core_web_sm', disable=['parser', 'ner'])
    list_stopWords = list(set(stopwords.words('english')))

    list_affects = ["fear", "anger", "anticip", "trust", "surprise", "positive", "negative", "sadness", "disgust", "joy"]
    list_textStats = ["charCount", "wordCount", "uqWordCount", "sentenceCount", "charsPerWord", "wordsPerSentence", "readability"]


    index_row_rootTweet = "EMPTY"
    id_rootTweet = "EMPTY"
    index_row_retweet = "EMPTY"
    id_retweet = "EMPTY"
    timestamp = "EMPTY"
    line = "EMPTY"


    index_row_retweet = len(df_retweets)

    for index_row_rootTweet in range(0, len(df_input_rootTweets)):

        print("index_row_rootTweet:")
        print(index_row_rootTweet)

        if index_row_rootTweet < index_inputDF_start or index_row_rootTweet > index_inputDF_end:
            print("index_row_rootTweet is not in the range of index of this run. Skip this index.")
            continue


        id_rootTweet = df_input_rootTweets.loc[index_row_rootTweet, "id_rootTweet"]

        print("id_rootTweet:")
        print(id_rootTweet)



        try:

            if (len(df_retweets) > 0) and (id_rootTweet in df_retweets["rootTweetIdStr"].tolist()):
                print("id_rootTweet already processed. Skip.")
                continue


            date_createdAt_rootTweet = datetime.strptime(df_input_rootTweets.loc[index_row_rootTweet, "str_createdAt_rootTweet"], "%a %b %d %H:%M:%S %z %Y")

            print("date_createdAt_rootTweet:")
            print(date_createdAt_rootTweet)

            absFilename_input_retweets = path_input_rootTweetRetweets + "rootTweetID=" + id_rootTweet + os.path.sep + "retweets_rootTweetID=" + id_rootTweet + ".txt"

            

            num_lines = 0
            file_input_retweets = open(absFilename_input_retweets, "r", encoding="utf-8")    
            for line in file_input_retweets:    
                num_lines += 1
            file_input_retweets.close()
            print("num_lines:")
            print(num_lines)

            file_input_retweets = open(absFilename_input_retweets, "r", encoding="utf-8")

            

            

            for line in file_input_retweets:

                dict_retweet = json.loads(line)

                # print("dict_retweet:")
                # print(dict_retweet)

                if "id_str" not in dict_retweet:
                    print("index_row_rootTweet:")
                    print(index_row_rootTweet)
                    print("id_rootTweet:")
                    print(id_rootTweet)
                    print("index_row_retweet:")
                    print(index_row_retweet)
                    print("dict_retweet[\"id_str\"]:")
                    print(dict_retweet["id_str"])
                    print("line:")
                    print(line)
                    print("id_str is not in the line. Skip.")
                    continue

                id_retweet = str(dict_retweet["id_str"])

                if dict_retweet["id_str"] == id_rootTweet:
                    print("index_row_rootTweet:")
                    print(index_row_rootTweet)
                    print("id_rootTweet:")
                    print(id_rootTweet)
                    print("index_row_retweet:")
                    print(index_row_retweet)
                    print("id_retweet:")
                    print(id_retweet)
                    print("line:")
                    print(line)
                    print("This is the root tweet, not a retweet of the root tweet. Skip.")
                    continue
                if "retweeted_status" not in dict_retweet:
                    print("index_row_rootTweet:")
                    print(index_row_rootTweet)
                    print("id_rootTweet:")
                    print(id_rootTweet)
                    print("index_row_retweet:")
                    print(index_row_retweet)
                    print("id_retweet:")
                    print(id_retweet)
                    print("line:")
                    print(line)
                    print("This is not a retweet. Skip.")
                    continue
                if dict_retweet["retweeted_status"]["id_str"] != id_rootTweet:
                    print("index_row_rootTweet:")
                    print(index_row_rootTweet)
                    print("id_rootTweet:")
                    print(id_rootTweet)
                    print("index_row_retweet:")
                    print(index_row_retweet)
                    print("id_retweet:")
                    print(id_retweet)
                    print("line:")
                    print(line)
                    print("This is a retweet, but not made to the root tweet. Skip.")
                    continue

                df_retweets.loc[index_row_retweet, "rootTweetIdStr"] = id_rootTweet

                date_createdAt_retweet = datetime.strptime(str(dict_retweet["created_at"]), "%a %b %d %H:%M:%S %z %Y")
                df_retweets.loc[index_row_retweet, "retweetAge_sec"] = (date_createdAt_retweet - date_createdAt_rootTweet).total_seconds()

                df_retweets.loc[index_row_retweet, "idStr"] = str(dict_retweet["id_str"])
                # df_retweets.loc[index_row_retweet, "createdAt"] = str(dict_retweet["created_at"])

                # if "full_text" in dict_retweet:
                #     df_retweets.loc[index_row_retweet, "fullText"] = str(dict_retweet["full_text"])
                # elif "text" in dict_retweet:
                #     df_retweets.loc[index_row_retweet, "fullText"] = str(dict_retweet["text"])

                # df_retweets.loc[index_row_retweet, "entities_hashtags"] = (t["text"] for t in dict_retweet["entities"]["hashtags"])
                # df_retweets.loc[index_row_retweet, "entities_hashtags"] = str(dict_retweet["entities"]["hashtags"])
                # df_retweets.loc[index_row_retweet, "entities_symbols"] = str(dict_retweet["entities"]["symbols"])
                # df_retweets.loc[index_row_retweet, "count_entities_userMentions"] = len(dict_retweet["entities"]["user_mentions"])
                # df_retweets.loc[index_row_retweet, "user_description"] = str(dict_retweet["user"]["description"])
                # df_retweets.loc[index_row_retweet, "user_protected"] = str(dict_retweet["user"]["protected"])
                df_retweets.loc[index_row_retweet, "user_followersCount"] = dict_retweet["user"]["followers_count"]
                df_retweets.loc[index_row_retweet, "user_friendsCount"] = dict_retweet["user"]["friends_count"]
                df_retweets.loc[index_row_retweet, "user_listedCount"] = dict_retweet["user"]["listed_count"]
                # df_retweets.loc[index_row_retweet, "user_createdAt"] = str(dict_retweet["user"]["created_at"])
                df_retweets.loc[index_row_retweet, "user_favouritesCount"] = dict_retweet["user"]["favourites_count"]
                # df_retweets.loc[index_row_retweet, "user_geoEnabled"] = str(dict_retweet["user"]["geo_enabled"])
                # df_retweets.loc[index_row_retweet, "user_verified"] = str(dict_retweet["user"]["verified"])
                df_retweets.loc[index_row_retweet, "user_statusesCount"] = dict_retweet["user"]["statuses_count"]
                # df_retweets.loc[index_row_retweet, "user_lang"] = str(dict_retweet["user"]["lang"])
                # df_retweets.loc[index_row_retweet, "user_profileBackgroundColor"] = str(dict_retweet["user"]["profile_background_color"])
                # df_retweets.loc[index_row_retweet, "user_profileTextColor"] = str(dict_retweet["user"]["profile_text_color"])
                # df_retweets.loc[index_row_retweet, "user_profileUseBackgroundImage"] = str(dict_retweet["user"]["profile_use_background_image"])
                # df_retweets.loc[index_row_retweet, "user_defaultProfile"] = str(dict_retweet["user"]["default_profile"])
                # df_retweets.loc[index_row_retweet, "user_following"] = str(dict_retweet["user"]["following"])
                # df_retweets.loc[index_row_retweet, "geo"] = str(dict_retweet["geo"])
                # df_retweets.loc[index_row_retweet, "coordinates"] = str(dict_retweet["coordinates"])
                # df_retweets.loc[index_row_retweet, "place"] = str(dict_retweet["place"])
                # if "possibly_sensitive" in dict_retweet:
                    # df_retweets.loc[index_row_retweet, "possiblySensitive"] = str(dict_retweet["possibly_sensitive"])

                

                date_user_createdAt_retweet = datetime.strptime(str(dict_retweet["user"]["created_at"]), "%a %b %d %H:%M:%S %z %Y")
                df_retweets.loc[index_row_retweet, "user_accountAge_day"] = (date_current - date_user_createdAt_retweet).total_seconds()/(60*60*24)

                dict_sentimentScores = analyzer.polarity_scores(str(dict_retweet["user"]["description"]))
                df_retweets.loc[index_row_retweet, "user_description_sentiment_neg"] = dict_sentimentScores["neg"]
                df_retweets.loc[index_row_retweet, "user_description_sentiment_neu"] = dict_sentimentScores["neu"]
                df_retweets.loc[index_row_retweet, "user_description_sentiment_pos"] = dict_sentimentScores["pos"]
                df_retweets.loc[index_row_retweet, "user_description_sentiment_compound"] = dict_sentimentScores["compound"]

                df_retweets.loc[index_row_retweet, "user_description_subjectivity"] = TextBlob(str(dict_retweet["user"]["description"])).sentiment.subjectivity

                str_text_preprocessed = preprocessText(str(dict_retweet["user"]["description"]), nlp, list_stopWords)
                results_emotion = NRCLex(str_text_preprocessed)
                for affect in list_affects:
                    df_retweets.loc[index_row_retweet, "user_description_emotion_" + affect] = results_emotion.affect_frequencies[affect]
                    # print(affect)
                    # print(df_retweets.loc[index_row_retweet, "user_description_emotion_" + affect])

                dict_textStats = textStats(str(dict_retweet["user"]["description"]))
                for metric in list_textStats:
                    df_retweets.loc[index_row_retweet, "user_description_textStats_" + metric] = dict_textStats[metric]

                
                index_row_retweet += 1

                if(index_row_retweet % 100 == 0):
                    print("index_row_retweet = " + str(index_row_retweet))
                    df_retweets.to_csv(absFilename_output_timeSeries, index=False, quoting=csv.QUOTE_ALL)

            file_input_retweets.close()

            print("absFilename_output_timeSeries:")
            print(absFilename_output_timeSeries)

            df_retweets.to_csv(absFilename_output_timeSeries, index=False, quoting=csv.QUOTE_ALL)

        
        except Exception as e:

            track = traceback.format_exc()
            print(track + "\n")

            print("index_row_rootTweet:")
            print(str(index_row_rootTweet))
            print("id_rootTweet:")
            print(str(id_rootTweet))
            print("index_row_retweet:")
            print(str(index_row_retweet))
            print("id_retweet:")
            print(id_retweet)
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
            file_log.write("index_row_retweet:\n")
            file_log.write(str(index_row_retweet) + "\n")
            file_log.write("id_retweet:\n")
            file_log.write(id_retweet + "\n")
            file_log.write("timestamp:\n")
            file_log.write(str(timestamp) + "\n")
            file_log.write("line:\n")
            file_log.write(str(line) + "\n")
            file_log.flush()

            continue

if __name__ == "__main__":
    main(sys.argv[1:])
