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

    opts, args = getopt.getopt(argv, '', ["absFilename_input_rootTweets=", "path_input_rootTweetReplies=", "absFilename_output_timeSeries=", "absFilename_log="])

    print(opts)

    for opt, arg in opts:
        if opt == '--absFilename_input_rootTweets':
            absFilename_input_rootTweets = arg
        if opt == '--path_input_rootTweetReplies':
            path_input_rootTweetReplies = arg
        if opt == '--absFilename_output_timeSeries':
            absFilename_output_timeSeries = arg
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
    # absFilename_output_timeSeries = "D:\\vmwareSharedFolder\\TwitterDataAnalysis\\results\\cascadeResults_temporal_replies.csv"
    # absFilename_log = "D:\\vmwareSharedFolder\\TwitterDataAnalysis\\programs\\logs\\07_02_cascadeAnalysis_replies.log"

    if not os.path.exists(os.path.dirname(absFilename_log)):
        os.makedirs(os.path.dirname(absFilename_log))
    file_log = open(absFilename_log, "w+")
    file_log.close()
    file_log = open(absFilename_log, "a")

    if os.path.exists(absFilename_output_timeSeries):
        df_replies = pd.read_csv(absFilename_output_timeSeries, dtype=str)
        print("Output file exists. Load and continue with this file.")
    else:
        df_replies = pd.DataFrame()
        print("Output file does not exists. Start from scratch.")
    
    print("len(df_replies):")
    print(len(df_replies))



    df_input_rootTweets = pd.read_csv(absFilename_input_rootTweets, dtype=str)

    df_input_rootTweets["id_rootTweet"] = df_input_rootTweets["tweet link"].apply(lambda x: x.split('/')[-1])


    print("len(df_input_rootTweets):")
    print(len(df_input_rootTweets))

    df_input_rootTweets = df_input_rootTweets.sort_values(by=["id_rootTweet"], ascending=True)
    df_input_rootTweets = df_input_rootTweets.reset_index(drop=True)

    print("len(df_input_rootTweets):")
    print(len(df_input_rootTweets))

    

    analyzer = SentimentIntensityAnalyzer()
    nlp = spacy.load('en_core_web_sm', disable=['parser', 'ner'])
    list_stopWords = list(set(stopwords.words('english')))


    list_affects = ["fear", "anger", "anticip", "trust", "surprise", "positive", "negative", "sadness", "disgust", "joy"]
    list_textStats = ["charCount", "wordCount", "uqWordCount", "sentenceCount", "charsPerWord", "wordsPerSentence", "readability"]

    index_row_rootTweet = "EMPTY"
    id_rootTweet = "EMPTY"
    index_row_reply = "EMPTY"
    id_reply = "EMPTY"
    timestamp = "EMPTY"
    line = "EMPTY"

    index_row_reply = len(df_replies)


    for index_row_rootTweet in range(0, len(df_input_rootTweets)):

        id_rootTweet = df_input_rootTweets.loc[index_row_rootTweet, "id_rootTweet"]

        print("id_rootTweet:")
        print(id_rootTweet)

        try:

            if (len(df_replies) > 0) and (id_rootTweet in df_replies["rootTweetIdStr"].tolist()):
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


          

            file_input_replies = open(absFilename_input_replies, "r", encoding="utf-8")        

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


                df_replies.loc[index_row_reply, "rootTweetIdStr"] = id_rootTweet

                date_createdAt_reply = datetime.strptime(str(dict_reply["created_at"]), "%a %b %d %H:%M:%S %z %Y")
                df_replies.loc[index_row_reply, "replyAge_sec"] = (date_createdAt_reply - date_createdAt_rootTweet).total_seconds()

                df_replies.loc[index_row_reply, "idStr"] = str(dict_reply["id_str"])

                # if "full_text" in dict_reply:
                #     df_replies.loc[index_row_reply, "fullText"] = str(dict_reply["full_text"])
                # elif "text" in dict_reply:
                #     df_replies.loc[index_row_reply, "fullText"] = str(dict_reply["text"])
                

                # df_replies.loc[index_row_reply, "count_entities_hashtags"] = len([t["text"] for t in dict_reply["entities"]["hashtags"]])
                # df_replies.loc[index_row_reply, "entities_hashtags"] = str(dict_reply["entities"]["hashtags"])
                # df_replies.loc[index_row_reply, "entities_symbols"] = str(dict_reply["entities"]["symbols"])
                # df_replies.loc[index_row_reply, "count_entities_userMentions"] = len(dict_reply["entities"]["user_mentions"])
                # df_replies.loc[index_row_reply, "user_description"] = str(dict_reply["user"]["description"])
                # df_replies.loc[index_row_reply, "user_protected"] = str(dict_reply["user"]["protected"])
                df_replies.loc[index_row_reply, "user_followersCount"] = dict_reply["user"]["followers_count"]
                df_replies.loc[index_row_reply, "user_friendsCount"] = dict_reply["user"]["friends_count"]
                df_replies.loc[index_row_reply, "user_listedCount"] = dict_reply["user"]["listed_count"]
                # df_replies.loc[index_row_reply, "user_createdAt"] = str(dict_reply["user"]["created_at"])
                df_replies.loc[index_row_reply, "user_favouritesCount"] = dict_reply["user"]["favourites_count"]
                # df_replies.loc[index_row_reply, "user_geoEnabled"] = str(dict_reply["user"]["geo_enabled"])
                # df_replies.loc[index_row_reply, "user_verified"] = str(dict_reply["user"]["verified"])
                df_replies.loc[index_row_reply, "user_statusesCount"] = dict_reply["user"]["statuses_count"]
                # df_replies.loc[index_row_reply, "user_lang"] = str(dict_reply["user"]["lang"])
                # df_replies.loc[index_row_reply, "user_profileBackgroundColor"] = str(dict_reply["user"]["profile_background_color"])
                # df_replies.loc[index_row_reply, "user_profileTextColor"] = str(dict_reply["user"]["profile_text_color"])
                # df_replies.loc[index_row_reply, "user_profileUseBackgroundImage"] = str(dict_reply["user"]["profile_use_background_image"])
                # df_replies.loc[index_row_reply, "user_defaultProfile"] = str(dict_reply["user"]["default_profile"])
                # df_replies.loc[index_row_reply, "user_following"] = str(dict_reply["user"]["following"])
                # df_replies.loc[index_row_reply, "geo"] = str(dict_reply["geo"])
                # df_replies.loc[index_row_reply, "coordinates"] = str(dict_reply["coordinates"])
                # df_replies.loc[index_row_reply, "place"] = str(dict_reply["place"])
                # if "possibly_sensitive" in dict_reply:
                #     df_replies.loc[index_row_reply, "possiblySensitive"] = str(dict_reply["possibly_sensitive"])


                # date_createdAt_reply = datetime.strptime(df_replies.loc[index_row_reply, "createdAt"], "%a %b %d %H:%M:%S %z %Y")
                # df_replies.loc[index_row_reply, "replyAge_min"] = (date_createdAt_reply - date_createdAt_rootTweet).total_seconds()/60

                date_user_createdAt_reply = datetime.strptime(str(dict_reply["user"]["created_at"]), "%a %b %d %H:%M:%S %z %Y")
                df_replies.loc[index_row_reply, "user_accountAge_day"] = (date_current - date_user_createdAt_reply).total_seconds()/(60*60*24)

                if "full_text" in dict_reply:
                    fullText = str(dict_reply["full_text"])
                elif "text" in dict_reply:
                    fullText = str(dict_reply["text"])

                dict_sentimentScores = analyzer.polarity_scores(fullText)
                df_replies.loc[index_row_reply, "fullText_sentiment_neg"] = dict_sentimentScores["neg"]
                df_replies.loc[index_row_reply, "fullText_sentiment_neu"] = dict_sentimentScores["neu"]
                df_replies.loc[index_row_reply, "fullText_sentiment_pos"] = dict_sentimentScores["pos"]
                df_replies.loc[index_row_reply, "fullText_sentiment_compound"] = dict_sentimentScores["compound"]

                df_replies.loc[index_row_reply, "fullText_subjectivity"] = TextBlob(fullText).sentiment.subjectivity

                str_text_preprocessed = preprocessText(fullText, nlp, list_stopWords)
                results_emotion = NRCLex(str_text_preprocessed)
                for affect in list_affects:
                    df_replies.loc[index_row_reply, "fullText_emotion_" + affect] = results_emotion.affect_frequencies[affect]
                    # print(affect)
                    # print(df_replies.loc[index_row_reply, "fullText_emotion_" + affect])

                dict_textStats = textStats(fullText)
                for metric in list_textStats:
                    df_replies.loc[index_row_reply, "fullText_textStats_" + metric] = dict_textStats[metric]



                dict_sentimentScores = analyzer.polarity_scores(str(dict_reply["user"]["description"]))
                df_replies.loc[index_row_reply, "user_description_sentiment_neg"] = dict_sentimentScores["neg"]
                df_replies.loc[index_row_reply, "user_description_sentiment_neu"] = dict_sentimentScores["neu"]
                df_replies.loc[index_row_reply, "user_description_sentiment_pos"] = dict_sentimentScores["pos"]
                df_replies.loc[index_row_reply, "user_description_sentiment_compound"] = dict_sentimentScores["compound"]

                df_replies.loc[index_row_reply, "user_description_subjectivity"] = TextBlob(str(dict_reply["user"]["description"])).sentiment.subjectivity

                str_text_preprocessed = preprocessText(str(dict_reply["user"]["description"]), nlp, list_stopWords)
                results_emotion = NRCLex(str_text_preprocessed)
                for affect in list_affects:
                    df_replies.loc[index_row_reply, "user_description_emotion_" + affect] = results_emotion.affect_frequencies[affect]
                    # print(affect)
                    # print(df_replies.loc[index_row_reply, "user_description_emotion_" + affect])

                dict_textStats = textStats(str(dict_reply["user"]["description"]))
                for metric in list_textStats:
                    df_replies.loc[index_row_reply, "user_description_textStats_" + metric] = dict_textStats[metric]

                
                index_row_reply += 1

                if(index_row_reply % 100 == 0):
                    print("index_row_reply = " + str(index_row_reply))
                    df_replies.to_csv(absFilename_output_timeSeries, index=False, quoting=csv.QUOTE_ALL)

            file_input_replies.close()

            print("absFilename_output_timeSeries:")
            print(absFilename_output_timeSeries)

            df_replies.to_csv(absFilename_output_timeSeries, index=False, quoting=csv.QUOTE_ALL)
        
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
