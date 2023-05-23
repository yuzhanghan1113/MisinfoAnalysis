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
import itertools
from collections import Counter




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
    # preprocess root tweet data:
    

    list_veracityLabels_true = ["<TR>", "<MOSTLY_TR>", "<CORRECT_ATTRIBUTION>", "<TR_BY_OPPOSING_FN>"]
    list_veracityLabels_unknown = ["<UNKNOWN>", "<UNPROVEN>"]
    list_veracityLabels_satire = ["<LABELED_SATIRE>"]

    # absFilename_input_rootTweets = "D:\\vmwareSharedFolder\\TwitterDataCollection\\data\\PFN\\snapshot\\dateRetrieval=20201111\\rootTweets_selected_factcheckArticleRep_20201111.csv"
    absFilename_input_rootTweets = "C:\\vmwareSharedFolder\\TwitterDataCollection\\data\\PFN\\snapshot\\dateRetrieval=20210209\\rootTweets_selected_factcheckArticleRep_annotated_20210209.csv"
    # absFilename_output_rootTweets = "D:\\vmwareSharedFolder\\TwitterDataAnalysis\\results\\rootTweets_selected_factcheckArticleRep_20201111_preprocessed.csv"
    absFilename_output_rootTweets = "C:\\vmwareSharedFolder\\TwitterDataAnalysis\\results\\rootTweets_selected_factcheckArticleRep_20210209_preprocessed.csv"

    df_input_rootTweets = pd.read_csv(absFilename_input_rootTweets, dtype=str, encoding="ISO-8859-1")

    print("len(df_input_rootTweets):")
    print(len(df_input_rootTweets))

    # print(df_input_rootTweets["tweet link"].str.split("/")[-1])
    # print(df_input_rootTweets["tweet link"].str.split("/"))
    df_input_rootTweets["id_rootTweet"] = df_input_rootTweets["tweet link"].apply(lambda x: x.split('/')[-1])
    # print(df_input_rootTweets[["id_rootTweet", "tweet link"]])
    # return


    df_input_rootTweets["id_rootTweet_float"] = df_input_rootTweets["id_rootTweet"].astype(float)

    df_input_rootTweets = df_input_rootTweets.drop_duplicates()
    df_input_rootTweets = df_input_rootTweets.sort_values(by=["id_rootTweet_float"], ascending=True)
    df_input_rootTweets = df_input_rootTweets.reset_index(drop=True)


    print("len(df_input_rootTweets):")
    print(len(df_input_rootTweets))

    # print(list(set(df_input_rootTweets["checker assessment_split"].tolist())))

    df_input_rootTweets["communicative intention_split"] = df_input_rootTweets["communicative intention 2"].apply(lambda x: "" if pd.isnull(x) else sorted(list(set(x.split(",")))))
    # print(df_input_rootTweets["communicative intention_split"])
    print("Counter(list(itertools.chain.from_iterable(df_input_rootTweets[\"communicative intention_split\"].tolist()))):")
    print(Counter(list(itertools.chain.from_iterable(df_input_rootTweets["communicative intention_split"].tolist()))))


    list_communicativeIntentions = ['REP', 'DIR', 'COM', 'EXP', 'DEC', 'QOU']

    for cm in list_communicativeIntentions:
        df_input_rootTweets["communicativeIntention_" + cm] = df_input_rootTweets["communicative intention_split"].apply(lambda x: "1" if cm in x else "0")
    
    df_input_rootTweets["communicativeIntention_REP_and_QOU"] = df_input_rootTweets["communicative intention_split"].apply(lambda x: "1" if ("REP" in x) and ("QOU" in x) and (len(list(set(x)))==2) else "0")
    df_input_rootTweets["communicativeIntention_REP_plus"] = df_input_rootTweets["communicative intention_split"].apply(lambda x: "1" if (("REP" in x) and (len(list(set(x)))>1)) else ("0" if (("REP" in x) and (len(list(set(x)))==1)) else ""))

    df_input_rootTweets["communicativeIntention_count"] = df_input_rootTweets["communicative intention_split"].apply(lambda x: len(list(set(x))))

    list_counts = [1, 2, 3, 4, 5, 6]

    for c in list_counts:
        df_input_rootTweets["communicativeIntention_count_EQ" + str(c)] = df_input_rootTweets["communicative intention_split"].apply(lambda x: "1" if len(list(set(x)))==c else "0")

    df_input_rootTweets["checker assessment_split"] = df_input_rootTweets["checker assessment"]

    df_input_rootTweets["checker assessment_split"] = df_input_rootTweets["checker assessment_split"].apply(lambda x: x.replace("not true", "<NOT_TR>"))
    df_input_rootTweets["checker assessment_split"] = df_input_rootTweets["checker assessment_split"].apply(lambda x: x.replace("not ture", "<NOT_TR>"))
    df_input_rootTweets["checker assessment_split"] = df_input_rootTweets["checker assessment_split"].apply(lambda x: x.replace("Not True", "<NOT_TR>"))
    df_input_rootTweets["checker assessment_split"] = df_input_rootTweets["checker assessment_split"].apply(lambda x: x.replace("half true", "<HALF_TR>"))
    df_input_rootTweets["checker assessment_split"] = df_input_rootTweets["checker assessment_split"].apply(lambda x: x.replace("mostly true", "<MOSTLY_TR>"))

    df_input_rootTweets["checker assessment_split"] = df_input_rootTweets["checker assessment_split"].apply(lambda x: x.replace("true by oppose to FN", "<TR_BY_OPPOSING_FN>"))
    df_input_rootTweets["checker assessment_split"] = df_input_rootTweets["checker assessment_split"].apply(lambda x: x.replace("true by opposite to FN", "<TR_BY_OPPOSING_FN>"))
    df_input_rootTweets["checker assessment_split"] = df_input_rootTweets["checker assessment_split"].apply(lambda x: x.replace("True by opposing to factcheking", "<TR_BY_OPPOSING_FN>"))
    df_input_rootTweets["checker assessment_split"] = df_input_rootTweets["checker assessment_split"].apply(lambda x: x.replace("TRUE by oppose to FN", "<TR_BY_OPPOSING_FN>"))
    df_input_rootTweets["checker assessment_split"] = df_input_rootTweets["checker assessment_split"].apply(lambda x: x.replace("TRUE by opposing fact check", "<TR_BY_OPPOSING_FN>"))
    df_input_rootTweets["checker assessment_split"] = df_input_rootTweets["checker assessment_split"].apply(lambda x: x.replace("TRUE", "<TR>"))
    df_input_rootTweets["checker assessment_split"] = df_input_rootTweets["checker assessment_split"].apply(lambda x: x.replace("true", "<TR>"))
    
    df_input_rootTweets["checker assessment_split"] = df_input_rootTweets["checker assessment_split"].apply(lambda x: x.replace("correct attribution", "<CORRECT_ATTRIBUTION>"))

    df_input_rootTweets["checker assessment_split"] = df_input_rootTweets["checker assessment_split"].apply(lambda x: x.replace("mixed", "<MIXED>"))
    df_input_rootTweets["checker assessment_split"] = df_input_rootTweets["checker assessment_split"].apply(lambda x: x.replace("mixture", "<MIXTURE>"))
    df_input_rootTweets["checker assessment_split"] = df_input_rootTweets["checker assessment_split"].apply(lambda x: x.replace("Mixture", "<MIXTURE>"))
    df_input_rootTweets["checker assessment_split"] = df_input_rootTweets["checker assessment_split"].apply(lambda x: x.replace("mixtiure", "<MIXTURE>"))
    df_input_rootTweets["checker assessment_split"] = df_input_rootTweets["checker assessment_split"].apply(lambda x: x.replace("unknown", "<UNKNOWN>"))
    df_input_rootTweets["checker assessment_split"] = df_input_rootTweets["checker assessment_split"].apply(lambda x: x.replace("unproven", "<UNPROVEN>"))
    df_input_rootTweets["checker assessment_split"] = df_input_rootTweets["checker assessment_split"].apply(lambda x: x.replace("Labeled Satire", "<LABELED_SATIRE>"))


    df_input_rootTweets["checker assessment_split"] = df_input_rootTweets["checker assessment_split"].apply(lambda x: x.replace("FALSE", "<FA>"))
    df_input_rootTweets["checker assessment_split"] = df_input_rootTweets["checker assessment_split"].apply(lambda x: x.replace("falase", "<FA>"))
    df_input_rootTweets["checker assessment_split"] = df_input_rootTweets["checker assessment_split"].apply(lambda x: x.replace("mostly false", "<MOSTLY_FA>"))
    df_input_rootTweets["checker assessment_split"] = df_input_rootTweets["checker assessment_split"].apply(lambda x: x.replace("false stories", "<FA_STORIES>"))
    df_input_rootTweets["checker assessment_split"] = df_input_rootTweets["checker assessment_split"].apply(lambda x: x.replace("false claim", "<FA_CLAIM>"))
    df_input_rootTweets["checker assessment_split"] = df_input_rootTweets["checker assessment_split"].apply(lambda x: x.replace("false", "<FA>"))

    df_input_rootTweets["checker assessment_split"] = df_input_rootTweets["checker assessment_split"].apply(lambda x: x.replace("outdated", "<OUTDATED>"))
    df_input_rootTweets["checker assessment_split"] = df_input_rootTweets["checker assessment_split"].apply(lambda x: x.replace("pants on fire", "<PANTS_ON_FIRE>"))
    df_input_rootTweets["checker assessment_split"] = df_input_rootTweets["checker assessment_split"].apply(lambda x: x.replace("pants-on-fire", "<PANTS_ON_FIRE>"))
    df_input_rootTweets["checker assessment_split"] = df_input_rootTweets["checker assessment_split"].apply(lambda x: x.replace("misattributed", "<MISATTRIBUTED>"))
    df_input_rootTweets["checker assessment_split"] = df_input_rootTweets["checker assessment_split"].apply(lambda x: x.replace("miscaptionated", "<MISCAPTIONATED>"))
    df_input_rootTweets["checker assessment_split"] = df_input_rootTweets["checker assessment_split"].apply(lambda x: x.replace("miscaptioned", "<MISCAPTIONATED>"))
    df_input_rootTweets["checker assessment_split"] = df_input_rootTweets["checker assessment_split"].apply(lambda x: x.replace("decontextualized", "<DECONTEXTUALIZED>"))
    df_input_rootTweets["checker assessment_split"] = df_input_rootTweets["checker assessment_split"].apply(lambda x: x.replace("scam", "<SCAM>"))

    df_input_rootTweets["checker assessment_split"] = df_input_rootTweets["checker assessment_split"].apply(lambda x: x.replace("ture", "<TR>"))


    # print(re.findall('"([^"]*)"', df_input_rootTweets["checker assessment_split"].str))

    df_input_rootTweets["checker assessment_split"] = df_input_rootTweets["checker assessment_split"].apply(lambda x: re.findall('<[A-Z_]+>', x))
    ## df_input_rootTweets["checker assessment_split"] = df_input_rootTweets["checker assessment_split"].apply(lambda x: [e.lower() for e in x])
    # print(df_input_rootTweets["checker assessment_split"])

    # return

    # print(df_input_rootTweets[["checker assessment_split", "checker assessment_split"]])
    # print(df_input_rootTweets.loc[df_input_rootTweets["checker assessment_split"].isnull(), "id_rootTweet"])

    # return


    print("Counter(list(itertools.chain.from_iterable(df_input_rootTweets[\"checker assessment_split\"].tolist()))):")
    print(Counter(list(itertools.chain.from_iterable(df_input_rootTweets["checker assessment_split"].tolist()))))
    
    # return 

    list_veracityLabels = list(set(list(itertools.chain.from_iterable(df_input_rootTweets["checker assessment_split"].tolist()))))

    list_veracityLabels_false = list(set(list_veracityLabels) - set(list_veracityLabels_true) - set(list_veracityLabels_unknown) - set(list_veracityLabels_satire))

    print("list_veracityLabels_false:")
    print(list_veracityLabels_false)

    # return

    df_input_rootTweets["veracityLabel_agg_misinformation"] = df_input_rootTweets["checker assessment_split"].apply(lambda x: "1" if any(i in x for i in list_veracityLabels_false) else "0")
    df_input_rootTweets["veracityLabel_agg_authentic"] = df_input_rootTweets["checker assessment_split"].apply(lambda x: "1" if any(i in x for i in list_veracityLabels_true) else "0")
    df_input_rootTweets["veracityLabel_agg_unknown"] = df_input_rootTweets["checker assessment_split"].apply(lambda x: "1" if any(i in x for i in list_veracityLabels_unknown) else "0")
    df_input_rootTweets["veracityLabel_agg_satire"] = df_input_rootTweets["checker assessment_split"].apply(lambda x: "1" if any(i in x for i in list_veracityLabels_satire) else "0")

    for label in list_veracityLabels:
        df_input_rootTweets["veracityLabel_" + label.upper()] = df_input_rootTweets["checker assessment_split"].apply(lambda x: "1" if label in x else "0")

    df_output_rootTweets = df_input_rootTweets.copy()

    df_output_rootTweets.columns = ["ROOTTWEETS_" + c for c in df_output_rootTweets.columns]

    print("len(df_output_rootTweets):")
    print(len(df_output_rootTweets))

    print("absFilename_output_rootTweets:")
    print(absFilename_output_rootTweets)

    df_output_rootTweets.to_csv(absFilename_output_rootTweets, index=False, quoting=csv.QUOTE_ALL)
    """

    # create annotations for producer approach:

    # absFilename_input_rootTweets = "C:\\vmwareSharedFolder\\TwitterDataAnalysis\\results\\results_20210209\\preprocessedData.csv"
    # absFilename_input_rootTweets = "C:\\vmwareSharedFolder\\TwitterDataCollection\\data\\PFN\\snapshot\\dateRetrieval=20210209\\rootTweets_selected_factcheckArticleRep_20210209.csv"
    absFilename_input_rootTweets = "C:\\vmwareSharedFolder\\TwitterDataCollection\\data\\PFN\\snapshot\\dateRetrieval=20210209\\rootTweets_selected_factcheckArticleRep_annotated_20210209.csv"

    absFilename_output_rootTweets = "C:\\vmwareSharedFolder\\TwitterDataAnalysis\\results\\results_20210209\\preprocessedData_proIntProApp.csv"

    df_input_rootTweets = pd.read_csv(absFilename_input_rootTweets, dtype=str, encoding="ISO-8859-1")

    print("len(df_input_rootTweets):")
    print(len(df_input_rootTweets))

        # print(list(set(df_input_rootTweets["checker assessment_split"].tolist())))

    var = np.nan
    print(isinstance(var, str))
    print(np.isnan(var))

    # print("df_input_rootTweets[\"ROOTTWEETS_production approach 1\"]:")
    # print(df_input_rootTweets["ROOTTWEETS_production approach 1"])

    # df_input_rootTweets["ROOTTWEETS_production approach 1"] = df_input_rootTweets["ROOTTWEETS_production approach 1"].apply(lambda x: "EMPTYCELL" if pd.isnull(x) else x)
    # df_input_rootTweets["ROOTTWEETS_production approach 1"] = df_input_rootTweets["ROOTTWEETS_production approach 1"].apply(lambda x: "EMPTYCELL" if pd.isnull(x) else x)
    # df_input_rootTweets["ROOTTWEETS_communicative intention 2"] = df_input_rootTweets["ROOTTWEETS_communicative intention 2"].apply(lambda x: "EMPTYCELL" if pd.isnull(x) else x)
    # df_input_rootTweets["production approach 1"] = df_input_rootTweets["production approach 1"].apply(lambda x: "EMPTYCELL" if pd.isnull(x) else x)
    df_input_rootTweets["communicative intention 2"] = df_input_rootTweets["communicative intention 2"].apply(lambda x: "EMPTYCELL" if pd.isnull(x) else x)

    # df_input_rootTweets["ROOTTWEETS_production approach 1"] = df_input_rootTweets["ROOTTWEETS_production approach 1"].apply(lambda x: x.replace("IMA", "IMG") if isinstance(x, str) else x if np.isnan(x) else x)
    # df_input_rootTweets["ROOTTWEETS_production approach 1"] = df_input_rootTweets["ROOTTWEETS_production approach 1"].apply(lambda x: x.replace("QUL", "QUA") if isinstance(x, str) else x if np.isnan(x) else x)
    # df_input_rootTweets["ROOTTWEETS_production approach 1"] = df_input_rootTweets["ROOTTWEETS_production approach 1"].apply(lambda x: x.replace("QOT", "QUT") if isinstance(x, str) else x if np.isnan(x) else x)
    # df_input_rootTweets["ROOTTWEETS_production approach 1"] = df_input_rootTweets["ROOTTWEETS_production approach 1"].apply(lambda x: x.replace("DEL", "REL") if isinstance(x, str) else x if np.isnan(x) else x)
    # df_input_rootTweets["ROOTTWEETS_production approach 1"] = df_input_rootTweets["ROOTTWEETS_production approach 1"].apply(lambda x: x.replace("OUA", "QUA") if isinstance(x, str) else x if np.isnan(x) else x)
    # df_input_rootTweets["ROOTTWEETS_production approach 1"] = df_input_rootTweets["ROOTTWEETS_production approach 1"].apply(lambda x: x.replace(" DEP", "DEP") if isinstance(x, str) else x if np.isnan(x) else x)
    # df_input_rootTweets["ROOTTWEETS_production approach 1"] = df_input_rootTweets["ROOTTWEETS_production approach 1"].apply(lambda x: x.replace(" IMG", "IMG") if isinstance(x, str) else x if np.isnan(x) else x)
    # df_input_rootTweets["ROOTTWEETS_production approach 1"] = df_input_rootTweets["ROOTTWEETS_production approach 1"].apply(lambda x: x.replace(" QUT", "QUT") if isinstance(x, str) else x if np.isnan(x) else x)
    # df_input_rootTweets["ROOTTWEETS_production approach 1"] = df_input_rootTweets["ROOTTWEETS_production approach 1"].apply(lambda x: x.replace("OUT", "QUT") if isinstance(x, str) else x if np.isnan(x) else x)
    # df_input_rootTweets["ROOTTWEETS_production approach 1"] = df_input_rootTweets["ROOTTWEETS_production approach 1"].apply(lambda x: x.replace("REP", "REL") if isinstance(x, str) else x if np.isnan(x) else x)
    
    # df_input_rootTweets["ROOTTWEETS_productionApproach_split"] = df_input_rootTweets["ROOTTWEETS_production approach 1"].apply(lambda x: sorted(list(set(x.split(",")))) if isinstance(x, str) else x if np.isnan(x) else x)

    # print(Counter(df_input_rootTweets["ROOTTWEETS_production approach 1"]))
    # print(len(df_input_rootTweets["ROOTTWEETS_production approach 1"]))
    # print(Counter(df_input_rootTweets["ROOTTWEETS_communicative intention 2"]))
    # print(len(df_input_rootTweets["ROOTTWEETS_communicative intention 2"]))

    # print(Counter(df_input_rootTweets["production approach 1"]))
    # print(len(df_input_rootTweets["production approach 1"]))
    print(Counter(df_input_rootTweets["communicative intention 2"]))
    print(len(df_input_rootTweets["communicative intention 2"]))

    # print("Counter(list(itertools.chain.from_iterable(df_input_rootTweets[\"ROOTTWEETS_productionApproach_split\"].tolist()))):")
    # print(Counter(list(itertools.chain.from_iterable(df_input_rootTweets["ROOTTWEETS_productionApproach_split"].tolist()))))

    return

    list_communicativeIntentions = ["QUT", "IMG", "DEP", "REL", "QUA"]

    for cm in list_communicativeIntentions:
        df_input_rootTweets["communicativeIntention_" + cm] = df_input_rootTweets["communicative intention_split"].apply(lambda x: "1" if cm in x else "0")
    
    df_input_rootTweets["communicativeIntention_REP_and_QOU"] = df_input_rootTweets["communicative intention_split"].apply(lambda x: "1" if ("REP" in x) and ("QOU" in x) and (len(list(set(x)))==2) else "0")
    df_input_rootTweets["communicativeIntention_REP_plus"] = df_input_rootTweets["communicative intention_split"].apply(lambda x: "1" if (("REP" in x) and (len(list(set(x)))>1)) else ("0" if (("REP" in x) and (len(list(set(x)))==1)) else ""))

    df_input_rootTweets["communicativeIntention_count"] = df_input_rootTweets["communicative intention_split"].apply(lambda x: len(list(set(x))))

    list_counts = [1, 2, 3, 4, 5, 6]

    for c in list_counts:
        df_input_rootTweets["communicativeIntention_count_EQ" + str(c)] = df_input_rootTweets["communicative intention_split"].apply(lambda x: "1" if len(list(set(x)))==c else "0")


    """
    # preprocess time series:

    task = "retweets"
    #task = "replies"

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
    """



    
    """
    
    # merge separate cascade temporal files into single ones and get cascade end data:        

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

    

    # absFilename_input_rootTweets = "D:\\vmwareSharedFolder\\TwitterDataCollection\\data\\PFN\\snapshot\\dateRetrieval=20201111\\rootTweets_selected_factcheckArticleRep_20201111.csv"
   

    absFilename_input_rootTweets = "C:\\vmwareSharedFolder\\TwitterDataCollection\\data\\PFN\\snapshot\\dateRetrieval=20210209\\rootTweets_selected_factcheckArticleRep_annotated_20210209.csv"
    absFilename_input_retweets = "C:\\vmwareSharedFolder\\TwitterDataAnalysis\\results\\results_20210209\\cascadeResults_temporal_retweets.csv"
    absFilename_input_replies = "C:\\vmwareSharedFolder\\TwitterDataAnalysis\\results\\results_20210209\\cascadeResults_temporal_replies.csv"
    absFilename_input_producerTweets = "C:\\vmwareSharedFolder\\TwitterDataAnalysis\\results\\results_20210209\\cascadeResults_producerTweets_timeWindow=7.csv"
    absFilename_output = "C:\\vmwareSharedFolder\\TwitterDataAnalysis\\results\\results_20210209\\cascadeResults_cascadeEnd.csv"


    # absFilename_input_TS = "D:\\vmwareSharedFolder\\TwitterDataAnalysis\\results\\timeSeries_retweets.csv"
    # absFilename_input_replies_TS = "D:\\vmwareSharedFolder\\TwitterDataAnalysis\\results\\timeSeries_replies.csv"
    # absFilename_output_retweets_TS = "D:\\vmwareSharedFolder\\TwitterDataAnalysis\\results\\timeSeries_retweets_preprocessed.csv"
    # absFilename_output_replies_TS = "D:\\vmwareSharedFolder\\TwitterDataAnalysis\\results\\imeSeries_replies_prepreocessed.csv"

    df_input_rootTweets = pd.read_csv(absFilename_input_rootTweets, dtype=str, engine='python')
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

    df_input_rootTweets["id_rootTweet"] = df_input_rootTweets["tweet link"].apply(lambda x: x.split('/')[-1])

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

    # list_features_output = ["ROOTTWEETS_id_rootTweet", "RETWEETS_rootTweetIdStr", "RETWEETS_cascadeAge_min", "REPLIES_rootTweetIdStr", "REPLIES_cascadeAge_min", "PRODUCERTWEETS_rootTweetIdStr", "PRODUCERTWEETS_producerScreenName"]

    
    # list_features_output += [feature_base]

    # for f in list_featuresToStandardize:
    #     list_features_output += [f]
    #     list_features_output += [f + "_stadardized"]


    # df_result = df_result[list_features_output]

    df_result.columns = ["CASCADEEND_" + c for c in df_result.columns]

    df_result.to_csv(absFilename_output, index=False, quoting=csv.QUOTE_ALL)


    print("len(df_result):")
    print(len(df_result))
    print("len(df_result[\"CASCADEEND_ROOTTWEETS_id_rootTweet\"]):")
    print(len(df_result["CASCADEEND_ROOTTWEETS_id_rootTweet"]))
    print("len(set(list(df_result[\"CASCADEEND_ROOTTWEETS_id_rootTweet\"]))):")
    print(len(set(list(df_result["CASCADEEND_ROOTTWEETS_id_rootTweet"]))))
    print("len(df_result[\"CASCADEEND_RETWEETS_rootTweetIdStr\"]):")
    print(len(df_result["CASCADEEND_RETWEETS_rootTweetIdStr"]))
    print("len(set(list(df_result[\"CASCADEEND_RETWEETS_rootTweetIdStr\"]))):")
    print(len(set(list(df_result["CASCADEEND_RETWEETS_rootTweetIdStr"]))))
    print("len(df_result[\"CASCADEEND_REPLIES_rootTweetIdStr\"]):")
    print(len(df_result["CASCADEEND_REPLIES_rootTweetIdStr"]))
    print("len(set(list(df_result[\"CASCADEEND_REPLIES_rootTweetIdStr\"]))):")
    print(len(set(list(df_result["CASCADEEND_REPLIES_rootTweetIdStr"]))))
    print("len(df_result[\"CASCADEEND_PRODUCERTWEETS_rootTweetIdStr\"]):")
    print(len(df_result["CASCADEEND_PRODUCERTWEETS_rootTweetIdStr"]))
    print("len(set(list(df_result[\"CASCADEEND_PRODUCERTWEETS_rootTweetIdStr\"]))):")
    print(len(set(list(df_result["CASCADEEND_PRODUCERTWEETS_rootTweetIdStr"]))))
    """
  
    
    """
    # flatten time series data

    # absFilename_output = "D:\\vmwareSharedFolder\\TwitterDataAnalysis\\results\\cascadeResults_timeSeries.csv"
    # absFilename_output = "D:\\vmwareSharedFolder\\TwitterDataAnalysis\\results\\results_20210209\\cascadeResults_timeSeries.csv"
    absFilename_output = "C:\\vmwareSharedFolder\\TwitterDataAnalysis\\results\\results_20210209\\cascadeResults_timeSeries.csv"


    df_output = pd.DataFrame()

    list_tasks = ["retweets", "replies"]

    for task in list_tasks:

        print("task:")
        print(task)

        # absFilename_input_TSResults = "D:\\vmwareSharedFolder\\TwitterDataAnalysis\\results\\timeSeries\\metrics\\temporal_" + task + ".csv"
        absFilename_input_TSResults = "C:\\vmwareSharedFolder\\TwitterDataAnalysis\\results\\timeSeries\\metrics\\temporal_" + task + ".csv"
        df_input_TSResults = pd.read_csv(absFilename_input_TSResults, dtype=str)

        print("len(df_input_TSResults):")
        print(len(df_input_TSResults))

        list_metrics = df_input_TSResults.columns
        
        print("len(list_metrics):")
        print(len(list_metrics))

        list_metrics = [e for e in list_metrics if e not in ["rootTweetIdStr", "feature"]]

        print("len(list_metrics):")
        print(len(list_metrics))


        list_rootTweetIdStr = df_input_TSResults["rootTweetIdStr"].tolist()
        print("len(list_rootTweetIdStr):")
        print(len(list_rootTweetIdStr))
        list_rootTweetIdStr = list(set(df_input_TSResults["rootTweetIdStr"].tolist()))
        print("len(list_rootTweetIdStr):")
        print(len(list_rootTweetIdStr))

        index_rootTweet = 0
        
        for rootTweetIdStr in list_rootTweetIdStr:

            print("task:")
            print(task)

            print("index_rootTweet:")
            print(index_rootTweet)

            print("rootTweetIdStr:")
            print(rootTweetIdStr)

            df_output.loc[rootTweetIdStr, "rootTweetIdStr"] = rootTweetIdStr

            df_input_TSResults_rootTweet = df_input_TSResults[df_input_TSResults["rootTweetIdStr"] == rootTweetIdStr]

            list_features = list(set(df_input_TSResults_rootTweet["feature"].tolist()))

            print("len(list_features):")
            print(len(list_features))

            for feature in list_features:
                for metric in list_metrics:
                    df_temp = df_input_TSResults_rootTweet.loc[(df_input_TSResults_rootTweet["rootTweetIdStr"] == rootTweetIdStr) & (df_input_TSResults_rootTweet["feature"] == feature), metric].reset_index(drop=True)
                    df_output.loc[rootTweetIdStr, task.upper() + "_" + feature + "-" + metric] = str(df_temp[0])



            index_rootTweet += 1

            print("absFilename_output:")
            print(absFilename_output)

            df_output.to_csv(absFilename_output, index=False, quoting=csv.QUOTE_ALL)
    """ 

    """
    
    # merge root tweets data, cascade end data, and time series data

    # absFilename_input_rootTweets = "C:\\vmwareSharedFolder\\TwitterDataAnalysis\\results\\rootTweets_selected_factcheckArticleRep_20201111_preprocessed.csv"
    # absFilename_input_cascadeEnd = "C:\\vmwareSharedFolder\\TwitterDataAnalysis\\results\\cascadeResults_cascadeEnd.csv"
    # absFilename_input_timeSeries = "C:\\vmwareSharedFolder\\TwitterDataAnalysis\\results\\cascadeResults_timeSeries.csv"
    # absFilename_output_allFeaturesStr = "C:\\vmwareSharedFolder\\TwitterDataAnalysis\\results\\preprocessedData_allFeaturesStr.csv"
    # absFilename_output = "C:\\vmwareSharedFolder\\TwitterDataAnalysis\\results\\preprocessedData.csv"

    absFilename_input_rootTweets = "C:\\vmwareSharedFolder\\TwitterDataAnalysis\\results\\results_20210209\\rootTweets_selected_factcheckArticleRep_20210209_preprocessed.csv"
    absFilename_input_cascadeEnd = "C:\\vmwareSharedFolder\\TwitterDataAnalysis\\results\\results_20210209\\cascadeResults_cascadeEnd.csv"
    absFilename_input_timeSeries = "C:\\vmwareSharedFolder\\TwitterDataAnalysis\\results\\results_20210209\\cascadeResults_timeSeries.csv"
    absFilename_output_allFeaturesStr = "C:\\vmwareSharedFolder\\TwitterDataAnalysis\\results\\results_20210209\\preprocessedData_allFeaturesStr.csv"
    absFilename_output = "C:\\vmwareSharedFolder\\TwitterDataAnalysis\\results\\results_20210209\\preprocessedData.csv"

    df_input_rootTweets = pd.read_csv(absFilename_input_rootTweets, dtype=str)
    df_input_cascadeEnd = pd.read_csv(absFilename_input_cascadeEnd, dtype=str)
    df_input_timeSeries = pd.read_csv(absFilename_input_timeSeries, dtype=str)

    if not df_input_timeSeries.columns[0].startswith("TIMESERIES_"):
        df_input_timeSeries.columns = ["TIMESERIES_" + c for c in df_input_timeSeries.columns]

    print("len(df_input_rootTweets):")
    print(len(df_input_rootTweets))
    print("len(list(set(df_input_rootTweets[\"ROOTTWEETS_id_rootTweet\"].tolist()))):")
    print(len(list(set(df_input_rootTweets["ROOTTWEETS_id_rootTweet"].tolist()))))

    print("len(df_input_cascadeEnd):")
    print(len(df_input_cascadeEnd))
    print("len(list(set(df_input_cascadeEnd[\"CASCADEEND_ROOTTWEETS_id_rootTweet\"].tolist()))):")
    print(len(list(set(df_input_cascadeEnd["CASCADEEND_ROOTTWEETS_id_rootTweet"].tolist()))))

    print("len(df_input_timeSeries):")
    print(len(df_input_timeSeries))
    print("len(list(set(df_input_timeSeries[\"TIMESERIES_rootTweetIdStr\"].tolist()))):")
    print(len(list(set(df_input_timeSeries["TIMESERIES_rootTweetIdStr"].tolist()))))

    if not df_input_rootTweets.loc[0, "ROOTTWEETS_id_rootTweet"].startswith("id"):
        df_input_rootTweets["ROOTTWEETS_id_rootTweet"] = "id" + df_input_rootTweets["ROOTTWEETS_id_rootTweet"].astype(str)

    if not df_input_cascadeEnd.loc[0, "CASCADEEND_ROOTTWEETS_id_rootTweet"].startswith("id"):
        df_input_cascadeEnd["CASCADEEND_ROOTTWEETS_id_rootTweet"] = "id" + df_input_cascadeEnd["CASCADEEND_ROOTTWEETS_id_rootTweet"].astype(str)

    if not df_input_timeSeries.loc[0, "TIMESERIES_rootTweetIdStr"].startswith("id"):
        df_input_timeSeries["TIMESERIES_rootTweetIdStr"] = "id" + df_input_timeSeries["TIMESERIES_rootTweetIdStr"].astype(str)

    df_output = pd.merge(df_input_rootTweets, df_input_cascadeEnd, how='left', left_on=["ROOTTWEETS_id_rootTweet"], right_on=["CASCADEEND_ROOTTWEETS_id_rootTweet"])
    df_output = pd.merge(df_output, df_input_timeSeries, how='left', left_on=["ROOTTWEETS_id_rootTweet"], right_on=["TIMESERIES_rootTweetIdStr"])

    print("len(df_output):")
    print(len(df_output))
    print("len(list(set(df_output[\"ROOTTWEETS_id_rootTweet\"].tolist()))):")
    print(len(list(set(df_output["ROOTTWEETS_id_rootTweet"].tolist()))))


    print("absFilename_output_allFeaturesStr:")
    print(absFilename_output_allFeaturesStr)

    df_output.to_csv(absFilename_output_allFeaturesStr, index=False, quoting=csv.QUOTE_ALL)

    # list_features_int = ["ROOTTWEETS_retrievalDate_int"]    
    # list_features_int += [c for c in df_output.columns if c.startswith("ROOTTWEETS_veracityLabel")]
    # list_features_int += [c for c in df_output.columns if c.startswith("ROOTTWEETS_communicativeIntention")]


    # list_features_str = ["ROOTTWEETS_link to checker article","ROOTTWEETS_retrieval date","ROOTTWEETS_communicativeIntention_count","ROOTTWEETS_communicativeIntention_REP","ROOTTWEETS_communicativeIntention_DIR","ROOTTWEETS_communicativeIntention_COM","ROOTTWEETS_communicativeIntention_EXP","ROOTTWEETS_communicativeIntention_DEC","ROOTTWEETS_communicativeIntention_QOU","ROOTTWEETS_communicativeIntention_count_EQ1","ROOTTWEETS_communicativeIntention_count_EQ2","ROOTTWEETS_communicativeIntention_count_EQ3","ROOTTWEETS_communicativeIntention_count_EQ4","ROOTTWEETS_communicativeIntention_count_EQ5","ROOTTWEETS_communicativeIntention_count_EQ6", "ROOTTWEETS_checker assessment","ROOTTWEETS_id_rootTweet","ROOTTWEETS_str_createdAt_rootTweet","ROOTTWEETS_tweet link","ROOTTWEETS_tweet link","ROOTTWEETS_communicative intention 1","ROOTTWEETS_note 1","ROOTTWEETS_communicative intention 2","ROOTTWEETS_communicative intention_split","ROOTTWEETS_checker assessment_split","CASCADEEND_ROOTTWEETS_id_rootTweet","CASCADEEND_RETWEETS_rootTweetIdStr","CASCADEEND_REPLIES_rootTweetIdStr","CASCADEEND_PRODUCERTWEETS_rootTweetIdStr","CASCADEEND_PRODUCERTWEETS_producerScreenName","TIMESERIES_rootTweetIdStr"]

    # list_features_float = list(set(df_output.columns) - set(list_features_int) - set(list_features_str))

    # df_output[list_features_float] = df_output[list_features_float].astype(float)
    # # df_output[list_features_float] = df_output[list_features_float].astype(dtype=float, errors="ignore")
    # df_output[list_features_int] = df_output[list_features_int].astype(float).astype(int)

    for c in df_output.columns:
        if "-" in c:
            df_output = df_output.rename({c: c.replace("-", "DASH")}, axis='columns')


    print("absFilename_output:")
    print(absFilename_output)

    df_output.to_csv(absFilename_output, index=False)
    

    """

    """
    
    # check final preprocessed data

    absFilename_input = "C:\\vmwareSharedFolder\\TwitterDataAnalysis\\results\\results_20210209\\preprocessedData.csv"

    df_input = pd.read_csv(absFilename_input, dtype=str)

    print("len(df_input[\"ROOTTWEETS_id_rootTweet\"]):")
    print(len(df_input["ROOTTWEETS_id_rootTweet"]))
    print("len(set(df_input[\"ROOTTWEETS_id_rootTweet\"].tolist())):")
    print(len(set(df_input["ROOTTWEETS_id_rootTweet"].tolist())))

    print("len(df_input[\"CASCADEEND_ROOTTWEETS_id_rootTweet\"]):")
    print(len(df_input["CASCADEEND_ROOTTWEETS_id_rootTweet"]))
    print("len(set(df_input[\"CASCADEEND_ROOTTWEETS_id_rootTweet\"].tolist())):")
    print(len(set(df_input["CASCADEEND_ROOTTWEETS_id_rootTweet"].tolist())))

    print("len(df_input[\"CASCADEEND_ROOTTWEETS_tweet link\"]):")
    print(len(df_input["CASCADEEND_ROOTTWEETS_tweet link"]))
    print("len(set(df_input[\"CASCADEEND_ROOTTWEETS_tweet link\"].tolist())):")
    print(len(set(df_input["CASCADEEND_ROOTTWEETS_tweet link"].tolist())))

    print("len(df_input[\"CASCADEEND_RETWEETS_rootTweetIdStr\"]):")
    print(len(df_input["CASCADEEND_RETWEETS_rootTweetIdStr"]))
    print("len(set(df_input[\"CASCADEEND_RETWEETS_rootTweetIdStr\"].tolist())):")
    print(len(set(df_input["CASCADEEND_RETWEETS_rootTweetIdStr"].tolist())))

    print("len(df_input[\"CASCADEEND_REPLIES_rootTweetIdStr\"]):")
    print(len(df_input["CASCADEEND_REPLIES_rootTweetIdStr"]))
    print("len(set(df_input[\"CASCADEEND_REPLIES_rootTweetIdStr\"].tolist())):")
    print(len(set(df_input["CASCADEEND_REPLIES_rootTweetIdStr"].tolist())))

    print("len(df_input[\"CASCADEEND_PRODUCERTWEETS_rootTweetIdStr\"]):")
    print(len(df_input["CASCADEEND_PRODUCERTWEETS_rootTweetIdStr"]))
    print("len(set(df_input[\"CASCADEEND_PRODUCERTWEETS_rootTweetIdStr\"].tolist())):")
    print(len(set(df_input["CASCADEEND_PRODUCERTWEETS_rootTweetIdStr"].tolist())))

    print("len(df_input[\"CASCADEEND_PRODUCERTWEETS_producerScreenName\"]):")
    print(len(df_input["CASCADEEND_PRODUCERTWEETS_producerScreenName"]))
    print("len(set(df_input[\"CASCADEEND_PRODUCERTWEETS_producerScreenName\"].tolist())):")
    print(len(set(df_input["CASCADEEND_PRODUCERTWEETS_producerScreenName"].tolist())))

    print("len(df_input[\"TIMESERIES_rootTweetIdStr\"]):")
    print(len(df_input["TIMESERIES_rootTweetIdStr"]))
    print("len(set(df_input[\"TIMESERIES_rootTweetIdStr\"].tolist())):")
    print(len(set(df_input["TIMESERIES_rootTweetIdStr"].tolist())))
    
    """

if __name__ == "__main__":
    main(sys.argv[1:])