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

    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)
    pd.set_option('display.max_colwidth', -1)

    random.seed(1113)
    
    
    # str_path_base = "D:\\vmwareSharedFolder\\TwitterDataCollection\\data\\PFN\\snapshot\\dateRetrieval=20200819" + os.path.sep
    # str_path_base = "D:\\vmwareSharedFolder\\TwitterDataCollection\\data\\PFN\\snapshot\\dateRetrieval=20201008" + os.path.sep
    # str_path_base = "D:\\vmwareSharedFolder\\TwitterDataCollection\\data\\PFN\\snapshot\\dateRetrieval=20201111" + os.path.sep
    str_path_base = "D:\\vmwareSharedFolder\\TwitterDataCollection\\data\\PFN\\snapshot\\dateRetrieval=20210209" + os.path.sep
    # absFilename_input_statistics_raw = str_path_base + "statistics_retweets_20201008.csv"
    absFilename_input_statistics_raw = str_path_base + "statistics_retweets_20201111.csv"
    # absFilename_output_statistics_selected = str_path_base + "statistics_retweets_selected1_20201008.csv"
    absFilename_output_statistics_selected = str_path_base + "statistics_retweets_selected1_20201111.csv"
    # absFilename_output_rootTweets_factcheckArticleRep = str_path_base + "rootTweets_selected_factcheckArticleRep_20201008.csv"
    absFilename_output_rootTweets_factcheckArticleRep = str_path_base + "rootTweets_selected_factcheckArticleRep_20201111.csv"
    list_absFilenames_input_rootTweets_raw = ["D:\\vmwareSharedFolder\\TwitterDataCollection\\data\\fake news\\fakenews_factcheck.csv", "D:\\vmwareSharedFolder\\TwitterDataCollection\\data\\fake news\\fakenews_politifact.csv", "D:\\vmwareSharedFolder\\TwitterDataCollection\\data\\fake news\\fakenews_snopes.csv", "D:\\vmwareSharedFolder\\TwitterDataCollection\\data\\fake news\\fakenews_truthorfiction.csv"]
    # list_absFilenames_input_rootTweets_raw = ["D:\\vmwareSharedFolder\\TwitterDataCollection\\data\\fake news\\fakenews_politifact.csv"]


    df_input_statistics_raw = pd.read_csv(absFilename_input_statistics_raw)

    df_input_statistics_raw["id_rootTweet"] = df_input_statistics_raw["id_rootTweet"].astype(str)

    num_rootTweets_total = len(df_input_statistics_raw)
    df_input_statistics_selected = df_input_statistics_raw.loc[df_input_statistics_raw["ptg_retweets_total_24hrs"] >= 0.5,]
    num_rootTweets_selected = len(df_input_statistics_selected)

    print("num_rootTweets_total:")
    print(num_rootTweets_total)
    print("num_rootTweets_selected:")
    print(num_rootTweets_selected)

    print("absFilename_output_statistics_selected:")
    print(absFilename_output_statistics_selected)

    df_input_statistics_selected.to_csv(absFilename_output_statistics_selected, index=False)

    num_rootTweets_selected = 0
    num_articles = 0
    df_output_rootTweets_factcheckArticleRep = pd.DataFrame()

    for absFilenames_input_rootTweets_raw in list_absFilenames_input_rootTweets_raw:

        print("absFilenames_input_rootTweets_raw:")
        print(absFilenames_input_rootTweets_raw)

        df_input_rootTweets_raw = pd.read_csv(absFilenames_input_rootTweets_raw, dtype=str)

        print("len(df_input_rootTweets_raw):")
        print(len(df_input_rootTweets_raw))

        df_input_rootTweets_raw = df_input_rootTweets_raw.dropna(subset=["tweet link"])

        print("len(df_input_rootTweets_raw):")
        print(len(df_input_rootTweets_raw))

        # df_input_rootTweets_raw["id_rootTweet"] = df_input_rootTweets_raw["tweet link"].str.rsplit('/', 1)[-1]
        df_input_rootTweets_raw["id_rootTweet"] = df_input_rootTweets_raw["tweet link"].apply(lambda x: x.split('/')[-1])

        # print("df_input_rootTweets_raw[\"id_rootTweet\"]:")
        # print(df_input_rootTweets_raw["id_rootTweet"].tolist())
        # return

        # print(df_input_statistics_selected["id_rootTweet"].tolist())

        # df_output_rootTweets_selected = df_input_rootTweets_raw.loc[df_input_rootTweets_raw["id_rootTweet"].isin(df_input_statistics_selected["id_rootTweet"].tolist()),]

        df_output_rootTweets_selected = pd.merge(left=df_input_rootTweets_raw, right=df_input_statistics_selected, how="inner", on=["id_rootTweet"])

        print("len(df_output_rootTweets_selected):")
        print(len(df_output_rootTweets_selected))

        df_output_rootTweets_selected["retrievalDate_int"] = df_output_rootTweets_selected["retrieval date"].apply(lambda x: datetime.strptime(x, "%m/%d/%Y").strftime('%Y%m%d'))
        # df_output_rootTweets_selected.sort_values(["retrievalDate_int", "id_rootTweet"],ascending=False).groupby("retrievalDate_int")
        df_output_rootTweets_selected = df_output_rootTweets_selected.sort_values(["retrievalDate_int", "link to checker article", "id_rootTweet"],ascending=True)
        df_output_rootTweets_selected = df_output_rootTweets_selected[["retrievalDate_int", "retrieval date", "checker assessment", "link to checker article", "id_rootTweet", "str_createdAt_rootTweet", "num_retweets_total", "num_retweets", "ptg_retweets_total_24hrs", "ptg_retweets_total_48hrs", "ptg_retweets_total_72hrs", "tweet link"]]
        df_output_rootTweets_selected = df_output_rootTweets_selected.reset_index(drop=True)
        num_articles += len(set(df_output_rootTweets_selected["link to checker article"].tolist()))
        num_rootTweets_selected += len(df_output_rootTweets_selected)

        absFilenames_output_rootTweets_raw = absFilenames_input_rootTweets_raw.replace(".csv", "_selected.csv")
        df_output_rootTweets_selected.to_csv(absFilenames_output_rootTweets_raw, index=False)

        print("print(pd.value_counts(df_output_rootTweets_selected[\"checker assessment\"])):")
        print(pd.value_counts(df_output_rootTweets_selected["checker assessment"]))


        df_temp = df_output_rootTweets_selected.groupby("link to checker article").first().reset_index()
        df_output_rootTweets_factcheckArticleRep = df_output_rootTweets_factcheckArticleRep.append(df_temp, ignore_index=True)
        # df_temp = df_output_rootTweets_selected.loc[df_output_rootTweets_selected.reset_index().groupby(["link to checker article"])["num_retweets"].idxmax()]
        df_temp = df_output_rootTweets_selected.loc[df_output_rootTweets_selected.groupby(["link to checker article"])["num_retweets"].idxmax()]
        df_output_rootTweets_factcheckArticleRep = df_output_rootTweets_factcheckArticleRep.append(df_temp, ignore_index=True)

    print("num_rootTweets_selected:")
    print(num_rootTweets_selected)
    print("num_articles:")
    print(num_articles)

    print("absFilename_output_rootTweets_factcheckArticleRep:")
    print(absFilename_output_rootTweets_factcheckArticleRep)
    print(len(set(df_output_rootTweets_factcheckArticleRep["link to checker article"].tolist())))
    print(len(df_output_rootTweets_factcheckArticleRep))

    df_output_rootTweets_factcheckArticleRep = df_output_rootTweets_factcheckArticleRep.drop_duplicates()
    df_output_rootTweets_factcheckArticleRep = df_output_rootTweets_factcheckArticleRep.sort_values(["retrievalDate_int", "link to checker article", "id_rootTweet"],ascending=True)
    df_output_rootTweets_factcheckArticleRep = df_output_rootTweets_factcheckArticleRep.reset_index(drop=True)

    print(len(df_output_rootTweets_factcheckArticleRep))

    df_output_rootTweets_factcheckArticleRep = df_output_rootTweets_factcheckArticleRep.sample(frac=1).reset_index(drop=True)

    print(len(df_output_rootTweets_factcheckArticleRep))


    df_output_rootTweets_factcheckArticleRep.to_csv(absFilename_output_rootTweets_factcheckArticleRep, index=False)
    

if __name__ == "__main__":
    main(sys.argv[1:])
