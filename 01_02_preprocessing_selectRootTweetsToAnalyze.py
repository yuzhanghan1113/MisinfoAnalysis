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
    str_path_base = "C:\\vmwareSharedFolder\\TwitterDataCollection\\data\\PFN\\snapshot\\dateRetrieval=20210209" + os.path.sep
    # absFilename_input_statistics_raw = str_path_base + "statistics_retweets_20201008.csv"
    # absFilename_input_statistics_raw = str_path_base + "statistics_retweets_20201111.csv"
    absFilename_input_statistics_raw = str_path_base + "statistics_retweets_20210209.csv"
    # absFilename_output_statistics_selected = str_path_base + "statistics_retweets_selected1_20201008.csv"
    # absFilename_output_statistics_selected = str_path_base + "statistics_retweets_selected1_20201111.csv"
    absFilename_output_statistics_selected = str_path_base + "statistics_retweets_selected_20210209.csv"
    # absFilename_output_rootTweets_factcheckArticleRep = str_path_base + "rootTweets_selected_factcheckArticleRep_20201008.csv"
    # absFilename_output_rootTweets_factcheckArticleRep = str_path_base + "rootTweets_selected_factcheckArticleRep_20201111.csv"
    absFilename_output_rootTweets_factcheckArticleRep = str_path_base + "rootTweets_selected_factcheckArticleRep_20210209.csv"
    absFilename_output_rootTweets_factcheckArticleRep_annotated = str_path_base + "rootTweets_selected_factcheckArticleRep_annotated_20210209.csv"
    # list_absFilenames_input_rootTweets_raw = ["D:\\vmwareSharedFolder\\TwitterDataCollection\\data\\fake news\\fakenews_factcheck.csv", "D:\\vmwareSharedFolder\\TwitterDataCollection\\data\\fake news\\fakenews_politifact.csv", "D:\\vmwareSharedFolder\\TwitterDataCollection\\data\\fake news\\fakenews_snopes.csv", "D:\\vmwareSharedFolder\\TwitterDataCollection\\data\\fake news\\fakenews_truthorfiction.csv"]
    # list_absFilenames_input_rootTweets_raw = ["D:\\vmwareSharedFolder\\TwitterDataCollection\\data\\fake news\\fakenews_politifact.csv"]
    list_absFilenames_input_rootTweets_raw = ["D:\\vmwareSharedFolder\\TwitterDataCollection\\data\\fake news\\fakenews_factcheck_20210209.csv", "D:\\vmwareSharedFolder\\TwitterDataCollection\\data\\fake news\\fakenews_politifact_20210209.csv", "D:\\vmwareSharedFolder\\TwitterDataCollection\\data\\fake news\\fakenews_snopes_20210209.csv", "D:\\vmwareSharedFolder\\TwitterDataCollection\\data\\fake news\\fakenews_truthorfiction_20210209.csv"]

    absFilename_input_rootTweets_factcheckArticleRep_existing_annotated = "D:\\vmwareSharedFolder\\TwitterDataCollection\\data\\PFN\\snapshot\\dateRetrieval=20201111\\rootTweets_selected_factcheckArticleRep_20201111.csv"

    df_input_statistics_raw = pd.read_csv(absFilename_input_statistics_raw)

    df_input_statistics_raw["id_rootTweet"] = df_input_statistics_raw["id_rootTweet"].astype(str)

    num_rootTweets_total = len(df_input_statistics_raw)
    # df_input_statistics_selected = df_input_statistics_raw.loc[df_input_statistics_raw["ptg_retweets_total_24hrs"] >= 0.5,]
    df_input_statistics_selected = df_input_statistics_raw.loc[(df_input_statistics_raw["ptg_retweets_total_24hrs"] >= 0.5) & (df_input_statistics_raw["num_retweets"] > 0)]
    num_rootTweets_selected = len(df_input_statistics_selected)

    print("num_rootTweets_total:")
    print(num_rootTweets_total)
    print("num_rootTweets_selected:")
    print(num_rootTweets_selected)

    df_input_statistics_selected = df_input_statistics_selected.sort_values(by=["id_rootTweet"], ascending=True)
    df_input_statistics_selected = df_input_statistics_selected.loc[df_input_statistics_selected.groupby("id_rootTweet")["ptg_retweets_total_24hrs"].idxmax()]
    df_input_statistics_selected = df_input_statistics_selected.reset_index(drop=True)

    print("after removing duplicate root tweets:")
    num_rootTweets_selected = len(df_input_statistics_selected)
    print("num_rootTweets_selected:")
    print(num_rootTweets_selected)

    print("absFilename_output_statistics_selected:")
    print(absFilename_output_statistics_selected)

    df_input_statistics_selected.to_csv(absFilename_output_statistics_selected, index=False)

    # print("df_input_statistics_selected.columns:")
    # print(df_input_statistics_selected.columns)

    # input("wait")


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

        # print("df_output_rootTweets_selected.columns:")
        # print(df_output_rootTweets_selected.columns)

        # input("wait")

        df_output_rootTweets_selected["retrievalDate_int"] = df_output_rootTweets_selected["retrieval date"].apply(lambda x: datetime.strptime(x, "%m/%d/%Y").strftime('%Y%m%d'))
        # df_output_rootTweets_selected.sort_values(["retrievalDate_int", "id_rootTweet"],ascending=False).groupby("retrievalDate_int")
        df_output_rootTweets_selected = df_output_rootTweets_selected.sort_values(["retrievalDate_int", "link to checker article", "id_rootTweet"],ascending=True)
        df_output_rootTweets_selected = df_output_rootTweets_selected[["retrievalDate_int", "retrieval date", "checker assessment", "link to checker article", "id_rootTweet", "str_createdAt_rootTweet", "str_fullText_rootTweet", "num_retweets_total", "num_retweets", "ptg_retweets_total_24hrs", "ptg_retweets_total_48hrs", "ptg_retweets_total_72hrs", "tweet link"]]
        df_output_rootTweets_selected = df_output_rootTweets_selected.reset_index(drop=True)
        num_articles += len(set(df_output_rootTweets_selected["link to checker article"].tolist()))
        num_rootTweets_selected += len(df_output_rootTweets_selected)

        absFilenames_output_rootTweets_raw = absFilenames_input_rootTweets_raw.replace(".csv", "_selected.csv")
        df_output_rootTweets_selected.to_csv(absFilenames_output_rootTweets_raw, index=False)

        print("pd.value_counts(df_output_rootTweets_selected[\"checker assessment\"]):")
        print(pd.value_counts(df_output_rootTweets_selected["checker assessment"]))

        # # group root tweets by checker articles and only pick the earliest root tweet in each group
        # df_temp = df_output_rootTweets_selected.groupby("link to checker article").first().reset_index()
        # df_output_rootTweets_factcheckArticleRep = df_output_rootTweets_factcheckArticleRep.append(df_temp, ignore_index=True)
        # # group root tweets by checker articles and only pick the root tweet with most retweets count in each group
        # df_temp = df_output_rootTweets_selected.loc[df_output_rootTweets_selected.groupby(["link to checker article"])["num_retweets"].idxmax()]
        # df_output_rootTweets_factcheckArticleRep = df_output_rootTweets_factcheckArticleRep.append(df_temp, ignore_index=True)

        df_output_rootTweets_factcheckArticleRep = df_output_rootTweets_factcheckArticleRep.append(df_output_rootTweets_selected, ignore_index=True)

    print("num_rootTweets_selected:")
    print(num_rootTweets_selected)
    print("num_articles:")
    print(num_articles)

    print("absFilename_output_rootTweets_factcheckArticleRep:")
    print(absFilename_output_rootTweets_factcheckArticleRep)
    print("len(set(df_output_rootTweets_factcheckArticleRep[\"link to checker article\"].tolist())):")
    print(len(set(df_output_rootTweets_factcheckArticleRep["link to checker article"].tolist())))
    print("len(df_output_rootTweets_factcheckArticleRep):")
    print(len(df_output_rootTweets_factcheckArticleRep))

    df_output_rootTweets_factcheckArticleRep = df_output_rootTweets_factcheckArticleRep.drop_duplicates()
    df_output_rootTweets_factcheckArticleRep = df_output_rootTweets_factcheckArticleRep.sort_values(["retrievalDate_int", "link to checker article", "id_rootTweet"],ascending=True)
    df_output_rootTweets_factcheckArticleRep = df_output_rootTweets_factcheckArticleRep.reset_index(drop=True)

    print("len(df_output_rootTweets_factcheckArticleRep):")
    print(len(df_output_rootTweets_factcheckArticleRep))

    # df_output_rootTweets_factcheckArticleRep = df_output_rootTweets_factcheckArticleRep.sample(frac=1).reset_index(drop=True)

    # print(len(df_output_rootTweets_factcheckArticleRep))

    df_input_rootTweets_factcheckArticleRep_existing_annotated = pd.read_csv(absFilename_input_rootTweets_factcheckArticleRep_existing_annotated, dtype=str)
    df_input_rootTweets_factcheckArticleRep_existing_annotated = df_input_rootTweets_factcheckArticleRep_existing_annotated[["tweet link", "communicative intention 1", "communicative intention 2", "production approach 1"]]
    
    df_output_rootTweets_factcheckArticleRep = pd.merge(left=df_output_rootTweets_factcheckArticleRep, right=df_input_rootTweets_factcheckArticleRep_existing_annotated, how="left", on=["tweet link"])
    
    # print("after removing duplicate root tweets:")
    # print("df_output_rootTweets_factcheckArticleRep.columns:")
    # print(df_output_rootTweets_factcheckArticleRep.columns)

    print("before removing duplicate root tweets:")
    print("len(df_output_rootTweets_factcheckArticleRep):")
    print(len(df_output_rootTweets_factcheckArticleRep))
    print("len(df_output_rootTweets_factcheckArticleRep[\"id_rootTweet\"].tolist()):")
    print(len(df_output_rootTweets_factcheckArticleRep["id_rootTweet"].tolist()))
    print("len(list(set(df_output_rootTweets_factcheckArticleRep[\"id_rootTweet\"].tolist()))):")
    print(len(list(set(df_output_rootTweets_factcheckArticleRep["id_rootTweet"].tolist()))))
    print("len(df_output_rootTweets_factcheckArticleRep[\"tweet link\"].tolist()):")
    print(len(df_output_rootTweets_factcheckArticleRep["tweet link"].tolist()))
    print("len(list(set(df_output_rootTweets_factcheckArticleRep[\"tweet link\"].tolist()))):")
    print(len(list(set(df_output_rootTweets_factcheckArticleRep["tweet link"].tolist()))))

    df_output_rootTweets_factcheckArticleRep = df_output_rootTweets_factcheckArticleRep.sort_values(by=["id_rootTweet"], ascending=True)
    df_output_rootTweets_factcheckArticleRep = df_output_rootTweets_factcheckArticleRep.loc[df_output_rootTweets_factcheckArticleRep.groupby("id_rootTweet")["ptg_retweets_total_24hrs"].idxmax()]
    df_output_rootTweets_factcheckArticleRep = df_output_rootTweets_factcheckArticleRep.reset_index(drop=True)

    print("after removing duplicate root tweets:")
    print("len(df_output_rootTweets_factcheckArticleRep):")
    print(len(df_output_rootTweets_factcheckArticleRep))
    print("len(df_output_rootTweets_factcheckArticleRep[\"id_rootTweet\"].tolist()):")
    print(len(df_output_rootTweets_factcheckArticleRep["id_rootTweet"].tolist()))
    print("len(list(set(df_output_rootTweets_factcheckArticleRep[\"id_rootTweet\"].tolist()))):")
    print(len(list(set(df_output_rootTweets_factcheckArticleRep["id_rootTweet"].tolist()))))
    print("len(df_output_rootTweets_factcheckArticleRep[\"tweet link\"].tolist()):")
    print(len(df_output_rootTweets_factcheckArticleRep["tweet link"].tolist()))
    print("len(list(set(df_output_rootTweets_factcheckArticleRep[\"tweet link\"].tolist()))):")
    print(len(list(set(df_output_rootTweets_factcheckArticleRep["tweet link"].tolist()))))

    df_output_rootTweets_factcheckArticleRep = df_output_rootTweets_factcheckArticleRep.sort_values(["str_fullText_rootTweet", "link to checker article", "id_rootTweet"],ascending=True)
    df_output_rootTweets_factcheckArticleRep = df_output_rootTweets_factcheckArticleRep[["communicative intention 2", "tweet link", "checker assessment",  "str_fullText_rootTweet", "link to checker article", "production approach 1", "communicative intention 1", "retrievalDate_int", "retrieval date", "id_rootTweet", "str_createdAt_rootTweet", "num_retweets_total", "num_retweets", "ptg_retweets_total_24hrs", "ptg_retweets_total_48hrs", "ptg_retweets_total_72hrs"]]

    # communicative intention 2   tweet link  checker assessment  str_fullText_rootTweet  link to checker article production approach 1   communicative intention 1   retrievalDate_int   retrieval date  id_rootTweet    str_createdAt_rootTweet num_retweets_total  num_retweets    ptg_retweets_total_24hrs    ptg_retweets_total_48hrs    ptg_retweets_total_72hrs


    df_output_rootTweets_factcheckArticleRep.to_csv(absFilename_output_rootTweets_factcheckArticleRep, index=False)
    df_output_rootTweets_factcheckArticleRep.to_csv(absFilename_output_rootTweets_factcheckArticleRep_annotated, index=False)
    

         


if __name__ == "__main__":
    main(sys.argv[1:])
