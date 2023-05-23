import pandas as pd

fn = "C:\\vmwareSharedFolder\\TwitterDataAnalysis\\results\\results_20210209\\cascadeResults_temporal_retweets.csv"
fn = "C:\\vmwareSharedFolder\\TwitterDataAnalysis\\results\\results_20210209\\cascadeResults_temporal_replies.csv"
fn = "C:\\vmwareSharedFolder\\TwitterDataAnalysis\\results\\results_20210209\\timeSeries_replies_indexStart=201_indexEnd=300.csv"
fn = "C:\\vmwareSharedFolder\\TwitterDataAnalysis\\results\\results_20210209\\timeSeries_retweets_indexStart=201_indexEnd=300.csv"

df = pd.read_csv(fn, dtype=str)

list_rootIdStrs = sorted(list(set(df["rootTweetIdStr"].tolist())))

print(len(list_rootIdStrs))

