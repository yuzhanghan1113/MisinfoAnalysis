import pandas as pd

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', -1)



absFilename_input = "C:\\vmwareSharedFolder\\TwitterDataAnalysis\\results\\mixed model results\\regression_producerIntentions_depVar=sigInt_rdmEff=ROOTTWEETS_userScreenName_producer_rootTweetGp=misinformation_coefThr=0p1_minNumSigfX=1.csv"
absFilename_output = absFilename_input.replace("regression_", "metrics_")

df_input = pd.read_csv(absFilename_input, dtype=str)

# print(df_input["feature_y"].str.contains("_curvature_"))
# print(True in df_input["feature_y"].str.contains("_curvature_"))
# input("stop...")

df_input_selected = df_input[df_input["feature_y"].str.contains("_curvature_")]

for e in df_input_selected["feature_y"].tolist():
	print(e)

list_featureY_prefix = ["TIMESERIES_RETWEETS_user_friendsCount", "TIMESERIES_RETWEETS_user_statusesCount", "TIMESERIES_RETWEETS_user_favouritesCount", "TIMESERIES_RETWEETS_user_followersCount", "TIMESERIES_RETWEETS_user_accountAge_day", "TIMESERIES_RETWEETS_user_description_textStats_readability", "TIMESERIES_RETWEETS_user_description_subjectivity", "TIMESERIES_RETWEETS_user_description_textStats_wordCount", "TIMESERIES_RETWEETS_user_description_emotion_anger", "TIMESERIES_RETWEETS_user_description_emotion_disgust", "TIMESERIES_RETWEETS_user_description_emotion_fear", "TIMESERIES_RETWEETS_user_description_emotion_joy", "TIMESERIES_RETWEETS_user_description_emotion_negative", "TIMESERIES_RETWEETS_user_description_emotion_positive", "TIMESERIES_RETWEETS_user_description_emotion_sadness", "TIMESERIES_RETWEETS_user_description_emotion_surprise", "TIMESERIES_RETWEETS_user_description_emotion_trust", "TIMESERIES_RETWEETS_user_description_sentiment_compound", "TIMESERIES_RETWEETS_user_description_sentiment_neg", "TIMESERIES_RETWEETS_user_description_sentiment_neu", "TIMESERIES_RETWEETS_user_description_sentiment_pos", "TIMESERIES_REPLIES_fullText_emotion_anger", "TIMESERIES_REPLIES_fullText_emotion_disgust", "TIMESERIES_REPLIES_fullText_emotion_fear", "TIMESERIES_REPLIES_fullText_emotion_joy", "TIMESERIES_REPLIES_fullText_emotion_negative", "TIMESERIES_REPLIES_fullText_emotion_positive", "TIMESERIES_REPLIES_fullText_emotion_sadness", "TIMESERIES_REPLIES_fullText_emotion_surprise", "TIMESERIES_REPLIES_fullText_sentiment_neu", "TIMESERIES_REPLIES_fullText_sentiment_pos", "TIMESERIES_REPLIES_fullText_sentiment_neg", "TIMESERIES_REPLIES_fullText_subjectivity", "TIMESERIES_REPLIES_fullText_textStats_readability", "TIMESERIES_REPLIES_fullText_textStats_wordCount", "TIMESERIES_REPLIES_user_accountAge_day", "TIMESERIES_REPLIES_user_description_emotion_anger", "TIMESERIES_REPLIES_user_description_emotion_fear", "TIMESERIES_REPLIES_user_description_emotion_negative", "TIMESERIES_REPLIES_user_description_emotion_sadness", "TIMESERIES_REPLIES_user_description_emotion_surprise", "TIMESERIES_REPLIES_user_description_emotion_trust", "TIMESERIES_REPLIES_user_description_sentiment_compound", "TIMESERIES_REPLIES_user_description_sentiment_neg", "TIMESERIES_REPLIES_user_description_sentiment_neu", "TIMESERIES_REPLIES_user_description_sentiment_pos", "TIMESERIES_REPLIES_user_description_subjectivity", "TIMESERIES_REPLIES_user_description_textStats_readability", "TIMESERIES_REPLIES_user_description_textStats_wordCount", "TIMESERIES_REPLIES_user_favouritesCount", "TIMESERIES_REPLIES_user_followersCount", "TIMESERIES_REPLIES_user_friendsCount", "TIMESERIES_REPLIES_user_listedCount", "TIMESERIES_REPLIES_user_statusesCount"]
list_metrics = ["stl_linearity_", "stl_curvature_", "stl_trend_", "stl_spike_", "nonlinearity_", "lumpiness_", "std1stDer_", "stability_"]


df_output = pd.DataFrame()

index_row = 0

for featureY_prefix in list_featureY_prefix:
	for metric in  list_metrics:

		df_output.loc[index_row, "dependentVariable"] = featureY_prefix

		str_depVarAndMetric = featureY_prefix + "DASH" + metric

		print(str_depVarAndMetric)
		
		if True in df_input["feature_y"].str.contains(str_depVarAndMetric).tolist():
			print(True)
			df_output.loc[index_row, metric] = "1"
		else:
			print(False)

	index_row += 1

print("absFilename_output:")
print(absFilename_output)

# df_output = df_output.sort_values(by=["feature_y"])
# df_output = df_output.reset_index(drop=True)
df_output.to_csv(absFilename_output, index=False)


