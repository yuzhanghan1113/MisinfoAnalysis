import pandas as pd

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', -1)

ROUNDING_NUM_DIGITS = 3

# x = "bbb"
# y = x
# y = y + "ddd"
# print("x:")
# print(x)
# print("y:")
# print(y)

# input("end")

# l = ["aaa", "bbb", "ccc"]
# print("cccc" in l)

# input()


absFilename_input = "C:\\vmwareSharedFolder\\TwitterDataAnalysis\\results\\mixed model results\\regression_producerIntentions_depVar=sigInt_rdmEff=ROOTTWEETS_userScreenName_producer_rootTweetGp=misinformation_coefThr=0p1_minNumSigfX=0.csv"
absFilename_output = absFilename_input.replace("regression_", "selectedResult_")
absFilename_output_rounded = absFilename_input.replace("regression_", "selectedResultRounded_")

df_input = pd.read_csv(absFilename_input, dtype=str)

list_producerIntentions = ["REP", "COM", "DIR", "EXP", "DEC", "QOU"]

list_metrics = []
list_metrics += ["CASCADEEND_mean", "CASCADEEND_median"]
list_metrics += ["TIMESERIES_stl_linearity", "TIMESERIES_stl_curvature", "TIMESERIES_lumpiness"]


# list_depVars = []
# list_depVars += ["CASCADEEND_RETWEETS_cascadeSize_stadardized", "CASCADEEND_RETWEETS_retweet_latest_retweetedStatus_favoriteCount"]
# list_depVars += ["TIMESERIES_RETWEETS_cascadeSize_stadardized"]


# list_depVars = []
# list_depVars += ["TIMESERIES_RETWEETS_user_accountAge_day", "TIMESERIES_RETWEETS_user_statusesCount", "TIMESERIES_RETWEETS_user_favouritesCount", "TIMESERIES_RETWEETS_user_friendsCount", "TIMESERIES_RETWEETS_user_followersCount"]
# list_depVars += ["TIMESERIES_RETWEETS_user_description_sentiment_pos", "TIMESERIES_RETWEETS_user_description_sentiment_neg"]
# list_depVars += ["TIMESERIES_RETWEETS_user_description_emotion_fear", "TIMESERIES_RETWEETS_user_description_emotion_joy", "TIMESERIES_RETWEETS_user_description_emotion_trust"]

# list_depVars += ["TIMESERIES_REPLIES_user_accountAge_day", "TIMESERIES_REPLIES_user_statusesCount", "TIMESERIES_REPLIES_user_favouritesCount", "TIMESERIES_REPLIES_user_friendsCount", "TIMESERIES_REPLIES_user_followersCount"]
# list_depVars += ["TIMESERIES_REPLIES_fullText_sentiment_pos", "TIMESERIES_REPLIES_fullText_sentiment_neg"]
# list_depVars += ["TIMESERIES_REPLIES_fullText_emotion_anger", "TIMESERIES_REPLIES_fullText_emotion_disgust", "TIMESERIES_REPLIES_fullText_emotion_surprise"]
# list_depVars += ["TIMESERIES_REPLIES_user_description_emotion_anger", "TIMESERIES_REPLIES_user_description_emotion_surprise", "TIMESERIES_REPLIES_user_description_emotion_trust"]
# list_depVars += ["TIMESERIES_REPLIES_user_description_sentiment_pos", "TIMESERIES_REPLIES_user_description_sentiment_neg"]

list_depVars = []
list_depVars += ["CASCADEEND_RETWEETS_cascadeSize_stadardized", "CASCADEEND_RETWEETS_retweet_latest_retweetedStatus_favoriteCount"]
list_depVars += ["TIMESERIES_RETWEETS_cascadeSize_stadardized"]

list_depVars += ["TYPE_RETWEETS_retweet_METRIC_user_accountAge_day", "TYPE_RETWEETS_retweet_METRIC_user_statusesCount", "TYPE_RETWEETS_retweet_METRIC_user_favouritesCount", "TYPE_RETWEETS_retweet_METRIC_user_friendsCount", "TYPE_RETWEETS_retweet_METRIC_user_followersCount"]
list_depVars += ["TYPE_RETWEETS_retweet_METRIC_user_description_sentiment_pos", "TYPE_RETWEETS_retweet_METRIC_user_description_sentiment_neg"]
list_depVars += ["TYPE_RETWEETS_retweet_METRIC_user_description_emotion_fear", "TYPE_RETWEETS_retweet_METRIC_user_description_emotion_joy", "TYPE_RETWEETS_retweet_METRIC_user_description_emotion_trust"]

list_depVars += ["TYPE_REPLIES_reply_METRIC_user_accountAge_day", "TYPE_REPLIES_reply_METRIC_user_statusesCount", "TYPE_REPLIES_reply_METRIC_user_favouritesCount", "TYPE_REPLIES_reply_METRIC_user_friendsCount", "TYPE_REPLIES_reply_METRIC_user_followersCount"]
list_depVars += ["TYPE_REPLIES_reply_METRIC_fullText_sentiment_pos", "TYPE_REPLIES_reply_METRIC_fullText_sentiment_neg"]
list_depVars += ["TYPE_REPLIES_reply_METRIC_fullText_emotion_anger", "TYPE_REPLIES_reply_METRIC_fullText_emotion_disgust", "TYPE_REPLIES_reply_METRIC_fullText_emotion_surprise"]
list_depVars += ["TYPE_REPLIES_reply_METRIC_user_description_emotion_anger", "TYPE_REPLIES_reply_METRIC_user_description_emotion_surprise", "TYPE_REPLIES_reply_METRIC_user_description_emotion_trust"]
list_depVars += ["TYPE_REPLIES_reply_METRIC_user_description_sentiment_pos", "TYPE_REPLIES_reply_METRIC_user_description_sentiment_neg"]

df_output = pd.DataFrame()
df_output_rounded = pd.DataFrame()

for depVar in list_depVars:


	for metric in list_metrics:

		temp_depVar = depVar
		temp_metric = metric.replace("TIMESERIES_", "")

		depVarAndMetric_cascadeEnd = "EMPTY"
		list_depVarAndMetrics_timeSeries = ["EMPTY", "EMPTY"]

		if depVar.startswith("CASCADEEND_"):
			temp_depVar = depVar
			depVarAndMetric_cascadeEnd = depVar
			print("depVarAndMetric_cascadeEnd:")
			print(depVarAndMetric_cascadeEnd)
		elif metric == "CASCADEEND_mean":
			temp_depVar = temp_depVar.replace("TYPE_", "CASCADEEND_")
			depVarAndMetric_cascadeEnd = temp_depVar.replace("_METRIC_", "_mean_")
			print("depVarAndMetric_cascadeEnd:")
			print(depVarAndMetric_cascadeEnd)
		elif metric == "CASCADEEND_median":
			temp_depVar = temp_depVar.replace("TYPE_", "CASCADEEND_")
			depVarAndMetric_cascadeEnd = temp_depVar.replace("_METRIC_", "_median_")
			print("depVarAndMetric_cascadeEnd:")
			print(depVarAndMetric_cascadeEnd)
		elif metric.startswith("TIMESERIES_"):
			temp_depVar = temp_depVar.replace("TYPE_", "TIMESERIES_")
			temp_depVar = temp_depVar.replace("_METRIC_", "_")
			temp_depVar = temp_depVar.replace("_retweet_", "_")
			temp_depVar = temp_depVar.replace("_reply_", "_")
			temp_depVar += "DASH" + temp_metric
			list_depVarAndMetrics_timeSeries[0] = temp_depVar + "_allTimeStamps"
			list_depVarAndMetrics_timeSeries[1] = temp_depVar + "_filledTimeStamps"

			print("list_depVarAndMetrics_timeSeries:")
			print(list_depVarAndMetrics_timeSeries)

		if metric.startswith("CASCADEEND_"):

			if depVarAndMetric_cascadeEnd in df_input["feature_y"].tolist():

				for producerIntention in list_producerIntentions:
					coef = df_input.loc[df_input["feature_y"]==depVarAndMetric_cascadeEnd, "ROOTTWEETS_communicativeIntention_" + producerIntention + "_regression_coef"].reset_index(drop=True)[0]
					coef_rounded = round(float(coef), ROUNDING_NUM_DIGITS)
					p_coef = df_input.loc[df_input["feature_y"]==depVarAndMetric_cascadeEnd, "ROOTTWEETS_communicativeIntention_" + producerIntention + "_regression_p"].reset_index(drop=True)[0]
					p_coef = float(p_coef)

					if p_coef < 0.0001:
						coef = str(coef) + "***"
						coef_rounded = str(coef_rounded) + "***"
					elif p_coef < 0.01:
						coef = str(coef) + "**"
						coef_rounded = str(coef_rounded) + "**"
					elif p_coef < 0.05:
						coef = str(coef) + "*"
						coef_rounded = str(coef_rounded) + "*"

					df_output.loc[depVar, metric + "_" + producerIntention] = coef
					df_output_rounded.loc[depVar, metric + "_" + producerIntention] = coef_rounded

		elif metric.startswith("TIMESERIES_"):

			list_numberOfSigCoef = [-1, -1]

			if list_depVarAndMetrics_timeSeries[0] in df_input["feature_y"].tolist():
				list_numberOfSigCoef[0] = df_input.loc[df_input["feature_y"]==list_depVarAndMetrics_timeSeries[0], "numSigfX"].reset_index(drop=True)[0]
				list_numberOfSigCoef[0] = int(list_numberOfSigCoef[0])
			elif list_depVarAndMetrics_timeSeries[1] in df_input["feature_y"].tolist():
				list_numberOfSigCoef[1] = df_input.loc[df_input["feature_y"]==list_depVarAndMetrics_timeSeries[1], "numSigfX"].reset_index(drop=True)[0]
				list_numberOfSigCoef[1] = int(list_numberOfSigCoef[1])

			if list_numberOfSigCoef[0] == -1 and list_numberOfSigCoef[1] == -1:
				continue
			
			depVarAndMetric_timeSeries = "EMPTY"

			if list_numberOfSigCoef[0] >= list_numberOfSigCoef[1]:
				depVarAndMetric_timeSeries = list_depVarAndMetrics_timeSeries[0]
			elif list_numberOfSigCoef[0] < list_numberOfSigCoef[1]:
				depVarAndMetric_timeSeries = list_depVarAndMetrics_timeSeries[1]
		
			if depVarAndMetric_timeSeries in df_input["feature_y"].tolist():

				for producerIntention in list_producerIntentions:
					coef = df_input.loc[df_input["feature_y"]==depVarAndMetric_timeSeries, "ROOTTWEETS_communicativeIntention_" + producerIntention + "_regression_coef"].reset_index(drop=True)[0]
					coef_rounded = round(float(coef), ROUNDING_NUM_DIGITS)
					p_coef = df_input.loc[df_input["feature_y"]==depVarAndMetric_timeSeries, "ROOTTWEETS_communicativeIntention_" + producerIntention + "_regression_p"].reset_index(drop=True)[0]
					p_coef = float(p_coef)

					if p_coef < 0.0001:
						coef = str(coef) + "***"
						coef_rounded = str(coef_rounded) + "***"
					elif p_coef < 0.01:
						coef = str(coef) + "**"
						coef_rounded = str(coef_rounded) + "**"
					elif p_coef < 0.05:
						coef = str(coef) + "*"
						coef_rounded = str(coef_rounded) + "*"

					df_output.loc[depVar, metric + "_" + producerIntention] = coef
					df_output_rounded.loc[depVar, metric + "_" + producerIntention] = coef_rounded

		# df_output["dependentVariable"] = df_output.index
		# df_output.to_csv(absFilename_output, index=False)

print("absFilename_output:")
print(absFilename_output)

df_output["dependentVariable"] = df_output.index
df_output = df_output[["dependentVariable", "CASCADEEND_mean_REP", "CASCADEEND_mean_COM", "CASCADEEND_mean_DIR", "CASCADEEND_mean_EXP", "CASCADEEND_mean_DEC", "CASCADEEND_mean_QOU", "CASCADEEND_median_REP", "CASCADEEND_median_COM", "CASCADEEND_median_DIR", "CASCADEEND_median_EXP", "CASCADEEND_median_DEC", "CASCADEEND_median_QOU", "TIMESERIES_stl_linearity_REP", "TIMESERIES_stl_linearity_COM", "TIMESERIES_stl_linearity_DIR", "TIMESERIES_stl_linearity_EXP", "TIMESERIES_stl_linearity_DEC", "TIMESERIES_stl_linearity_QOU", "TIMESERIES_stl_curvature_REP", "TIMESERIES_stl_curvature_COM", "TIMESERIES_stl_curvature_DIR", "TIMESERIES_stl_curvature_EXP", "TIMESERIES_stl_curvature_DEC", "TIMESERIES_stl_curvature_QOU", "TIMESERIES_lumpiness_REP", "TIMESERIES_lumpiness_COM", "TIMESERIES_lumpiness_DIR", "TIMESERIES_lumpiness_EXP", "TIMESERIES_lumpiness_DEC", "TIMESERIES_lumpiness_QOU"]]
# df_output = df_output.sort_values(by=["feature_y"])
# df_output = df_output.reset_index(drop=True)
df_output.to_csv(absFilename_output, index=False)

print("absFilename_output_rounded:")
print(absFilename_output_rounded)
df_output_rounded["dependentVariable"] = df_output_rounded.index
df_output_rounded = df_output_rounded[["dependentVariable", "CASCADEEND_mean_REP", "CASCADEEND_mean_COM", "CASCADEEND_mean_DIR", "CASCADEEND_mean_EXP", "CASCADEEND_mean_DEC", "CASCADEEND_mean_QOU", "CASCADEEND_median_REP", "CASCADEEND_median_COM", "CASCADEEND_median_DIR", "CASCADEEND_median_EXP", "CASCADEEND_median_DEC", "CASCADEEND_median_QOU", "TIMESERIES_stl_linearity_REP", "TIMESERIES_stl_linearity_COM", "TIMESERIES_stl_linearity_DIR", "TIMESERIES_stl_linearity_EXP", "TIMESERIES_stl_linearity_DEC", "TIMESERIES_stl_linearity_QOU", "TIMESERIES_stl_curvature_REP", "TIMESERIES_stl_curvature_COM", "TIMESERIES_stl_curvature_DIR", "TIMESERIES_stl_curvature_EXP", "TIMESERIES_stl_curvature_DEC", "TIMESERIES_stl_curvature_QOU", "TIMESERIES_lumpiness_REP", "TIMESERIES_lumpiness_COM", "TIMESERIES_lumpiness_DIR", "TIMESERIES_lumpiness_EXP", "TIMESERIES_lumpiness_DEC", "TIMESERIES_lumpiness_QOU"]]
df_output_rounded.to_csv(absFilename_output_rounded, index=False)


# df_output.loc[:, df_output.columns != "dependentVariable"].astype(float).round(2).to_csv(absFilename_output_rounded, index=False)











