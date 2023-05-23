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
from scipy import stats
import statsmodels.api as sm


def main(argv):

	pd.set_option('display.max_rows', None)
	pd.set_option('display.max_columns', None)
	pd.set_option('display.width', None)
	pd.set_option('display.max_colwidth', -1)
    
	random.seed(1113)

	# absFilename_input = "D:\\vmwareSharedFolder\\TwitterDataAnalysis\\results\\preprocessedData.csv"
	absFilename_input = "C:\\vmwareSharedFolder\\TwitterDataAnalysis\\results\\preprocessedData.csv"

	THRESHOLD_COR = 0.1
	THRESHOLD_REG_COEF = 0.1
	THRESHOLD_REG_R2 = 0.2
	# MIN_NUM_SIGNIFICANT_X = 3
	MIN_NUM_SIGNIFICANT_X = 0

	list_features_toExclude = ["CASCADEEND_RETWEETS_rootTweetIdStr", "CASCADEEND_RETWEETS_rootTweet_createdAt", "CASCADEEND_RETWEETS_rootTweet_idStr", "CASCADEEND_RETWEETS_cascadeAge_min", "CASCADEEND_RETWEETS_rootTweet_fullText", "CASCADEEND_RETWEETS_rootTweet_entities_hashtags", "CASCADEEND_RETWEETS_rootTweet_entities_symbols", "CASCADEEND_RETWEETS_rootTweet_entities_userMentions_screenName", "CASCADEEND_RETWEETS_rootTweet_inReplyToStatusIdStr", "CASCADEEND_RETWEETS_rootTweet_inReplyToScreenName", "CASCADEEND_RETWEETS_rootTweet_user_screenName", "CASCADEEND_RETWEETS_rootTweet_user_location", "CASCADEEND_RETWEETS_rootTweet_user_description", "CASCADEEND_RETWEETS_rootTweet_user_url", "CASCADEEND_RETWEETS_rootTweet_user_createdAt", "CASCADEEND_RETWEETS_rootTweet_user_lang", "CASCADEEND_RETWEETS_rootTweet_user_profileBackgroundColor", "CASCADEEND_RETWEETS_rootTweet_user_profileTextColor", "CASCADEEND_RETWEETS_rootTweet_user_following", "CASCADEEND_RETWEETS_rootTweet_user_translatorType", "CASCADEEND_RETWEETS_rootTweet_geo", "CASCADEEND_RETWEETS_rootTweet_coordinates", "CASCADEEND_RETWEETS_rootTweet_place", "CASCADEEND_RETWEETS_rootTweet_contributors", "CASCADEEND_RETWEETS_rootTweet_lang", "CASCADEEND_PRODUCERTWEETS_producerScreenName", "CASCADEEND_PRODUCERTWEETS_rootTweetIdStr", "TIMESERIES_rootTweetIdStr"]
	list_features_toExclude += [f.replace("_RETWEETS", "_REPLIES") for f in list_features_toExclude]


	

	# significant y-features:
	# CASCADEEND_RETWEETS_retweet_mean_user_followersCount, 

	df_input = pd.read_csv(absFilename_input)

	print("Full input data:")

	print("len(df_input):")
	print(len(df_input))
	print("len(list(set(df_input[\"ROOTTWEETS_id_rootTweet\"].tolist()))):")
	print(len(list(set(df_input["ROOTTWEETS_id_rootTweet"].tolist()))))

	# list_features_int = ["ROOTTWEETS_retrievalDate_int", "ROOTTWEETS_veracityLabel_agg_misinformation","ROOTTWEETS_veracityLabel_agg_authentic","ROOTTWEETS_veracityLabel_agg_unknown","ROOTTWEETS_veracityLabel_agg_satire","ROOTTWEETS_veracityLabel_OUTDATED","ROOTTWEETS_veracityLabel_MISCAPTIONATED","ROOTTWEETS_veracityLabel_UNKNOWN","ROOTTWEETS_veracityLabel_MIXED","ROOTTWEETS_veracityLabel_MOSTLY FALSE","ROOTTWEETS_veracityLabel_MOSTLY TRUE","ROOTTWEETS_veracityLabel_FALSE STORIES","ROOTTWEETS_veracityLabel_NOT TRUE","ROOTTWEETS_veracityLabel_TRUE","ROOTTWEETS_veracityLabel_DECONTEXTUALIZED","ROOTTWEETS_veracityLabel_FALSE","ROOTTWEETS_veracityLabel_MISATTRIBUTED","ROOTTWEETS_veracityLabel_FALSE CLAIM","ROOTTWEETS_veracityLabel_HALF TRUE","ROOTTWEETS_veracityLabel_CORRECT ATTRIBUTION","ROOTTWEETS_veracityLabel_UNPROVEN","ROOTTWEETS_veracityLabel_MIXTURE","ROOTTWEETS_veracityLabel_LABELED SATIRE","ROOTTWEETS_veracityLabel_PANTS ON FIRE"]
	# list_features_str = ["ROOTTWEETS_link to checker article","ROOTTWEETS_retrieval date","ROOTTWEETS_checker assessment","ROOTTWEETS_id_rootTweet","ROOTTWEETS_str_createdAt_rootTweet","ROOTTWEETS_tweet link","ROOTTWEETS_tweet link","ROOTTWEETS_communicative intention 1","ROOTTWEETS_note 1","ROOTTWEETS_communicative intention 2","ROOTTWEETS_checker assessment_split","CASCADEEND_ROOTTWEETS_id_rootTweet","CASCADEEND_RETWEETS_rootTweetIdStr","CASCADEEND_REPLIES_rootTweetIdStr","CASCADEEND_PRODUCERTWEETS_rootTweetIdStr","CASCADEEND_PRODUCERTWEETS_producerScreenName","TIMESERIES_rootTweetIdStr"]

	# list_features_float = list(set(df_input.columns) - set(list_features_int) - set(list_features_str))

	# df_input[list_features_float] = df_input[list_features_float].astype(float)
	# df_input[list_features_int] = df_input[list_features_int].astype(float).astype(int)

	# list_featureX = ["ROOTTWEETS_communicativeIntention_QOU", "ROOTTWEETS_communicativeIntention_EXP", "ROOTTWEETS_communicativeIntention_COM", "ROOTTWEETS_communicativeIntention_DIR", "ROOTTWEETS_communicativeIntention_DEC", "ROOTTWEETS_communicativeIntention_REP", "ROOTTWEETS_communicativeIntention_REP_and_QOU", "ROOTTWEETS_communicativeIntention_REP_plus", "ROOTTWEETS_communicativeIntention_count"]
	
	# list_featureX = ["ROOTTWEETS_communicativeIntention_QOU", "ROOTTWEETS_communicativeIntention_EXP", "ROOTTWEETS_communicativeIntention_COM", "ROOTTWEETS_communicativeIntention_DIR", "ROOTTWEETS_communicativeIntention_DEC", "ROOTTWEETS_communicativeIntention_REP", "ROOTTWEETS_communicativeIntention_count"]
	# str_depVar = "sigIntAndIntCnt"

	# list_featureX = ["ROOTTWEETS_communicativeIntention_QOU", "ROOTTWEETS_communicativeIntention_EXP", "ROOTTWEETS_communicativeIntention_COM", "ROOTTWEETS_communicativeIntention_DIR", "ROOTTWEETS_communicativeIntention_DEC", "ROOTTWEETS_communicativeIntention_REP"]
	# str_depVar = "sigInt"

	# list_featureX = ["ROOTTWEETS_communicativeIntention_EXP", "ROOTTWEETS_communicativeIntention_COM", "ROOTTWEETS_communicativeIntention_DIR", "ROOTTWEETS_communicativeIntention_DEC", "ROOTTWEETS_communicativeIntention_REP"]
	# str_depVar = "sigIntNoQUO"

	# list_featureX = ["ROOTTWEETS_communicativeIntention_count", "ROOTTWEETS_communicativeIntention_QOU", "ROOTTWEETS_communicativeIntention_EXP", "ROOTTWEETS_communicativeIntention_COM", "ROOTTWEETS_communicativeIntention_DIR", "ROOTTWEETS_communicativeIntention_DEC", "ROOTTWEETS_communicativeIntention_REP"]
	# str_depVar = "sigIntAndIntCnt"

	list_featureX = ["ROOTTWEETS_communicativeIntention_count", "ROOTTWEETS_communicativeIntention_QOU", "ROOTTWEETS_communicativeIntention_EXP", "ROOTTWEETS_communicativeIntention_COM", "ROOTTWEETS_communicativeIntention_DIR", "ROOTTWEETS_communicativeIntention_DEC", "ROOTTWEETS_communicativeIntention_REP"]
	str_depVar = "intnCntSigIntn"

	# list_featureX = ["ROOTTWEETS_communicativeIntention_count", "ROOTTWEETS_communicativeIntention_QOU", "ROOTTWEETS_communicativeIntention_EXP", "ROOTTWEETS_communicativeIntention_COM", "ROOTTWEETS_communicativeIntention_DIR", "ROOTTWEETS_communicativeIntention_DEC", "ROOTTWEETS_communicativeIntention_REP"]
	# str_depVar = "intnCntSigIntnNo"

	# list_featureX = ["ROOTTWEETS_communicativeIntention_count", "ROOTTWEETS_communicativeIntention_EXP", "ROOTTWEETS_communicativeIntention_COM", "ROOTTWEETS_communicativeIntention_DIR", "ROOTTWEETS_communicativeIntention_DEC", "ROOTTWEETS_communicativeIntention_REP"]
	# str_depVar = "intnCntSigIntnNoQUO"

	# list_featureX = ["ROOTTWEETS_communicativeIntention_count", "ROOTTWEETS_communicativeIntention_QOU", "ROOTTWEETS_communicativeIntention_COM", "ROOTTWEETS_communicativeIntention_DIR", "ROOTTWEETS_communicativeIntention_DEC", "ROOTTWEETS_communicativeIntention_REP"]
	# str_depVar = "intnCntSigIntnNoEXP"

	# list_featureX = ["ROOTTWEETS_communicativeIntention_count", "ROOTTWEETS_communicativeIntention_QOU", "ROOTTWEETS_communicativeIntention_EXP", "ROOTTWEETS_communicativeIntention_DIR", "ROOTTWEETS_communicativeIntention_DEC", "ROOTTWEETS_communicativeIntention_REP"]
	# str_depVar = "intnCntSigIntnNoCOM"

	# list_featureX = ["ROOTTWEETS_communicativeIntention_count", "ROOTTWEETS_communicativeIntention_QOU", "ROOTTWEETS_communicativeIntention_EXP", "ROOTTWEETS_communicativeIntention_COM", "ROOTTWEETS_communicativeIntention_DEC", "ROOTTWEETS_communicativeIntention_REP"]
	# str_depVar = "intnCntSigIntnNoDIR"

	# list_featureX = ["ROOTTWEETS_communicativeIntention_count", "ROOTTWEETS_communicativeIntention_QOU", "ROOTTWEETS_communicativeIntention_EXP", "ROOTTWEETS_communicativeIntention_COM", "ROOTTWEETS_communicativeIntention_DIR", "ROOTTWEETS_communicativeIntention_REP"]
	# str_depVar = "intnCntSigIntnNoDEC"

	# list_featureX = ["ROOTTWEETS_communicativeIntention_count", "ROOTTWEETS_communicativeIntention_QOU", "ROOTTWEETS_communicativeIntention_EXP", "ROOTTWEETS_communicativeIntention_COM", "ROOTTWEETS_communicativeIntention_DIR", "ROOTTWEETS_communicativeIntention_DEC"]
	# str_depVar = "intnCntSigIntnNoREP"

	rootTweetGroup_feature = "ROOTTWEETS_veracityLabel_agg_misinformation"
	rootTweetGroup_value = 1

	print("rootTweetGroup_feature:")
	print(rootTweetGroup_feature)
	print("rootTweetGroup_value:")
	print(rootTweetGroup_value)

	str_rootTweetGp = ""

	if rootTweetGroup_feature=="ROOTTWEETS_veracityLabel_agg_misinformation" and rootTweetGroup_value==1:
		str_rootTweetGp = "misinformation"

	# absFilename_output = "D:\\vmwareSharedFolder\\TwitterDataAnalysis\\results\\featureSelection\\correlation_depVar=" + feature_x + "_rootTweetGp=" + str_rootTweetGp + "_corThr=" + str(THRESHOLD_COR).replace(".", "p") + ".csv"
	absFilename_output = "C:\\vmwareSharedFolder\\TwitterDataAnalysis\\results\\featureSelection\\regression_producerIntentions_depVar=" + str_depVar + "_rootTweetGp=" + str_rootTweetGp + "_coefThr=" + str(THRESHOLD_REG_COEF).replace(".", "p") + "_minNumSigfX=" + str(MIN_NUM_SIGNIFICANT_X) + ".csv"

	print("absFilename_output:")
	print(absFilename_output)



	df_output = pd.DataFrame()
	index_row_output = 0

	list_features_y = [f for f in df_input.columns if f.startswith("CASCADEEND") or f.startswith("TIMESERIES")]
	list_features_y = [f for f in list_features_y if not f.startswith("CASCADEEND_ROOTTWEETS")]

	print("len(list_features_y):")
	print(len(list_features_y))

	# feature_x = "ROOTTWEETS_communicativeIntention_QOU"
	# feature_x = "ROOTTWEETS_communicativeIntention_REP_and_QOU"
	# feature_y = "CASCADEEND_RETWEETS_cascadeSize_stadardized"
	# feature_y = "CASCADEEND_RETWEETS_cascadeSize"
	# feature_y = "CASCADEEND_RETWEETS_retweet_mean_user_followersCount"
	# feature_y = "CASCADEEND_RETWEETS_retweet_median_user_followersCount"
	# feature_y = "CASCADEEND_RETWEETS_retweet_mean_user_friendsCount"


	# print("only the misinformation:")

	df_input_toAnalyze_allFeatures = df_input[df_input[rootTweetGroup_feature]==rootTweetGroup_value].copy()

	print("\nData to analyze:")
	print("len(df_input_toAnalyze_allFeatures):")
	print(len(df_input_toAnalyze_allFeatures))
	print("len(list(set(df_input_toAnalyze_allFeatures[\"ROOTTWEETS_id_rootTweet\"].tolist()))):")
	print(len(list(set(df_input_toAnalyze_allFeatures["ROOTTWEETS_id_rootTweet"].tolist()))))

	for feature_y in list_features_y:

		if "_PRODUCERTWEETS_" in feature_y:
			continue
		if feature_y in list_features_toExclude:
			continue


		# print("\n")
		# print("index_row_output:")
		# print(index_row_output)
		# print("feature_x:")
		# print(feature_x)
		# print("feature_y:")
		# print(feature_y)

		df_input_toAnalyze = df_input_toAnalyze_allFeatures[list_featureX + [feature_y]].copy()
		df_input_toAnalyze_org = df_input_toAnalyze.copy()
		df_input_toAnalyze = df_input_toAnalyze.dropna()


		# print("feature_x:")
		# print(feature_x)
		# print("feature_y:")
		# print(feature_y)

		df_x = df_input_toAnalyze[list_featureX]
		list_y = df_input_toAnalyze[feature_y].tolist()

		# print("Counter(list_x):")
		# print(Counter(list_x))

		# print("\n")
		# print("len(list_x):")
		# print(len(list_x))

		# print("len(list_y):")
		# print(len(list_y))
		try:

			model = sm.OLS(list_y,df_x)
			model_fitted = model.fit()

			regression_R2 = model_fitted.summary2().tables[0][1][6]
			# regression_coef = model_fitted.summary2().tables[1]["Coef."][0]
			# regression_p = model_fitted.summary2().tables[1]["P>|t|"][0]
			# print(model_fitted.summary2())
			# print(regression_coef)
			# print(regression_p)

			# return

			# result = stats.pointbiserialr(list_x, list_y)
			# correlation_stat = result.correlation
			# correlation_p = result.pvalue

		except Exception as e:
			track = traceback.format_exc()
			print(track + "\n")
			continue	

		# print(result)

		# print("Point Biserial Correlation:")
		# print(result.correlation)
		# print("p-value:")
		# print(result.pvalue)

		# if np.abs((mean_positive - mean_negative)/mean_negative)>=0.1 or np.abs((median_positive - median_negative)/median_negative)>=0.1 or result.correlation>=0.1 or result.pvalue<=0.05:
		# if abs(result.correlation)>=THRESHOLD_COR and result.pvalue<=0.05:
		# if abs(regression_coef)>=THRESHOLD_REG_COEF and regression_p<=0.05:

		# print("regression_R2:")
		# print(regression_R2)

		# if "inf" not in str(regression_R2) and float(regression_R2)>=THRESHOLD_REG_R2:

		# 	print("\n")
		# 	print("-------------- feature selected --------------")
		# 	print("index_row_output:")
		# 	print(index_row_output)
		# 	print("list_featureX:")
		# 	print(list_featureX)
		# 	print("feature_y:")
		# 	print(feature_y)

		# 	regression_R2 = float(regression_R2)


			

		# 	# df_temp_positive = df_input_toAnalyze_org[df_input_toAnalyze_org[feature_x]==1]
		# 	# df_temp_negative = df_input_toAnalyze_org[df_input_toAnalyze_org[feature_x]==0]
		# 	# print("df_temp_positive:")
		# 	# print(df_temp_positive)
		# 	# print("len(df_temp_positive):")
		# 	# print(len(df_temp_positive))
		# 	# print("df_temp_negative:")
		# 	# print(df_temp_negative)
		# 	# print("len(df_temp_negative):")
		# 	# print(len(df_temp_negative))

		# 	df_output.loc[index_row_output, "rootTweetGroup_feature"] = rootTweetGroup_feature
		# 	df_output.loc[index_row_output, "rootTweetGroup_value"] = str(rootTweetGroup_value)
		# 	# df_output.loc[index_row_output, "correlationThreshold"] = str(THRESHOLD_COR)
		# 	df_output.loc[index_row_output, "regCoefThreshold"] = str(THRESHOLD_REG_COEF)
		# 	df_output.loc[index_row_output, "feature_x"] = str(list_featureX)
			
		# 	list_strings = feature_y.split("_")
		# 	list_strings = [s for s in list_strings if len(s)>0]

		# 	df_output.loc[index_row_output, "feature_y_type"] = list_strings[0] + "_" + list_strings[1]
		# 	df_output.loc[index_row_output, "feature_y"] = feature_y

		# 	df_output.loc[index_row_output, "regression_R2"] = regression_R2

		# 	for feature_x in list_featureX:

		# 		# print(model_fitted.summary2().tables[1])
		# 		regression_coef = model_fitted.summary2().tables[1]["Coef."][feature_x]
		# 		df_output.loc[index_row_output, feature_x + "_regression_coef"] = regression_coef

		# 		regression_p = model_fitted.summary2().tables[1]["P>|t|"][feature_x]
		# 		df_output.loc[index_row_output, feature_x + "_regression_p"] = regression_p

		# 	# regression_coef = model_fitted.summary2().tables[1]["Coef."]
		# 	# regression_p = model_fitted.summary2().tables[1]["P>|t|"]


		# 	# return


		# 	# df_output.loc[index_row_output, "regression_coef_abs"] = np.abs(regression_coef)
		# 	# df_output.loc[index_row_output, "regression_coef"] = regression_coef
		# 	# df_output.loc[index_row_output, "regression_p"] = regression_p

		# 	# try:

		# 	# 	result = stats.pointbiserialr(list_x, list_y)
		# 	# 	correlation_stat = result.correlation
		# 	# 	correlation_p = result.pvalue

		# 	# 	df_output.loc[index_row_output, "correlation_abs"] = np.abs(correlation_stat)
		# 	# 	df_output.loc[index_row_output, "correlation_stat"] = correlation_stat
		# 	# 	df_output.loc[index_row_output, "correlation_p"] = correlation_p

		# 	# except Exception as e:
		# 	# 	track = traceback.format_exc()
		# 	# 	print(track + "\n")
		# 	# 	continue	

			

		# 	# list_xValues = sorted(list(set(df_input_toAnalyze[feature_x].tolist())))

		# 	# for xValue in list_xValues:

		# 	# 	# print("\n")
		# 	# 	list_yValues = df_input_toAnalyze.loc[df_input_toAnalyze[feature_x]==xValue, feature_y].tolist()
		# 	# 	df_output.loc[index_row_output, "countY_x=" + str(xValue)] = len(list_yValues)

		# 	# for xValue in list_xValues:

		# 	# 	# print("\n")
		# 	# 	list_yValues = df_input_toAnalyze.loc[df_input_toAnalyze[feature_x]==xValue, feature_y].tolist()
		# 	# 	df_output.loc[index_row_output, "meanY_x=" + str(xValue)] = np.mean(list_yValues)

		# 	# for xValue in list_xValues:

		# 	# 	# print("\n")
		# 	# 	list_yValues = df_input_toAnalyze.loc[df_input_toAnalyze[feature_x]==xValue, feature_y].tolist()
		# 	# 	df_output.loc[index_row_output, "medianY_x=" + str(xValue)] = np.median(list_yValues)
			

		# 	index_row_output += 1

		# 	# print("absFilename_output:")
		#  	# print(absFilename_output)
			
		# 	df_output.to_csv(absFilename_output, index=False)

		# 	# input('user input:\n')

		dict_featureXToCoef = {}
		dict_featureXToCoefP = {}

		num_significantX = 0

		for feature_x in list_featureX:

			# print(model_fitted.summary2().tables[1])
			regression_coef = model_fitted.summary2().tables[1]["Coef."][feature_x]
			regression_coef = float(regression_coef)
			dict_featureXToCoef[feature_x] = regression_coef

			regression_p = model_fitted.summary2().tables[1]["P>|t|"][feature_x]
			regression_p = float(regression_p)
			dict_featureXToCoefP[feature_x] = regression_p

			if np.abs(regression_coef) >= THRESHOLD_REG_COEF and regression_p <= 0.05:
				num_significantX += 1


		if num_significantX >= MIN_NUM_SIGNIFICANT_X:

			print("\n")
			print("-------------- feature selected --------------")
			print("index_row_output:")
			print(index_row_output)
			print("list_featureX:")
			print(list_featureX)
			print("feature_y:")
			print(feature_y)

			regression_R2 = float(regression_R2)


			

			# df_temp_positive = df_input_toAnalyze_org[df_input_toAnalyze_org[feature_x]==1]
			# df_temp_negative = df_input_toAnalyze_org[df_input_toAnalyze_org[feature_x]==0]
			# print("df_temp_positive:")
			# print(df_temp_positive)
			# print("len(df_temp_positive):")
			# print(len(df_temp_positive))
			# print("df_temp_negative:")
			# print(df_temp_negative)
			# print("len(df_temp_negative):")
			# print(len(df_temp_negative))

			df_output.loc[index_row_output, "rootTweetGroup_feature"] = rootTweetGroup_feature
			df_output.loc[index_row_output, "rootTweetGroup_value"] = str(rootTweetGroup_value)
			# df_output.loc[index_row_output, "correlationThreshold"] = str(THRESHOLD_COR)
			df_output.loc[index_row_output, "regCoefThreshold"] = str(THRESHOLD_REG_COEF)
			df_output.loc[index_row_output, "minNumSigfX"] = str(MIN_NUM_SIGNIFICANT_X)
			df_output.loc[index_row_output, "feature_x"] = str(list_featureX)
			
			list_strings = feature_y.split("_")
			list_strings = [s for s in list_strings if len(s)>0]

			df_output.loc[index_row_output, "feature_y_type"] = list_strings[0] + "_" + list_strings[1]
			df_output.loc[index_row_output, "feature_y"] = feature_y

			df_output.loc[index_row_output, "regression_R2"] = regression_R2
			df_output.loc[index_row_output, "numSigfX"] = num_significantX
			

			# regression_coef = model_fitted.summary2().tables[1]["Coef."]
			# regression_p = model_fitted.summary2().tables[1]["P>|t|"]


			# return


			# df_output.loc[index_row_output, "regression_coef_abs"] = np.abs(regression_coef)
			# df_output.loc[index_row_output, "regression_coef"] = regression_coef
			# df_output.loc[index_row_output, "regression_p"] = regression_p

			# try:

			# 	result = stats.pointbiserialr(list_x, list_y)
			# 	correlation_stat = result.correlation
			# 	correlation_p = result.pvalue

			# 	df_output.loc[index_row_output, "correlation_abs"] = np.abs(correlation_stat)
			# 	df_output.loc[index_row_output, "correlation_stat"] = correlation_stat
			# 	df_output.loc[index_row_output, "correlation_p"] = correlation_p

			# except Exception as e:
			# 	track = traceback.format_exc()
			# 	print(track + "\n")
			# 	continue	

			

			# list_xValues = sorted(list(set(df_input_toAnalyze[feature_x].tolist())))

			# for xValue in list_xValues:

			# 	# print("\n")
			# 	list_yValues = df_input_toAnalyze.loc[df_input_toAnalyze[feature_x]==xValue, feature_y].tolist()
			# 	df_output.loc[index_row_output, "countY_x=" + str(xValue)] = len(list_yValues)

			# for xValue in list_xValues:

			# 	# print("\n")
			# 	list_yValues = df_input_toAnalyze.loc[df_input_toAnalyze[feature_x]==xValue, feature_y].tolist()
			# 	df_output.loc[index_row_output, "meanY_x=" + str(xValue)] = np.mean(list_yValues)

			# for xValue in list_xValues:

			# 	# print("\n")
			# 	list_yValues = df_input_toAnalyze.loc[df_input_toAnalyze[feature_x]==xValue, feature_y].tolist()
			# 	df_output.loc[index_row_output, "medianY_x=" + str(xValue)] = np.median(list_yValues)

			for feature_x in list_featureX:

				# print(model_fitted.summary2().tables[1])
				regression_coef = dict_featureXToCoef[feature_x]
				df_output.loc[index_row_output, feature_x + "_regression_coef"] = str(regression_coef)
				df_output.loc[index_row_output, feature_x + "_regression_coef_abs"] = np.abs(regression_coef)

				regression_p = dict_featureXToCoefP[feature_x]
				df_output.loc[index_row_output, feature_x + "_regression_p"] = str(regression_p)
			

			index_row_output += 1

			# print("absFilename_output:")
		 	# print(absFilename_output)
			
			df_output = df_output.sort_values(by=["feature_y_type", "numSigfX", "regression_R2"], ascending=[True, False, False])
			df_output = df_output.reset_index(drop=True)
			df_output.to_csv(absFilename_output, index=False)

			# input('user input:\n')

	print("absFilename_output:")
	print(absFilename_output)

	# df_output = df_output.sort_values(by=["feature_y_type", "correlation_abs"], ascending=[True, False])
	# df_output = df_output.sort_values(by=["feature_y_type", "regression_coef_abs"], ascending=[True, False])
	# df_output = df_output.sort_values(by=["feature_y_type", "numSigfX", "regression_R2"], ascending=[True, False, False])
	df_output = df_output.sort_values(by=["feature_y"])
	df_output = df_output.reset_index(drop=True)
	df_output.to_csv(absFilename_output, index=False)

	print("X feature processed.")
	print("program exits.")



if __name__ == "__main__":
    main(sys.argv[1:])







