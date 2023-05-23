rm(list=ls())

library(data.table)
library(stringr)
library(tidyr)
library(MuMIn)
library(psych)

options(max.print = 999999)
# options(error = traceback)
options(error = recover)


absFilename_input = "C:/vmwareSharedFolder/TwitterDataAnalysis/results/preprocessedData.csv"
absFilename_output = "C:/vmwareSharedFolder/TwitterDataAnalysis/results/preprocessedData_factorMeasures.csv"

THRESHOLD_COR = 0.1
THRESHOLD_REG_COEF = 0.1
THRESHOLD_REG_R2 = 0.2
# MIN_NUM_SIGNIFICANT_X = 3
MIN_NUM_SIGNIFICANT_X = 0
# MIN_NUM_SIGNIFICANT_X = 1



list_features_toExclude = c("CASCADEEND_RETWEETS_rootTweetIdStr", "CASCADEEND_RETWEETS_rootTweet_createdAt", "CASCADEEND_RETWEETS_rootTweet_idStr", "CASCADEEND_RETWEETS_cascadeAge_min", "CASCADEEND_RETWEETS_rootTweet_fullText", "CASCADEEND_RETWEETS_rootTweet_entities_hashtags", "CASCADEEND_RETWEETS_rootTweet_entities_symbols", "CASCADEEND_RETWEETS_rootTweet_entities_userMentions_screenName", "CASCADEEND_RETWEETS_rootTweet_inReplyToStatusIdStr", "CASCADEEND_RETWEETS_rootTweet_inReplyToScreenName", "CASCADEEND_RETWEETS_rootTweet_user_screenName", "CASCADEEND_RETWEETS_rootTweet_user_location", "CASCADEEND_RETWEETS_rootTweet_user_description", "CASCADEEND_RETWEETS_rootTweet_user_url", "CASCADEEND_RETWEETS_rootTweet_user_createdAt", "CASCADEEND_RETWEETS_rootTweet_user_lang", "CASCADEEND_RETWEETS_rootTweet_user_profileBackgroundColor", "CASCADEEND_RETWEETS_rootTweet_user_profileTextColor", "CASCADEEND_RETWEETS_rootTweet_user_following", "CASCADEEND_RETWEETS_rootTweet_user_translatorType", "CASCADEEND_RETWEETS_rootTweet_geo", "CASCADEEND_RETWEETS_rootTweet_coordinates", "CASCADEEND_RETWEETS_rootTweet_place", "CASCADEEND_RETWEETS_rootTweet_contributors", "CASCADEEND_RETWEETS_rootTweet_lang", "CASCADEEND_PRODUCERTWEETS_producerScreenName", "CASCADEEND_PRODUCERTWEETS_rootTweetIdStr", "TIMESERIES_rootTweetIdStr")
list_features_toExclude = union(gsub("_RETWEETS", "_REPLIES", list_features_toExclude), list_features_toExclude)

df_input =
  data.frame(fread(
    absFilename_input,
    # nrows = 50000,
    # sep = ",",
    header = TRUE
    # select = vector_attributesToLoad_rereports,
    # colClasses = c("character")
  ))

df_input[, "index"] <- rownames(df_input)

print("Full input data:")

print("length(df_input):")
print(length(df_input))
print("length(unique(as.vector(df_input[, \"ROOTTWEETS_id_rootTweet\"]))):")
print(length(unique(as.vector(df_input[, "ROOTTWEETS_id_rootTweet"]))))

df_input[, "ROOTTWEETS_userScreenName_producer"] = str_split(df_input[, "ROOTTWEETS_tweet.link"], "/", simplify = TRUE)[,4]
list_userScreenNames_producer = sort(unique(as.vector(df_input[, "ROOTTWEETS_userScreenName_producer"])))

print(list_userScreenNames_producer)
print("length(list_userScreenNames_producer):")
print(length(list_userScreenNames_producer))

df_input[, "ROOTTWEETS_idStr_rootTweet"] = str_split(df_input[, "ROOTTWEETS_tweet.link"], "/", simplify = TRUE)[,6]

print(df_input[, "ROOTTWEETS_idStr_rootTweet"])
print("length(df_input[, \"ROOTTWEETS_idStr_rootTweet\"]):")
print(length(df_input[, "ROOTTWEETS_idStr_rootTweet"]))

list_featureX = c("ROOTTWEETS_communicativeIntention_QOU", "ROOTTWEETS_communicativeIntention_EXP", "ROOTTWEETS_communicativeIntention_COM", "ROOTTWEETS_communicativeIntention_DIR", "ROOTTWEETS_communicativeIntention_DEC", "ROOTTWEETS_communicativeIntention_REP")
str_depVar = "sigInt"

# list_featureX = c("ROOTTWEETS_communicativeIntention_EXP", "ROOTTWEETS_communicativeIntention_COM", "ROOTTWEETS_communicativeIntention_DIR", "ROOTTWEETS_communicativeIntention_DEC", "ROOTTWEETS_communicativeIntention_REP")
# str_depVar = "sigIntNoQOU"
# 
# list_featureX = c("ROOTTWEETS_communicativeIntention_QOU", "ROOTTWEETS_communicativeIntention_COM", "ROOTTWEETS_communicativeIntention_DIR", "ROOTTWEETS_communicativeIntention_DEC", "ROOTTWEETS_communicativeIntention_REP")
# str_depVar = "sigIntNoEXP"

rootTweetGroup_feature = "ROOTTWEETS_veracityLabel_agg_misinformation"
rootTweetGroup_value = 1

print("rootTweetGroup_feature:")
print(rootTweetGroup_feature)
print("rootTweetGroup_value:")
print(rootTweetGroup_value)

str_rootTweetGp = ""

if (rootTweetGroup_feature=="ROOTTWEETS_veracityLabel_agg_misinformation" && rootTweetGroup_value==1) {
  str_rootTweetGp = "misinformation"
}

# #Spreader Persona - activeness - cascade end
# list_items = c(
#   # "CASCADEEND_RETWEETS_retweet_mean_user_accountAge_day",
#   "CASCADEEND_RETWEETS_retweet_median_user_accountAge_day",
#   # "CASCADEEND_RETWEETS_retweet_mean_user_followersCount",
#   "CASCADEEND_RETWEETS_retweet_median_user_followersCount",
#   # "CASCADEEND_RETWEETS_retweet_mean_user_friendsCount",
#   "CASCADEEND_RETWEETS_retweet_median_user_friendsCount",
#   # "CASCADEEND_RETWEETS_retweet_mean_user_favouritesCount",
#   "CASCADEEND_RETWEETS_retweet_median_user_favouritesCount",
#   # "CASCADEEND_RETWEETS_retweet_mean_user_statusesCount",
#   "CASCADEEND_RETWEETS_retweet_median_user_statusesCount"
# )
# NUM_FACTORS = 1
# measurePrefix = "measure_SpreaderPersona_activeness_cascadeEnd_"

# #Spreader Persona - sentiment/emotion - cascade end
# list_items = c(
#   # "CASCADEEND_RETWEETS_retweet_mean_user_description_sentiment_neg",
#   "CASCADEEND_RETWEETS_retweet_median_user_description_sentiment_neg",
#   # "CASCADEEND_RETWEETS_retweet_mean_user_description_sentiment_neu",
#   "CASCADEEND_RETWEETS_retweet_median_user_description_sentiment_neu",
#   # "CASCADEEND_RETWEETS_retweet_mean_user_description_sentiment_pos",
#   "CASCADEEND_RETWEETS_retweet_median_user_description_sentiment_pos",
#   # "CASCADEEND_RETWEETS_retweet_mean_user_description_sentiment_compound",
#   # "CASCADEEND_RETWEETS_retweet_median_user_description_sentiment_compound",
#   # "CASCADEEND_RETWEETS_retweet_mean_user_description_subjectivity",
#   "CASCADEEND_RETWEETS_retweet_median_user_description_subjectivity",
#   # "CASCADEEND_RETWEETS_retweet_mean_user_description_emotion_fear",
#   "CASCADEEND_RETWEETS_retweet_median_user_description_emotion_fear",
#   # "CASCADEEND_RETWEETS_retweet_mean_user_description_emotion_anger",
#   "CASCADEEND_RETWEETS_retweet_median_user_description_emotion_anger",
#   # "CASCADEEND_RETWEETS_retweet_mean_user_description_emotion_anticip",
#   # "CASCADEEND_RETWEETS_retweet_median_user_description_emotion_anticip",
#   # "CASCADEEND_RETWEETS_retweet_mean_user_description_emotion_trust",
#   "CASCADEEND_RETWEETS_retweet_median_user_description_emotion_trust",
#   # "CASCADEEND_RETWEETS_retweet_mean_user_description_emotion_surprise",
#   "CASCADEEND_RETWEETS_retweet_median_user_description_emotion_surprise",
#   # "CASCADEEND_RETWEETS_retweet_mean_user_description_emotion_positive",
#   # "CASCADEEND_RETWEETS_retweet_median_user_description_emotion_positive",
#   # "CASCADEEND_RETWEETS_retweet_mean_user_description_emotion_negative",
#   # "CASCADEEND_RETWEETS_retweet_median_user_description_emotion_negative",
#   # "CASCADEEND_RETWEETS_retweet_mean_user_description_emotion_sadness",
#   "CASCADEEND_RETWEETS_retweet_median_user_description_emotion_sadness",
#   # "CASCADEEND_RETWEETS_retweet_mean_user_description_emotion_disgust",
#   "CASCADEEND_RETWEETS_retweet_median_user_description_emotion_disgust",
#   # "CASCADEEND_RETWEETS_retweet_mean_user_description_emotion_joy",
#   "CASCADEEND_RETWEETS_retweet_median_user_description_emotion_joy"
# )
# NUM_FACTORS = 3

#Spreader Persona - activeness - time series
list_itemPrefixes = c(
  "TIMESERIES_RETWEETS_user_accountAge_day",
  "TIMESERIES_RETWEETS_user_friendsCount",
  "TIMESERIES_RETWEETS_user_followersCount",
  "TIMESERIES_RETWEETS_user_favouritesCount",
  "TIMESERIES_RETWEETS_user_statusesCount"
)
measurePrefix = "measure_test_"


# 
# #Spreader Persona - activeness - time series - non-STL metrics
# list_tsMetrics = c(
#   "LR_coef",
#   # "nonlinearity",
#   # "entropy",
#   "stability",
#   "lumpiness",
#   "maxLevelShift",
#   "maxVarShift",
#   "crossingPoints",
#   "flatSpots",
#   # "KPSS_statistic",
#   # "KPSS_LT10pct",
#   # "KPSS_LT5pct",
#   # "KPSS_LT2p5pct",
#   # "KPSS_LT1pct",
#   # "std1stDer",
#   "histogramMode"
# )
# NUM_FACTORS = 4
# 
# 
#Spreader Persona - activeness - time series - STL metrics
list_tsMetrics = c(
  "stl_linearity",
  "stl_curvature",
  "stl_trend",
  "stl_spike",
  # "stl_seasonalPeriod",
  "stl_eacf"
  # "stl_nperiods"
)
NUM_FACTORS = 4

str_tsSurfix = "_allTimeStamps"
# str_tsSurfix = "_filledTimeStamps"


list_items = c()
for (f in colnames(df_input)) {
  if (any(startsWith(f, list_itemPrefixes)) && any(str_detect(f, list_tsMetrics)) && endsWith(f, str_tsSurfix)) {
      list_items = c(list_items, f)
  }
}




# list_items = list_items[!startsWith(list_items, "CASCADEEND_ROOTTWEETS")]

# any(startsWith(list_features_y, "CASCADEEND_ROOTTWEETS"))
# any(startsWith(list_features_y, "CASCADEEND"))
# any(startsWith(list_features_y, "TIMESERIES"))



print("list_items:")
print(list_items)
print("length(list_items):")
print(length(list_items))



df_input_toAnalyze_withIndex = copy(df_input[df_input[, rootTweetGroup_feature]==rootTweetGroup_value, c("index", list_items)])
df_input_toAnalyze_allFeatures = copy(df_input_toAnalyze_withIndex[, list_items])
df_input_toAnalyze_allFeatures = data.frame(lapply(df_input_toAnalyze_allFeatures, as.numeric))

print("Raw data:")
describe(df_input_toAnalyze_allFeatures)
df_input_toAnalyze_standardized = data.frame(scale(df_input_toAnalyze_allFeatures, center=TRUE, scale=TRUE))
print("df_input_toAnalyze_standardized:")
describe(df_input_toAnalyze_standardized)

# df_input_toAnalyze_standardized[,"index"] = rownames(df_input_toAnalyze_standardized)

print("\nData to analyze:")
print("colnames(df_input_toAnalyze_standardized):")
print(colnames(df_input_toAnalyze_standardized))
print("nrow(df_input_toAnalyze_standardized):")
print(nrow(df_input_toAnalyze_standardized))

# EFAresult1 = factanal(~ ., data=df_input_toAnalyze_allFeatures, factors = 10, rotation = "none", na.action = na.exclude)
# EFAresult1 = factanal(~ ., data=df_input_toAnalyze_allFeatures, factors = 2, rotation = "none")
# describe(df_input_toAnalyze_allFeatures)

print("NUM_FACTORS:")
print(NUM_FACTORS)

# EFAresult1 = factanal(~ ., data=df_input_toAnalyze_allFeatures, factors = NUM_FACTORS, rotation = "varimax", na.action = na.exclude, lower = 0.01, scores = c("regression"))
EFAresult1 = factanal(~ ., data=df_input_toAnalyze_standardized, factors = NUM_FACTORS, rotation = "varimax", na.action = na.exclude, lower = 0.01, scores = c("regression"))
EFAresult1
EFAresult1$scores

df_toMerge = cbind(df_input_toAnalyze_withIndex[, "index"], EFAresult1$scores)
colnames(df_toMerge)[1] = "index"
colnames(df_toMerge)[2:length(colnames(df_toMerge))] = paste(measurePrefix, colnames(df_toMerge)[2:length(colnames(df_toMerge))], sep="")

df_output = merge(df_input, df_toMerge, by.x = "index", by.y = "index", all.x = TRUE, all.y = FALSE)

# df_output[, c("index", paste(measurePrefix, "Factor1", sep=""))]

nrow(df_output)
# colnames(df_output)

print("absFilename_output:")
print(absFilename_output)

write.table(
  df_output,
  absFilename_output,
  sep = ",",
  row.names = FALSE,
  col.names = TRUE,
  quote = TRUE
)
