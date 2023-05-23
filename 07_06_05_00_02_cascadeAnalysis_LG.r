rm(list=ls())

library(data.table)
library(stringr)
library(tidyr)
library(lme4) # for the analysis
library(lmerTest)# to get p-value estimations that are not part of the standard lme4 packages
library(MuMIn)

absFilename_input = "C:/vmwareSharedFolder/TwitterDataAnalysis/results/preprocessedData_factorMeasures.csv"

df_input =
  data.frame(fread(
    absFilename_input,
    # nrows = 50000,
    # sep = ",",
    header = TRUE
    # select = vector_attributesToLoad_rereports,
    # colClasses = c("character")
  ))

# # list_featureX = c("ROOTTWEETS_communicativeIntention_QOU", "ROOTTWEETS_communicativeIntention_EXP", "ROOTTWEETS_communicativeIntention_COM", "ROOTTWEETS_communicativeIntention_DIR", "ROOTTWEETS_communicativeIntention_DEC", "ROOTTWEETS_communicativeIntention_REP")
# list_featureX = c("ROOTTWEETS_communicativeIntention_QOU")
# feature_y = "measure_SpreaderPersona_retweeters_activeness_cascadeEnd"

# list_featureX = c("ROOTTWEETS_communicativeIntention_QOU", "ROOTTWEETS_communicativeIntention_EXP", "ROOTTWEETS_communicativeIntention_COM", "ROOTTWEETS_communicativeIntention_DIR", "ROOTTWEETS_communicativeIntention_DEC", "ROOTTWEETS_communicativeIntention_REP")
# feature_y = "measure_SpreaderPersona_activeness_cascadeEnd_Factor1"
# feature_y = "CASCADEEND_RETWEETS_retweet_median_user_accountAge_day" not good
# feature_y = "CASCADEEND_RETWEETS_retweet_median_user_followersCount" not good
# feature_y = "CASCADEEND_RETWEETS_retweet_median_user_friendsCount" not good
# feature_y = "CASCADEEND_RETWEETS_retweet_median_user_favouritesCount" not good
# feature_y = "CASCADEEND_RETWEETS_retweet_median_user_statusesCount"

# list_featureX = c("ROOTTWEETS_communicativeIntention_QOU", "ROOTTWEETS_communicativeIntention_EXP", "ROOTTWEETS_communicativeIntention_COM", "ROOTTWEETS_communicativeIntention_DIR", "ROOTTWEETS_communicativeIntention_DEC", "ROOTTWEETS_communicativeIntention_REP")
# list_featureX = c("measureIndependent_test_Factor1", "measureIndependent_test_Factor2", "measureIndependent_test_Factor3")
# list_featureX = c("measureIndependent_test_Factor1", "measureIndependent_test_Factor2")
# list_featureX = c("measureIndependent_test_Factor2")
list_featureX = c("ROOTTWEETS_communicativeIntention_QOU")

# feature_y = "measure_test_Factor1"
# feature_y = "CASCADEEND_RETWEETS_retweet_median_user_statusesCount"
# feature_y = "CASCADEEND_RETWEETS_retweet_mean_user_followersCount"
# feature_y = "CASCADEEND_RETWEETS_retweet_mean_user_friendsCount"
# feature_y = "CASCADEEND_RETWEETS_retweet_median_user_friendsCount"
# feature_y = "CASCADEEND_RETWEETS_retweet_mean_user_description_sentiment_neg"
# feature_y = "CASCADEEND_RETWEETS_retweet_median_user_description_sentiment_neg"
feature_y = "CASCADEEND_RETWEETS_retweet_mean_user_description_emotion_trust"




list_features_randomEffect = c("ROOTTWEETS_userScreenName_producer", "ROOTTWEETS_idStr_rootTweet", "index")
feature_randomEffect = list_features_randomEffect[1]
# feature_randomEffect = list_features_randomEffect[1]

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

df_input_toAnalyze_allFeatures = copy(df_input[df_input[, rootTweetGroup_feature]==rootTweetGroup_value,])
print("nrow(df_input_toAnalyze_allFeatures):")
print(nrow(df_input_toAnalyze_allFeatures))

# print(colnames(df_input_toAnalyze_allFeatures))
# print(df_input_toAnalyze_allFeatures$SpreaderPersona_retweeters_sentEmo_cascadeEnd)



# df_input_toAnalyze_allFeatures[, c(feature_y, "measure")]

# df_input_toAnalyze_allFeatures[, list_featureX]
# df_input_toAnalyze_allFeatures[, feature_y]

str_formula_full = feature_y
str_formula_full = paste(str_formula_full, " ~ ", list_featureX[1], sep="")
if (length(list_featureX) > 1) {
  for (i in 2:length(list_featureX))
    str_formula_full = paste(str_formula_full, " + ", list_featureX[i], sep="")
}
str_formula_full = paste(str_formula_full, " + (1|", feature_randomEffect, ")" , sep="")
# str_formula_full = paste(str_formula_full, " + ROOTTWEETS_communicativeIntention_REP*ROOTTWEETS_communicativeIntention_QOU" , sep="")
print("str_formula_full:")
print(str_formula_full)

# df_input_toAnalyze_allFeatures[, feature_y]

# model <- lm(str_formula_full, data = df_input_toAnalyze_allFeatures)
model = lmer(str_formula_full, data=df_input_toAnalyze_allFeatures, REML = FALSE)
modelingResult = summary(model)
modelingResult

