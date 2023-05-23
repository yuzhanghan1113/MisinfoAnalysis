rm(list=ls())

library(data.table)
library(stringr)
library(tidyr)
library(lme4) # for the analysis
library(lmerTest)# to get p-value estimations that are not part of the standard lme4 packages
library(MuMIn)
library(lmtest)
library(sandwich)

options(max.print = 999999)
# options(error = traceback)
options(error = recover)

# method = "mxdEff"
method = "linRgr"

# rSE = "F"
rSE = "T"


modelingTask = "fullAndReduced"
# modelingTask = "full"
# modelingTask = "reduced"

# THRESHOLD_COEF = 0.01
THRESHOLD_COEF = -1

absFilename_input = "C:/vmwareSharedFolder/TwitterDataAnalysis/results/results_20210209/preprocessedData.csv"
# path_output = "C:/vmwareSharedFolder/TwitterDataAnalysis/results/results_20210209/mixedModelResults/producerSpreader/multivariate/"
path_output = "C:/vmwareSharedFolder/TwitterDataAnalysis/results/results_20210209/linearRegressionResults/producerSpreader/multivariate/"

# 1 all features, for full models

vector_producerFeatures_user = c(
  "CASCADEEND_RETWEETS_rootTweet_user_followersCount",
  "CASCADEEND_RETWEETS_rootTweet_user_friendsCount",
  "CASCADEEND_RETWEETS_rootTweet_user_listedCount",
  "CASCADEEND_RETWEETS_rootTweet_user_favouritesCount",
  "CASCADEEND_RETWEETS_rootTweet_user_geoEnabled",
  "CASCADEEND_RETWEETS_rootTweet_user_verified",
  "CASCADEEND_RETWEETS_rootTweet_user_statusesCount"
)
vector_producerFeatures_userDescription = c(
  "CASCADEEND_RETWEETS_rootTweet_user_description_sentiment_neg",
  "CASCADEEND_RETWEETS_rootTweet_user_description_sentiment_neu",
  "CASCADEEND_RETWEETS_rootTweet_user_description_sentiment_pos",
  "CASCADEEND_RETWEETS_rootTweet_user_description_sentiment_compound",
  "CASCADEEND_RETWEETS_rootTweet_user_description_subjectivity",
  "CASCADEEND_RETWEETS_rootTweet_user_description_emotion_fear",
  "CASCADEEND_RETWEETS_rootTweet_user_description_emotion_anger",
  "CASCADEEND_RETWEETS_rootTweet_user_description_emotion_anticip",
  "CASCADEEND_RETWEETS_rootTweet_user_description_emotion_trust",
  "CASCADEEND_RETWEETS_rootTweet_user_description_emotion_surprise",
  "CASCADEEND_RETWEETS_rootTweet_user_description_emotion_positive",
  "CASCADEEND_RETWEETS_rootTweet_user_description_emotion_negative",
  "CASCADEEND_RETWEETS_rootTweet_user_description_emotion_sadness",
  "CASCADEEND_RETWEETS_rootTweet_user_description_emotion_disgust",
  "CASCADEEND_RETWEETS_rootTweet_user_description_emotion_joy",
  "CASCADEEND_RETWEETS_rootTweet_user_description_textStats_charCount",
  "CASCADEEND_RETWEETS_rootTweet_user_description_textStats_wordCount",
  "CASCADEEND_RETWEETS_rootTweet_user_description_textStats_uqWordCount",
  "CASCADEEND_RETWEETS_rootTweet_user_description_textStats_sentenceCount",
  "CASCADEEND_RETWEETS_rootTweet_user_description_textStats_charsPerWord",
  "CASCADEEND_RETWEETS_rootTweet_user_description_textStats_wordsPerSentence",
  "CASCADEEND_RETWEETS_rootTweet_user_description_textStats_readability"
)
vector_rootTweetFeatures = c(
  "CASCADEEND_RETWEETS_cascadeSize",
  "CASCADEEND_RETWEETS_rootTweet_favoriteCount",
  "CASCADEEND_RETWEETS_rootTweet_possiblySensitive"
)
vector_rootTweetFeatures_fullText = c(
  "CASCADEEND_RETWEETS_rootTweet_fullText_sentiment_neg",
  "CASCADEEND_RETWEETS_rootTweet_fullText_sentiment_neu",
  "CASCADEEND_RETWEETS_rootTweet_fullText_sentiment_pos",
  "CASCADEEND_RETWEETS_rootTweet_fullText_sentiment_compound",
  "CASCADEEND_RETWEETS_rootTweet_fullText_subjectivity",
  "CASCADEEND_RETWEETS_rootTweet_fullText_emotion_fear",
  "CASCADEEND_RETWEETS_rootTweet_fullText_emotion_anger",
  "CASCADEEND_RETWEETS_rootTweet_fullText_emotion_anticip",
  "CASCADEEND_RETWEETS_rootTweet_fullText_emotion_trust",
  "CASCADEEND_RETWEETS_rootTweet_fullText_emotion_surprise",
  "CASCADEEND_RETWEETS_rootTweet_fullText_emotion_positive",
  "CASCADEEND_RETWEETS_rootTweet_fullText_emotion_negative",
  "CASCADEEND_RETWEETS_rootTweet_fullText_emotion_sadness",
  "CASCADEEND_RETWEETS_rootTweet_fullText_emotion_disgust",
  "CASCADEEND_RETWEETS_rootTweet_fullText_emotion_joy",
  "CASCADEEND_RETWEETS_rootTweet_fullText_textStats_charCount",
  "CASCADEEND_RETWEETS_rootTweet_fullText_textStats_wordCount",
  "CASCADEEND_RETWEETS_rootTweet_fullText_textStats_uqWordCount",
  "CASCADEEND_RETWEETS_rootTweet_fullText_textStats_sentenceCount",
  "CASCADEEND_RETWEETS_rootTweet_fullText_textStats_charsPerWord",
  "CASCADEEND_RETWEETS_rootTweet_fullText_textStats_wordsPerSentence",
  "CASCADEEND_RETWEETS_rootTweet_fullText_textStats_readability"
)
vector_retweeterFeatures_user = c(
  "CASCADEEND_RETWEETS_retweet_ptg_user_protected",
  "CASCADEEND_RETWEETS_retweet_mean_user_followersCount",
  "CASCADEEND_RETWEETS_retweet_median_user_followersCount",
  "CASCADEEND_RETWEETS_retweet_mean_user_friendsCount",
  "CASCADEEND_RETWEETS_retweet_median_user_friendsCount",
  "CASCADEEND_RETWEETS_retweet_mean_user_listedCount",
  "CASCADEEND_RETWEETS_retweet_median_user_listedCount",
  "CASCADEEND_RETWEETS_retweet_median_user_accountAge_day",
  "CASCADEEND_RETWEETS_retweet_mean_user_favouritesCount",
  "CASCADEEND_RETWEETS_retweet_median_user_favouritesCount",
  "CASCADEEND_RETWEETS_retweet_mean_user_statusesCount",
  "CASCADEEND_RETWEETS_retweet_median_user_statusesCount",
  "CASCADEEND_RETWEETS_retweet_ptg_user_geoEnabled",
  "CASCADEEND_RETWEETS_retweet_ptg_user_verified"
)
vector_retweeterFeatures_userDescription = c(
  "CASCADEEND_RETWEETS_retweet_mean_user_description_sentiment_neg",
  "CASCADEEND_RETWEETS_retweet_median_user_description_sentiment_neg",
  "CASCADEEND_RETWEETS_retweet_mean_user_description_sentiment_neu",
  "CASCADEEND_RETWEETS_retweet_median_user_description_sentiment_neu",
  "CASCADEEND_RETWEETS_retweet_mean_user_description_sentiment_pos",
  "CASCADEEND_RETWEETS_retweet_median_user_description_sentiment_pos",
  "CASCADEEND_RETWEETS_retweet_mean_user_description_sentiment_compound",
  "CASCADEEND_RETWEETS_retweet_median_user_description_sentiment_compound",
  "CASCADEEND_RETWEETS_retweet_mean_user_description_subjectivity",
  "CASCADEEND_RETWEETS_retweet_median_user_description_subjectivity",
  "CASCADEEND_RETWEETS_retweet_mean_user_description_emotion_fear",
  "CASCADEEND_RETWEETS_retweet_median_user_description_emotion_fear",
  "CASCADEEND_RETWEETS_retweet_mean_user_description_emotion_anger",
  "CASCADEEND_RETWEETS_retweet_median_user_description_emotion_anger",
  "CASCADEEND_RETWEETS_retweet_mean_user_description_emotion_anticip",
  "CASCADEEND_RETWEETS_retweet_median_user_description_emotion_anticip",
  "CASCADEEND_RETWEETS_retweet_mean_user_description_emotion_trust",
  "CASCADEEND_RETWEETS_retweet_median_user_description_emotion_trust",
  "CASCADEEND_RETWEETS_retweet_mean_user_description_emotion_surprise",
  "CASCADEEND_RETWEETS_retweet_median_user_description_emotion_surprise",
  "CASCADEEND_RETWEETS_retweet_mean_user_description_emotion_positive",
  "CASCADEEND_RETWEETS_retweet_median_user_description_emotion_positive",
  "CASCADEEND_RETWEETS_retweet_mean_user_description_emotion_negative",
  "CASCADEEND_RETWEETS_retweet_median_user_description_emotion_negative",
  "CASCADEEND_RETWEETS_retweet_mean_user_description_emotion_sadness",
  "CASCADEEND_RETWEETS_retweet_median_user_description_emotion_sadness",
  "CASCADEEND_RETWEETS_retweet_mean_user_description_emotion_disgust",
  "CASCADEEND_RETWEETS_retweet_median_user_description_emotion_disgust",
  "CASCADEEND_RETWEETS_retweet_mean_user_description_emotion_joy",
  "CASCADEEND_RETWEETS_retweet_median_user_description_emotion_joy",
  "CASCADEEND_RETWEETS_retweet_mean_user_description_textStats_charCount",
  "CASCADEEND_RETWEETS_retweet_median_user_description_textStats_charCount",
  "CASCADEEND_RETWEETS_retweet_mean_user_description_textStats_wordCount",
  "CASCADEEND_RETWEETS_retweet_median_user_description_textStats_wordCount",
  "CASCADEEND_RETWEETS_retweet_mean_user_description_textStats_uqWordCount",
  "CASCADEEND_RETWEETS_retweet_median_user_description_textStats_uqWordCount",
  "CASCADEEND_RETWEETS_retweet_mean_user_description_textStats_sentenceCount",
  "CASCADEEND_RETWEETS_retweet_median_user_description_textStats_sentenceCount",
  "CASCADEEND_RETWEETS_retweet_mean_user_description_textStats_charsPerWord",
  "CASCADEEND_RETWEETS_retweet_median_user_description_textStats_charsPerWord",
  "CASCADEEND_RETWEETS_retweet_mean_user_description_textStats_wordsPerSentence",
  "CASCADEEND_RETWEETS_retweet_median_user_description_textStats_wordsPerSentence",
  "CASCADEEND_RETWEETS_retweet_mean_user_description_textStats_readability",
  "CASCADEEND_RETWEETS_retweet_median_user_description_textStats_readability"
)
vector_producerIntention = c(
"ROOTTWEETS_communicativeIntention_DIR",
"ROOTTWEETS_communicativeIntention_COM",
"ROOTTWEETS_communicativeIntention_EXP",
"ROOTTWEETS_communicativeIntention_DEC",
"ROOTTWEETS_communicativeIntention_QOU",
"ROOTTWEETS_communicativeIntention_count"
)

# # # 2 features selected by reduced models of 1, for full models
# #
# vector_producerFeatures_user = c(
#   "CASCADEEND_RETWEETS_rootTweet_user_friendsCount",
#   "CASCADEEND_RETWEETS_rootTweet_user_favouritesCount",
#   "CASCADEEND_RETWEETS_rootTweet_user_statusesCount"
# )
# vector_producerFeatures_userDescription = c(
#   "CASCADEEND_RETWEETS_rootTweet_user_description_sentiment_neg",
#   "CASCADEEND_RETWEETS_rootTweet_user_description_sentiment_neu",
#   "CASCADEEND_RETWEETS_rootTweet_user_description_sentiment_pos",
#   "CASCADEEND_RETWEETS_rootTweet_user_description_sentiment_compound",
#   "CASCADEEND_RETWEETS_rootTweet_user_description_emotion_positive",
#   "CASCADEEND_RETWEETS_rootTweet_user_description_emotion_sadness",
#   "CASCADEEND_RETWEETS_rootTweet_user_description_emotion_joy",
#   "CASCADEEND_RETWEETS_rootTweet_user_description_emotion_fear",
#   "CASCADEEND_RETWEETS_rootTweet_user_description_emotion_anger",
#   "CASCADEEND_RETWEETS_rootTweet_user_description_emotion_trust",
#   "CASCADEEND_RETWEETS_rootTweet_user_description_emotion_surprise",
#   "CASCADEEND_RETWEETS_rootTweet_user_description_emotion_negative",
#   "CASCADEEND_RETWEETS_rootTweet_user_description_emotion_disgust",
#   "CASCADEEND_RETWEETS_rootTweet_user_description_subjectivity",
#   "CASCADEEND_RETWEETS_rootTweet_user_description_textStats_charCount",
#   "CASCADEEND_RETWEETS_rootTweet_user_description_textStats_wordCount",
#   "CASCADEEND_RETWEETS_rootTweet_user_description_textStats_sentenceCount",
#   "CASCADEEND_RETWEETS_rootTweet_user_description_textStats_charsPerWord",
#   "CASCADEEND_RETWEETS_rootTweet_user_description_textStats_wordsPerSentence",
#   "CASCADEEND_RETWEETS_rootTweet_user_description_textStats_uqWordCount"
# )
# vector_rootTweetFeatures_fullText = c(
#   "CASCADEEND_RETWEETS_rootTweet_fullText_sentiment_neg",
#   "CASCADEEND_RETWEETS_rootTweet_fullText_sentiment_neu",
#   "CASCADEEND_RETWEETS_rootTweet_fullText_sentiment_pos",
#   "CASCADEEND_RETWEETS_rootTweet_fullText_sentiment_compound",
#   "CASCADEEND_RETWEETS_rootTweet_fullText_emotion_positive",
#   "CASCADEEND_RETWEETS_rootTweet_fullText_emotion_negative",
#   "CASCADEEND_RETWEETS_rootTweet_fullText_emotion_fear",
#   "CASCADEEND_RETWEETS_rootTweet_fullText_emotion_anger",
#   "CASCADEEND_RETWEETS_rootTweet_fullText_emotion_trust",
#   "CASCADEEND_RETWEETS_rootTweet_fullText_emotion_sadness",
#   "CASCADEEND_RETWEETS_rootTweet_fullText_emotion_disgust",
#   "CASCADEEND_RETWEETS_rootTweet_fullText_emotion_joy",
#   "CASCADEEND_RETWEETS_rootTweet_fullText_textStats_charCount",
#   "CASCADEEND_RETWEETS_rootTweet_fullText_textStats_wordCount",
#   "CASCADEEND_RETWEETS_rootTweet_fullText_textStats_sentenceCount",
#   "CASCADEEND_RETWEETS_rootTweet_fullText_textStats_charsPerWord",
#   "CASCADEEND_RETWEETS_rootTweet_fullText_textStats_wordsPerSentence",
#   "CASCADEEND_RETWEETS_rootTweet_fullText_textStats_readability",
#   "CASCADEEND_RETWEETS_rootTweet_fullText_subjectivity",
#   "CASCADEEND_RETWEETS_rootTweet_fullText_textStats_uqWordCount"
# )
# vector_retweeterFeatures_user = c(
#   "CASCADEEND_RETWEETS_retweet_median_user_accountAge_day",
#   "CASCADEEND_RETWEETS_retweet_mean_user_followersCount",
#   "CASCADEEND_RETWEETS_retweet_mean_user_friendsCount",
#   "CASCADEEND_RETWEETS_retweet_mean_user_listedCount",
#   "CASCADEEND_RETWEETS_retweet_mean_user_favouritesCount",
#   "CASCADEEND_RETWEETS_retweet_mean_user_statusesCount",
#   "CASCADEEND_RETWEETS_retweet_ptg_user_geoEnabled"
# )
# vector_retweeterFeatures_userDescription = c(
#   "CASCADEEND_RETWEETS_retweet_mean_user_description_sentiment_neg",
#   "CASCADEEND_RETWEETS_retweet_median_user_description_sentiment_neu",
#   "CASCADEEND_RETWEETS_retweet_median_user_description_sentiment_pos",
#   "CASCADEEND_RETWEETS_retweet_median_user_description_sentiment_compound",
#   "CASCADEEND_RETWEETS_retweet_mean_user_description_emotion_fear",
#   "CASCADEEND_RETWEETS_retweet_mean_user_description_emotion_surprise",
#   "CASCADEEND_RETWEETS_retweet_mean_user_description_emotion_sadness",
#   "CASCADEEND_RETWEETS_retweet_mean_user_description_emotion_anger",
#   "CASCADEEND_RETWEETS_retweet_mean_user_description_emotion_trust",
#   "CASCADEEND_RETWEETS_retweet_mean_user_description_emotion_positive",
#   "CASCADEEND_RETWEETS_retweet_mean_user_description_emotion_negative",
#   "CASCADEEND_RETWEETS_retweet_mean_user_description_emotion_disgust",
#   "CASCADEEND_RETWEETS_retweet_mean_user_description_emotion_joy",
#   "CASCADEEND_RETWEETS_retweet_median_user_description_subjectivity",
#   "CASCADEEND_RETWEETS_retweet_median_user_description_textStats_charCount",
#   "CASCADEEND_RETWEETS_retweet_median_user_description_textStats_wordCount",
#   "CASCADEEND_RETWEETS_retweet_median_user_description_textStats_uqWordCount",
#   "CASCADEEND_RETWEETS_retweet_median_user_description_textStats_sentenceCount",
#   "CASCADEEND_RETWEETS_retweet_median_user_description_textStats_charsPerWord",
#   "CASCADEEND_RETWEETS_retweet_median_user_description_textStats_wordsPerSentence",
#   "CASCADEEND_RETWEETS_retweet_median_user_description_textStats_readability"
# )
# vector_rootTweetFeatures = c(
#   "CASCADEEND_RETWEETS_cascadeSize",
#   "CASCADEEND_RETWEETS_rootTweet_favoriteCount",
#   "CASCADEEND_RETWEETS_rootTweet_possiblySensitive"
# )
# vector_producerIntention = c(
# "ROOTTWEETS_communicativeIntention_DIR",
# "ROOTTWEETS_communicativeIntention_COM",
# "ROOTTWEETS_communicativeIntention_EXP",
# "ROOTTWEETS_communicativeIntention_DEC",
# "ROOTTWEETS_communicativeIntention_QOU",
# "ROOTTWEETS_communicativeIntention_count"
# )

print(absFilename_input)
print("absFilename_input:")

df_input =
  data.frame(fread(
    absFilename_input,
    # nrows = 50000,
    # sep = ",",
    header = TRUE
    # select = vector_attributesToLoad_rereports,
    # colClasses = c("character")
  ))

print("rownames(df_input):")
# print(rownames(df_input))
# print("rnames(df_input):")
# print(colnames(df_input))

# print(df_input[, c(
#                   "CASCADEEND_ROOTTWEETS_ptg_retweets_total_48hrs",
#                    "CASCADEEND_ROOTTWEETS_num_retweets",
#                    "CASCADEEND_ROOTTWEETS_num_retweets_total",
#                    "CASCADEEND_RETWEETS_cascadeSize"
#                    
#                    )])

# sum(is.na(df_input[, c("CASCADEEND_RETWEETS_cascadeSize")]))

feature_randomEffect = "CASCADEEND_RETWEETS_rootTweet_user_screenName"

dir.create(file.path(path_output, "fm"))
dir.create(file.path(paste(path_output, "fm", sep=""), "cpl"))
dir.create(file.path(path_output, "rm"))
dir.create(file.path(paste(path_output, "rm", sep=""), "prt"))

vector_veracities = c("mis", "aut")

if (modelingTask=="full" |
    modelingTask=="fullAndReduced") {
  ## produce the full models with all independent variables
  
  for (veracity in vector_veracities) {
    # veracity = "mis"
    # veracity = "aut"
    
    misinfo = "999"
    if (veracity == "mis") {
      misinfo = "1"
    } else if (veracity == "aut") {
      misinfo = "0"
    }
    
    for (index_task in 0:18) {
      print("index_task:")
      print(index_task)
      
      df_output = data.frame()
      df_output_complete = data.frame()
      
      if (index_task == 0) {
        vector_featureX = vector_rootTweetFeatures_fullText
        str_name_X = "rootTweetFeaturesFullText"
        vector_featureY = vector_rootTweetFeatures
        str_name_Y = "rootTweetFeatures"
      } else if (index_task == 1) {
        vector_featureX = vector_rootTweetFeatures_fullText
        str_name_X = "rootTweetFeaturesFullText"
        vector_featureY = vector_retweeterFeatures_user
        str_name_Y = "retweeterFeaturesUser"
      } else if (index_task == 2) {
        vector_featureX = vector_rootTweetFeatures_fullText
        str_name_X = "rootTweetFeaturesFullText"
        vector_featureY = vector_retweeterFeatures_userDescription
        str_name_Y = "retweeterFeaturesUserDescription"
      } else if (index_task == 3) {
        vector_featureX = vector_rootTweetFeatures
        str_name_X = "rootTweetFeatures"
        vector_featureY = vector_retweeterFeatures_user
        str_name_Y = "retweeterFeaturesUser"
      } else if (index_task == 4) {
        vector_featureX = vector_rootTweetFeatures
        str_name_X = "rootTweetFeatures"
        vector_featureY = vector_retweeterFeatures_userDescription
        str_name_Y = "retweeterFeaturesUserDescription"
      } else if (index_task == 5) {
        vector_featureX = vector_producerFeatures_user
        str_name_X = "producerFeaturesUser"
        vector_featureY = vector_rootTweetFeatures
        str_name_Y = "rootTweetFeatures"
      } else if (index_task == 6) {
        vector_featureX = vector_producerFeatures_user
        str_name_X = "producerFeaturesUser"
        vector_featureY = vector_rootTweetFeatures_fullText
        str_name_Y = "rootTweetFeaturesFullText"
      } else if (index_task == 7) {
        vector_featureX = vector_producerFeatures_userDescription
        str_name_X = "producerFeaturesUserDescription"
        vector_featureY = vector_rootTweetFeatures
        str_name_Y = "rootTweetFeatures"
      } else if (index_task == 8) {
        vector_featureX = vector_producerFeatures_userDescription
        str_name_X = "producerFeaturesUserDescription"
        vector_featureY = vector_rootTweetFeatures_fullText
        str_name_Y = "rootTweetFeaturesFullText"
      } else if (index_task == 9) {
        vector_featureX = vector_producerFeatures_user
        str_name_X = "producerFeaturesUser"
        vector_featureY = vector_retweeterFeatures_user
        str_name_Y = "retweeterFeaturesUser"
      } else if (index_task == 10) {
        vector_featureX = vector_producerFeatures_user
        str_name_X = "producerFeaturesUser"
        vector_featureY = vector_retweeterFeatures_userDescription
        str_name_Y = "retweeterFeaturesUserDescription"
      }  else if (index_task == 11) {
        vector_featureX = vector_producerFeatures_userDescription
        str_name_X = "producerFeaturesUserDescription"
        vector_featureY = vector_retweeterFeatures_user
        str_name_Y = "retweeterFeaturesUser"
      } else if (index_task == 12) {
        vector_featureX = vector_producerFeatures_userDescription
        str_name_X = "producerFeaturesUserDescription"
        vector_featureY = vector_retweeterFeatures_userDescription
        str_name_Y = "retweeterFeaturesUserDescription"
      } else if (index_task == 13) {
        vector_featureX = vector_producerIntention
        str_name_X = "producerIntention"
        vector_featureY = vector_producerFeatures_user
        str_name_Y = "producerFeaturesUser"
      } else if (index_task == 14) {
        vector_featureX = vector_producerIntention
        str_name_X = "producerIntention"
        vector_featureY = vector_producerFeatures_userDescription
        str_name_Y = "producerFeaturesUserDescription"
      } else if (index_task == 15) {
        vector_featureX = vector_producerIntention
        str_name_X = "producerIntention"
        vector_featureY = vector_rootTweetFeatures
        str_name_Y = "rootTweetFeatures"
      } else if (index_task == 16) {
        vector_featureX = vector_producerIntention
        str_name_X = "producerIntention"
        vector_featureY = vector_rootTweetFeatures_fullText
        str_name_Y = "rootTweetFeaturesFullText"
      } else if (index_task == 17) {
        vector_featureX = vector_producerIntention
        str_name_X = "producerIntention"
        vector_featureY = vector_retweeterFeatures_user
        str_name_Y = "retweeterFeaturesUser"
      } else if (index_task == 18) {
        vector_featureX = vector_producerIntention
        str_name_X = "producerIntention"
        vector_featureY = vector_retweeterFeatures_userDescription
        str_name_Y = "retweeterFeaturesUserDescription"
      } else {
        vector_featureX = c("WrongX")
        str_name_X = "WrongX"
        vector_featureY = c("WrongY")
        str_name_Y = "WrongY"
      }
      
      absFilename_output = paste(
        path_output,
        "fm/",
        "rg_prodSprd_var=",
        veracity,
        "_med=",
        method,
        "_rSE=",
        rSE,
        "_X=",
        str_name_X,
        "_Y=",
        str_name_Y,
        "_fm",
        ".csv",
        sep = ""
      )
      
      absFilename_output_complete = paste(
        path_output,
        "fm/cpl/",
        "rg_prodSprd_var=",
        veracity,
        "_med=",
        method,
        "_rSE=",
        rSE,
        "_X=",
        str_name_X,
        "_Y=",
        str_name_Y,
        "_rm_cpl",
        ".csv",
        sep = ""
      )
      
      for (feature_y in vector_featureY) {
        df_input_toAnalyze = df_input[df_input$ROOTTWEETS_veracityLabel_agg_misinformation ==
                                        misinfo, c(
                                          vector_featureX,
                                          feature_y,
                                          feature_randomEffect,
                                          "ROOTTWEETS_veracityLabel_agg_misinformation"
                                        )]
        
        print(
          "table(df_input_toAnalyze$ROOTTWEETS_veracityLabel_agg_misinformation):"
        )
        print(table(
          df_input_toAnalyze$ROOTTWEETS_veracityLabel_agg_misinformation
        ))
        
        index_row = paste(str_name_X, "-", feature_y, sep = "")
        
        str_formula = paste(feature_y,
                                 " ~ ",
                                 sep = "")
        
        str_formula = paste(str_formula,
                                 vector_featureX[1],
                                 sep = "")
        
        if(length(vector_featureX) >= 2) {
          for (feature_x in vector_featureX[2:length(vector_featureX)]) {
            str_formula = paste(str_formula,
                                     " + ",
                                     feature_x,
                                     sep = "")
          }
        }
        
        if (method == "mxdEff") {
          str_formula = paste(str_formula,
                                   " + (1|",
                                   feature_randomEffect,
                                   ")",
                                   sep = "")
        }
        
        str_formula = as.formula(str_formula)
        
        print("str_formula:")
        print(str_formula)
        
        tryCatch({
          
          if (method == "mxdEff") {
            model_regression = lmer(str_formula, data = df_input_toAnalyze, REML = FALSE)
          } else if (method == "linRgr") {
            model_regression = lm(str_formula, data = df_input_toAnalyze)
          }
          
          if (rSE == "T") {
            df_modelingResults = coeftest(model_regression, vcov. = vcovHC(model_regression, type = 'HC1'))
          } else if (rSE == "F") {
            modelingResult = summary(model_regression)
            df_modelingResults = modelingResult$coefficients
          }
          
          # print("df_modelingResults:")
          # print(df_modelingResults)
          
          # result_RSquared = r.squaredGLMM(model_regression)
          # RSquared_marginal = result_RSquared[[1, "R2m"]]
          # RSquared_conditional = result_RSquared[[1, "R2c"]]
          
          df_output[index_row, "method"] = method
          df_output[index_row, "robustStandardErrors"] = rSE
          df_output[index_row, "veracity"] = veracity
          df_output[index_row, "varGrp_X"] = str_name_X
          df_output[index_row, "varGrp_Y"] = str_name_Y
          
          df_output_complete[index_row, "method"] = method
          df_output_complete[index_row, "robustStandardErrors"] = rSE
          df_output_complete[index_row, "veracity"] = veracity
          df_output_complete[index_row, "varGrp_X"] = str_name_X
          df_output_complete[index_row, "varGrp_Y"] = str_name_Y
          
          df_output[index_row, "feature_y"] = feature_y
          df_output_complete[index_row, "feature_y"] = feature_y
          
          num_observations = nrow(df_input_toAnalyze[, c(vector_featureX, feature_y, feature_randomEffect)]) - sum(apply(df_input_toAnalyze[, c(vector_featureX, feature_y, feature_randomEffect)], 1, anyNA))
          df_output[index_row, "num_observations"] = num_observations
          df_output_complete[index_row, "num_observations"] = num_observations
          
          
          num_xFeatures_useful = 0
          
          for (feature_x in vector_featureX) {
            if (feature_x %in% rownames(df_modelingResults)) {
              regression_coef = df_modelingResults[feature_x, "Estimate"]
              regression_p = df_modelingResults[feature_x, "Pr(>|t|)"]
              
              if (!is.nan(regression_p) & regression_p < 0.05 &
                  !is.nan(regression_coef) & abs(regression_coef) >= THRESHOLD_COEF) {
                num_xFeatures_useful = num_xFeatures_useful + 1
              }
            }
          }
          
          df_output[index_row, "num_xFeatures_useful"] = num_xFeatures_useful
          df_output_complete[index_row, "num_xFeatures_useful"] = num_xFeatures_useful
          
          for (feature_x in vector_featureX) {
            df_output[index_row, feature_x] = NA
            df_output_complete[index_row, feature_x] = NA
            
            if (feature_x %in% rownames(df_modelingResults)) {
              print("regression runnable")
              
              regression_coef = df_modelingResults[feature_x, "Estimate"]
              regression_p = df_modelingResults[feature_x, "Pr(>|t|)"]
              
              # print("regression_p 1:")
              # print(regression_p)
              
              if (!is.nan(regression_p) & regression_p < 0.05) {
                print("p-value < 0.05")
                df_output_complete[index_row, feature_x] = regression_coef
              }
              
              if (!is.nan(regression_p) & regression_p < 0.05 &
                  !is.nan(regression_coef) & abs(regression_coef) >= THRESHOLD_COEF) {
                print("p-value < 0.05 and abs(coef) >= threshold")
                df_output[index_row, feature_x] = regression_coef
              }
            }
          }
          
          print("absFilename_output:")
          print(absFilename_output)
          
          write.table(
            df_output_complete,
            absFilename_output_complete,
            sep = ",",
            row.names = FALSE,
            col.names = TRUE,
            quote = TRUE
          )
          
          write.table(
            df_output,
            absFilename_output,
            sep = ",",
            row.names = FALSE,
            col.names = TRUE,
            quote = TRUE
          )
          
        }, warning = function(war) {
          print("Get warning")
          print("war:")
          print(war)
          readline(prompt="Press [enter] to continue")
        }, error = function(err) {
          print("Get error")
          print("err:")
          print(err)
          readline(prompt="Press [enter] to continue")
        }, finally = {
        })
      }
      
      print("absFilename_output_complete:")
      print(absFilename_output_complete)
      
      write.table(
        df_output_complete,
        absFilename_output_complete,
        sep = ",",
        row.names = FALSE,
        col.names = TRUE,
        quote = TRUE
      )
      
      if (nrow(df_output_complete) <= 0) {
        file.remove(absFilename_output_complete)
        print("No row in output-complete. File deleted.")
      }
      
      df_output = df_output[df_output$num_xFeatures_useful > 0, ]
      df_output = df_output[, colSums(is.na(df_output)) < nrow(df_output)]
      
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
      
      if (nrow(df_output) <= 0) {
        file.remove(absFilename_output)
        print("No row in output. File deleted.")
      }
    }
  }
}

if (modelingTask=="reduced" | modelingTask=="fullAndReduced") {

  ## produce the reduced models with only independent considred "userful" in the full models
  ## usful means: p < 0.05 and coef >= threshold
  
  for (veracity in vector_veracities) {
    # veracity = "mis"
    # veracity = "aut"
    
    misinfo = "999"
    if (veracity == "mis") {
      misinfo = "1"
    } else if (veracity == "aut") {
      misinfo = "0"
    }
    
    for (index_task in 0:18) {
      
      print("index_task:")
      print(index_task)
      
      df_output = data.frame()
      
      if (index_task == 0) {
        # vector_featureX = vector_rootTweetFeatures_fullText
        str_name_X = "rootTweetFeaturesFullText"
        vector_featureY = vector_rootTweetFeatures
        str_name_Y = "rootTweetFeatures"
      } else if(index_task==1) {
        # vector_featureX = vector_rootTweetFeatures_fullText
        str_name_X = "rootTweetFeaturesFullText"
        vector_featureY = vector_retweeterFeatures_user
        str_name_Y = "retweeterFeaturesUser"
      } else if (index_task == 2) {
        # vector_featureX = vector_rootTweetFeatures_fullText
        str_name_X = "rootTweetFeaturesFullText"
        vector_featureY = vector_retweeterFeatures_userDescription
        str_name_Y = "retweeterFeaturesUserDescription"
      } else if (index_task == 3) {
        # vector_featureX = vector_rootTweetFeatures
        str_name_X = "rootTweetFeatures"
        vector_featureY = vector_retweeterFeatures_user
        str_name_Y = "retweeterFeaturesUser"
      } else if (index_task == 4) {
        # vector_featureX = vector_rootTweetFeatures
        str_name_X = "rootTweetFeatures"
        vector_featureY = vector_retweeterFeatures_userDescription
        str_name_Y = "retweeterFeaturesUserDescription"
      } else if (index_task == 5) {
        # vector_featureX = vector_producerFeatures_user
        str_name_X = "producerFeaturesUser"
        vector_featureY = vector_rootTweetFeatures
        str_name_Y = "rootTweetFeatures"
      } else if (index_task == 6) {
        # vector_featureX = vector_producerFeatures_user
        str_name_X = "producerFeaturesUser"
        vector_featureY = vector_rootTweetFeatures_fullText
        str_name_Y = "rootTweetFeaturesFullText"
      } else if (index_task == 7) {
        # vector_featureX = vector_producerFeatures_userDescription
        str_name_X = "producerFeaturesUserDescription"
        vector_featureY = vector_rootTweetFeatures
        str_name_Y = "rootTweetFeatures"
      } else if (index_task == 8) {
        # vector_featureX = vector_producerFeatures_userDescription
        str_name_X = "producerFeaturesUserDescription"
        vector_featureY = vector_rootTweetFeatures_fullText
        str_name_Y = "rootTweetFeaturesFullText"
      } else if (index_task == 9) {
        # vector_featureX = vector_producerFeatures_user
        str_name_X = "producerFeaturesUser"
        vector_featureY = vector_retweeterFeatures_user
        str_name_Y = "retweeterFeaturesUser"
      } else if (index_task == 10) {
        # vector_featureX = vector_producerFeatures_user
        str_name_X = "producerFeaturesUser"
        vector_featureY = vector_retweeterFeatures_userDescription
        str_name_Y = "retweeterFeaturesUserDescription"
      }  else if (index_task == 11) {
        # vector_featureX = vector_producerFeatures_userDescription
        str_name_X = "producerFeaturesUserDescription"
        vector_featureY = vector_retweeterFeatures_user
        str_name_Y = "retweeterFeaturesUser"
      } else if (index_task == 12) {
        # vector_featureX = vector_producerFeatures_userDescription
        str_name_X = "producerFeaturesUserDescription"
        vector_featureY = vector_retweeterFeatures_userDescription
        str_name_Y = "retweeterFeaturesUserDescription"
      } else if (index_task == 13) {
        # vector_featureX = vector_producerIntention
        str_name_X = "producerIntention"
        vector_featureY = vector_producerFeatures_user
        str_name_Y = "producerFeaturesUser"
      } else if (index_task == 14) {
        # vector_featureX = vector_producerIntention
        str_name_X = "producerIntention"
        vector_featureY = vector_producerFeatures_userDescription
        str_name_Y = "producerFeaturesUserDescription"
      } else if (index_task == 15) {
        # vector_featureX = vector_producerIntention
        str_name_X = "producerIntention"
        vector_featureY = vector_rootTweetFeatures
        str_name_Y = "rootTweetFeatures"
      } else if (index_task == 16) {
        # vector_featureX = vector_producerIntention
        str_name_X = "producerIntention"
        vector_featureY = vector_rootTweetFeatures_fullText
        str_name_Y = "rootTweetFeaturesFullText"
      } else if (index_task == 17) {
        # vector_featureX = vector_producerIntention
        str_name_X = "producerIntention"
        vector_featureY = vector_retweeterFeatures_user
        str_name_Y = "retweeterFeaturesUser"
      } else if (index_task == 18) {
        # vector_featureX = vector_producerIntention
        str_name_X = "producerIntention"
        vector_featureY = vector_retweeterFeatures_userDescription
        str_name_Y = "retweeterFeaturesUserDescription"
      } else {
        # vector_featureX = c("WrongX")
        str_name_X = "WrongX"
        vector_featureY = c("WrongY")
        str_name_Y = "WrongY"
      }
      
      vector_featureX = c("WrongX")
      
      absFilename_input_fullModel = paste(
        path_output, "fm/",
        "rg_prodSprd_var=",
        veracity,
        "_med=",
        method,
        "_rSE=",
        rSE,
        "_X=",
        str_name_X,
        "_Y=",
        str_name_Y,
        "_fm",
        ".csv",
        sep = ""
      )
      
      
      
      print("absFilename_input_fullModel:")
      print(absFilename_input_fullModel)
      
      if (!file.exists(absFilename_input_fullModel)) {
        print("Full model file does not exist. Skip this task.")
        next
      }
      
      df_input_fullModel =
        data.frame(fread(
          absFilename_input_fullModel,
          # nrows = 50000,
          # sep = ",",
          header = TRUE
          # select = vector_attributesToLoad_rereports,
          # colClasses = c("character")
        ))
      
      if (nrow(df_input_fullModel) <= 0) {
        print("The full model has 0 rows. Skip this task.")
        next
      }
      
      vector_columnsToRemove = c("veracity", "robustStandardErrors", "method", "varGrp_X", "varGrp_Y","feature_y", "num_observations", "num_xFeatures_useful")
      vector_featureX = colnames(df_input_fullModel)
      
      # print("df_input_fullModel:")
      # print(df_input_fullModel)
      
      print("vector_featureX:")
      print(vector_featureX)
      
      vector_featureX = vector_featureX[!(vector_featureX %in% vector_columnsToRemove)]
      
      print("vector_featureX:")
      print(vector_featureX)
      
      if (length(vector_featureX) <= 0) {
        print("The full model has 0 x features. Skip this task.")
        next
      }
      
      absFilename_output = paste(
        path_output, "rm/",
        "rg_prodSprd_var=",
        veracity,
        "_med=",
        method,
        "_rSE=",
        rSE,
        "_X=",
        str_name_X,
        "_Y=",
        str_name_Y,
        "_rm",
        ".csv",
        sep = ""
      )
      
      for (feature_y in vector_featureY) {
        
        # print("vector_featureX:")
        # print(vector_featureX)
        # 
        # print("feature_y:")
        # print(feature_y)
        
        
        
        df_input_toAnalyze = df_input[df_input$ROOTTWEETS_veracityLabel_agg_misinformation ==
                                        misinfo, c(vector_featureX, feature_y, feature_randomEffect, "ROOTTWEETS_veracityLabel_agg_misinformation")]
        
        print("table(df_input_toAnalyze$ROOTTWEETS_veracityLabel_agg_misinformation):")
        print(table(df_input_toAnalyze$ROOTTWEETS_veracityLabel_agg_misinformation))
        
        index_row = paste(str_name_X, "-", feature_y, sep = "")
        
        str_formula = paste(feature_y,
                                 " ~ ",
                                 sep = "")
        
        str_formula = paste(str_formula,
                                 vector_featureX[1],
                                 sep = "")
        
        if(length(vector_featureX) >= 2) {
          for (feature_x in vector_featureX[2:length(vector_featureX)]) {
            str_formula = paste(str_formula,
                                     " + ",
                                     feature_x,
                                     sep = "")
          }
        }
        
        if (method == "mxdEff") {
          str_formula = paste(str_formula,
                                   " + (1|",
                                   feature_randomEffect,
                                   ")",
                                   sep = "")
        }
        
        print("str_formula:")
        print(str_formula)
        
        tryCatch({
          
          if (method == "mxdEff") {
            model_regression = lmer(str_formula, data = df_input_toAnalyze, REML = FALSE)
          } else if (method == "linRgr") {
            model_regression = lm(str_formula, data = df_input_toAnalyze)
          }
          
          if (rSE == "T") {
            df_modelingResults = coeftest(model_regression, vcov. = vcovHC(model_regression, type = 'HC1'))
          } else if (rSE == "F") {
            modelingResult = summary(model_regression)
            df_modelingResults = modelingResult$coefficients
          }
          
          # print("df_modelingResults:")
          # print(df_modelingResults)
          
          # result_RSquared = r.squaredGLMM(model_regression)
          # RSquared_marginal = result_RSquared[[1, "R2m"]]
          # RSquared_conditional = result_RSquared[[1, "R2c"]]
          
          df_output[index_row, "method"] = method
          df_output[index_row, "robustStandardErrors"] = rSE
          df_output[index_row, "veracity"] = veracity
          df_output[index_row, "varGrp_X"] = str_name_X
          df_output[index_row, "varGrp_Y"] = str_name_Y
          
          df_output[index_row, "feature_y"] = feature_y
          
          num_observations = nrow(df_input_toAnalyze[, c(vector_featureX, feature_y, feature_randomEffect)]) - sum(apply(df_input_toAnalyze[, c(vector_featureX, feature_y, feature_randomEffect)], 1, anyNA))
          df_output[index_row, "num_observations"] = num_observations
          
          num_xFeatures_useful = 0
          
          
          
          for (feature_x in vector_featureX) {
            
            if (feature_x %in% rownames(df_modelingResults)) {
              
              regression_coef = df_modelingResults[feature_x, "Estimate"]
              regression_p = df_modelingResults[feature_x, "Pr(>|t|)"]
              
              if (!is.nan(regression_p) & regression_p < 0.05 & !is.nan(regression_coef) & abs(regression_coef) >= THRESHOLD_COEF) {
                num_xFeatures_useful = num_xFeatures_useful + 1
              }
            }
          }
          
          df_output[index_row, "num_xFeatures_useful"] = num_xFeatures_useful
          
          for (feature_x in vector_featureX) {
            
            df_output[index_row, feature_x] = NA
            
            if (feature_x %in% rownames(df_modelingResults)) {
              print("regression runnable")
              
              regression_coef = df_modelingResults[feature_x, "Estimate"]
              regression_p = df_modelingResults[feature_x, "Pr(>|t|)"]
              
              if (!is.nan(regression_p) & regression_p < 0.05 & !is.nan(regression_coef) & abs(regression_coef) >= THRESHOLD_COEF) {
                print("p-value < 0.05 and abs(coef) >= threshold")
                df_output[index_row, feature_x] = regression_coef
              }
            }
          }
          
          print("index_task 1:")
          print(index_task)

          print("absFilename_output 1:")
          print(absFilename_output)

          write.table(
            df_output,
            absFilename_output,
            sep = ",",
            row.names = FALSE,
            col.names = TRUE,
            quote = TRUE
          )
          
        }, warning = function(war) {
          print("Get warning")
          print("war:")
          print(war)
          readline(prompt="Press [enter] to continue")
        }, error = function(err) {
          print("Get error")
          print("err:")
          print(err)
          readline(prompt="Press [enter] to continue")
        }, finally = {
        })
      }
      
      absFilename_output_print = paste(
        path_output,
        "fm/pt/",
        "rg_prodSprd_var=",
        veracity,
        "_med=",
        method,
        "_X=",
        str_name_X,
        "_Y=",
        str_name_Y,
        "_fm",
        ".csv",
        sep = ""
      )
      
      
      df_output = df_output[df_output$num_xFeatures_useful>0,]
      # df_output = df_output[,colSums(is.na(df_output))<nrow(df_output)]
      
      print("index_task 2:")
      print(index_task)
      
      print("absFilename_output 2:")
      print(absFilename_output)
      
      write.table(
        df_output,
        absFilename_output,
        sep = ",",
        row.names = FALSE,
        col.names = TRUE,
        quote = TRUE
      )
      
      # print("vector_featureX:")
      # print(vector_featureX)
      
      df_output_print = df_output
      
      for(col in vector_featureX) {
        df_output_print[, col] = format(round(df_output_print[, col], 3), nsmall = 3)
      }

      for (index_col in 1:ncol(df_output_print)) {
        
        colName = colnames(df_output_print)[index_col]
        
        colName = gsub("CASCADEEND_RETWEETS_rootTweet_user_description_", "", colName)
        colName = gsub("CASCADEEND_RETWEETS_rootTweet_user_", "", colName)
        colName = gsub("CASCADEEND_RETWEETS_rootTweet_fullText_", "", colName)
        colName = gsub("CASCADEEND_RETWEETS_retweet_", "", colName)
        colName = gsub("CASCADEEND_RETWEETS_", "", colName)
        colName = gsub("_user_description", "", colName)
        colName = gsub("textStats_", "", colName)
        
        colnames(df_output_print)[index_col] <- colName
      }
      
      df_output_print$feature_y = gsub("CASCADEEND_RETWEETS_rootTweet_user_description_", "", df_output_print$feature_y)
      df_output_print$feature_y = gsub("CASCADEEND_RETWEETS_rootTweet_user_", "", df_output_print$feature_y)
      df_output_print$feature_y = gsub("CASCADEEND_RETWEETS_rootTweet_fullText_", "", df_output_print$feature_y)
      df_output_print$feature_y = gsub("CASCADEEND_RETWEETS_retweet_", "", df_output_print$feature_y)
      df_output_print$feature_y = gsub("CASCADEEND_RETWEETS_", "", df_output_print$feature_y)
      df_output_print$feature_y = gsub("_user_description", "", df_output_print$feature_y)
      df_output_print$feature_y = gsub("textStats_", "", df_output_print$feature_y)
      
      # df_output_print[is.na(df_output_print)] = ""
      df_output_print = sapply(df_output_print, function(x) {x = gsub("^\\s*NA", "", x)})
      
      absFilename_output_print = paste(
        path_output, "rm/prt/",
        "rg_prodSprd_var=",
        veracity,
        "_med=",
        method,
        "_rSE=",
        rSE,
        "_X=",
        str_name_X,
        "_Y=",
        str_name_Y,
        "_rm_prt",
        ".csv",
        sep = ""
      )
      
      print("index_task 2:")
      print(index_task)
      
      print("absFilename_output_print 2:")
      print(absFilename_output_print)
      
      write.table(
        df_output_print,
        absFilename_output_print,
        sep = ",",
        row.names = FALSE,
        col.names = TRUE,
        quote = TRUE
      )
      
    }
  }
}



print("program exits.")
