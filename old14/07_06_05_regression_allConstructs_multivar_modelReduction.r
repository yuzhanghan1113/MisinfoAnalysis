rm(list=ls())

library(data.table)
library(stringr)
library(tidyr)
library(lme4) # for the analysis
library(lmerTest)# to get p-value estimations that are not part of the standard lme4 packages
library(MuMIn)
library(miceadds)
library(digest)

options(max.print = 999999)
# options(error = traceback)
options(error = recover)

# method = "mxdEff"
method = "linRgr"

# rSE = "F"
rSE = "T"

# transformX = "T"
transformX = "F"
# transformY = "T"
transformY = "F"

mainXInteraction = "T"
# mainXInteraction = "F"

mainXInteraction_selectedTerms = "T"
# mainXInteraction_selectedTerms = "F"


modelingTask = "fullAndReduced"
# modelingTask = "full"
# modelingTask = "reduced"

# THRESHOLD_COEF = 0.01
THRESHOLD_COEF = -1

vector_columns_meta = c("veracity", "robustStandardErrors", "transformX", "transformY", "method", "mainXInteraction", "mainXInteraction_selectedTerms", "varGrp_X", "varGrp_Y", "num_singleTerms_sig", "num_interTerms_sig", "feature_y", "num_observations", "R2", "R2Adj")


absFilename_input = "C:/vmwareSharedFolder/TwitterDataAnalysis/results/results_20210209/preprocessedData.csv"
path_output = "C:/vmwareSharedFolder/TwitterDataAnalysis/results/dissertation/linearRegressionResults/multivariate/"

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
  "CASCADEEND_RETWEETS_cascadeSize",
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

vector_featuresToLog = c(
  "CASCADEEND_RETWEETS_rootTweet_user_followersCount",
  "CASCADEEND_RETWEETS_rootTweet_user_friendsCount",
  "CASCADEEND_RETWEETS_rootTweet_user_listedCount",
  "CASCADEEND_RETWEETS_rootTweet_user_favouritesCount",
  "CASCADEEND_RETWEETS_rootTweet_user_statusesCount",
  "CASCADEEND_RETWEETS_rootTweet_user_description_textStats_charCount",
  "CASCADEEND_RETWEETS_rootTweet_user_description_textStats_wordCount",
  "CASCADEEND_RETWEETS_rootTweet_user_description_textStats_uqWordCount",
  "CASCADEEND_RETWEETS_rootTweet_user_description_textStats_sentenceCount",
  "CASCADEEND_RETWEETS_rootTweet_favoriteCount",
  "CASCADEEND_RETWEETS_rootTweet_fullText_textStats_charCount",
  "CASCADEEND_RETWEETS_rootTweet_fullText_textStats_wordCount",
  "CASCADEEND_RETWEETS_rootTweet_fullText_textStats_uqWordCount",
  "CASCADEEND_RETWEETS_rootTweet_fullText_textStats_sentenceCount",
  "CASCADEEND_RETWEETS_cascadeSize",
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
  "CASCADEEND_RETWEETS_retweet_mean_user_description_textStats_charCount",
  "CASCADEEND_RETWEETS_retweet_median_user_description_textStats_charCount",
  "CASCADEEND_RETWEETS_retweet_mean_user_description_textStats_wordCount",
  "CASCADEEND_RETWEETS_retweet_median_user_description_textStats_wordCount",
  "CASCADEEND_RETWEETS_retweet_mean_user_description_textStats_uqWordCount",
  "CASCADEEND_RETWEETS_retweet_median_user_description_textStats_uqWordCount",
  "CASCADEEND_RETWEETS_retweet_mean_user_description_textStats_sentenceCount",
  "CASCADEEND_RETWEETS_retweet_median_user_description_textStats_sentenceCount"
)

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

# feature_randomEffect = "CASCADEEND_RETWEETS_rootTweet_user_screenName"
feature_randomEffect = "CASCADEEND_RETWEETS_rootTweet_user_screenName"
feature_x_clusterOn_org = "CASCADEEND_RETWEETS_rootTweet_user_screenName"
feature_x_clusterOn = "CASCADEEND_RETWEETS_rootTweet_user_screenName_int"

regression = function(feature_y, vector_featureX, df_input) {
  
  feature_y_formula = "wrongY"
  if (transformY == "T" & feature_y %in% vector_featuresToLog){
    feature_y_formula = paste("sqrt(", feature_y, ")", sep="")
  }
  else {
    feature_y_formula = feature_y
  }
  
  num_indVarInFormula = 0
  
  str_formula = paste(feature_y_formula,
                      " ~ ",
                      sep = "")
  
  feature_x_formula = "wrongX"
  if (transformX == "T" & vector_featureX[1] %in% vector_featuresToLog){
    feature_x_formula = paste("sqrt(", vector_featureX[1], ")", sep="")
  }
  else {
    feature_x_formula = vector_featureX[1]
  }
  
  str_formula = paste(str_formula,
                      feature_x_formula,
                      sep = "")
  num_indVarInFormula = num_indVarInFormula + 1
  
  if(length(vector_featureX) >= 2) {
    for (feature_x in vector_featureX[2:length(vector_featureX)]) {
      
      feature_x_formula = "wrongX"
      if (transformX == "T" & feature_x %in% vector_featuresToLog){
        feature_x_formula = paste("sqrt(", feature_x, ")", sep="")
      }
      else {
        feature_x_formula = feature_x
      }
      str_formula = paste(str_formula,
                          " + ",
                          feature_x_formula,
                          sep = "")
      num_indVarInFormula = num_indVarInFormula + 1
    }
  }
  
  # if (mainXInteraction == "T") {
  #   for (feature_x1 in vector_featureX) {
  #     for (feature_x2 in vector_featureX) {
  #       # if (feature_x1!=feature_x2 & feature_x1%in%vector_featureXPool_main & feature_x2%in%vector_featureXPool_main) {
  #       if (feature_x1!=feature_x2) {
  #         
  #         if (mainXInteraction_selectedTerms == "T") {
  #           str_term = paste(feature_x1, ":", feature_x2, sep="")
  #           if (!str_term%in%vector_interTerms_selected) {
  #             next
  #           }
  #         }
  #         
  #         str_formula = paste(str_formula,
  #                             " + ",
  #                             feature_x1,
  #                             "*",
  #                             feature_x2,
  #                             sep = "")
  #         num_indVarInFormula = num_indVarInFormula + 1
  #       }
  #     }
  #   }
  # }
  
  if (method == "mxdEff") {
    str_formula = paste(str_formula,
                        " + (1|",
                        feature_randomEffect,
                        ")",
                        sep = "")
  }
  
  print("str_formula:")
  print(str_formula)
  
  if (method == "mxdEff") {
    model_regression = lmer(str_formula, data = df_input_toAnalyze, REML = FALSE)
    modelingResult = summary(model_regression)
    df_modelingResults = modelingResult$coefficients
  } else if (method == "linRgr") {
    if (rSE == "T") {
      model_regression = miceadds::lm.cluster(formula=str_formula, data=df_input_toAnalyze, cluster=feature_x_clusterOn)
      df_modelingResults = summary(model_regression)
      
      lm_res = stats::lm(formula=str_formula, data=df_input_toAnalyze, weights=NULL)
      RSquared = summary(lm_res)$r.squared
    } else if (rSE == "F") {
      model_regression = lm(formula=str_formula, data=df_input_toAnalyze)
      modelingResult = summary(model_regression)
      df_modelingResults = modelingResult$coefficients
    }
  }
  
  return(df_modelingResults)
}






df_input[, feature_x_clusterOn] = digest2int(df_input[, feature_x_clusterOn_org])

# table(df_input[, feature_x_clusterOn_org])
# table(df_input[, feature_x_clusterOn])
# df_input[, c(feature_x_clusterOn_org, feature_x_clusterOn)]

dir.create(file.path(path_output, "fm"))
dir.create(file.path(paste(path_output, "fm", sep=""), "cpl"))
dir.create(file.path(path_output, "rm"))
dir.create(file.path(paste(path_output, "rm", sep=""), "prt"))

vector_veracities = c("mis", "aut")
# vector_veracities = c("mis")

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
    
    for (index_task in 1:5) {
      print("index_task:")
      print(index_task)
      
      df_output = data.frame()
      df_output_complete = data.frame()
      
      if (index_task == 1) {
        vector_featureX = c(vector_producerFeatures_user, vector_producerFeatures_userDescription)
        vector_featureXPool_main = c(vector_producerFeatures_user, vector_producerFeatures_userDescription)
        str_name_X = "producer"
        vector_featureY = vector_rootTweetFeatures
        str_name_Y = "rootTweetMeta"
        vector_interTerms_selected = c("CASCADEEND_RETWEETS_rootTweet_user_followersCount:CASCADEEND_RETWEETS_rootTweet_user_description_emotion_fear", "CASCADEEND_RETWEETS_rootTweet_user_statusesCount:CASCADEEND_RETWEETS_rootTweet_user_description_emotion_fear")
      } else if (index_task == 2) {
        vector_featureX = c(vector_producerFeatures_user, vector_producerFeatures_userDescription)
        vector_featureXPool_main = c(vector_producerFeatures_user, vector_producerFeatures_userDescription)
        str_name_X = "producer"
        vector_featureY = vector_rootTweetFeatures_fullText
        str_name_Y = "rootTweetText"
        vector_interTerms_selected = c("CASCADEEND_RETWEETS_rootTweet_user_statusesCount:CASCADEEND_RETWEETS_rootTweet_user_description_sentiment_neg", "CASCADEEND_RETWEETS_rootTweet_user_description_sentiment_neg:CASCADEEND_RETWEETS_rootTweet_user_description_textStats_wordCount", "CASCADEEND_RETWEETS_rootTweet_user_description_emotion_sadness:CASCADEEND_RETWEETS_rootTweet_user_description_textStats_wordCount")
      } else if (index_task == 3) {
        vector_featureX = c(vector_rootTweetFeatures_fullText, vector_producerFeatures_user, vector_producerFeatures_userDescription)
        vector_featureXPool_main = c(vector_rootTweetFeatures_fullText, vector_producerFeatures_user, vector_producerFeatures_userDescription)
        vector_featureX = c(vector_rootTweetFeatures_fullText, vector_producerFeatures_user, vector_producerFeatures_userDescription)
        str_name_X = "rootTweetTextCtlProducer"
        vector_featureY = vector_rootTweetFeatures
        str_name_Y = "rootTweetMeta"
        vector_interTerms_selected = c("CASCADEEND_RETWEETS_rootTweet_fullText_sentiment_neg:CASCADEEND_RETWEETS_rootTweet_fullText_emotion_fear", "CASCADEEND_RETWEETS_rootTweet_fullText_subjectivity:CASCADEEND_RETWEETS_rootTweet_fullText_emotion_sadness", "CASCADEEND_RETWEETS_rootTweet_fullText_sentiment_neg:CASCADEEND_RETWEETS_rootTweet_fullText_subjectivity", "CASCADEEND_RETWEETS_rootTweet_user_followersCount:CASCADEEND_RETWEETS_rootTweet_user_listedCount", "CASCADEEND_RETWEETS_rootTweet_user_listedCount:CASCADEEND_RETWEETS_rootTweet_user_description_emotion_fear", "CASCADEEND_RETWEETS_rootTweet_user_followersCount:CASCADEEND_RETWEETS_rootTweet_user_description_emotion_fear")
      } else if (index_task == 4) {
        vector_featureX = c(vector_rootTweetFeatures, vector_rootTweetFeatures_fullText, vector_producerFeatures_user, vector_producerFeatures_userDescription)
        str_name_X = "rootTweetCtlProducer"
        vector_featureY = vector_retweeterFeatures_user
        str_name_Y = "retweeterMeta"
        vector_interTerms_selected = c("CASCADEEND_RETWEETS_rootTweet_fullText_sentiment_neg:CASCADEEND_RETWEETS_rootTweet_fullText_emotion_fear", "CASCADEEND_RETWEETS_rootTweet_fullText_subjectivity:CASCADEEND_RETWEETS_rootTweet_fullText_emotion_sadness", "CASCADEEND_RETWEETS_rootTweet_favoriteCount:CASCADEEND_RETWEETS_rootTweet_fullText_sentiment_neg", "CASCADEEND_RETWEETS_rootTweet_user_followersCount:CASCADEEND_RETWEETS_rootTweet_user_description_emotion_joy", "CASCADEEND_RETWEETS_rootTweet_user_listedCount:CASCADEEND_RETWEETS_rootTweet_user_description_textStats_charsPerWord", "CASCADEEND_RETWEETS_rootTweet_user_followersCount:CASCADEEND_RETWEETS_rootTweet_user_description_textStats_charsPerWord")
      } else if (index_task == 5) {
        vector_featureX = c(vector_rootTweetFeatures, vector_rootTweetFeatures_fullText, vector_producerFeatures_user, vector_producerFeatures_userDescription)
        vector_featureXPool_main = c(vector_rootTweetFeatures, vector_rootTweetFeatures_fullText)
        str_name_X = "rootTweetCtlProducer"
        vector_featureY = vector_retweeterFeatures_userDescription
        str_name_Y = "retweeterUserDesc"
        vector_interTerms_selected = c("CASCADEEND_RETWEETS_rootTweet_fullText_sentiment_neg:CASCADEEND_RETWEETS_rootTweet_fullText_subjectivity", "CASCADEEND_RETWEETS_rootTweet_favoriteCount:CASCADEEND_RETWEETS_rootTweet_fullText_textStats_uqWordCount", "CASCADEEND_RETWEETS_rootTweet_fullText_emotion_trust:CASCADEEND_RETWEETS_rootTweet_fullText_textStats_sentenceCount", "CASCADEEND_RETWEETS_rootTweet_user_description_sentiment_neg:CASCADEEND_RETWEETS_rootTweet_user_description_textStats_charsPerWord", "CASCADEEND_RETWEETS_rootTweet_user_description_sentiment_neu:CASCADEEND_RETWEETS_rootTweet_user_description_textStats_charsPerWord", "CASCADEEND_RETWEETS_rootTweet_user_description_sentiment_pos:CASCADEEND_RETWEETS_rootTweet_user_description_textStats_charsPerWord")
      } else {
        vector_featureX = c("WrongX")
        str_name_X = "WrongX"
        vector_featureY = c("WrongY")
        str_name_Y = "WrongY"
        vector_interTerms_selected = c("WrongTerms")
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
      
      vector_featureX = c(vector_featureX, feature_x_clusterOn)
      
      for (feature_y in vector_featureY) {
        
        tryCatch({
          
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
        
          
          ## call function
          df_modelingResults = regression(feature_y, vector_featureX, df_input)
          # print("df_modelingResults:")
          # print(df_modelingResults)
          
          # result_RSquared = r.squaredGLMM(model_regression)
          # RSquared_marginal = result_RSquared[[1, "R2m"]]
          # RSquared_conditional = result_RSquared[[1, "R2c"]]
          
          df_output[index_row, "method"] = method
          df_output[index_row, "robustStandardErrors"] = rSE
          df_output[index_row, "transformX"] = transformX
          df_output[index_row, "transformY"] = transformY
          df_output[index_row, "mainXInteraction"] = mainXInteraction
          df_output[index_row, "mainXInteraction_selectedTerms"] = mainXInteraction_selectedTerms
          df_output[index_row, "veracity"] = veracity
          df_output[index_row, "varGrp_X"] = str_name_X
          df_output[index_row, "varGrp_Y"] = str_name_Y
          
          df_output_complete[index_row, "method"] = method
          df_output_complete[index_row, "robustStandardErrors"] = rSE
          df_output_complete[index_row, "transformX"] = transformX
          df_output_complete[index_row, "transformY"] = transformY
          df_output_complete[index_row, "mainXInteraction"] = mainXInteraction
          df_output_complete[index_row, "mainXInteraction_selectedTerms"] = mainXInteraction_selectedTerms
          df_output_complete[index_row, "veracity"] = veracity
          df_output_complete[index_row, "varGrp_X"] = str_name_X
          df_output_complete[index_row, "varGrp_Y"] = str_name_Y
          
          df_output[index_row, "feature_y"] = feature_y
          df_output_complete[index_row, "feature_y"] = feature_y
          
          num_observations = nrow(df_input_toAnalyze[, c(vector_featureX, feature_y, feature_randomEffect)]) - sum(apply(df_input_toAnalyze[, c(vector_featureX, feature_y, feature_randomEffect)], 1, anyNA))
          df_output[index_row, "num_observations"] = num_observations
          df_output_complete[index_row, "num_observations"] = num_observations
          
          num_singleTerms_sig = 0
          
          for (feature_x in vector_featureX) {
            if (feature_x %in% rownames(df_modelingResults)) {
              regression_coef = df_modelingResults[feature_x, "Estimate"]
              regression_p = df_modelingResults[feature_x, "Pr(>|t|)"]
              
              if (!is.nan(regression_p) & regression_p < 0.05 &
                  !is.nan(regression_coef) & abs(regression_coef) >= THRESHOLD_COEF) {
                num_singleTerms_sig = num_singleTerms_sig + 1
              }
            }
          }
          
          df_output[index_row, "num_singleTerms_sig"] = num_singleTerms_sig
          df_output_complete[index_row, "num_singleTerms_sig"] = num_singleTerms_sig
          
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
      
      df_output = df_output[df_output$num_singleTerms_sig > 0, ]
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

print("program exits.")


