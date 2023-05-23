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

vector_columns_meta = c("veracity", "robustStandardErrors", "transformX", "transformY", "method", "mainXInteraction", "mainXInteraction_selectedTerms", "varGrp_X", "varGrp_Y", "num_sigMainVar_singleTerm", "num_sigMainVar_interTerms", "num_sigCtrlVar_singleTerm", "feature_y", "num_observations", "R2", "R2Adj")


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
vector_rootTweetFeatures_meta = c(
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

df_input[, feature_x_clusterOn] = digest2int(df_input[, feature_x_clusterOn_org])

# table(df_input[, feature_x_clusterOn_org])
# table(df_input[, feature_x_clusterOn])
# df_input[, c(feature_x_clusterOn_org, feature_x_clusterOn)]

dir.create(file.path(path_output, "fm"))
dir.create(file.path(paste(path_output, "fm", sep=""), "cpl"))
dir.create(file.path(path_output, "rm"))
dir.create(file.path(paste(path_output, "rm", sep=""), "prt"))


regression = function(feature_y, vector_featureX, df_input_toAnalyze) {
  
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
  
  if (mainXInteraction == "T") {
    for (interTerm in vector_interTerms_selected) {

      str_formula = paste(str_formula,
                          " + ",
                          interTerm,
                          sep = "")
      num_indVarInFormula = num_indVarInFormula + 1
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
  
  df_modelingResults = data.frame()
  RSquared = -999
  
  if (method == "mxdEff") {
    model_regression = lmer(str_formula, data = df_input_toAnalyze, REML = FALSE)
    modelingResult = summary(model_regression)
    df_modelingResults = modelingResult$coefficients
  } else if (method == "linRgr") {
    if (rSE == "T") {
      model_regression = miceadds::lm.cluster(formula=str_formula, data=df_input_toAnalyze, cluster=feature_x_clusterOn)
      df_modelingResults = summary(model_regression)

      # lm_res = stats::lm(formula=str_formula, data=df_input_toAnalyze, weights=NULL)
      
      lm_res = stats::lm(formula=str_formula, data=df_input_toAnalyze)
      RSquared = summary(lm_res)$r.squared
      
    } else if (rSE == "F") {
      model_regression = lm(formula=str_formula, data=df_input_toAnalyze)
      modelingResult = summary(model_regression)
      df_modelingResults = modelingResult$coefficients
    }
  }
  
  print("RSquared:")
  print(RSquared)
  
  list_results = list("df_modelingResults"=df_modelingResults, "RSquared"=RSquared, "num_indVarInFormula"=num_indVarInFormula)
  
  return(list_results)
}

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
    
    # for (index_task in 6:7) {
    # for (index_task in 1:5) {
    for (index_task in 1:9) {
        
      print("index_task:")
      print(index_task)
      
      df_output = data.frame()
      df_output_complete = data.frame()
      
      if (index_task == 1) {
        vector_featureX_main = c(vector_producerFeatures_user, vector_producerFeatures_userDescription)
        vector_featureX_control = c()
        vector_featureX = c(vector_featureX_main, vector_featureX_control)
        str_name_X = "producer"
        vector_featureY = vector_rootTweetFeatures_meta
        str_name_Y = "rootTweetMeta"
      } else if (index_task == 2) {
        vector_featureX_main = c(vector_producerFeatures_user, vector_producerFeatures_userDescription)
        vector_featureX_control = c()
        vector_featureX = c(vector_featureX_main, vector_featureX_control)
        str_name_X = "producer"
        vector_featureY = vector_rootTweetFeatures_fullText
        str_name_Y = "rootTweetText"
      } else if (index_task == 3) {
        vector_featureX_main = c(vector_rootTweetFeatures_fullText)
        vector_featureX_control = c(vector_producerFeatures_user, vector_producerFeatures_userDescription)
        vector_featureX = c(vector_featureX_main, vector_featureX_control)
        str_name_X = "rootTweetTextCtlProducer"
        vector_featureY = vector_rootTweetFeatures_meta
        str_name_Y = "rootTweetMeta"
      } else if (index_task == 4) {
        vector_featureX_main = c(vector_rootTweetFeatures_meta, vector_rootTweetFeatures_fullText)
        vector_featureX_control = c(vector_producerFeatures_user, vector_producerFeatures_userDescription)
        vector_featureX = c(vector_featureX_main, vector_featureX_control)
        str_name_X = "rootTweetCtlProducer"
        vector_featureY = vector_retweeterFeatures_user
        str_name_Y = "retweeterMeta"
      } else if (index_task == 5) {
        vector_featureX_main = c(vector_rootTweetFeatures_meta, vector_rootTweetFeatures_fullText)
        vector_featureX_control = c(vector_producerFeatures_user, vector_producerFeatures_userDescription)
        vector_featureX = c(vector_featureX_main, vector_featureX_control)
        str_name_X = "rootTweetCtlProducer"
        vector_featureY = vector_retweeterFeatures_userDescription
        str_name_Y = "retweeterUserDesc"
      } else if (index_task == 6) {
        vector_featureX_main = c(vector_producerIntention)
        vector_featureX_control = c(vector_producerFeatures_user, vector_producerFeatures_userDescription)
        vector_featureX = c(vector_featureX_main, vector_featureX_control)
        str_name_X = "producerIntentionCtlProducer"
        vector_featureY = vector_rootTweetFeatures_meta
        str_name_Y = "rootTweetMeta"
      } else if (index_task == 7) {
        vector_featureX_main = c(vector_producerIntention)
        vector_featureX_control = c(vector_producerFeatures_user, vector_producerFeatures_userDescription)
        vector_featureX = c(vector_featureX_main, vector_featureX_control)
        str_name_X = "producerIntentionCtlProducer"
        vector_featureY = vector_rootTweetFeatures_fullText
        str_name_Y = "rootTweetText"
      } else if (index_task == 8) {
        vector_featureX_main = c(vector_producerIntention)
        vector_featureX_control = c(vector_producerFeatures_user, vector_producerFeatures_userDescription, vector_rootTweetFeatures_meta, vector_rootTweetFeatures_fullText)
        vector_featureX = c(vector_featureX_main, vector_featureX_control)
        str_name_X = "producerIntentionCtlProducerRootTweet"
        vector_featureY = vector_retweeterFeatures_user
        str_name_Y = "retweeterMeta"
      } else if (index_task == 9) {
        vector_featureX_main = c(vector_producerIntention)
        vector_featureX_control = c(vector_producerFeatures_user, vector_producerFeatures_userDescription, vector_rootTweetFeatures_meta, vector_rootTweetFeatures_fullText)
        vector_featureX = c(vector_featureX_main, vector_featureX_control)
        str_name_X = "producerIntentionCtlProducerRootTweet"
        vector_featureY = vector_retweeterFeatures_userDescription
        str_name_Y = "retweeterUserDesc"
      } else {
        vector_featureX = c("WrongX")
        str_name_X = "WrongX"
        vector_featureY = c("WrongY")
        str_name_Y = "WrongY"
      }
      if (index_task %in% c(1, 2, 3, 4, 5)) {
        vector_interTerms_selected = c()
      } else if (index_task %in% c(6, 7, 8, 9)) {
        
        vector_interTerms_selected = c()
        
        for (index_feature1 in 1:(length(vector_producerIntention)-1)) {
          for (index_feature2  in (index_feature1+1):length(vector_producerIntention)) {
            vector_interTerms_selected = c(vector_interTerms_selected, paste(vector_producerIntention[index_feature1], ":", vector_producerIntention[index_feature2], sep=""))
          }
        }
        # print(length(vector_interTerms_selected))
      }
      
      absFilename_output = paste(
        path_output,
        "rm/",
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
      
      vector_featureX = c(vector_featureX, feature_x_clusterOn)
      
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
        
        tryCatch({
          
          # df_modelingResults = regression(feature_y, vector_featureX, df_input_toAnalyze)
          # df_modelingResults = regression1(feature_y, vector_featureX, df_input_toAnalyze)
          list_results = regression(feature_y, vector_featureX, df_input_toAnalyze)
          df_modelingResults = list_results[["df_modelingResults"]]
            
          # print("df_modelingResults:")
          # print(df_modelingResults)
          
          # result_RSquared = r.squaredGLMM(model_regression)
          # RSquared_marginal = result_RSquared[[1, "R2m"]]
          # RSquared_conditional = result_RSquared[[1, "R2c"]]
          
          df_output[index_row, "method"] = method
          df_output[index_row, "robustStandardErrors"] = rSE
          df_output[index_row, "transformX"] = transformX
          df_output[index_row, "transformY"] = transformY
          df_output[index_row, "veracity"] = veracity
          df_output[index_row, "mainXInteraction"] = mainXInteraction
          df_output[index_row, "mainXInteraction_selectedTerms"] = mainXInteraction_selectedTerms
          df_output[index_row, "varGrp_X"] = str_name_X
          df_output[index_row, "varGrp_Y"] = str_name_Y
          df_output[index_row, "feature_y"] = feature_y
          
          num_observations = nrow(df_input_toAnalyze[, c(vector_featureX, feature_y, feature_randomEffect)]) - sum(apply(df_input_toAnalyze[, c(vector_featureX, feature_y, feature_randomEffect)], 1, anyNA))
          df_output[index_row, "num_observations"] = num_observations
            
          RSquared = list_results[["RSquared"]]
          num_indVarInFormula = list_results[["num_indVarInFormula"]]
          RSquared_adjusted = 1 - (1-RSquared)*(num_observations-1)/(num_observations-num_indVarInFormula-1)

          df_output[index_row, "R2"] = RSquared
          df_output[index_row, "R2Adj"] = RSquared_adjusted
          
          num_sigMainVar_singleTerm = 0
          num_sigCtrlVar_singleTerm = 0
          
          for (feature_x in vector_featureX) {
            
            if (feature_x %in% rownames(df_modelingResults)) {
              
              regression_coef = df_modelingResults[feature_x, "Estimate"]
              regression_p = df_modelingResults[feature_x, "Pr(>|t|)"]
              
              if (!is.nan(regression_p) & regression_p < 0.05 & !is.nan(regression_coef) & abs(regression_coef) >= THRESHOLD_COEF) {
                
                if (feature_x %in% vector_featureX_main) {
                  num_sigMainVar_singleTerm = num_sigMainVar_singleTerm + 1
                } else if (feature_x %in% vector_featureX_control) {
                  num_sigCtrlVar_singleTerm = num_sigCtrlVar_singleTerm + 1
                }
              }
            }
          }
          
          df_output[index_row, "num_sigMainVar_singleTerm"] = num_sigMainVar_singleTerm
        
          num_sigMainVar_interTerms = 0
          
          if (mainXInteraction == "T") {
            for (str_term in vector_interTerms_selected) {
                  
              if (str_term %in% rownames(df_modelingResults)) {
                
                regression_coef = df_modelingResults[str_term, "Estimate"]
                regression_p = df_modelingResults[str_term, "Pr(>|t|)"]
                regression_SE = df_modelingResults[str_term, "Std. Error"]
                
                if (!is.nan(regression_p) & regression_p < 0.05 & !is.nan(regression_coef) & abs(regression_coef) >= THRESHOLD_COEF) {
                  num_sigMainVar_interTerms = num_sigMainVar_interTerms + 1
                }
              }
            }
          }
          
          df_output[index_row, "num_sigMainVar_interTerms"] = num_sigMainVar_interTerms
          
          df_output[index_row, "num_sigCtrlVar_singleTerm"] = num_sigCtrlVar_singleTerm
          
          
          for (feature_x in vector_featureX) {
            
            df_output[index_row, feature_x] = NA
            df_output[index_row, paste("p_", feature_x, sep="")] = NA
            
            if (feature_x %in% rownames(df_modelingResults)) {
              print("regression runnable")
              
              regression_coef = df_modelingResults[feature_x, "Estimate"]
              regression_p = df_modelingResults[feature_x, "Pr(>|t|)"]
              regression_SE = df_modelingResults[feature_x, "Std. Error"]
              
              if (!is.nan(regression_coef)) {
                df_output[index_row, feature_x] = regression_coef
              }
              if (!is.nan(regression_p)) {
                df_output[index_row, paste("p_", feature_x, sep="")] = regression_p
              }
              if (!is.nan(regression_SE)) {
                df_output[index_row, paste("SE_", feature_x, sep="")] = regression_SE
              }
            }
          }
          
          
          if (mainXInteraction == "T") {
            for (feature_x1 in vector_featureX) {
              for (feature_x2 in vector_featureX) {
                # if (feature_x1!=feature_x2 & feature_x1%in%vector_featureXPool_main & feature_x2%in%vector_featureXPool_main) {
                if (feature_x1!=feature_x2) {
                  
                  str_term = paste(feature_x1, ":", feature_x2, sep="")
                  
                  if (mainXInteraction_selectedTerms == "T") {
                    if (!str_term%in%vector_interTerms_selected) {
                      next
                    }
                  }
                  
                  df_output[index_row, str_term] = NA
                  df_output[index_row, paste("p_", str_term, sep="")] = NA
                  df_output[index_row, paste("SE_", str_term, sep="")] = NA
                  
                  if (str_term %in% rownames(df_modelingResults)) {
                    print("regression runnable")
                    
                    regression_coef = df_modelingResults[str_term, "Estimate"]
                    regression_p = df_modelingResults[str_term, "Pr(>|t|)"]
                    regression_SE = df_modelingResults[str_term, "Std. Error"]
                    
                    if (!is.nan(regression_coef)) {
                      df_output[index_row, str_term] = regression_coef
                    }
                    if (!is.nan(regression_p)) {
                      df_output[index_row, paste("p_", str_term, sep="")] = regression_p
                    }
                    if (!is.nan(regression_SE)) {
                      df_output[index_row, paste("SE_", str_term, sep="")] = regression_SE
                    }
                  }
                }
              }
            }
          }
          
          print("index_task 1:")
          print(index_task)
          
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
      
      
      
      df_output = df_output[df_output$num_sigMainVar_singleTerm+df_output$num_sigMainVar_interTerms > 0, ]
      # df_output = df_output[, colSums(is.na(df_output)) < nrow(df_output)]
      
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
      
      # if (nrow(df_output) <= 0) {
      #   file.remove(absFilename_output)
      #   print("No row in output. File deleted.")
      # }
      
      print("vector_featureX:")
      print(vector_featureX)
      
      # df_output_print = df_output
      df_output_print = data.frame()
      
      for(index_row in rownames(df_output)) {
        for(index_col in vector_columns_meta) {
          if (index_col %in% colnames(df_output)) {
            df_output_print[index_row, index_col] = df_output[index_row, index_col]
          }
        }
        
        # R2_org = as.numeric(df_output_print[index_row, "R2"])
        # df_output_print[index_row, "R2"] = format(round(R2_org, 3), nsmall = 3)
        # 
        # R2Adj_org = as.numeric(df_output_print[index_row, "R2Adj"])
        # df_output_print[index_row, "R2Adj"] = format(round(R2Adj_org, 3), nsmall = 3)
      }
      
      vector_columns_all = colnames(df_output)
      vector_columns_values = vector_columns_all[!(vector_columns_all %in% vector_columns_meta)]
      
      for(index_col in vector_columns_values) {
        
        if(startsWith(index_col, "p_") | startsWith(index_col, "SE_")) {
          next
        }
        
        for(index_row in rownames(df_output)){
          
          coef_org = as.numeric(df_output[index_row, index_col])
          
          str_value = ""
          
          if (!is.nan(coef_org) & !is.na(coef_org)) {
            
            num_digits = 3
            
            while (TRUE) {
              coef_rounded = round(coef_org, num_digits)
              
              if (abs(coef_rounded) > 0) {
                str_value = as.character(coef_rounded)
                break
              }
              else {
                num_digits = num_digits + 1
              }
            }
            
            # p = as.numeric(df_output[index_row, paste("p_", index_col, sep="")])
            p = df_output[index_row, paste("p_", index_col, sep="")]
            str_surfix = ""
            
            # if (!is.nan(p)) {
            if (!is.null(p)) {
              
                p = as.numeric(p)
                
                if (p < 0.001) {
                str_surfix = "***"
              } else if (p < 0.01) {
                str_surfix = "**"
              } else if (p < 0.05) {
                str_surfix = "*"
              }
            }
            
            str_value = paste("X", str_value, str_surfix, sep="")
            
            # SE_org = as.numeric(df_output[index_row, paste("SE_", index_col, sep="")])
            SE_org = df_output[index_row, paste("SE_", index_col, sep="")]
            
            # if (!is.nan(SE_org) & !is.na(SE_org)) {
            if (!is.null(SE_org)) {
              
              SE_org = as.numeric(SE_org)

              SE_rounded = format(round(SE_org, 3), nsmall = 3)
              str_value = paste(str_value, " (", SE_rounded, ")", sep="")
            }
          }
          
          df_output_print[index_row, index_col] = str_value
        }
      }
      
      # for (index_col in 1:ncol(df_output_print)) {
      #   
      #   colName = colnames(df_output_print)[index_col]
      #   
      #   colName = gsub("CASCADEEND_RETWEETS_rootTweet_user_description_", "proDes_", colName)
      #   colName = gsub("CASCADEEND_RETWEETS_rootTweet_user_", "proMet_", colName)
      #   colName = gsub("CASCADEEND_RETWEETS_rootTweet_fullText_", "rooTex_", colName)
      #   colName = gsub("CASCADEEND_RETWEETS_rootTweet_", "rooMet_", colName)
      #   colName = gsub("CASCADEEND_RETWEETS_retweet_", "ret_", colName)
      #   colName = gsub("CASCADEEND_RETWEETS_", "RET_", colName)
      #   # colName = gsub("_user_description", "", colName)
      #   colName = gsub("textStats_", "", colName)
      #   
      #   colName = gsub("followers", "fol", colName)
      #   colName = gsub("friends", "fri", colName)
      #   colName = gsub("listed", "lis", colName)
      #   colName = gsub("favorite", "fav", colName)
      #   colName = gsub("favourites", "fav", colName)
      #   colName = gsub("statuses", "sta", colName)
      #   colName = gsub("emotion", "emo", colName)
      #   colName = gsub("sentiment", "sen", colName)
      #   colName = gsub("Count", "Cnt", colName)
      #   
      #   colName = gsub("subjectivity", "sub", colName)
      #   colName = gsub("_trust", "Tru", colName)
      #   colName = gsub("_positive", "Pos", colName)
      #   colName = gsub("_negative", "Neg", colName)
      #   colName = gsub("_compound", "Com", colName)
      #   colName = gsub("_sadness", "Sad", colName)
      #   colName = gsub("_disgust", "Dis", colName)
      #   colName = gsub("_fear", "Fea", colName)
      #   colName = gsub("_anger", "Ang", colName)
      #   colName = gsub("_surprise", "Sur", colName)
      #   colName = gsub("_joy", "Joy", colName)
      #   colName = gsub("_neg", "Neg", colName)
      #   colName = gsub("_neu", "Neu", colName)
      #   colName = gsub("_pos", "Pos", colName)
      #   colName = gsub("uqWord", "uniWor", colName)
      #   colName = gsub("wordsPerSentence", "worSen", colName)
      #   colName = gsub("word", "wor", colName)
      #   colName = gsub("sentence", "sen", colName)
      #   colName = gsub("charsPerWord", "chaWor", colName)
      #   colName = gsub("readability", "rea", colName)
      #   colName = gsub("cascadeSize", "casSiz", colName)
      #   colName = gsub("median", "med", colName)
      #   colName = gsub("mean", "avg", colName)
      #   colName = gsub("_user", "_usr", colName)
      #   colName = gsub("accountAge_day", "accAge", colName)
      #   colName = gsub("geoEnabled", "geoEna", colName)
      #   colName = gsub("verified", "ver", colName)
      #   colName = gsub("char", "cha", colName)
      #   
      #   colName = gsub(":", " * ", colName)
      #   
      #   
      #   colnames(df_output_print)[index_col] <- colName
      # }
      # 
      # df_output_print$feature_y = gsub("CASCADEEND_RETWEETS_rootTweet_user_description_", "", df_output_print$feature_y)
      # df_output_print$feature_y = gsub("CASCADEEND_RETWEETS_rootTweet_user_", "", df_output_print$feature_y)
      # df_output_print$feature_y = gsub("CASCADEEND_RETWEETS_rootTweet_fullText_", "", df_output_print$feature_y)
      # df_output_print$feature_y = gsub("CASCADEEND_RETWEETS_rootTweet_", "", df_output_print$feature_y)
      # df_output_print$feature_y = gsub("CASCADEEND_RETWEETS_retweet", "", df_output_print$feature_y)
      # df_output_print$feature_y = gsub("CASCADEEND_RETWEETS_", "", df_output_print$feature_y)
      # df_output_print$feature_y = gsub("_user_description", "", df_output_print$feature_y)
      # df_output_print$feature_y = gsub("textStats_", "", df_output_print$feature_y)
      # 
      # df_output_print$feature_y = gsub("followers", "fol", df_output_print$feature_y)
      # df_output_print$feature_y = gsub("friends", "fri", df_output_print$feature_y)
      # df_output_print$feature_y = gsub("listed", "lis", df_output_print$feature_y)
      # df_output_print$feature_y = gsub("favorite", "fav", df_output_print$feature_y)
      # df_output_print$feature_y = gsub("favourites", "fav", df_output_print$feature_y)
      # df_output_print$feature_y = gsub("statuses", "sta", df_output_print$feature_y)
      # df_output_print$feature_y = gsub("emotion", "emo", df_output_print$feature_y)
      # df_output_print$feature_y = gsub("sentiment", "sen", df_output_print$feature_y)
      # df_output_print$feature_y = gsub("Count", "Cnt", df_output_print$feature_y)
      # 
      # df_output_print$feature_y = gsub("subjectivity", "sub", df_output_print$feature_y)
      # df_output_print$feature_y = gsub("_trust", "Tru", df_output_print$feature_y)
      # df_output_print$feature_y = gsub("_positive", "Pos", df_output_print$feature_y)
      # df_output_print$feature_y = gsub("_negative", "Neg", df_output_print$feature_y)
      # df_output_print$feature_y = gsub("_compound", "Com", df_output_print$feature_y)
      # df_output_print$feature_y = gsub("_sadness", "Sad", df_output_print$feature_y)
      # df_output_print$feature_y = gsub("_disgust", "Dis", df_output_print$feature_y)
      # df_output_print$feature_y = gsub("_fear", "Fea", df_output_print$feature_y)
      # df_output_print$feature_y = gsub("_anger", "Ang", df_output_print$feature_y)
      # df_output_print$feature_y = gsub("_surprise", "Sur", df_output_print$feature_y)
      # df_output_print$feature_y = gsub("_joy", "Joy", df_output_print$feature_y)
      # df_output_print$feature_y = gsub("_neg", "Neg", df_output_print$feature_y)
      # df_output_print$feature_y = gsub("_neu", "Neu", df_output_print$feature_y)
      # df_output_print$feature_y = gsub("_pos", "Pos", df_output_print$feature_y)
      # df_output_print$feature_y = gsub("uqWord", "uniWor", df_output_print$feature_y)
      # df_output_print$feature_y = gsub("wordsPerSentence", "worSen", df_output_print$feature_y)
      # df_output_print$feature_y = gsub("word", "wor", df_output_print$feature_y)
      # df_output_print$feature_y = gsub("sentence", "sen", df_output_print$feature_y)
      # df_output_print$feature_y = gsub("charsPerWord", "chaWor", df_output_print$feature_y)
      # df_output_print$feature_y = gsub("readability", "rea", df_output_print$feature_y)
      # df_output_print$feature_y = gsub("cascadeSize", "casSiz", df_output_print$feature_y)
      # df_output_print$feature_y = gsub("_ptg_user_", "ptg.", df_output_print$feature_y)
      # df_output_print$feature_y = gsub("_median_user_", "med.", df_output_print$feature_y)
      # df_output_print$feature_y = gsub("_mean_user_", "avg.", df_output_print$feature_y)
      # df_output_print$feature_y = gsub("_median_", "med.", df_output_print$feature_y)
      # df_output_print$feature_y = gsub("_mean_", "avg.", df_output_print$feature_y)
      # df_output_print$feature_y = gsub("accountAge_day", "accAge", df_output_print$feature_y)
      # df_output_print$feature_y = gsub("geoEnabled", "geoEna", df_output_print$feature_y)
      # df_output_print$feature_y = gsub("verified", "ver", df_output_print$feature_y)
      # df_output_print$feature_y = gsub("char", "cha", df_output_print$feature_y)
      
      
      # df_output_print[is.na(df_output_print)] = ""
      # df_output_print = sapply(df_output_print, function(x) {x = gsub("^\\s*NA", "", x)})
      
      vector_columns_exist = c()
      for(col in c(vector_columns_meta, vector_featureX_main, vector_interTerms_selected, vector_featureX_control)) {
        if (col %in% colnames(df_output_print)) {
          vector_columns_exist = c(vector_columns_exist, col)
        }
      }
      df_output_print = df_output_print[, vector_columns_exist]
        
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
        # df_output_print,
        t(df_output_print),
        absFilename_output_print,
        sep = ",",
        # row.names = FALSE,
        # col.names = TRUE,
        row.names = TRUE,
        col.names = FALSE,
        # quote = TRUE
        quote = FALSE
      )
    }
  }
}





print("program exits.")
