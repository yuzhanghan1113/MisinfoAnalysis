rm(list=ls())

library(data.table)
library(stringr)
library(tidyr)
library(lme4) # for the analysis
library(lmerTest)# to get p-value estimations that are not part of the standard lme4 packages
library(MuMIn)

options(max.print = 999999)
# options(error = traceback)
options(error = recover)

modelingTask = "fullAndReduced"
# modelingTask = "full"
# modelingTask = "reduced"

THRESHOLD_COEF = 0.01

absFilename_input = "C:/vmwareSharedFolder/TwitterDataAnalysis/results/results_20210209/preprocessedData.csv"
path_output = "C:/vmwareSharedFolder/TwitterDataAnalysis/results/results_20210209/mixedModelResults/producerSpreader/multivariate/"

# # 1 all features, for full models
# 
# vector_producerFeatures_user = c(
#   "CASCADEEND_RETWEETS_rootTweet_user_followersCount",
#   "CASCADEEND_RETWEETS_rootTweet_user_friendsCount",
#   "CASCADEEND_RETWEETS_rootTweet_user_listedCount",
#   "CASCADEEND_RETWEETS_rootTweet_user_favouritesCount",
#   "CASCADEEND_RETWEETS_rootTweet_user_geoEnabled",
#   "CASCADEEND_RETWEETS_rootTweet_user_verified",
#   "CASCADEEND_RETWEETS_rootTweet_user_statusesCount"
# )
# vector_producerFeatures_userDescription = c(
#   "CASCADEEND_RETWEETS_rootTweet_user_description_sentiment_neg",
#   "CASCADEEND_RETWEETS_rootTweet_user_description_sentiment_neu",
#   "CASCADEEND_RETWEETS_rootTweet_user_description_sentiment_pos",
#   "CASCADEEND_RETWEETS_rootTweet_user_description_sentiment_compound",
#   "CASCADEEND_RETWEETS_rootTweet_user_description_subjectivity",
#   "CASCADEEND_RETWEETS_rootTweet_user_description_emotion_fear",
#   "CASCADEEND_RETWEETS_rootTweet_user_description_emotion_anger",
#   "CASCADEEND_RETWEETS_rootTweet_user_description_emotion_anticip",
#   "CASCADEEND_RETWEETS_rootTweet_user_description_emotion_trust",
#   "CASCADEEND_RETWEETS_rootTweet_user_description_emotion_surprise",
#   "CASCADEEND_RETWEETS_rootTweet_user_description_emotion_positive",
#   "CASCADEEND_RETWEETS_rootTweet_user_description_emotion_negative",
#   "CASCADEEND_RETWEETS_rootTweet_user_description_emotion_sadness",
#   "CASCADEEND_RETWEETS_rootTweet_user_description_emotion_disgust",
#   "CASCADEEND_RETWEETS_rootTweet_user_description_emotion_joy",
#   "CASCADEEND_RETWEETS_rootTweet_user_description_textStats_charCount",
#   "CASCADEEND_RETWEETS_rootTweet_user_description_textStats_wordCount",
#   "CASCADEEND_RETWEETS_rootTweet_user_description_textStats_uqWordCount",
#   "CASCADEEND_RETWEETS_rootTweet_user_description_textStats_sentenceCount",
#   "CASCADEEND_RETWEETS_rootTweet_user_description_textStats_charsPerWord",
#   "CASCADEEND_RETWEETS_rootTweet_user_description_textStats_wordsPerSentence",
#   "CASCADEEND_RETWEETS_rootTweet_user_description_textStats_readability"
# )
# vector_rootTweetFeatures = c(
#   "CASCADEEND_RETWEETS_cascadeSize",
#   "CASCADEEND_RETWEETS_rootTweet_favoriteCount",
#   "CASCADEEND_RETWEETS_rootTweet_possiblySensitive"
# )
# vector_rootTweetFeatures_fullText = c(
#   "CASCADEEND_RETWEETS_rootTweet_fullText_sentiment_neg",
#   "CASCADEEND_RETWEETS_rootTweet_fullText_sentiment_neu",
#   "CASCADEEND_RETWEETS_rootTweet_fullText_sentiment_pos",
#   "CASCADEEND_RETWEETS_rootTweet_fullText_sentiment_compound",
#   "CASCADEEND_RETWEETS_rootTweet_fullText_subjectivity",
#   "CASCADEEND_RETWEETS_rootTweet_fullText_emotion_fear",
#   "CASCADEEND_RETWEETS_rootTweet_fullText_emotion_anger",
#   "CASCADEEND_RETWEETS_rootTweet_fullText_emotion_anticip",
#   "CASCADEEND_RETWEETS_rootTweet_fullText_emotion_trust",
#   "CASCADEEND_RETWEETS_rootTweet_fullText_emotion_surprise",
#   "CASCADEEND_RETWEETS_rootTweet_fullText_emotion_positive",
#   "CASCADEEND_RETWEETS_rootTweet_fullText_emotion_negative",
#   "CASCADEEND_RETWEETS_rootTweet_fullText_emotion_sadness",
#   "CASCADEEND_RETWEETS_rootTweet_fullText_emotion_disgust",
#   "CASCADEEND_RETWEETS_rootTweet_fullText_emotion_joy",
#   "CASCADEEND_RETWEETS_rootTweet_fullText_textStats_charCount",
#   "CASCADEEND_RETWEETS_rootTweet_fullText_textStats_wordCount",
#   "CASCADEEND_RETWEETS_rootTweet_fullText_textStats_uqWordCount",
#   "CASCADEEND_RETWEETS_rootTweet_fullText_textStats_sentenceCount",
#   "CASCADEEND_RETWEETS_rootTweet_fullText_textStats_charsPerWord",
#   "CASCADEEND_RETWEETS_rootTweet_fullText_textStats_wordsPerSentence",
#   "CASCADEEND_RETWEETS_rootTweet_fullText_textStats_readability"
# )
# vector_retweeterFeatures_user = c(
#   "CASCADEEND_RETWEETS_retweet_ptg_user_protected",
#   "CASCADEEND_RETWEETS_retweet_mean_user_followersCount",
#   "CASCADEEND_RETWEETS_retweet_median_user_followersCount",
#   "CASCADEEND_RETWEETS_retweet_mean_user_friendsCount",
#   "CASCADEEND_RETWEETS_retweet_median_user_friendsCount",
#   "CASCADEEND_RETWEETS_retweet_mean_user_listedCount",
#   "CASCADEEND_RETWEETS_retweet_median_user_listedCount",
#   "CASCADEEND_RETWEETS_retweet_median_user_accountAge_day",
#   "CASCADEEND_RETWEETS_retweet_mean_user_favouritesCount",
#   "CASCADEEND_RETWEETS_retweet_median_user_favouritesCount",
#   "CASCADEEND_RETWEETS_retweet_mean_user_statusesCount",
#   "CASCADEEND_RETWEETS_retweet_median_user_statusesCount",
#   "CASCADEEND_RETWEETS_retweet_ptg_user_geoEnabled",
#   "CASCADEEND_RETWEETS_retweet_ptg_user_verified"
# )
# vector_retweeterFeatures_userDescription = c(
#   "CASCADEEND_RETWEETS_retweet_mean_user_description_sentiment_neg",
#   "CASCADEEND_RETWEETS_retweet_median_user_description_sentiment_neg",
#   "CASCADEEND_RETWEETS_retweet_mean_user_description_sentiment_neu",
#   "CASCADEEND_RETWEETS_retweet_median_user_description_sentiment_neu",
#   "CASCADEEND_RETWEETS_retweet_mean_user_description_sentiment_pos",
#   "CASCADEEND_RETWEETS_retweet_median_user_description_sentiment_pos",
#   "CASCADEEND_RETWEETS_retweet_mean_user_description_sentiment_compound",
#   "CASCADEEND_RETWEETS_retweet_median_user_description_sentiment_compound",
#   "CASCADEEND_RETWEETS_retweet_mean_user_description_subjectivity",
#   "CASCADEEND_RETWEETS_retweet_median_user_description_subjectivity",
#   "CASCADEEND_RETWEETS_retweet_mean_user_description_emotion_fear",
#   "CASCADEEND_RETWEETS_retweet_median_user_description_emotion_fear",
#   "CASCADEEND_RETWEETS_retweet_mean_user_description_emotion_anger",
#   "CASCADEEND_RETWEETS_retweet_median_user_description_emotion_anger",
#   "CASCADEEND_RETWEETS_retweet_mean_user_description_emotion_anticip",
#   "CASCADEEND_RETWEETS_retweet_median_user_description_emotion_anticip",
#   "CASCADEEND_RETWEETS_retweet_mean_user_description_emotion_trust",
#   "CASCADEEND_RETWEETS_retweet_median_user_description_emotion_trust",
#   "CASCADEEND_RETWEETS_retweet_mean_user_description_emotion_surprise",
#   "CASCADEEND_RETWEETS_retweet_median_user_description_emotion_surprise",
#   "CASCADEEND_RETWEETS_retweet_mean_user_description_emotion_positive",
#   "CASCADEEND_RETWEETS_retweet_median_user_description_emotion_positive",
#   "CASCADEEND_RETWEETS_retweet_mean_user_description_emotion_negative",
#   "CASCADEEND_RETWEETS_retweet_median_user_description_emotion_negative",
#   "CASCADEEND_RETWEETS_retweet_mean_user_description_emotion_sadness",
#   "CASCADEEND_RETWEETS_retweet_median_user_description_emotion_sadness",
#   "CASCADEEND_RETWEETS_retweet_mean_user_description_emotion_disgust",
#   "CASCADEEND_RETWEETS_retweet_median_user_description_emotion_disgust",
#   "CASCADEEND_RETWEETS_retweet_mean_user_description_emotion_joy",
#   "CASCADEEND_RETWEETS_retweet_median_user_description_emotion_joy",
#   "CASCADEEND_RETWEETS_retweet_mean_user_description_textStats_charCount",
#   "CASCADEEND_RETWEETS_retweet_median_user_description_textStats_charCount",
#   "CASCADEEND_RETWEETS_retweet_mean_user_description_textStats_wordCount",
#   "CASCADEEND_RETWEETS_retweet_median_user_description_textStats_wordCount",
#   "CASCADEEND_RETWEETS_retweet_mean_user_description_textStats_uqWordCount",
#   "CASCADEEND_RETWEETS_retweet_median_user_description_textStats_uqWordCount",
#   "CASCADEEND_RETWEETS_retweet_mean_user_description_textStats_sentenceCount",
#   "CASCADEEND_RETWEETS_retweet_median_user_description_textStats_sentenceCount",
#   "CASCADEEND_RETWEETS_retweet_mean_user_description_textStats_charsPerWord",
#   "CASCADEEND_RETWEETS_retweet_median_user_description_textStats_charsPerWord",
#   "CASCADEEND_RETWEETS_retweet_mean_user_description_textStats_wordsPerSentence",
#   "CASCADEEND_RETWEETS_retweet_median_user_description_textStats_wordsPerSentence",
#   "CASCADEEND_RETWEETS_retweet_mean_user_description_textStats_readability",
#   "CASCADEEND_RETWEETS_retweet_median_user_description_textStats_readability"
# )

# # 2 features selected by reduced models of 1, for full models
# 
vector_producerFeatures_user = c(
  "CASCADEEND_RETWEETS_rootTweet_user_friendsCount",
  "CASCADEEND_RETWEETS_rootTweet_user_favouritesCount",
  "CASCADEEND_RETWEETS_rootTweet_user_statusesCount"
)
vector_producerFeatures_userDescription = c(
  "CASCADEEND_RETWEETS_rootTweet_user_description_sentiment_neg",
  "CASCADEEND_RETWEETS_rootTweet_user_description_sentiment_neu",
  "CASCADEEND_RETWEETS_rootTweet_user_description_sentiment_pos",
  "CASCADEEND_RETWEETS_rootTweet_user_description_sentiment_compound",
  "CASCADEEND_RETWEETS_rootTweet_user_description_emotion_positive",
  "CASCADEEND_RETWEETS_rootTweet_user_description_emotion_sadness",
  "CASCADEEND_RETWEETS_rootTweet_user_description_emotion_joy",
  "CASCADEEND_RETWEETS_rootTweet_user_description_emotion_fear",
  "CASCADEEND_RETWEETS_rootTweet_user_description_emotion_anger",
  "CASCADEEND_RETWEETS_rootTweet_user_description_emotion_trust",
  "CASCADEEND_RETWEETS_rootTweet_user_description_emotion_surprise",
  "CASCADEEND_RETWEETS_rootTweet_user_description_emotion_negative",
  "CASCADEEND_RETWEETS_rootTweet_user_description_emotion_disgust",
  "CASCADEEND_RETWEETS_rootTweet_user_description_subjectivity",
  "CASCADEEND_RETWEETS_rootTweet_user_description_textStats_charCount",
  "CASCADEEND_RETWEETS_rootTweet_user_description_textStats_wordCount",
  "CASCADEEND_RETWEETS_rootTweet_user_description_textStats_sentenceCount",
  "CASCADEEND_RETWEETS_rootTweet_user_description_textStats_charsPerWord",
  "CASCADEEND_RETWEETS_rootTweet_user_description_textStats_wordsPerSentence",
  "CASCADEEND_RETWEETS_rootTweet_user_description_textStats_uqWordCount"
)
vector_rootTweetFeatures_fullText = c(
  "CASCADEEND_RETWEETS_rootTweet_fullText_sentiment_neg",
  "CASCADEEND_RETWEETS_rootTweet_fullText_sentiment_neu",
  "CASCADEEND_RETWEETS_rootTweet_fullText_sentiment_pos",
  "CASCADEEND_RETWEETS_rootTweet_fullText_sentiment_compound",
  "CASCADEEND_RETWEETS_rootTweet_fullText_emotion_positive",
  "CASCADEEND_RETWEETS_rootTweet_fullText_emotion_negative",
  "CASCADEEND_RETWEETS_rootTweet_fullText_emotion_fear",
  "CASCADEEND_RETWEETS_rootTweet_fullText_emotion_anger",
  "CASCADEEND_RETWEETS_rootTweet_fullText_emotion_trust",
  "CASCADEEND_RETWEETS_rootTweet_fullText_emotion_sadness",
  "CASCADEEND_RETWEETS_rootTweet_fullText_emotion_disgust",
  "CASCADEEND_RETWEETS_rootTweet_fullText_emotion_joy",
  "CASCADEEND_RETWEETS_rootTweet_fullText_textStats_charCount",
  "CASCADEEND_RETWEETS_rootTweet_fullText_textStats_wordCount",
  "CASCADEEND_RETWEETS_rootTweet_fullText_textStats_sentenceCount",
  "CASCADEEND_RETWEETS_rootTweet_fullText_textStats_charsPerWord",
  "CASCADEEND_RETWEETS_rootTweet_fullText_textStats_wordsPerSentence",
  "CASCADEEND_RETWEETS_rootTweet_fullText_textStats_readability",
  "CASCADEEND_RETWEETS_rootTweet_fullText_subjectivity",
  "CASCADEEND_RETWEETS_rootTweet_fullText_textStats_uqWordCount"
)
vector_retweeterFeatures_user = c(
  "CASCADEEND_RETWEETS_retweet_median_user_accountAge_day",
  "CASCADEEND_RETWEETS_retweet_mean_user_followersCount",
  "CASCADEEND_RETWEETS_retweet_mean_user_friendsCount",
  "CASCADEEND_RETWEETS_retweet_mean_user_listedCount",
  "CASCADEEND_RETWEETS_retweet_mean_user_favouritesCount",
  "CASCADEEND_RETWEETS_retweet_mean_user_statusesCount",
  "CASCADEEND_RETWEETS_retweet_ptg_user_geoEnabled"
)
vector_retweeterFeatures_userDescription = c(
  "CASCADEEND_RETWEETS_retweet_mean_user_description_sentiment_neg",
  "CASCADEEND_RETWEETS_retweet_median_user_description_sentiment_neu",
  "CASCADEEND_RETWEETS_retweet_median_user_description_sentiment_pos",
  "CASCADEEND_RETWEETS_retweet_median_user_description_sentiment_compound",
  "CASCADEEND_RETWEETS_retweet_mean_user_description_emotion_fear",
  "CASCADEEND_RETWEETS_retweet_mean_user_description_emotion_surprise",
  "CASCADEEND_RETWEETS_retweet_mean_user_description_emotion_sadness",
  "CASCADEEND_RETWEETS_retweet_mean_user_description_emotion_anger",
  "CASCADEEND_RETWEETS_retweet_mean_user_description_emotion_trust",
  "CASCADEEND_RETWEETS_retweet_mean_user_description_emotion_positive",
  "CASCADEEND_RETWEETS_retweet_mean_user_description_emotion_negative",
  "CASCADEEND_RETWEETS_retweet_mean_user_description_emotion_disgust",
  "CASCADEEND_RETWEETS_retweet_mean_user_description_emotion_joy",
  "CASCADEEND_RETWEETS_retweet_median_user_description_subjectivity",
  "CASCADEEND_RETWEETS_retweet_median_user_description_textStats_charCount",
  "CASCADEEND_RETWEETS_retweet_median_user_description_textStats_wordCount",
  "CASCADEEND_RETWEETS_retweet_median_user_description_textStats_uqWordCount",
  "CASCADEEND_RETWEETS_retweet_median_user_description_textStats_sentenceCount",
  "CASCADEEND_RETWEETS_retweet_median_user_description_textStats_charsPerWord",
  "CASCADEEND_RETWEETS_retweet_median_user_description_textStats_wordsPerSentence",
  "CASCADEEND_RETWEETS_retweet_median_user_description_textStats_readability"
)
vector_rootTweetFeatures = c(
  "CASCADEEND_RETWEETS_cascadeSize",
  "CASCADEEND_RETWEETS_rootTweet_favoriteCount",
  "CASCADEEND_RETWEETS_rootTweet_possiblySensitive"
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

method = "mixedEffect"
feature_randomEffect = "CASCADEEND_RETWEETS_rootTweet_user_screenName"

vector_varacities = c("mis", "aut")

if (modelingTask=="full" |
    modelingTask=="fullAndReduced") {
  ## produce the full models with all independent variables
  
  for (varacity in vector_varacities) {
    # varacity = "mis"
    # varacity = "aut"
    
    misinfo = "999"
    if (varacity == "mis") {
      misinfo = "1"
    } else if (varacity == "aut") {
      misinfo = "0"
    }
    
    for (index_task in 1:12) {
      print("index_task:")
      print(index_task)
      
      df_output = data.frame()
      df_output_complete = data.frame()
      
      if (index_task == 1) {
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
      } else {
        vector_featureX = c("WrongX")
        str_name_X = "WrongX"
        vector_featureY = c("WrongY")
        str_name_Y = "WrongY"
      }
      
      absFilename_output = paste(
        path_output,
        "fullMdl/",
        "rg_prodSprd_var=",
        varacity,
        "_med=",
        method,
        "_X=",
        str_name_X,
        "_Y=",
        str_name_Y,
        "_fullMdl",
        ".csv",
        sep = ""
      )
      
      absFilename_output_complete = paste(
        path_output,
        "fullMdl/cpl/",
        "rg_prodSprd_var=",
        varacity,
        "_med=",
        method,
        "_X=",
        str_name_X,
        "_Y=",
        str_name_Y,
        "_fullMdl_cpl",
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
        
        str_formula_full = paste(feature_y,
                                 " ~ ",
                                 sep = "")
        
        str_formula_full = paste(str_formula_full,
                                 vector_featureX[1],
                                 sep = "")
        
        for (feature_x in vector_featureX[2:length(vector_featureX)]) {
          str_formula_full = paste(str_formula_full,
                                   " + ",
                                   feature_x,
                                   sep = "")
        }
        
        str_formula_full = paste(str_formula_full,
                                 " + (1|",
                                 feature_randomEffect,
                                 ")",
                                 sep = "")
        
        print("str_formula_full:")
        print(str_formula_full)
        
        try({
          model_full <-
            lmer(str_formula_full, data = df_input_toAnalyze, REML = FALSE)
          
          modelingResult = summary(model_full)
          
          # print("modelingResult:")
          # print(modelingResult)
          
          # result_RSquared = r.squaredGLMM(model_full)
          # RSquared_marginal = result_RSquared[[1, "R2m"]]
          # RSquared_conditional = result_RSquared[[1, "R2c"]]
          
          df_output[index_row, "varGrp_X"] = str_name_X
          df_output[index_row, "varGrp_Y"] = str_name_Y
          
          df_output_complete[index_row, "varGrp_X"] = str_name_X
          df_output_complete[index_row, "varGrp_Y"] = str_name_Y
          
          df_output[index_row, "feature_y"] = feature_y
          df_output_complete[index_row, "feature_y"] = feature_y
          
          num_observations = nrow(df_input_toAnalyze[, c(vector_featureX, feature_y, feature_randomEffect)]) - sum(apply(df_input_toAnalyze[, c(vector_featureX, feature_y, feature_randomEffect)], 1, anyNA))
          df_output[index_row, "num_observations"] = num_observations
          df_output_complete[index_row, "num_observations"] = num_observations
          
          
          
          num_xFeatures_useful = 0
          
          for (feature_x in vector_featureX) {
            if (feature_x %in% rownames(modelingResult$coefficients)) {
              regression_coef = modelingResult$coefficients[feature_x, "Estimate"]
              regression_p = modelingResult$coefficients[feature_x, "Pr(>|t|)"]
              
              if (regression_p < 0.05 &
                  abs(regression_coef) >= THRESHOLD_COEF) {
                num_xFeatures_useful = num_xFeatures_useful + 1
              }
            }
          }
          
          df_output[index_row, "num_xFeatures_useful"] = num_xFeatures_useful
          df_output_complete[index_row, "num_xFeatures_useful"] = num_xFeatures_useful
          
          for (feature_x in vector_featureX) {
            df_output[index_row, feature_x] = NA
            df_output_complete[index_row, feature_x] = NA
            
            if (feature_x %in% rownames(modelingResult$coefficients)) {
              print("regression runnable")
              
              regression_coef = modelingResult$coefficients[feature_x, "Estimate"]
              regression_p = modelingResult$coefficients[feature_x, "Pr(>|t|)"]
              
              if (regression_p < 0.05) {
                print("p-value < 0.05")
                df_output_complete[index_row, feature_x] = regression_coef
              }
              
              if (regression_p < 0.05 &
                  abs(regression_coef) >= THRESHOLD_COEF) {
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
  
  for (varacity in vector_varacities) {
    # varacity = "mis"
    # varacity = "aut"
    
    misinfo = "999"
    if (varacity == "mis") {
      misinfo = "1"
    } else if (varacity == "aut") {
      misinfo = "0"
    }
    
    for (index_task in 1:12) {
      
      print("index_task:")
      print(index_task)
      
      df_output = data.frame()
      
      if(index_task==1) {
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
      } else {
        # vector_featureX = c("WrongX")
        str_name_X = "WrongX"
        vector_featureY = c("WrongY")
        str_name_Y = "WrongY"
      }
      
      vector_featureX = c("WrongX")
      
      absFilename_input_fullModel = paste(
        path_output, "fullMdl/",
        "rg_prodSprd_var=",
        varacity,
        "_med=",
        method,
        "_X=",
        str_name_X,
        "_Y=",
        str_name_Y,
        "_fullMdl",
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
      
      vector_columnsToRemove = c("varGrp_X", "varGrp_Y","feature_y", "num_observations", "num_xFeatures_useful")
      vector_featureX = colnames(df_input_fullModel)
      
      # print("df_input_fullModel:")
      # print(df_input_fullModel)
      
      print("vector_featureX:")
      print(vector_featureX)
      
      vector_featureX = vector_featureX[!(vector_featureX %in% vector_columnsToRemove)]
      
      print("vector_featureX:")
      print(vector_featureX)
      
      absFilename_output = paste(
        path_output, "reducedMdl/",
        "rg_prodSprd_var=",
        varacity,
        "_med=",
        method,
        "_X=",
        str_name_X,
        "_Y=",
        str_name_Y,
        "_reducedMdl",
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
        
        str_formula_full = paste(feature_y,
                                 " ~ ",
                                 sep = "")
        
        str_formula_full = paste(str_formula_full,
                                 vector_featureX[1],
                                 sep = "")
        
        for (feature_x in vector_featureX[2:length(vector_featureX)]) {
          str_formula_full = paste(str_formula_full,
                                   " + ",
                                   feature_x,
                                   sep = "")
        }
        
        str_formula_full = paste(str_formula_full,
                                 " + (1|",
                                 feature_randomEffect,
                                 ")",
                                 sep = "")
        
        print("str_formula_full:")
        print(str_formula_full)
        
        try({
          model_full <-
            lmer(str_formula_full, data = df_input_toAnalyze, REML = FALSE)
          
          modelingResult = summary(model_full)
          
          # print("modelingResult:")
          # print(modelingResult)
          
          # result_RSquared = r.squaredGLMM(model_full)
          # RSquared_marginal = result_RSquared[[1, "R2m"]]
          # RSquared_conditional = result_RSquared[[1, "R2c"]]
          
          df_output[index_row, "varGrp_X"] = str_name_X
          df_output[index_row, "varGrp_Y"] = str_name_Y
          
          df_output[index_row, "feature_y"] = feature_y
          
          num_observations = nrow(df_input_toAnalyze[, c(vector_featureX, feature_y, feature_randomEffect)]) - sum(apply(df_input_toAnalyze[, c(vector_featureX, feature_y, feature_randomEffect)], 1, anyNA))
          df_output[index_row, "num_observations"] = num_observations
          
          num_xFeatures_useful = 0
          
          
          
          for (feature_x in vector_featureX) {
            
            if (feature_x %in% rownames(modelingResult$coefficients)) {
              
              regression_coef = modelingResult$coefficients[feature_x, "Estimate"]
              regression_p = modelingResult$coefficients[feature_x, "Pr(>|t|)"]
              
              if (regression_p < 0.05 & abs(regression_coef) >= THRESHOLD_COEF) {
                num_xFeatures_useful = num_xFeatures_useful + 1
              }
            }
          }
          
          df_output[index_row, "num_xFeatures_useful"] = num_xFeatures_useful
          
          for (feature_x in vector_featureX) {
            
            df_output[index_row, feature_x] = NA
            
            if (feature_x %in% rownames(modelingResult$coefficients)) {
              print("regression runnable")
              
              regression_coef = modelingResult$coefficients[feature_x, "Estimate"]
              regression_p = modelingResult$coefficients[feature_x, "Pr(>|t|)"]
              
              if (regression_p < 0.05 & abs(regression_coef) >= THRESHOLD_COEF) {
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
          
        })
      }
      
      absFilename_output_print = paste(
        path_output,
        "fullMdl/pt/",
        "rg_prodSprd_var=",
        varacity,
        "_med=",
        method,
        "_X=",
        str_name_X,
        "_Y=",
        str_name_Y,
        "_fullMdl",
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
        df_output_print[, col] = format(round(df_output_print[, col], 2), nsmall = 2)
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
        path_output, "reducedMdl/prt/",
        "rg_prodSprd_var=",
        varacity,
        "_med=",
        method,
        "_X=",
        str_name_X,
        "_Y=",
        str_name_Y,
        "_reducedMdl_prt",
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
