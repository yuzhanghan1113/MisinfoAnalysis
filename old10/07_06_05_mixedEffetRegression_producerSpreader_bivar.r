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

absFilename_input = "C:/vmwareSharedFolder/TwitterDataAnalysis/results/results_20210209/preprocessedData.csv"
path_output = "C:/vmwareSharedFolder/TwitterDataAnalysis/results/results_20210209/mixedModelResults/producerSpreader/"

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

vector_varacities = c("mis", "aut")

for (varacity in vector_varacities) {
  # varacity = "mis"
  # varacity = "aut"
  
  misinfo = "999"
  if (varacity == "mis") {
    misinfo = "1"
  } else if (varacity == "aut") {
    misinfo = "0"
  }
  
  for (index_task in 1:8) {
    
    print("index_task:")
    print(index_task)
    
    df_output = data.frame()
    
    method = "mixedEffect"
    feature_randomEffect = "CASCADEEND_RETWEETS_rootTweet_user_screenName"
    
    if(index_task==1) {
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
    } else {
      vector_featureX = c("WrongX")
      str_name_X = "WrongX"
      vector_featureY = c("WrongY")
      str_name_Y = "WrongY"
    }
    
    absFilename_output = paste(
      path_output,
      "regression_producerSpreader_var=",
      varacity,
      "_med=",
      method,
      "_X=",
      str_name_X,
      "_Y=",
      str_name_Y,
      ".csv",
      sep = ""
    )
    
    for (feature_x in vector_featureX) {
      for (feature_y in vector_featureY) {
        
        df_input_toAnalyze = df_input[df_input$ROOTTWEETS_veracityLabel_agg_misinformation ==
                                        misinfo, c(feature_x, feature_y, feature_randomEffect, "ROOTTWEETS_veracityLabel_agg_misinformation")]
        
        print("table(df_input_toAnalyze$ROOTTWEETS_veracityLabel_agg_misinformation):")
        print(table(df_input_toAnalyze$ROOTTWEETS_veracityLabel_agg_misinformation))
        
        index_row = paste(feature_x, "_", feature_y, sep = "")
        
        str_formula_full = paste(feature_y,
                                 " ~ ",
                                 feature_x,
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
          num_observations = nrow(df_input_toAnalyze[, c(feature_x, feature_y)]) - sum(apply(df_input_toAnalyze[, c(feature_x, feature_y)], 1, anyNA))
          
          if (feature_x %in% rownames(modelingResult$coefficients)) {
            print("regression runnable")
            
            regression_coef = modelingResult$coefficients[feature_x, "Estimate"]
            regression_p = modelingResult$coefficients[feature_x, "Pr(>|t|)"]
            
            if (regression_p < 0.05) {
              print("p-value < 0.05")
              
              df_output[index_row, "feature_x"] = feature_x
              df_output[index_row, "feature_y"] = feature_y
              df_output[index_row, "coef_x"] = regression_coef
              df_output[index_row, "p_x"] = regression_p
              df_output[index_row, "num_observations"] = num_observations
              
              write.table(
                df_output,
                absFilename_output,
                sep = ",",
                row.names = FALSE,
                col.names = TRUE,
                quote = TRUE
              )
            }
          }
          
        })
      }
    }
    
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
  }
  
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
}



print("program exits.")
