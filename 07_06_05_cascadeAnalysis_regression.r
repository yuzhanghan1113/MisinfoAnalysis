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


# absFilename_input = "C:/vmwareSharedFolder/TwitterDataAnalysis/results/preprocessedData.csv"
absFilename_input = "C:/vmwareSharedFolder/TwitterDataAnalysis/results/results_20210209/preprocessedData.csv"

THRESHOLD_COR = 0.1
THRESHOLD_REG_COEF = 0.1
THRESHOLD_REG_R2 = 0.2
# MIN_NUM_SIGNIFICANT_X = 3
MIN_NUM_SIGNIFICANT_X = 0
# MIN_NUM_SIGNIFICANT_X = 1

list_features_randomEffect = c("ROOTTWEETS_userScreenName_producer", "ROOTTWEETS_idStr_rootTweet", "index")
feature_randomEffect = list_features_randomEffect[1]
# feature_randomEffect = list_features_randomEffect[1]

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

absFilename_output = paste("C:/vmwareSharedFolder/TwitterDataAnalysis/results/results_20210209/mixedModelResults/regression_producerIntentions_depVar=", str_depVar, "_rdmEff=", feature_randomEffect, "_rootTweetGp=", str_rootTweetGp, "_coefThr=", gsub("\\.", "p", as.character(THRESHOLD_REG_COEF)), "_minNumSigfX=", as.character(MIN_NUM_SIGNIFICANT_X), ".csv", sep="")

print("absFilename_output:")
print(absFilename_output)

df_output = data.frame()
index_row_output = 1

list_features_y = c()
for (f in colnames(df_input)) {
  if (startsWith(f, "CASCADEEND") || startsWith(f, "TIMESERIES"))
    list_features_y = c(list_features_y, f)
}

list_features_y = list_features_y[!startsWith(list_features_y, "CASCADEEND_ROOTTWEETS")]

# any(startsWith(list_features_y, "CASCADEEND_ROOTTWEETS"))
# any(startsWith(list_features_y, "CASCADEEND"))
# any(startsWith(list_features_y, "TIMESERIES"))

print("length(list_features_y):")
print(length(list_features_y))

df_input_toAnalyze_allFeatures = copy(df_input[df_input[, rootTweetGroup_feature]==rootTweetGroup_value,])
df_input_toAnalyze_allFeatures[,"index"] = rownames(df_input_toAnalyze_allFeatures)

print("\nData to analyze:")
print("nrow(df_input_toAnalyze_allFeatures):")
print(nrow(df_input_toAnalyze_allFeatures))
print("length(as.vector(df_input_toAnalyze_allFeatures[, \"ROOTTWEETS_id_rootTweet\"])):")
print(length(as.vector(df_input_toAnalyze_allFeatures[,"ROOTTWEETS_id_rootTweet"])))
print("length(unique(as.vector(df_input_toAnalyze_allFeatures[, \"ROOTTWEETS_id_rootTweet\"]))):")
print(length(unique(as.vector(df_input_toAnalyze_allFeatures[, "ROOTTWEETS_id_rootTweet"]))))
print("length(unique(as.vector((df_input_toAnalyze_allFeatures[, \"index\"])))):")
print(length(unique(as.vector((df_input_toAnalyze_allFeatures[, "index"])))))


for (feature_y in list_features_y) {
  
  
  
  # if (feature_y != "CASCADEEND_RETWEETS_cascadeSize_stadardized" && feature_y != "TIMESERIES_RETWEETS_user_description_textStats_charsPerWordDASHstl_curvature_filledTimeStamps")
  #   next
  
  # print("Current feature_y:")
  # print(feature_y)
  
  
  # 
  # if ("_PRODUCERTWEETS_" %in% feature_y) {
  #   # print("feature_y:")
  #   # print(feature_y)
  #   # print("skip")
  #   next
  # }
  if (feature_y %in% list_features_toExclude) {
    # print("feature_y:")
    # print(feature_y)
    # print("skip")
    next
  }
  
  df_input_toAnalyze = copy(df_input_toAnalyze_allFeatures[, c(list_featureX, feature_y, list_features_randomEffect)])
  df_input_toAnalyze_org = copy(df_input_toAnalyze)
  # colnames(df_input_toAnalyze)
  # colnames(df_input_toAnalyze_org)
  df_input_toAnalyze[, feature_y]
  df_input_toAnalyze = df_input_toAnalyze %>% drop_na()
  
  df_x = df_input_toAnalyze[, list_featureX]
  list_y = as.vector(df_input_toAnalyze[, feature_y])
  
  try({
  
  str_formula_full = feature_y
  str_formula_full = paste(str_formula_full, " ~ ", list_featureX[1], sep="")
  for (i in 2:length(list_featureX))
    str_formula_full = paste(str_formula_full, " + ", list_featureX[i], sep="")
  str_formula_full = paste(str_formula_full, " + (1|", feature_randomEffect, ")" , sep="")
  
  print("str_formula_full:")
  print(str_formula_full)
  
  
  
  
  
  model_full <- lmer(str_formula_full, data=df_input_toAnalyze, REML = FALSE)
  # model <- glmer(str_formula, data=df_input_toAnalyze, family=binomial)
  modelingResult = summary(model_full)
  
  result_RSquared = r.squaredGLMM(model_full)
  RSquared_marginal = result_RSquared[[1, "R2m"]]
  RSquared_conditional = result_RSquared[[1, "R2c"]]
  
  dict_featureXToCoef = list()
  dict_featureXToCoefP = list()
  
  num_significantX = 0
  
  rownames(modelingResult$coefficients)
  colnames(modelingResult$coefficients)
  
  for (feature_x in list_featureX) {
    
    if (feature_x %in% rownames(modelingResult$coefficients)) {
      
      regression_coef = modelingResult$coefficients[feature_x,"Estimate"]
      dict_featureXToCoef[feature_x] = regression_coef
      regression_p = modelingResult$coefficients[feature_x,"Pr(>|t|)"]
      dict_featureXToCoefP[feature_x] = regression_p
      
      if (abs(regression_coef) >= THRESHOLD_REG_COEF && regression_p < 0.05)
        num_significantX = num_significantX + 1
    }
    else {
      dict_featureXToCoef[feature_x] = NA
      dict_featureXToCoefP[feature_x] = NA
    }
  }
  
  str_formula_reduced = paste(feature_y, " ~ 1 + (1|", feature_randomEffect, ")", sep="")
  
  print("str_formula_reduced:")
  print(str_formula_reduced)
  
  model_reduced <- lmer(str_formula_reduced, data=df_input_toAnalyze, REML = FALSE)
  # model_reduced <- glmer(str_formula_reduced, data=df_input_toAnalyze, family=binomial)
  
  anovaResult = anova(model_reduced, model_full)
  
  mdlFullVSMdlReduced_anova_AIC = anovaResult["model_full", "AIC"]
  mdlFullVSMdlReduced_anova_BIC = anovaResult["model_full", "BIC"]
  mdlFullVSMdlReduced_anova_logLik = anovaResult["model_full", "logLik"]
  mdlFullVSMdlReduced_anova_Chisq = anovaResult["model_full", "Chisq"]
  mdlFullVSMdlReduced_anova_Df = anovaResult["model_full", "Df"]
  mdlFullVSMdlReduced_anova_p = anovaResult["model_full", "Pr(>Chisq)"]
  
  
  if (num_significantX >= MIN_NUM_SIGNIFICANT_X) {
    
    print("\n")
    print("-------------- feature selected --------------")
    print("index_row_output:")
    print(index_row_output)
    print("list_featureX:")
    print(list_featureX)
    print("feature_y:")
    print(feature_y)
    
    df_output[index_row_output, "rootTweetGroup_feature"] = rootTweetGroup_feature
    df_output[index_row_output, "rootTweetGroup_value"] = rootTweetGroup_value
    df_output[index_row_output, "regCoefThreshold"] = THRESHOLD_REG_COEF
    df_output[index_row_output, "minNumSigfX"] = MIN_NUM_SIGNIFICANT_X
    df_output[index_row_output, "feature_x"] = paste("[", paste(list_featureX, collapse=","), "]", sep="")
    
    list_strings = str_split(feature_y, "_")[[1]]
    # list_strings = lapply(list_strings, function(z){ z[!is.na(z) & z != ""]})
    
    df_output[index_row_output, "feature_y_type"] = paste(list_strings[1], "_", list_strings[2], sep="")
    df_output[index_row_output, "feature_y"] = feature_y
    df_output[index_row_output, "mdlFullVSMdlReduced_anova_p"] = mdlFullVSMdlReduced_anova_p
    df_output[index_row_output, "R2_marginal"] = RSquared_marginal
    df_output[index_row_output, "R2_conditional"] = RSquared_conditional
    df_output[index_row_output, "numSigfX"] = num_significantX
    
    for (feature_x in list_featureX) {
      
      regression_coef = dict_featureXToCoef[feature_x][[1]]
      df_output[index_row_output, paste(feature_x, "_regression_coef", sep="")] = regression_coef
      df_output[index_row_output, paste(feature_x, "_regression_coef_abs", sep="")] = abs(regression_coef)
      
      regression_p = dict_featureXToCoefP[feature_x][[1]]
      df_output[index_row_output, paste(feature_x, "_regression_p", sep="")] = regression_p
    }
    
    df_output[index_row_output, "mdlFullVSMdlReduced_anova_AIC"] = mdlFullVSMdlReduced_anova_AIC
    df_output[index_row_output, "mdlFullVSMdlReduced_anova_BIC"] = mdlFullVSMdlReduced_anova_BIC
    df_output[index_row_output, "mdlFullVSMdlReduced_anova_logLik"] = mdlFullVSMdlReduced_anova_logLik
    df_output[index_row_output, "mdlFullVSMdlReduced_anova_Chisq"] = mdlFullVSMdlReduced_anova_Chisq
    df_output[index_row_output, "mdlFullVSMdlReduced_anova_Df"] = mdlFullVSMdlReduced_anova_Df
    
    
    
    index_row_output = index_row_output + 1
    
    df_output = df_output[order(df_output[, "feature_y"]),]
    rownames(df_output) = 1:nrow(df_output)
    write.table(
      df_output,
      absFilename_output,
      sep = ",",
      row.names = FALSE,
      col.names = TRUE,
      quote = TRUE
    )
  }
  })
}

print("absFilename_output:")
print(absFilename_output)

df_output = df_output[order(df_output[, "feature_y"]),]
rownames(df_output) = 1:nrow(df_output)
write.table(
  df_output,
  absFilename_output,
  sep = ",",
  row.names = FALSE,
  col.names = TRUE,
  quote = TRUE
)

print("X feature processed.")
print("program exits.")





