rm(list=ls())

library(data.table)
library(stringr)
library(tidyr)
library(MuMIn)
library(nnet)
library(psych)

absFilename_input = "C:/vmwareSharedFolder/TwitterDataAnalysis/results/preprocessedData_clusteredMeasures.csv"

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
# feature_y = "measure_SpreaderPersona_retweeters_mixed"

list_featureX = c("ROOTTWEETS_communicativeIntention_QOU", "ROOTTWEETS_communicativeIntention_EXP", "ROOTTWEETS_communicativeIntention_COM", "ROOTTWEETS_communicativeIntention_DIR", "ROOTTWEETS_communicativeIntention_DEC", "ROOTTWEETS_communicativeIntention_REP")
feature_y = "measure_SpreaderPersona_retweeters_test"


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

df_input_toAnalyze_allFeatures[, "measure"] = ifelse(df_input_toAnalyze_allFeatures[, feature_y]==0, "A",
                                                     ifelse(df_input_toAnalyze_allFeatures[, feature_y]==1, "B",
                                                            "C"
                                                            # "B"
                                                     )
)

# df_input_toAnalyze_allFeatures[, c(feature_y, "measure")]

# df_input_toAnalyze_allFeatures[, list_featureX]
# df_input_toAnalyze_allFeatures[, feature_y]

str_formula_full = feature_y
# str_formula_full = "measure"
str_formula_full = paste(str_formula_full, " ~ ", list_featureX[1], sep="")
if (length(list_featureX) > 1) {
  for (i in 2:length(list_featureX))
    str_formula_full = paste(str_formula_full, " + ", list_featureX[i], sep="")
}
print("str_formula_full:")
print(str_formula_full)

model <- multinom(str_formula_full, data = df_input_toAnalyze_allFeatures)
summary(model)

z <- summary(model)$coefficients/summary(model)$standard.errors
# z
p <- (1 - pnorm(abs(z), 0, 1))*2
p
# exp(coef(model))
table(df_input_toAnalyze_allFeatures[, feature_y])

