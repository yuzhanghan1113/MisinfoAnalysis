rm(list = ls())

library(data.table)
library(lavaan)
library(semPlot)

absFilename_input = "D:/vmwareSharedFolder/TwitterDataAnalysis/results/preprocessedData.csv"

df_input <-
  data.frame(fread(
    absFilename_input,
    # nrows = 50000,
    sep = ",",
    header = TRUE
    # select = vector_attributesToLoad_rereports,
    # colClasses = c("character")
  ))

print("nrow(df_input):")
print(nrow(df_input))


vector_indicators_ProducerIntention <- c(
  "ROOTTWEETS_communicativeIntention_REP_and_QOU"
)

vector_indicators_SpreaderPersona <- c(
  "CASCADEEND_RETWEETS_retweet_mean_user_listedCount",
  "CASCADEEND_RETWEETS_retweet_ptg_user_geoEnabled",
  # "CASCADEEND_RETWEETS_retweet_mean_user_friendsCount"
  "CASCADEEND_RETWEETS_retweet_mean_user_description_textStats_readability",
  "CASCADEEND_RETWEETS_retweet_mean_user_accountAge_day",
  "CASCADEEND_RETWEETS_retweet_median_user_accountAge_day",
  "TIMESERIES_RETWEETS_user_statusesCountDASHstl_eacf10_allTimeStamps",
  "TIMESERIES_RETWEETS_user_description_emotion_trustDASHlumpiness_allTimeStamps",
  "TIMESERIES_RETWEETS_user_description_emotion_surpriseDASHstd1stDer_allTimeStamps",
  # "TIMESERIES_RETWEETS_user_listedCountDASHstd1stDer_filledTimeStamps"
  # "TIMESERIES_RETWEETS_user_listedCountDASHstl_linearity_filledTimeStamps"
  # "TIMESERIES_RETWEETS_user_listedCountDASHlumpiness_allTimeStamps"
  "TIMESERIES_RETWEETS_user_description_sentiment_neuDASHstl_eacf10_allTimeStamps"
  # "TIMESERIES_RETWEETS_user_description_emotion_disgustDASHlumpiness_allTimeStamps",
  # "TIMESERIES_RETWEETS_user_statusesCountDASHstd1stDer_filledTimeStamps"
  # "TIMESERIES_RETWEETS_user_description_emotion_sadnessDASHKPSS_statistic_allTimeStamps"
  # "TIMESERIES_RETWEETS_user_description_emotion_sadnessDASHKPSS_statistic_filledTimeStamps"
)

formula <- ""

formula <- "# measurement model"
formula <- paste(formula, "\nProducerIntention =~ ", sep="")
for (index in 1:length(vector_indicators_ProducerIntention)) {
  if (index >= 2) {
    formula <- paste(formula, " + ", sep="")
  }
  formula <- paste(formula, vector_indicators_ProducerIntention[index], sep="")
}

formula <- paste(formula, "\nSpreaderPersona =~ ", sep="")
for (index in 1:length(vector_indicators_SpreaderPersona)) {
  if (index >= 2) {
    formula <- paste(formula, " + ", sep="")
  }
  formula <- paste(formula, vector_indicators_SpreaderPersona[index], sep="")
}


formula <- paste(formula, "\n# regressions", sep="")
formula <- paste(formula, "", sep="\nSpreaderPersona ~ ProducerIntention")

"ProducerIntention =~ ROOTTWEETS_communicativeIntention_REP_and_QOU\nSpreaderPersona =~ "


# rm(list = ls()) 
# formula <- "# measurement model\nProducerIntention =~ ROOTTWEETS_communicativeIntention_REP_and_QOU\nSpreaderPersona =~ CASCADEEND_RETWEETS_retweet_mean_user_friendsCount + CASCADEEND_RETWEETS_retweet_median_user_accountAge_day"

cat("formula:\n")
cat(formula)

# fit <- sem(formula, data=df_input, std.lv = FALSE)
# summary(fit, standardized=TRUE)
fit <- sem(formula, data=df_input, std.lv = TRUE)
summary(fit, standardized=TRUE)
