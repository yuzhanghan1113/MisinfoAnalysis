rm(list=ls())

library(tsfeatures)
library(data.table)
library(argparse)
library(urca)
# library(pracma)

options(max.print = 999999)
# options(error = traceback)
options(error = recover)

# task <- "retweets"
task <- "replies"

absFilename_input_rootTweets <-
  "D:/vmwareSharedFolder/TwitterDataCollection/data/PFN/snapshot/dateRetrieval=20201111/rootTweets_selected_factcheckArticleRep_20201111.csv"

absFilename_output <-
  paste(
    "D:/vmwareSharedFolder/TwitterDataAnalysis/results/timeSeries/metrics/temporal_",
    task,
    ".csv",
    sep = ""
  )

print("absFilename_output:")
print(absFilename_output)

df_input_rootTweets <-
  data.frame(fread(
    absFilename_input_rootTweets,
    # nrows = 50000,
    sep = ",",
    header = TRUE,
    # select = vector_attributesToLoad_rereports,
    colClasses = c("character")
  ))

print("nrow(df_input_rootTweets):")
print(nrow(df_input_rootTweets))

print("length(df_input_rootTweets[, \"id_rootTweet\"]):")
print(length(df_input_rootTweets[, "id_rootTweet"]))

vector_rootTweetIdStrs <-
  unique(df_input_rootTweets[, "id_rootTweet"])

print("length(vector_rootTweetIdStrs):")
print(length(vector_rootTweetIdStrs))

vector_rootTweetIdStr_existing <- c()

if (file.exists(absFilename_output)) {
  
  print("output file exists already:")
  print("absFilename_output:")
  print(absFilename_output)
  
  df_output <-
    data.frame(fread(
      absFilename_output,
      # nrows = 50000,
      sep = ",",
      header = TRUE,
      # select = vector_attributesToLoad_rereports,
      colClasses = c("character")
    ))
  
  print("nrow(df_output):")
  print(nrow(df_output))
  
  print("length(df_output[, \"rootTweetIdStr\"]):")
  print(length(df_output[, "rootTweetIdStr"]))
  
  vector_rootTweetIdStr_existing <-
    unique(df_output[, "rootTweetIdStr"])
  print("vector_rootTweetIdStr_existing:")
  print(vector_rootTweetIdStr_existing)
  print("length(vector_rootTweetIdStr_existing):")
  print(length(vector_rootTweetIdStr_existing))
  
  if (length(df_output) <= 0)
    df_output <- data.frame()
} else {
  print("Output file does not exist. Creates new df_output.")
  df_output <- data.frame()
}

index_row <- nrow(df_output) + 1

for (rootTweetIdStr in vector_rootTweetIdStrs) {
  print("index_row:")
  print(index_row)
  
  # rootTweetIdStr <- "1287105426283274241"
  
  print("rootTweetIdStr:")
  print(rootTweetIdStr)
  
  if (paste("id", rootTweetIdStr, sep = "") %in% vector_rootTweetIdStr_existing) {
    print("rootTweetIdStr has been included in existing output file. Skip.")
    next
  }
  
  absFilename_input <-
    paste(
      "D:/vmwareSharedFolder/TwitterDataAnalysis/results/timeSeries/preprocessed/",
      task,
      "/timeSeries_",
      task,
      "_preprocessed_rootTweetID=",
      rootTweetIdStr,
      ".csv",
      sep = ""
    )
  
  print("absFilename_input:")
  print(absFilename_input)
  
  if (!file.exists(absFilename_input)) {
    print("File does not exist. Skip this root tweet.")
    # index_row <- index_row + 1
    next
  }
  
  df_input <-
    data.frame(fread(
      absFilename_input,
      # nrows = 50000,
      sep = ",",
      header = TRUE,
      # select = vector_attributesToLoad_rereports,
      colClasses = c("character")
    ))
  
  print("nrow(df_input):")
  print(nrow(df_input))
  # print(head(df_input, n=5))
  #print(df_input[1, df_input$retweetAge_sec])
  
  vector_features <- colnames(df_input)
  vector_features <-
    vector_features[vector_features != "rootTweetIdStr"]
  df_input[, vector_features] <-
    sapply(df_input[, vector_features], as.numeric)
  
  if (task == "retweets") {
    vector_features <-
    vector_features[vector_features != "retweetAge_sec"]
  } else if (task == "replies") {
    vector_features <-
    vector_features[vector_features != "replyAge_sec"]
  }
  
  # print(head(df_input, n=5))
  
  for (feature_toMeasure in vector_features) {
    print("index_row:")
    print(index_row)
    print("rootTweetIdStr:")
    print(rootTweetIdStr)
    print("feature_toMeasure:")
    print(feature_toMeasure)
    print("length(df_input[, feature_toMeasure]):")
    print(length(df_input[, feature_toMeasure]))
    print("length(unique(df_input[, feature_toMeasure])):")
    print(length(unique(df_input[, feature_toMeasure])))
    
    # df_output[index_row, "test"] <- "testnumber"
    
    df_output[index_row, "rootTweetIdStr"] <-
      paste("id", rootTweetIdStr, sep = "")
    df_output[index_row, "feature"] <- feature_toMeasure
    
    
    vector_timeSeries_original <- df_input[, feature_toMeasure]
    print("length(vector_timeSeries_original):")
    print(length(vector_timeSeries_original))
    vector_timeSeries_nonNA <-
      vector_timeSeries_original[!is.na(vector_timeSeries_original)]
    print("length(vector_timeSeries_nonNA):")
    print(length(vector_timeSeries_nonNA))
    # print("vector_timeSeries_nonNA:")
    # print(vector_timeSeries_nonNA)
    
    feature_x <- NA
    if (task == "retweets") {
      feature_x <- "retweetAge_sec"
    } else if (task == "replies") {
      feature_x <- "replyAge_sec"
    }
    
    print("linear regression vector_timeSeries_original:")
    if (length(vector_timeSeries_nonNA) <= 1 ||
        all(is.na(vector_timeSeries_original))) {
      df_output[index_row, "LR_coef_allTimeStamps"] <- NA
      df_output[index_row, "LR_p_allTimeStamps"] <- NA
    } else {
      linearModel <- NA
      linearModel <-
        lm(vector_timeSeries_original ~ df_input[, feature_x])
      result <- NA
      result <- summary(linearModel)
      df_output[index_row, "LR_coef_allTimeStamps"] <-
        result$coefficients["df_input[, feature_x]", "Estimate"]
      df_output[index_row, "LR_p_allTimeStamps"] <-
        result$coefficients["df_input[, feature_x]", "Pr(>|t|)"]
    }
    print("linear regression vector_timeSeries_nonNA:")
    if (length(vector_timeSeries_nonNA) <= 1) {
      df_output[index_row, "LR_coef_filledTimeStamps"] <- NA
      df_output[index_row, "LR_p_filledTimeStamps"] <- NA
    } else {
      vector_x <- seq(1, length(vector_timeSeries_nonNA))
      linearModel <- NA
      linearModel <- lm(vector_timeSeries_nonNA ~ vector_x)
      result <- NA
      result <- summary(linearModel)
      df_output[index_row, "LR_coef_filledTimeStamps"] <-
        result$coefficients["vector_x", "Estimate"]
      df_output[index_row, "LR_p_filledTimeStamps"] <-
        result$coefficients["vector_x", "Pr(>|t|)"]
    }
    
    # print("linear regression vector_timeSeries_original:")
    # linearModel <- NA
    # linearModel <- lm(vector_timeSeries_original ~ df_output[, feature_x])
    # result <- NA
    # result <- summary(linearModel)
    # df_output[index_row, "LR_coef_allTimeStamps"] <- result$coefficients["df_output[, feature_x]", "Estimate"]
    # df_output[index_row, "LR_p_allTimeStamps"] <- result$coefficients["df_output[, feature_x]", "Pr(>|t|)"]
    #
    # print("linear regression vector_timeSeries_nonNA:")
    # vector_x <- seq(1, length(vector_timeSeries_nonNA))
    # linearModel <- NA
    # linearModel <- lm(vector_timeSeries_nonNA ~ vector_x)
    # result <- NA
    # result <- summary(linearModel)
    # df_output[index_row, "LR_coef_allTimeStamps"] <- result$coefficients["df_output[, feature_x]", "Estimate"]
    # df_output[index_row, "LR_p_allTimeStamps"] <- result$coefficients["df_output[, feature_x]", "Pr(>|t|)"]
    
    
    print("entropy vector_timeSeries_original:")
    if (length(vector_timeSeries_original) == 0 ||
        any(is.na(vector_timeSeries_original)) ||
        all(vector_timeSeries_nonNA == vector_timeSeries_nonNA[1])) {
      df_output[index_row, "entropy_allTimeStamps"] <- NA
    } else {
      result <- NA
      result <- entropy(vector_timeSeries_original)
      df_output[index_row, "entropy_allTimeStamps"] <- result
    }
    print("entropy vector_timeSeries_nonNA:")
    if (length(vector_timeSeries_nonNA) == 0 ||
        all(vector_timeSeries_nonNA == vector_timeSeries_nonNA[1])) {
      df_output[index_row, "entropy_filledTimeStamps"] <- NA
    } else {
      result <- NA
      result <- entropy(vector_timeSeries_nonNA)
      df_output[index_row, "entropy_filledTimeStamps"] <- result
    }
    
    print("stability condition:")
    if (length(vector_timeSeries_nonNA) <= 0) {
      df_output[index_row, "stability_allTimeStamps"] <- NA
      df_output[index_row, "stability_filledTimeStamps"] <- NA
    } else {
      print("stability vector_timeSeries_original:")
      result <- NA
      result <- stability(vector_timeSeries_original)
      df_output[index_row, "stability_allTimeStamps"] <- result
      print("stability vector_timeSeries_nonNA:")
      result <- NA
      result <- stability(vector_timeSeries_nonNA)
      df_output[index_row, "stability_filledTimeStamps"] <- result
    }
    
    print("lumpiness condition:")
    if (length(vector_timeSeries_nonNA) <= 0) {
      df_output[index_row, "lumpiness_allTimeStamps"] <- NA
      df_output[index_row, "lumpiness_filledTimeStamps"] <- NA
    } else {
      print("lumpiness vector_timeSeries_original:")
      result <- NA
      result <- lumpiness(vector_timeSeries_original)
      df_output[index_row, "lumpiness_allTimeStamps"] <- result
      print("lumpiness vector_timeSeries_nonNA:")
      result <- NA
      result <- lumpiness(vector_timeSeries_nonNA)
      df_output[index_row, "lumpiness_filledTimeStamps"] <- result
    }
    
    print("max_level_shift:")
    result <- NA
    result <- max_level_shift(vector_timeSeries_original)
    df_output[index_row, "maxLevelShift_maxLevelShift_allTimeStamps"] <-
      result["max_level_shift"]
    df_output[index_row, "maxLevelShift_timeLevelShift_allTimeStamps"] <-
      result["time_level_shift"]
    result <- NA
    result <- max_level_shift(vector_timeSeries_nonNA)
    df_output[index_row, "maxLevelShift_maxLevelShift_filledTimeStamps"] <-
      result["max_level_shift"]
    df_output[index_row, "maxLevelShift_timeLevelShift_filledTimeStamps"] <-
      result["time_level_shift"]
    
    print("max_var_shift:")
    result <- NA
    result <- max_var_shift(vector_timeSeries_original)
    df_output[index_row, "maxVarShift_maxVarShift_allTimeStamps"] <-
      result["max_var_shift"]
    df_output[index_row, "maxVarShift_timeVarShift_allTimeStamps"] <-
      result["time_var_shift"]
    result <- NA
    result <- max_var_shift(vector_timeSeries_nonNA)
    df_output[index_row, "maxVarShift_maxVarShift_filledTimeStamps"] <-
      result["max_var_shift"]
    df_output[index_row, "maxVarShift_timeVarShift_filledTimeStamps"] <-
      result["time_var_shift"]
    
    # max_kl_shift(vector_timeSeries_original)
    
    print("crossing_points condition:")
    if (length(vector_timeSeries_nonNA) <= 0) {
      df_output[index_row, "crossingPoints_allTimeStamps"] <- NA
      df_output[index_row, "crossingPoints_filledTimeStamps"] <-
        NA
    } else {
      print("crossing_points vector_timeSeries_original:")
      result <- NA
      result <- crossing_points(vector_timeSeries_original)
      df_output[index_row, "crossingPoints_allTimeStamps"] <- result
      print("crossing_points vector_timeSeries_nonNA:")
      result <- NA
      result <- crossing_points(vector_timeSeries_nonNA)
      df_output[index_row, "crossingPoints_filledTimeStamps"] <-
        result
    }
    
    print("flat_spots:")
    result <- NA
    result <- flat_spots(vector_timeSeries_original)
    df_output[index_row, "flatSpots_allTimeStamps"] <- result
    result <- NA
    result <- flat_spots(vector_timeSeries_nonNA)
    df_output[index_row, "flatSpots_filledTimeStamps"] <- result
    
    # hurst(vector_timeSeries_original)
    # hurst(vector_timeSeries_nonNA)
    # result <- hurstexp(vector_timeSeries_original)
    # result <- hurstexp(vector_timeSeries_nonNA)
    # print(result)
    
    # result <- unitroot_kpss(df_input[, feature_toMeasure])
    
    print("ur.kpss:")
    result <- NA
    result <- ur.kpss(vector_timeSeries_original)
    df_output[index_row, "KPSS_statistic_allTimeStamps"] <-
      result@teststat
    df_output[index_row, "KPSS_LT10pct_allTimeStamps"] <-
      ifelse (result@teststat > result@cval[1], 1, 0)
    df_output[index_row, "KPSS_LT5pct_allTimeStamps"] <-
      ifelse (result@teststat > result@cval[2], 1, 0)
    df_output[index_row, "KPSS_LT2p5pct_allTimeStamps"] <-
      ifelse (result@teststat > result@cval[3], 1, 0)
    df_output[index_row, "KPSS_LT1pct_allTimeStamps"] <-
      ifelse (result@teststat > result@cval[4], 1, 0)
    result <- NA
    result <- ur.kpss(vector_timeSeries_nonNA)
    df_output[index_row, "KPSS_statistic_filledTimeStamps"] <-
      result@teststat
    df_output[index_row, "KPSS_LT10pct_filledTimeStamps"] <-
      ifelse (result@teststat > result@cval[1], 1, 0)
    df_output[index_row, "KPSS_LT5pct_filledTimeStamps"] <-
      ifelse (result@teststat > result@cval[2], 1, 0)
    df_output[index_row, "KPSS_LT2p5pct_filledTimeStamps"] <-
      ifelse (result@teststat > result@cval[3], 1, 0)
    df_output[index_row, "KPSS_LT1pct_filledTimeStamps"] <-
      ifelse (result@teststat > result@cval[4], 1, 0)
    
    # result <- unitroot_pp(df_input[, feature_toMeasure])
    # print(result)
    # result <- ur.pp(df_input[, feature_toMeasure], type="Z-tau")
    # summary(result)
    # result@teststat
    # result@cval
    
    print("stl_features condition:")
    if (length(vector_timeSeries_nonNA) <= 1) {
      df_output[index_row, "stl_nperiods_allTimeStamps"] <- NA
      df_output[index_row, "stl_seasonalPeriod_allTimeStamps"] <- NA
      df_output[index_row, "stl_trend_allTimeStamps"] <- NA
      df_output[index_row, "stl_spike_allTimeStamps"] <- NA
      df_output[index_row, "stl_linearity_allTimeStamps"] <- NA
      df_output[index_row, "stl_curvature_allTimeStamps"] <- NA
      df_output[index_row, "stl_eacf1_allTimeStamps"] <- NA
      df_output[index_row, "stl_eacf10_allTimeStamps"] <- NA
      
      df_output[index_row, "stl_nperiods_filledTimeStamps"] <- NA
      df_output[index_row, "stl_seasonalPeriod_filledTimeStamps"] <-
        NA
      df_output[index_row, "stl_trend_filledTimeStamps"] <- NA
      df_output[index_row, "stl_spike_filledTimeStamps"] <- NA
      df_output[index_row, "stl_linearity_filledTimeStamps"] <- NA
      df_output[index_row, "stl_curvature_filledTimeStamps"] <- NA
      df_output[index_row, "stl_eacf1_filledTimeStamps"] <- NA
      df_output[index_row, "stl_eacf10_filledTimeStamps"] <- NA
    } else {
      print("stl_features vector_timeSeries_original:")
      result <- NA
      result <- stl_features(vector_timeSeries_original)
      df_output[index_row, "stl_nperiods_allTimeStamps"] <-
        result[["nperiods"]]
      df_output[index_row, "stl_seasonalPeriod_allTimeStamps"] <-
        result[["seasonal_period"]]
      df_output[index_row, "stl_trend_allTimeStamps"] <-
        result[["trend"]]
      df_output[index_row, "stl_spike_allTimeStamps"] <-
        result[["spike"]]
      df_output[index_row, "stl_linearity_allTimeStamps"] <-
        result[["linearity"]]
      df_output[index_row, "stl_curvature_allTimeStamps"] <-
        result[["curvature"]]
      df_output[index_row, "stl_eacf1_allTimeStamps"] <-
        result[["e_acf1"]]
      df_output[index_row, "stl_eacf10_allTimeStamps"] <-
        result[["e_acf10"]]
      
      print("stl_features vector_timeSeries_nonNA:")
      result <- NA
      result <- stl_features(vector_timeSeries_nonNA)
      df_output[index_row, "stl_nperiods_filledTimeStamps"] <-
        result[["nperiods"]]
      df_output[index_row, "stl_seasonalPeriod_filledTimeStamps"] <-
        result[["seasonal_period"]]
      df_output[index_row, "stl_trend_filledTimeStamps"] <-
        result[["trend"]]
      df_output[index_row, "stl_spike_filledTimeStamps"] <-
        result[["spike"]]
      df_output[index_row, "stl_linearity_filledTimeStamps"] <-
        result[["linearity"]]
      df_output[index_row, "stl_curvature_filledTimeStamps"] <-
        result[["curvature"]]
      df_output[index_row, "stl_eacf1_filledTimeStamps"] <-
        result[["e_acf1"]]
      df_output[index_row, "stl_eacf10_filledTimeStamps"] <-
        result[["e_acf10"]]
    }
    
    print("nonlinearity:")
    result <- NA
    result <- nonlinearity(vector_timeSeries_original)
    df_output[index_row, "nonlinearity_allTimeStamps"] <- result
    result <- NA
    result <- nonlinearity(vector_timeSeries_nonNA)
    df_output[index_row, "nonlinearity_filledTimeStamps"] <- result
    
    # sampen_first(df_input[, feature_toMeasure])
    # sampenc(df_input[, feature_toMeasure], M = 5, r = 0.3)
    
    print("std1st_der condition:")
    if (length(vector_timeSeries_nonNA) <= 1) {
      df_output[index_row, "std1stDer_allTimeStamps"] <- NA
      df_output[index_row, "std1stDer_filledTimeStamps"] <- NA
    } else {
      print("std1st_der vector_timeSeries_original:")
      result <- NA
      result <- std1st_der(vector_timeSeries_original)
      df_output[index_row, "std1stDer_allTimeStamps"] <- result
      print("std1st_der vector_timeSeries_nonNA:")
      result <- NA
      result <- std1st_der(vector_timeSeries_nonNA)
      df_output[index_row, "std1stDer_filledTimeStamps"] <- result
    }
    
    print("histogram_mode condition:")
    if (length(vector_timeSeries_nonNA) <= 0) {
      df_output[index_row, "histogramMode_allTimeStamps"] <- NA
      df_output[index_row, "histogramMode_filledTimeStamps"] <- NA
    } else {
      print("histogram_mode vector_timeSeries_original:")
      result <- NA
      result <-
        histogram_mode(vector_timeSeries_original, numBins = 10)
      df_output[index_row, "histogramMode_allTimeStamps"] <- result
      print("histogram_mode vector_timeSeries_nonNA:")
      result <- NA
      result <-
        histogram_mode(vector_timeSeries_nonNA, numBins = 10)
      df_output[index_row, "histogramMode_filledTimeStamps"] <- result
    }
    
    # fluctanal_prop_r1(df_input[, feature_toMeasure])
    
    index_row <- index_row + 1
  }
  
  
  vector_temp <- colnames(df_output)
  for (f in vector_temp) {
    df_output[, f] <- as.character(df_output[, f])
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

vector_temp <- colnames(df_output)
for (f in vector_temp) {
  df_output[, f] <- as.character(df_output[, f])
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
