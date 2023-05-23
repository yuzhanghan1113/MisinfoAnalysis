import json
import os
import getopt
import sys
import random
import pandas as pd
import numpy as np
import traceback
import gc 
import time
from datetime import datetime, timezone
from collections import Counter
import re
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from textblob import TextBlob
import preprocessor
import nltk
from nltk.tokenize import TweetTokenizer
from nltk import sent_tokenize, FreqDist
from nltk.corpus import stopwords
from nltk.stem import LancasterStemmer
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nrclex import NRCLex
import spacy
import textstat
import string
from nltk.tokenize import sent_tokenize
import csv


def preprocessText(text, nlp, list_stopWords):

    text = re.sub(r"#", "", text)
    # print(text)
    text = preprocessor.clean(text)
    text = text.lower()
    text = re.sub(r"\d+", "", text)
    
    list_words = nlp(text)
    list_words = [token.lemma_ for token in list_words]
    list_words = [w for w in list_words if w != "-PRON-"]
    list_words = [w for w in list_words if w not in list_stopWords]
    text = " ".join(list_words)
    
    return text

def textStats(text:str):

    text_org = str(text)

    dict_result = {}
    text = text.lower()
    text = re.sub(r"#", "", text)
    text = preprocessor.clean(text)
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"[0-9]", " ", text)
    list_words = re.split(" +|\t+|\n+", text)
    list_words = [w for w in list_words if w != ""]
    temp_text = re.sub(r" +|\t+|\n+", "", text)

    # print("text_org:")
    # print(text_org)
    # print("text:")
    # print(text)
    # print("temp_text:")
    # print(temp_text)

    dict_result["charCount"] = len(temp_text)
    dict_result["wordCount"] = len(list_words)
    dict_result["uqWordCount"] = len(list(set(list_words)))

    temp_sentences = [s for s in re.split(r"[.!?\n]+", text_org) if s != ""]
    # print(temp_sentences)
    dict_result["sentenceCount"] = len(temp_sentences)

    if dict_result["wordCount"] <= 0:
        dict_result["charsPerWord"] = np.nan
    else:
        dict_result["charsPerWord"] = dict_result["charCount"]/dict_result["wordCount"]

    if dict_result["sentenceCount"] <= 0:
        dict_result["wordsPerSentence"] = np.nan
    else:
        dict_result["wordsPerSentence"] = dict_result["wordCount"]/dict_result["sentenceCount"]

    dict_result["readability"] = textstat.flesch_reading_ease(text_org)

    return dict_result



# absFilename_input = "D:\\vmwareSharedFolder\\TwitterDataAnalysis\\programs\\util\\retweets_rootTweetID=1251000911595536384_230min.txt"
absFilename_input = "D:\\vmwareSharedFolder\\TwitterDataAnalysis\\programs\\util\\testTweets.txt"




file_input = open(absFilename_input, "r")

num_lines = 0
for line in file_input:
	num_lines += 1

file_input.close()

print("num_lines:")
print(num_lines)

str_date_current = "Thu Nov 12 00:00:00 +0000 2020"
date_current = datetime.strptime(str_date_current, "%a %b %d %H:%M:%S %z %Y")

str_createdAt_rootTweet = "Fri Apr 17 04:13:54 +0000 2020"
date_createdAt_rootTweet = datetime.strptime(str_createdAt_rootTweet, "%a %b %d %H:%M:%S %z %Y")

list_attributes = ["followers_count", "friends_count", "listed_count", "favourites_count", "statuses_count"]

for att in list_attributes:

	print(att + ":")
	list_values = []

	file_input = open(absFilename_input, "r")

	for line in file_input:
		dict_tweet = json.loads(line)
		value = float(dict_tweet["user"][att])
		list_values += [value]

	file_input.close()


	print("mean: " + str(np.nanmean(list_values)))
	print("median: " + str(np.nanmedian(list_values)))



list_attributes = ["followers_count", "friends_count", "listed_count", "favourites_count", "statuses_count"]

print("attributes:")
print("dict_tweet[\"retweeted_status\"][\"user\"]:")
print(list_attributes)

for att in list_attributes:

	print(att + ":")
	list_values = []

	file_input = open(absFilename_input, "r")

	for line in file_input:
		dict_tweet = json.loads(line)
		if "retweeted_status" in dict_tweet:
			value = float(dict_tweet["retweeted_status"]["user"][att])
			list_values += [value]

	file_input.close()


	print("mean: " + str(np.nanmean(list_values)))
	print("median: " + str(np.nanmedian(list_values)))





list_values = []

print("retweet_mean_user_accountAge_day:")

file_input = open(absFilename_input, "r")

for line in file_input:
	dict_tweet = json.loads(line)
	date_createdAt_tweet = datetime.strptime(dict_tweet["user"]["created_at"], "%a %b %d %H:%M:%S %z %Y")
	value = (date_current - date_createdAt_tweet).total_seconds()/(60*60*24)
	list_values += [value]

file_input.close()

print("mean: " + str(np.nanmean(list_values)))
print("median: " + str(np.nanmedian(list_values)))


list_attributes = ["geo_enabled", "verified", "profile_use_background_image", "default_profile"]

list_values = []

for att in list_attributes:

	print(att + ":")
	list_values = []

	file_input = open(absFilename_input, "r")

	for line in file_input:
		dict_tweet = json.loads(line)
		value = str(dict_tweet["user"][att])
		list_values += [value]

	file_input.close()

	print(list_values)




list_attributes = ["geo", "coordinates"]

list_values = []

for att in list_attributes:

	print(att + ":")
	list_values = []

	file_input = open(absFilename_input, "r")

	for line in file_input:
		dict_tweet = json.loads(line)
		value = str(dict_tweet[att])
		list_values += [value]

	file_input.close()

	print(list_values)




list_attributes = ["retweet_count", "favorite_count"]

list_values = []

for att in list_attributes:

	print(att + ":")
	list_values = []

	file_input = open(absFilename_input, "r")

	for line in file_input:
		dict_tweet = json.loads(line)
		if "retweeted_status" in dict_tweet:
			value = str(dict_tweet["retweeted_status"][att])
			list_values += [value]

	file_input.close()

	print(list_values)





analyzer = SentimentIntensityAnalyzer()

list_attributes = ["description"]

list_metrics = ["neg", "neu", "pos", "compound"]

list_values = []

for att in list_attributes:

	print(att + ":")

	for metric in list_metrics:

		print(metric + ":")

		# print(att + ":")
		list_values = []

		file_input = open(absFilename_input, "r")

		for line in file_input:
			dict_tweet = json.loads(line)
			value = str(dict_tweet["user"][att])

			dict_scores = analyzer.polarity_scores(value)
			list_values += [dict_scores[metric]]

		file_input.close()

		print("mean: " + str(np.nanmean(list_values)))
		print("median: " + str(np.nanmedian(list_values)))





list_attributes = ["description"]

for att in list_attributes:

	print(att + ":")
	print("subjectivity:")

	list_values = []

	file_input = open(absFilename_input, "r")

	for line in file_input:
		dict_tweet = json.loads(line)
		value = str(dict_tweet["user"][att])
		score = TextBlob(value).sentiment.subjectivity
		list_values += [score]

	file_input.close()

	print("mean: " + str(np.nanmean(list_values)))
	print("median: " + str(np.nanmedian(list_values)))


nlp = spacy.load('en', disable=['parser', 'ner'])
list_stopWords = list(set(stopwords.words('english')))

list_attributes = ["description"]

list_metrics = ["fear", "anger", "anticip", "trust", "surprise", "positive", "negative", "sadness", "disgust", "joy"]

list_values = []

for att in list_attributes:

	print(att + ":")

	for metric in list_metrics:

		print(metric + ":")

		# print(att + ":")
		list_values = []

		file_input = open(absFilename_input, "r")

		for line in file_input:
			dict_tweet = json.loads(line)
			value = str(dict_tweet["user"][att])

			dict_scores = NRCLex(preprocessText(value, nlp, list_stopWords))
			list_values += [dict_scores.affect_frequencies[metric]]

		file_input.close()

		print("mean: " + str(np.nanmean(list_values)))
		print("median: " + str(np.nanmedian(list_values)))




list_attributes = ["description"]

list_metrics = ["charCount", "wordCount", "uqWordCount", "sentenceCount", "charsPerWord", "wordsPerSentence", "readability", "readability"]

list_values = []

for att in list_attributes:

	print(att + ":")

	for metric in list_metrics:

		print(metric + ":")

		# print(att + ":")
		list_values = []

		file_input = open(absFilename_input, "r")

		for line in file_input:
			dict_tweet = json.loads(line)
			value = str(dict_tweet["user"][att])

			dict_scores = textStats(value)
			list_values += [dict_scores[metric]]

		file_input.close()
		print(list_values)
		print("mean: " + str(np.nanmean(list_values)))
		print("median: " + str(np.nanmedian(list_values)))




analyzer = SentimentIntensityAnalyzer()

list_attributes = ["full_text"]

list_metrics = ["neg", "neu", "pos", "compound"]

list_values = []

for att in list_attributes:

	print(att + ":")

	for metric in list_metrics:

		print(metric + ":")

		# print(att + ":")
		list_values = []

		file_input = open(absFilename_input, "r")

		for line in file_input:
			dict_tweet = json.loads(line)
		file_input.close()

		if "retweeted_status" in dict_tweet:
			value = str(dict_tweet["retweeted_status"][att])

			dict_scores = analyzer.polarity_scores(value)
			print(dict_scores[metric])




analyzer = SentimentIntensityAnalyzer()

list_attributes = ["full_text"]

list_metrics = ["neg", "neu", "pos", "compound"]

print("attributes:")
print("dict_tweet[\"retweeted_status\"][att]")
print(list_attributes)
print(list_metrics)

list_values = []

for att in list_attributes:

	print(att + ":")

	for metric in list_metrics:

		print(metric + ":")

		# print(att + ":")
		list_values = []

		file_input = open(absFilename_input, "r")

		for line in file_input:
			dict_tweet = json.loads(line)
			if "retweeted_status" in dict_tweet:
				value = str(dict_tweet["retweeted_status"][att])
				dict_scores = analyzer.polarity_scores(value)
				list_values += [dict_scores[metric]]
		file_input.close()

		print(list_values)
		print("mean: " + str(np.nanmean(list_values)))
		print("median: " + str(np.nanmedian(list_values)))

		
analyzer = SentimentIntensityAnalyzer()

list_attributes = ["full_text"]

list_metrics = ["neg", "neu", "pos", "compound"]

print("attributes:")
print("dict_tweet[att]")
print(list_attributes)
print(list_metrics)

list_values = []

for att in list_attributes:

	print(att + ":")

	for metric in list_metrics:

		print(metric + ":")

		# print(att + ":")
		list_values = []

		file_input = open(absFilename_input, "r")

		for line in file_input:
			dict_tweet = json.loads(line)
			value = str(dict_tweet[att])
			dict_scores = analyzer.polarity_scores(value)
			list_values += [dict_scores[metric]]
		file_input.close()

		print(list_values)
		print("mean: " + str(np.nanmean(list_values)))
		print("median: " + str(np.nanmedian(list_values)))



analyzer = SentimentIntensityAnalyzer()

list_attributes = ["full_text"]

for att in list_attributes:

	print(att + ":")

	file_input = open(absFilename_input, "r")

	for line in file_input:
		dict_tweet = json.loads(line)
	file_input.close()

	if "retweeted_status" in dict_tweet:
		value = str(dict_tweet["retweeted_status"][att])

		dict_scores = TextBlob(value).sentiment.subjectivity
		print(dict_scores)



analyzer = SentimentIntensityAnalyzer()

list_attributes = ["description"]

list_metrics = ["neg", "neu", "pos", "compound"]

list_values = []

for att in list_attributes:

	print(att + ":")

	for metric in list_metrics:

		print(metric + ":")

		# print(att + ":")
		list_values = []

		file_input = open(absFilename_input, "r")

		for line in file_input:
			dict_tweet = json.loads(line)
		file_input.close()

		if "retweeted_status" in dict_tweet:
			value = str(dict_tweet["retweeted_status"]["user"][att])

			dict_scores = analyzer.polarity_scores(value)
			print(dict_scores[metric])



analyzer = SentimentIntensityAnalyzer()

list_attributes = ["description"]

for att in list_attributes:

	print(att + ":")

	file_input = open(absFilename_input, "r")

	for line in file_input:
		dict_tweet = json.loads(line)
	file_input.close()

	if "retweeted_status" in dict_tweet:
		value = str(dict_tweet["retweeted_status"]["user"][att])

		dict_scores = TextBlob(value).sentiment.subjectivity
		print(dict_scores)


nlp = spacy.load('en', disable=['parser', 'ner'])
list_stopWords = list(set(stopwords.words('english')))

list_attributes = ["full_text"]

list_metrics = ["fear", "anger", "anticip", "trust", "surprise", "positive", "negative", "sadness", "disgust", "joy"]

list_values = []

for att in list_attributes:

	print(att + ":")

	for metric in list_metrics:

		print(metric + ":")

		# print(att + ":")
		list_values = []

		file_input = open(absFilename_input, "r")

		for line in file_input:
			dict_tweet = json.loads(line)
		file_input.close()

		if "retweeted_status" in dict_tweet:
			value = str(dict_tweet["retweeted_status"][att])

			dict_scores = NRCLex(preprocessText(value, nlp, list_stopWords))
			print(dict_scores.affect_frequencies[metric])



nlp = spacy.load('en', disable=['parser', 'ner'])
list_stopWords = list(set(stopwords.words('english')))

list_attributes = ["description"]

list_metrics = ["fear", "anger", "anticip", "trust", "surprise", "positive", "negative", "sadness", "disgust", "joy"]

list_values = []

for att in list_attributes:

	print(att + ":")

	for metric in list_metrics:

		print(metric + ":")

		# print(att + ":")
		list_values = []

		file_input = open(absFilename_input, "r")

		for line in file_input:
			dict_tweet = json.loads(line)
		file_input.close()

		if "retweeted_status" in dict_tweet:
			value = str(dict_tweet["retweeted_status"]["user"][att])

			dict_scores = NRCLex(preprocessText(value, nlp, list_stopWords))
			print(dict_scores.affect_frequencies[metric])







list_attributes = ["full_text"]

list_metrics = ["charCount", "wordCount", "uqWordCount", "sentenceCount", "charsPerWord", "wordsPerSentence", "readability", "readability"]

list_values = []

for att in list_attributes:

	print(att + ":")

	for metric in list_metrics:

		print(metric + ":")

		# print(att + ":")
		list_values = []

		file_input = open(absFilename_input, "r")

		for line in file_input:
			dict_tweet = json.loads(line)
		file_input.close()

		if "retweeted_status" in dict_tweet:
			value = str(dict_tweet["retweeted_status"][att])

			dict_scores = textStats(value)
			print(dict_scores[metric])





list_attributes = ["description"]

list_metrics = ["charCount", "wordCount", "uqWordCount", "sentenceCount", "charsPerWord", "wordsPerSentence", "readability", "readability"]

list_values = []

for att in list_attributes:

	print(att + ":")

	for metric in list_metrics:

		print(metric + ":")

		# print(att + ":")
		list_values = []

		file_input = open(absFilename_input, "r")

		for line in file_input:
			dict_tweet = json.loads(line)
		file_input.close()

		if "retweeted_status" in dict_tweet:
			value = str(dict_tweet["retweeted_status"]["user"][att])

			dict_scores = textStats(value)
			print(dict_scores[metric])