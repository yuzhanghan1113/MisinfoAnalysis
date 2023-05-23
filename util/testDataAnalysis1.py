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



absFilename_input = "D:\\vmwareSharedFolder\\TwitterDataCollection\\data\\PFN\\snapshot\\dateRetrieval=20201111\\retweets\\rootTweetID=1258750892448387074\\retweets_rootTweetID=1258750892448387074.txt"

file_input = open(absFilename_input, "r", encoding="utf-8")

fullText = "EMPTY"
userDescription = "EMPTY"

for line in file_input:

	dict_retweet = json.loads(line)

	if dict_retweet["id_str"] == "1258761859781574656":

		# fullText = str(dict_retweet["full_text"])
		fullText = str(dict_retweet["text"])
		userDescription = str(dict_retweet["user"]["description"])

print("fullText:")
print(fullText)

print("userDescription:")
print(userDescription)
	

list_strings = [fullText, userDescription]

analyzer = SentimentIntensityAnalyzer()

list_metrics = ["neg", "neu", "pos", "compound"]


for string in list_strings:

	print(string + ":")

	dict_scores = analyzer.polarity_scores(string)

	for metric in list_metrics:

		print(metric + ":")

		result = dict_scores[metric]

		print(result)

	result = TextBlob(string).sentiment.subjectivity

	print("subjectivity:")

	print(result)




nlp = spacy.load('en', disable=['parser', 'ner'])
list_stopWords = list(set(stopwords.words('english')))

list_metrics = ["fear", "anger", "anticip", "trust", "surprise", "positive", "negative", "sadness", "disgust", "joy"]

for string in list_strings:

	print(string + ":")

	dict_scores = NRCLex(preprocessText(string, nlp, list_stopWords))

	for metric in list_metrics:

		print(metric + ":")

		result = dict_scores.affect_frequencies[metric]

		print(result)


	

list_metrics = ["charCount", "wordCount", "uqWordCount", "sentenceCount", "charsPerWord", "wordsPerSentence", "readability", "readability"]

for string in list_strings:

	print(string + ":")

	dict_scores = textStats(string)

	for metric in list_metrics:

		print(metric + ":")

		result = dict_scores[metric]

		print(result)




"""		





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


"""

