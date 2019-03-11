import string
import re
import numpy as np
from nltk.corpus import words
from nltk.tokenize import word_tokenize
from nltk.stem.snowball import SnowballStemmer
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer

def getWordVector(word, glove_word_vec_dict):
	if word in glove_word_vec_dict:
		return glove_word_vec_dict[word]
	return np.zeros_like(glove_word_vec_dict["dummy"])

def sumVectors(finalList, glove_word_vec_dict):
	numNonZero = 0
	vector = np.zeros_like(glove_word_vec_dict["dummy"])
	for word in finalList:
		vect = getWordVector(word,glove_word_vec_dict)
		if vect.sum() != 0:
			vector += vect
			numNonZero += 1
	if numNonZero:
		vector = vector/numNonZero
	return vector

def simplify(word):
	dump = ''
	temp = []
	listOfWords = filter(None,re.split("([A-Z][^A-Z]*)",word))
	if len(listOfWords) == len(word):
		return word.lower()
	for i in range(len(listOfWords)):
		listOfWords[i] = listOfWords[i].lower()
		if len(listOfWords[i]) == 1:
			dump = dump + listOfWords[i]
			if dump in words.words() and len(dump) > 2:
				temp.append(dump)
				dump = ''
		else:
			temp.append(listOfWords[i])
	return temp

def glove(glove_word_vec_dict, trainTweets, testTweets):
	def createTokens(data,glove_word_vec_dict):
		listOfTweets = []
		listOfStances = []
		tweetVector = []
		for ind, row in data.iterrows():
			# Create a sentence using target and the tweet. Word vector will be formed from this.
			example_sentence = str(row["Target"]) + " " + str(row["Tweet"])
			# Remove punctuation
			final_sentence = example_sentence.translate(None, string.punctuation)
			wordList = word_tokenize(final_sentence)
			finalList = []
			s = ' '.join([i for i in wordList if i.isalpha()])
			# create tokens from the string and stem them
			wordList = word_tokenize(s)
			for word in wordList:
				#to break any combined word into its components for eg, hashtags
				finalList += simplify(word)
			final_sentence = ' '.join(finalList)
			# All tweets from training data
			listOfTweets.append(final_sentence)
			# All stances from training data
			listOfStances.append(row["Stance"])
			tweetVector.append(sumVectors(finalList,glove_word_vec_dict))
		return listOfTweets, listOfStances, tweetVector
	# Remove punctuation from and tokenize the training tweets
	listOfTweets, listOfStances, trainTweetVector = createTokens(trainTweets, glove_word_vec_dict)
	# Remove punctuation from and tokenize the test tweets
	listOfTestTweets, listOfTestStances, testTweetVector = createTokens(testTweets, glove_word_vec_dict)
	Xtrain = np.asarray(trainTweetVector)
	Ytrain = np.asarray(listOfStances)
	Xtest = np.asarray(testTweetVector)
	Ytest = np.asarray(listOfTestStances)

	return Xtrain, Ytrain, Xtest, Ytest

def tfidf(trainTweets, testTweets):
	stemmer = SnowballStemmer("english")
	def createTokens(data):
		listOfTweets = []
		listOfStances = []
		for ind, row in data.iterrows():
			# Create a sentence using target and the tweet. Word vector will be formed from this.
			example_sentence = str(row["Target"]) + " " + str(row["Tweet"])
			# Remove punctuation
			final_sentence = example_sentence.translate(None, string.punctuation)
			wordList = word_tokenize(final_sentence)
			simpleWordList = []
			s = ' '.join([i for i in wordList if i.isalpha()])
			# create tokens from the string and stem them
			wordList = word_tokenize(s)
			for word in wordList:
				#to break any combined word into its components for eg, hashtags
				simpleWordList += simplify(word)
			# we stem the words to reduce the number of unique words by reducing the words to their origins
			final_sentence = ' '.join(stemmer.stem(word) for word in simpleWordList)
			# All tweets from training data
			listOfTweets.append(final_sentence)
			# All stances from training data
			listOfStances.append(row["Stance"])
		return listOfTweets, listOfStances
	# Remove punctuation from and tokenize the training tweets
	listOfTweets, listOfStances = createTokens(trainTweets)
	# Remove punctuation from and tokenize the test tweets
	listOfTestTweets, listOfTestStances = createTokens(testTweets)
	# We vectorize the tweets into unigrams and bigrams after removing the stopwords
	count_vect = CountVectorizer(ngram_range = (1,2), stop_words="english")
	tfidf_transformer = TfidfTransformer()
	# build a document term matrix of training data and then convert it to a tfidf matrix
	X_train_counts = count_vect.fit_transform(listOfTweets)
	X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)
	Xtrain = X_train_tfidf.toarray()
	Ytrain = np.asarray(listOfStances)
	# build a vector based on words from training data and get a document term matrix on test data
	# and then change it to a tfidf matrix
	X_test_counts = count_vect.fit(listOfTweets).transform(listOfTestTweets)
	X_test_tfidf = tfidf_transformer.fit_transform(X_test_counts)
	Xtest = X_test_tfidf.toarray()
	Ytest = np.asarray(listOfTestStances)

	return Xtrain, Ytrain, Xtest, Ytest