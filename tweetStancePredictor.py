import sys
import datetime
import numpy as np
import pandas as pd
import vectorize
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.model_selection import cross_validate as cross_validation, ShuffleSplit, cross_val_score

def labelStance(labelDict, data):
	for key, val in labelDict.items():
		data.loc[data["Stance"] == val, "Stance"] = int(key)
	return data

def readGlobalVecData(glove_word_vec_file):
	file = open(glove_word_vec_file, 'r')
	rawData = file.readlines()
	glove_word_vec_dict = {}
	for line in rawData:
		line = line.strip().split()
		tag = line[0]
		vec = line[1:]
		glove_word_vec_dict[tag] = np.array(vec, dtype=float)
	return glove_word_vec_dict

if __name__ == "__main__":
	start = datetime.datetime.now()
	classifier = "svc"
	training = "Dataset-SemEval2016/training.txt"
	test = "Dataset-SemEval2016/test-gold.txt"
	gloveFile = "/home/siddeshlc8/siddeshlc/Glove Vectorization/glove.twitter.27B.200d.txt"
	logFile = "log"+classifier.upper()+".txt"
	log = open(logFile, 'a')
	logMsg = "Timestamp: "+str(datetime.datetime.now())+"\n"
	vect = raw_input("\nSelect choice for vectorization\n'g' for using glove\n't' for using tfidf\n: ").strip().lower()
	if vect not in ['g', 't']:
		print "Wrong choice for vectorization choice.. terminating..."
		sys.exit(0)
	elif vect == 'g':
		print "\nLoading glove data..."
		glove_word_vec_dict = readGlobalVecData(gloveFile)
	choice = 'a'
	print "Thanks for your inputs...processing input now..."
	logMsg += "Vectorization: "+vect+"\n"+"Evaluation: "+choice+"\n"

	trainTweets = pd.read_csv(training, sep='\t',header=0)
	testTweets = pd.read_csv(test, sep='\t',header=0)
	uniqTrainTargets = trainTweets.Target.unique()

	# For converting all the stances into numerical values in both training and test data
	labelDict = {0:"AGAINST", 1:"FAVOR", 2:"NONE"}
	trainTweets = labelStance(labelDict, trainTweets)
	testTweets = labelStance(labelDict, testTweets)

	totalAcc = 0
	for target in uniqTrainTargets:
		print "Vectorizing the input and building model for "+target+"..."
		if vect == "g":
			Xtrain, Ytrain, Xtest, Ytest = vectorize.glove(glove_word_vec_dict, trainTweets[trainTweets["Target"]==target], testTweets[testTweets["Target"]==target])
			# print Xtrain
			# print Xtest
		else:
			Xtrain, Ytrain, Xtest, Ytest = vectorize.tfidf(trainTweets[trainTweets["Target"]==target], testTweets[testTweets["Target"]==target])
			# print Xtrain
			# print Xtest

		if choice == 'a':
			if classifier.lower() == "svc":
				clf = SVC(kernel="rbf").fit(Xtrain, Ytrain)
				acc = clf.score(Xtest, Ytest)
				print "Test accuracy score by SVC for "+target+":", acc
		totalAcc += acc
		logMsg += target+": "+ str(round(acc*100,2))+"%"+"\n"
	overallAcc = totalAcc/len(uniqTrainTargets)
	logMsg += "Overall accuracy: "+str(round(overallAcc*100,2))+"%"+"\n"
	print "\nOverall accuracy: "+str(round(overallAcc*100,2))+"%"
	logMsg += str(clf)+"\n"
	logMsg += "*"*150+"\n"+"\n"
	log.write(logMsg)
	log.close()

	print "Total execution time:", (datetime.datetime.now() - start).total_seconds(), "seconds"