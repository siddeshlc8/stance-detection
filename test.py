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

	glove_word_vec_dict = readGlobalVecData(gloveFile)

	trainTweets = pd.read_csv(training, sep='\t',header=0)
	testTweets = pd.read_csv(test, sep='\t',header=0)
	uniqTrainTargets = trainTweets.Target.unique()

	# For converting all the stances into numerical values in both training and test data
	labelDict = {0:"AGAINST", 1:"FAVOR", 2:"NONE"}
	trainTweets = labelStance(labelDict, trainTweets)
	testTweets = labelStance(labelDict, testTweets)
	i=0
	for target in uniqTrainTargets:
		print "Vectorizing the input and building model for "+target+"..."
		Xtrain, Ytrain, Xtest, Ytest = vectorize.glove(glove_word_vec_dict, trainTweets[trainTweets["Target"]==target], testTweets[testTweets["Target"]==target])
		np.savetxt("Glove_X_train"+str(i)+".txt", Xtrain)
		np.savetxt("Glove_y_train"+str(i)+".txt", Ytrain)
		np.savetxt("Glove_X_test"+str(i)+".txt", Xtest)
		np.savetxt("Glove_y_test"+str(i)+".txt", Ytest)
		i=i+1
	
	# data = np.loadtxt("vector.txt",skiprows=0)
	# data = ((data+1/2)/2/2)*256
	# data = data.astype(int)
	# print data[0]

	# from PIL import Image 
	# img1 = Image.fromarray(data[1].reshape(10,20), 'L')
	# img2 = Image.fromarray(data[8].reshape(10,20), 'L')

	# import matplotlib.pyplot as plt
	# import matplotlib.image as mpimg
	# imgplot = plt.imshow(img1)
	# plt.show()
	# imgplot = plt.imshow(img2)
	# plt.show()
