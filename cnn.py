
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
import numpy as np


def Model1(X_train, X_test, y_train, y_test):
	model = Sequential()
	model.add(Conv2D(32, kernel_size=(3, 3),
	                 activation='relu',
	                 input_shape=(10,20,3)))
	model.add(Conv2D(64, (3, 3), activation='relu'))
	model.add(MaxPooling2D(pool_size=(2, 2)))
	model.add(Dropout(0.25))
	model.add(Flatten())
	model.add(Dense(128, activation='relu'))
	model.add(Dropout(0.5))
	model.add(Dense(num_classes, activation='softmax'))

	model.compile(loss=keras.losses.categorical_crossentropy,
	              optimizer=keras.optimizers.Adadelta(),
	              metrics=['accuracy'])

	model.fit(X_train, y_train,
	          batch_size=batch_size,
	          epochs=epochs,
	          verbose=1,
	          validation_data=(X_test, y_test))
	score = model.evaluate(X_test, y_test, verbose=0)
	# print('Test loss:', score[0])
	# print('Test accuracy:', score[1])
	return score[1]

def Model1(X_train, X_test, y_train, y_test):
	model = Sequential()
	model.add(Conv2D(32, kernel_size=(3, 3),
	                 activation='relu',
	                 input_shape=(10,20,1)))
	model.add(Conv2D(64, (3, 3), activation='relu'))
	model.add(MaxPooling2D(pool_size=(2, 2)))
	model.add(Dropout(0.25))
	model.add(Flatten())
	model.add(Dense(128, activation='relu'))
	model.add(Dropout(0.5))
	model.add(Dense(num_classes, activation='softmax'))

	model.compile(loss=keras.losses.categorical_crossentropy,
	              optimizer=keras.optimizers.Adadelta(),
	              metrics=['accuracy'])

	model.fit(X_train, y_train,
	          batch_size=batch_size,
	          epochs=epochs,
	          verbose=1,
	          validation_data=(X_test, y_test))
	score = model.evaluate(X_test, y_test, verbose=0)
	# print('Test loss:', score[0])
	# print('Test accuracy:', score[1])
	return score[1]


batch_size = 10
num_classes = 3
epochs = 12

# input image dimensions
img_rows, img_cols = 10, 20

from PIL import Image 

Tacc = 0
for i in range(0,5):
	X_train = np.loadtxt("DataSet-After-Glove-Vectorization/Glove_X_train"+str(i)+".txt",skiprows=0)
	X_test = np.loadtxt("DataSet-After-Glove-Vectorization/Glove_X_test"+str(i)+".txt",skiprows=0)
	y_train = np.loadtxt("DataSet-After-Glove-Vectorization/Glove_y_train"+str(i)+".txt",skiprows=0)
	y_test = np.loadtxt("DataSet-After-Glove-Vectorization/Glove_y_test"+str(i)+".txt",skiprows=0)


	mylist = []
	for item in X_train:
		img = Image.fromarray(item.reshape(10,20), 'L')
		img = np.asarray( img, dtype="int32" )
		mylist.append(img)
	X_train = np.array(mylist)


	mylist = []
	for item in X_test:
		img = Image.fromarray(item.reshape(10,20), 'L')
		img = np.asarray( img, dtype="int32" )
		mylist.append(img)
	X_test = np.array(mylist)

	X_train = X_train.reshape(X_train.shape[0],10,20,1)
	X_test = X_test.reshape(X_test.shape[0],10,20,1)

	# convert class vectors to binary class matrices
	y_train = keras.utils.to_categorical(y_train, num_classes)
	y_test = keras.utils.to_categorical(y_test, num_classes)

	acc = Model1(X_train, X_test, y_train, y_test)
	Tacc += acc
print "Total accuracy : ", Tacc/5


