import numpy as np
import pandas as pd
from util import *
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import PReLU
from keras.layers.recurrent import SimpleRNN
from keras.utils import np_utils, generic_utils

from keras.wrappers.scikit_learn import KerasClassifier, KerasRegressor
from sklearn import metrics, cross_validation
from sklearn.cross_validation import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, make_scorer, f1_score, precision_recall_curve, average_precision_score, roc_auc_score
from sklearn.metrics import mean_squared_error

list_ = ["20160523", "20160524", "20160525", "20160527", "20160531", "20160601", "20160602"]

predmove = True
X, y = load_multiple_data(list_, study_period=3, ahead=2, delta=True, pred_movement=predmove, daily = True, finer_subzone=False) # basic + daily()


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=10)

print X_train.shape

## build keras model

def create_baseline():
	
	model = Sequential()

	#input layer
	model.add(Dense(56, input_shape = (X_train.shape[1],)))


	# output layer
	model.add(Dense(1))
	model.add(BatchNormalization())
	model.add(Dropout(0.5))
	model.add(Activation('sigmoid'))
	model.compile(loss='binary_crossentropy', optimizer="adam", metrics=['accuracy'])
	return model

#thank god for wrappers

def nn_model():
	return KerasClassifier(build_fn=create_baseline, nb_epoch=20, batch_size=50, verbose = 1)


model = KerasClassifier(build_fn=create_baseline, nb_epoch=10, batch_size=80, verbose = 0)

model.fit(X_train, y_train, nb_epoch=7, batch_size=300, validation_split=0.1, show_accuracy=True)
scores = cross_validation.cross_val_score(model, X, y, cv = 5, scoring = "accuracy", n_jobs = -1, verbose = 1)
model.fit(X_train, y_train, verbose=2)

y_pred = model.predict(X_test)
'''
print y_pred
print y_test

print mean_squared_error(y_test, y_pred)


'''
#scores = roc_auc_score(y_test,y_pred)
print scores
#print f1_score(y_test, y_pred, average='macro')
print classification_report(y_test, y_pred)
print f1_score(y_test, y_pred, average='binary')
cm = confusion_matrix(y_test, y_pred)

print "Confusion Matrix:"
print cm




