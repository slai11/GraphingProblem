# _*_ coding: utf-8 _*_
import csv
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd
import xgboost as xgb
from sklearn.preprocessing import MinMaxScaler, scale
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC, SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import ExtraTreesClassifier, GradientBoostingClassifier, RandomForestClassifier, AdaBoostClassifier, VotingClassifier, BaggingClassifier
from sklearn import metrics, cross_validation
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.feature_selection import SelectPercentile, chi2
from sklearn.dummy import DummyClassifier


from graph import *



############
# Movement #
############
def dummy():
	scaler = MinMaxScaler()
	du = DummyClassifier(strategy='most_frequent', random_state=0)
	pipeline = Pipeline([('scale', scaler), ('dummy', du)])
	return (pipeline, pipeline)

def log_reg_model():
	select = SelectPercentile(score_func=chi2, percentile=65) #gridsearched
	log = LogisticRegression(class_weight='balanced', penalty='l2', tol=1e-8, C=64) #gridsearched
	scaler = MinMaxScaler()

	select_status = SelectPercentile(score_func=chi2, percentile=41) #gridsearched
	log_status = LogisticRegression(class_weight='balanced', penalty='l2', tol=0.1, C=0.16) #gridsearched

	pipeline1 = Pipeline([('scale', scaler), ('select', select_status), ('logre', log_status)])
	pipeline2 = Pipeline([('scale', scaler), ('select', select), ('logre', log)])
	return (pipeline1, pipeline1)

def linear_svc_model():
	select = SelectPercentile(score_func=chi2, percentile=66) #gridsearched
	svc = LinearSVC(dual=False, class_weight='balanced', penalty='l2', C=1024, tol=0.1)  #gridsearched
	#(dual=False, class_weight='balanced', C=1024, penalty='l2', tol=0.1)  #gridsearched
	scaler = MinMaxScaler()

	select_status = SelectPercentile(score_func=chi2, percentile=31) #gridsearched
	svc_status = LinearSVC(dual=False, class_weight='balanced', penalty='l1', C=0.25, tol=1e-8)  #gridsearched


	pipeline1 = Pipeline([('scale', scaler), ('select', select_status), ('linsvc', svc_status)])
	pipeline2 = Pipeline([('scale', scaler), ('select', select), ('linsvc', svc)])
	return (pipeline1, pipeline2)

def rbf_svc_model():
	select = SelectPercentile(score_func=chi2, percentile=93) #gridsearched
	svc = SVC(class_weight='balanced', gamma=1, kernel='rbf', C=4, tol=1e-8)  #gridsearched
	scaler = MinMaxScaler()

	select_status = SelectPercentile(score_func=chi2, percentile=41) #gridsearched
	svc_status = SVC(class_weight='balanced', gamma=9, kernel='rbf', C=64, tol=1e-8)  #gridsearched

	pipeline1 = Pipeline([('scale', scaler), ('select', select_status), ('svc', svc_status)])
	pipeline2 = Pipeline([('scale', scaler), ('select', select), ('svc', svc)])
	return (pipeline1, pipeline2)

def random_forest_model():
	rf = RandomForestClassifier(class_weight='balanced', max_depth=30, max_features='auto', min_samples_split=81,n_estimators=21)#gridsearched
	select = SelectPercentile(score_func=chi2, percentile=80)#gridsearched

	rf_status = RandomForestClassifier(class_weight='balanced', max_depth=None, max_features=None, min_samples_split=1,n_estimators=91, criterion='entropy')#gridsearched
	select_status = SelectPercentile(score_func=chi2, percentile=10)#gridsearched

	scaler = MinMaxScaler()
	pipeline1 = Pipeline([('scale', scaler), ('select',select_status),('randf', rf_status)])
	pipeline2 = Pipeline([('scale', scaler), ('select',select),('randf', rf)])
	return (pipeline1, pipeline2)

def extra_trees_model():
	select = SelectPercentile(score_func=chi2, percentile=70) #gridsearched
	et = ExtraTreesClassifier(class_weight='balanced', max_features='auto', n_estimators=1) #gridsearched
	select_status = SelectPercentile(score_func=chi2, percentile=30) #gridsearched
	et_status = ExtraTreesClassifier(class_weight='balanced', max_features=None, n_estimators=41) #gridsearched
	scaler = MinMaxScaler()
	pipeline1 = Pipeline([('scale', scaler), ('select', select_status), ('extra', et_status)])
	pipeline2 = Pipeline([('scale', scaler), ('select', select), ('extra', et)])
	return (pipeline1, pipeline2)

def k_nearest_model():
	select = SelectPercentile(score_func=chi2, percentile=91)
	knc = KNeighborsClassifier(weights='uniform', algorithm='auto', n_neighbors=1)

	select_status = SelectPercentile(score_func=chi2, percentile=1)
	knc_status = KNeighborsClassifier(weights='uniform', algorithm='auto', n_neighbors=3)

	scaler = MinMaxScaler()
	pipeline1 = Pipeline([('scale', scaler), ('select', select_status) , ('knear', knc_status)])
	pipeline2 = Pipeline([('scale', scaler), ('select', select) , ('knear', knc)])
	return (pipeline1, pipeline2)

def gradient_boost_model():
	gb = GradientBoostingClassifier(learning_rate=13, loss='exponential',\
									max_depth=5, max_features='sqrt', random_state=1)#gridsearched
	select = SelectPercentile(score_func=chi2, percentile=71)#gridsearched
	scaler = MinMaxScaler()
	pipeline = Pipeline([('scale', scaler), ('select', select), ('gb', gb)])
	return (pipeline, pipeline)

def ada_boost_model():
	select = SelectPercentile(score_func=chi2, percentile=90)#gridsearched
	scaler = MinMaxScaler()
	#clf=RandomForestClassifier(class_weight='balanced', max_depth=30, max_features='auto', min_samples_split=81,n_estimators=21)#gridsearched
	clf = RandomForestClassifier(class_weight='balanced', max_depth=None, max_features=None, min_samples_split=1,n_estimators=91, criterion='entropy')#gridsearched
	ada = AdaBoostClassifier(clf)
	#pipeline = Pipeline([('scale', scaler), ('select', select), ('ada', ada)])
	return ada

def xgboost_model():
	xgboost = xgb.XGBClassifier(gamma=0.1, max_depth=7, subsample=2, colsample_bytree=0.9)
	xgboost_status = xgb.XGBClassifier(n_estimators =90,reg_lambda=0.011, gamma=0.5, max_depth=16, subsample=1, colsample_bytree=1)
	tempxgboost = xgb.XGBClassifier(gamma=0, max_depth=16, subsample=0.85, colsample_bytree=0.7, scale_pos_weight=1, min_child_weight=1, reg_alpha=1e-5, n_estimators=37)
	scaler = MinMaxScaler()
	pipeline1 = Pipeline([('scale', scaler), ('xgb', tempxgboost)])
	pipeline2 = Pipeline([('scale', scaler), ('xgb', xgboost)])
	return (pipeline1, pipeline2)
	

def voting_ensemble():
	select = SelectPercentile(score_func=chi2, percentile=26)#gridsearched
	scaler = MinMaxScaler
	clf1 = gradient_boost_model()[1]
	clf2 = linear_svc_model()[1]
	clf3 = log_reg_model()[1]

	clf11 = xgboost_model()[0]
	clf21 = random_forest_model()[0]
	clf31 = k_nearest_model()[0]
	
	model1=[('1', clf11),('2', clf21),('3', clf31)]
	model2=[('1', clf1),('2', clf2),('3', clf3)]
	
	vote1 = VotingClassifier(estimators=model1, voting='hard')
	vote2 = VotingClassifier(estimators=model2, voting='hard')
	#pipeline = Pipeline([('scale', scaler), ('select', select), ('vote', vote)])
	return (vote1, vote2)

def bag_model():
	clf = k_nearest_model()[0]
	bag = BaggingClassifier(clf, max_samples=0.5, max_features=0.1, n_estimators=5)
	return bag


