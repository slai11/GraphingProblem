# _*_ coding: utf-8 _*_
import csv
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, scale
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC, SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import ExtraTreesClassifier, GradientBoostingClassifier, RandomForestClassifier
from sklearn import metrics, cross_validation
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.feature_selection import SelectPercentile, chi2
from sklearn.dummy import DummyClassifier

from graph import *





def dummy():
	scaler = MinMaxScaler()
	du = DummyClassifier(strategy='most_frequent', random_state=0)
	pipeline = Pipeline([('scale', scaler), ('dummy', du)])
	return pipeline

def log_reg_model():
	select = SelectPercentile(score_func=chi2, percentile=66) #gridsearched
	log = LogisticRegression(class_weight='balanced',C=262144.0, tol=0.1, penalty='l1')     #gridsearched
	scaler = MinMaxScaler()

	pipeline = Pipeline([('scale', scaler), ('select', select), ('logre', log)])
	return pipeline

def linear_svc_model():
	select = SelectPercentile(score_func=chi2, percentile=66) #gridsearched
	svc = LinearSVC(dual=False, class_weight='balanced', penalty='l1', C=1024, tol=0.1)  #gridsearched
	#(dual=False, class_weight='balanced', C=1024, penalty='l2', tol=0.1)  #gridsearched
	scaler = MinMaxScaler()
	pipeline = Pipeline([('scale', scaler), ('select', select), ('linsvc', svc)])
	return pipeline

def svc_model():
	select = SelectPercentile(score_func=chi2, percentile=21) #gridsearched
	svc = SVC(class_weight='balanced', gamma=0, kernel='rbf', C=0.25, tol=0.1)  #gridsearched
	scaler = MinMaxScaler()

	pipeline = Pipeline([('scale', scaler), ('select', select), ('svc', svc)])
	return pipeline

def random_forest_model():
	rf = RandomForestClassifier(max_features='log2', min_samples_split=11, n_estimators=71, max_depth=30, class_weight='balanced')#gridsearched
	select = SelectPercentile(score_func=chi2, percentile=20)#gridsearched
	scaler = MinMaxScaler()
	pipeline = Pipeline([('scale', scaler),('randf', rf)])
	return pipeline

def extra_trees_model():
	select = SelectPercentile(score_func=chi2, percentile=90)
	et = ExtraTreesClassifier(n_estimators=11, max_features='log2', class_weight='balanced')
	#n_estimators=10, max_depth=None, min_samples_split=1, random_state=0
	scaler = MinMaxScaler()
	pipeline = Pipeline([('scale', scaler), ('extra', et)])
	return pipeline

def k_nearest_model():
	select = SelectPercentile(score_func=chi2, percentile=45)
	knc = KNeighborsClassifier(weights='uniform', algorithm='auto', n_neighbors=3)
	scaler = MinMaxScaler()
	pipeline = Pipeline([('scale', scaler),  ('knear', knc)])
	return pipeline

def gradient_boost_model():
	gb = GradientBoostingClassifier(learning_rate=4, loss='exponential',\
									max_depth=3, max_features='sqrt', random_state=1)#gridsearched
	select = SelectPercentile(score_func=chi2, percentile=26)#gridsearched
	scaler = MinMaxScaler()
	pipeline = Pipeline([('scale', scaler), ('select', select), ('gb', gb)])
	return pipeline

