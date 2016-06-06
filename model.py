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

from features import BasicFeatureBuilder, GeospatialEffect, HitsChange, BadNeighbours

network_data = pd.read_csv("Data/network_20160511.csv")
wkend_network_data = pd.read_csv("Data/Original/20160514_network.csv")
subzone_data = pd.read_csv("Data/Processed/subzonedatav5.csv")
GG = GraphGenerator(network_data, subzone_data)
GG2 = GraphGenerator(wkend_network_data, subzone_data)
G, OG, BH = GG.get_graphs()
WG, WOG, WBH = GG2.get_graphs()
X = []




def dummy():
	scaler = MinMaxScaler()
	du = DummyClassifier(strategy='most_frequent', random_state=0)
	pipeline = Pipeline([('scale', scaler), ('dummy', du)])
	return pipeline	

def log_reg_model():
	select = SelectPercentile(score_func=chi2, percentile=64) #gridsearched
	log = LogisticRegression(tol=1e-8, penalty='l1', C=16384.0, class_weight='balanced')     #gridsearched
	scaler = MinMaxScaler()

	pipeline = Pipeline([('scale', scaler), ('logre', log)])
	return pipeline

def linear_svc_model():
	select = SelectPercentile(score_func=chi2, percentile=87) #gridsearched
	svc = LinearSVC(dual=False, class_weight='balanced', C=1024, penalty='l2', tol=0.1)  #gridsearched
	scaler = MinMaxScaler()
	pipeline = Pipeline([('scale', scaler), ('select', select),('linsvc', svc)])
	return pipeline

def svc_model():
	select = SelectPercentile(score_func=chi2, percentile=21) #gridsearched
	svc = SVC(class_weight='balanced', gamma=0, kernel='rbf', C=0.25, tol=0.1)  #gridsearched
	scaler = MinMaxScaler()

	BFB = BasicFeatureBuilder(G, OG, BH)
	BN = BadNeighbours(OG, BH)
	HC = HitsChange(OG, WOG)
	y = BFB.get_y()

	FU = FeatureUnion([('fb', BFB), ('bn',BN), ('hc',HC)])
	F = FU.fit_transform(X,y)
	
	pipeline = Pipeline([('scale', scaler), ('select', select),('svc', svc)])
	return (F, y, pipeline)

def random_forest_model():
	rf = RandomForestClassifier(max_features='log2', min_samples_split=11, n_estimators=71, max_depth=30, class_weight='balanced')#gridsearched
	select = SelectPercentile(score_func=chi2, percentile=20)#gridsearched
	scaler = MinMaxScaler()
	pipeline = Pipeline([('scale', scaler),('select', select), ('randf', rf)])
	return pipeline

def extra_trees_model():
	select = SelectPercentile(score_func=chi2, percentile=90)
	et = ExtraTreesClassifier(n_estimators=11, max_features='log2', class_weight='balanced')
	#n_estimators=10, max_depth=None, min_samples_split=1, random_state=0
	scaler = MinMaxScaler()
	pipeline = Pipeline([('scale', scaler), ('select', select), ('extra', et)])
	return pipeline

def k_nearest_model():
	select = SelectPercentile(score_func=chi2, percentile=45)
	knc = KNeighborsClassifier(weights='uniform', algorithm='auto', n_neighbors=3)
	scaler = MinMaxScaler()
	pipeline = Pipeline([('scale', scaler),('select', select),  ('knear', knc)])
	return pipeline

def gradient_boost_model():
	gb = GradientBoostingClassifier(learning_rate=4, loss='exponential',\
									max_depth=3, max_features='sqrt', random_state=1)#gridsearched
	select = SelectPercentile(score_func=chi2, percentile=26)#gridsearched
	scaler = MinMaxScaler()
	pipeline = Pipeline([('scale', scaler), ('select', select) ,('gb', gb)])
	return pipeline	

