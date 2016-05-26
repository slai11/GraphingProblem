import csv
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler, scale
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import ExtraTreesClassifier, GradientBoostingClassifier, RandomForestClassifier
from sklearn import metrics, cross_validation
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectPercentile, chi2, SelectKBest
from graph import *
from features import *


def log_reg_model():
	select = SelectPercentile(score_func=chi2, percentile=1)
	log = LogisticRegression(tol=1e-1, penalty='l1', C=95)
	scaler = MinMaxScaler()
	pipeline = Pipeline([('scale', scaler), ('logre', log)])
	return pipeline

def linear_svc_model():
	select = SelectPercentile(score_func=chi2, percentile=1)
	svc = LinearSVC(C=1, penalty='l2', loss = 'squared_hinge')
	scaler = MinMaxScaler()
	pipeline = Pipeline([('scale', scaler), ('linsvc', svc)])
	return pipeline

def random_forest_model():
	rf = RandomForestClassifier()
	#n_estimators=10,max_depth=None, min_samples_split=20, max_features = 3
	select = SelectPercentile(score_func=chi2, percentile=80)
	scaler = MinMaxScaler()
	pipeline = Pipeline([('scale', scaler),('select', select), ('randf', rf)])
	return pipeline

def extra_trees_model():
	select = SelectPercentile(score_func=chi2, percentile=50)
	et = ExtraTreesClassifier(n_estimators=10, max_depth=None, min_samples_split=1, random_state=0)
	scaler = MinMaxScaler()
	pipeline = Pipeline([('scale', scaler), ('extra', et)])
	return pipeline

def k_nearest_model():
	select = SelectPercentile(score_func=chi2, percentile=50)
	knc = KNeighborsClassifier()
	scaler = MinMaxScaler()
	pipeline = Pipeline([('scale', scaler), ('knear', knc)])
	return pipeline

def gradient_boost_model():
	gb = GradientBoostingClassifier(n_estimators=10)
	scaler = MinMaxScaler()
	pipeline = Pipeline([('scale', scaler), ('gb', gb)])
	return pipeline	

'''
full_graph , subgraph = get_graphs()
nodelist = sorted(full_graph.nodes())
#output = nx.to_numpy_matrix(subgraph, weight = 'weight')
mylist = []
for node in full_graph.nodes():
	print node
	mylist.append((full_graph.node[node]['longitude'], full_graph.node[node]['latitude']))
X = np.array(mylist)

km = KMeans(n_clusters = 4)  # this is by proximity only
y_pred=km.fit_predict(X)

plt.scatter(X[:, 0], X[:, 1], c = y_pred)
plt.show()

#print nx.info(full_graph)
#cluster differently??
'''