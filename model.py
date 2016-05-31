import csv
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, scale
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import ExtraTreesClassifier, GradientBoostingClassifier, RandomForestClassifier
from sklearn import metrics, cross_validation
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectPercentile, chi2
from sklearn.dummy import DummyClassifier


def dummy():
	scaler = MinMaxScaler()
	du = DummyClassifier(strategy='most_frequent', random_state=0)
	pipeline = Pipeline([('scale', scaler), ('dummy', du)])
	return pipeline	

def log_reg_model():
	select = SelectPercentile(score_func=chi2, percentile=20) #gridsearched
	log = LogisticRegression(C=64, penalty='l1', tol=0.1, class_weight='balanced')     #gridsearched
	scaler = MinMaxScaler()
	pipeline = Pipeline([('scale', scaler), ('select', select), ('logre', log)])
	return pipeline

def linear_svc_model():
	select = SelectPercentile(score_func=chi2, percentile=11) #gridsearched
	svc = LinearSVC(dual=False, class_weight='balanced', penalty='l2', C=0.015625, tol=1e-8)  #gridsearched
	scaler = MinMaxScaler()
	pipeline = Pipeline([('scale', scaler), ('select', select),('linsvc', svc)])
	return pipeline

def random_forest_model():
	rf = RandomForestClassifier(n_estimators=31, min_samples_split=11, max_features='auto', max_depth=20, class_weight='balanced')#gridsearched
	select = SelectPercentile(score_func=chi2, percentile=60)#gridsearched
	scaler = MinMaxScaler()
	pipeline = Pipeline([('scale', scaler),('select', select), ('randf', rf)])
	return pipeline

def extra_trees_model():
	select = SelectPercentile(score_func=chi2, percentile=60)
	et = ExtraTreesClassifier(n_estimators=51, max_features='auto', class_weight='balanced')
	#n_estimators=10, max_depth=None, min_samples_split=1, random_state=0
	scaler = MinMaxScaler()
	pipeline = Pipeline([('scale', scaler), ('select', select), ('extra', et)])
	return pipeline

def k_nearest_model():
	select = SelectPercentile(score_func=chi2, percentile=61)#gridsearched macro
	knc = KNeighborsClassifier(weights='distance', algorithm='auto', n_neighbors=6)#gridsearched macro
	#knear__weights': 'uniform', 'knear__algorithm': 'auto', 'knear__n_neighbors': 4, 'select__percentile': 11
	#knear__weights': 'uniform', 'knear__algorithm': 'auto', 'knear__n_neighbors': 1, 'select__percentile': 1
	#(weights='distance', algorithm='auto', n_neighbors=3)#gridsearched perc 67
	scaler = MinMaxScaler()
	pipeline = Pipeline([('scale', scaler),('select', select),  ('knear', knc)])
	return pipeline

def gradient_boost_model():
	gb = GradientBoostingClassifier(learning_rate=64, loss='exponential',\
									max_depth=11, max_features='log2', random_state=10)#gridsearched
	select = SelectPercentile(score_func=chi2, percentile=91)#gridsearched
	scaler = MinMaxScaler()
	pipeline = Pipeline([('scale', scaler), ('select', select) ,('gb', gb)])
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