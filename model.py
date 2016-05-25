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
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn import metrics, cross_validation
from sklearn.pipeline import Pipeline
from graph import *
from features import *

X, y = build_feature()


rf = RandomForestClassifier(n_estimators=10, max_depth=None, min_samples_split=1, random_state=0)
scaler = MinMaxScaler()
clf3 = Pipeline([('select', scaler), ('randf', rf)])
scores = cross_validation.cross_val_score(clf3, X, y, cv = 6, scoring = 'accuracy', n_jobs = -1)
print "Random Forest Classifier"
print scores
print "Accuracy: %0.4f (+/- %0.2f)" % (scores.mean(), scores.std() * 2)

'''
svc = LinearSVC()
scaler = MinMaxScaler()
clf = Pipeline([('select', scaler), ('linsvc', svc)])
scores = cross_validation.cross_val_score(clf, X, y, cv = 6, scoring = 'accuracy', n_jobs = -1)
print "Linear SVC"
print scores
print "Accuracy: %0.4f (+/- %0.2f)" % (scores.mean(), scores.std() * 2)

log = LogisticRegression(tol=1e-8, penalty='l2', C=3)
scaler = MinMaxScaler()
clf2 = Pipeline([('select', scaler), ('logre', log)])
scores = cross_validation.cross_val_score(clf2, X, y, cv = 6, scoring = 'accuracy', n_jobs = -1)
print "Logistic Regression"
print scores
print "Accuracy: %0.4f (+/- %0.2f)" % (scores.mean(), scores.std() * 2)



et = ExtraTreesClassifier(n_estimators=10, max_depth=None, min_samples_split=1, random_state=0)
scaler = MinMaxScaler()
clf5 = Pipeline([('select', scaler), ('extra', et)])
scores = cross_validation.cross_val_score(clf5, X, y, cv = 6, scoring = 'accuracy', n_jobs = -1)
print "Extra Trees Classifier"
print scores
print "Accuracy: %0.4f (+/- %0.2f)" % (scores.mean(), scores.std() * 2)

knc = KNeighborsClassifier()
scaler = MinMaxScaler()
clf6 = Pipeline([('select', scaler), ('knear', knc)])
scores = cross_validation.cross_val_score(clf6, X, y, cv = 6, scoring = 'accuracy', n_jobs = -1)
print "K-Neighbors Classifier"
print scores
print "Accuracy: %0.4f (+/- %0.2f)" % (scores.mean(), scores.std() * 2)
'''

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