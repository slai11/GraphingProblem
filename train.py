import numpy as np
from scipy import stats
from sklearn import metrics, cross_validation
from sklearn.pipeline import Pipeline
from sklearn.grid_search import GridSearchCV, RandomizedSearchCV
from sklearn.cross_validation import train_test_split
from sklearn.metrics import classification_report
from model import *
import warnings
warnings.filterwarnings("ignore")

def apply_model():
	clf1 = log_reg_model()
	clf2 = linear_svc_model()
	clf3 = random_forest_model()
	clf4 = extra_trees_model()
	clf5 = k_nearest_model()
	clf6 = gradient_boost_model()

	model = [clf1, clf2, clf3, clf4, clf5, clf6]

	for i, clf in enumerate(model):
		scores = cross_validation.cross_val_score(clf, X, y, cv = 4, scoring = 'f1_weighted', n_jobs = -1)
		print "Model %d " % (i+1) + "Accuracy: %0.4f (+/- %0.2f)" % (scores.mean(), scores.std() * 2)
		#print scores

def search_grid():
	model = gradient_boost_model()

	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=10)
	param_grid = dict(select__)

	#dict(select__percentile=np.arange(1,99,1), gb__loss=['deviance', 'exponential'], gb__learning_rate = (2.0**np.arange(-10,10,4)), gb__max_depth= np.arange(1,20,2), gb__max_features=[None, "auto", "sqrt", "log2"])
	#dict(select__percentile=np.arange(1,99,1), knear__algorithm=['auto', 'ball_tree', 'kd_tree', 'brute'], knear__n_neighbors=np.arange(1,20,1), knear__weights=['uniform', 'distance'])
	#dict(select__percentile=np.arange(1,99,1), linsvc__C=(2.0**np.arange(-10,20,4)), linsvc__penalty=['l1','l2'], linsvc__tol=[1e-8, 1e-4, 1e-2, 1e-1])
	#dict(select__percentile=np.arange(10,99,10), randf__max_features=["auto", "log2", None], randf__n_estimators=np.arange(1,100,10), randf__min_samples_split=np.arange(1,100,10), randf__max_depth=[None, 1, 10, 20, 30])
	#dict(select__percentile=np.arange(1,99,1), logre__penalty=['l1', 'l2'], logre__C = (2.0**np.arange(-10, 20, 4)), logre__tol=[1e-8, 1e-4, 1e-1])
	grid = GridSearchCV(model, cv=4, param_grid=param_grid, scoring="f1_weighted", n_jobs=-1, verbose=4)
	grid.fit(X,y)

	
	for i in grid.grid_scores_:
		print i
	print "Best params: " + str(grid.best_params_)
	print "Best score: " + str(grid.best_score_)

if __name__ == '__main__':
	X, y = build_feature()
	apply_model()
	#search_grid()
	'''
	clf = k_nearest_model()
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=10)
	clf.fit(X_train, y_train)
	pred = clf.predict(X_test)

	target_names = ['passive', 'active']

	print classification_report(y_test, pred, target_names=target_names)
	'''

	#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=10)
	#print metrics.classification_report(y_true, y_pred)