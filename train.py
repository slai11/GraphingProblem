import numpy as np
from scipy import stats
from sklearn import metrics, cross_validation
from sklearn.pipeline import Pipeline
from sklearn.grid_search import GridSearchCV, RandomizedSearchCV
from sklearn.cross_validation import train_test_split
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
		#print "Accuracy: %0.4f (+/- %0.2f)" % (scores.mean(), scores.std() * 2)

def search_grid():
	model = random_forest_model()

	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=10)
	param_grid = dict(select__percentile=np.arange(10,99,10), randf__max_features=["auto", "sqrt", "log2", None], randf__n_estimators=np.arange(1,200,10), randf__min_samples_split=np.arange(1,200,5), randf__max_depth=[None, 1, 10, 20, 30, 40, 50])

	#dict(select__percentile=np.arange(1,99,1), logre__penalty=['l1', 'l2'], logre__C=np.arange(1, 100, 2), logre__tol=[1e-8, 1e-4, 1e-1])	
	grid = GridSearchCV(model, cv=5, param_grid=param_grid, scoring="f1_weighted", n_jobs=-1, verbose=4)
	grid.fit(X,y)

	
	for i in grid.grid_scores_:
		print i
	print "Best params: " + str(grid.best_params_)
	print "Best score: " + str(grid.best_score_)

if __name__ == '__main__':
	X, y = build_feature()
	#apply_model()
	search_grid()

	#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=10)
	#print metrics.classification_report(y_true, y_pred)