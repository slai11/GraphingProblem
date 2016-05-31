import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from sklearn import metrics, cross_validation
from sklearn.pipeline import Pipeline
from sklearn.grid_search import GridSearchCV, RandomizedSearchCV
from sklearn.cross_validation import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, make_scorer
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import ExtraTreesClassifier
from model import *
from features import *
import warnings
warnings.filterwarnings("ignore")


def my_custom_loss_func(ground_truth, pred):
	diff = np.abs(ground_truth - pred).max()
	return np.log(1 + diff)

def apply_model(X, y):
	clf0 = dummy()
	clf1 = log_reg_model()
	clf2 = linear_svc_model()
	clf3 = random_forest_model()
	clf4 = extra_trees_model()
	clf5 = k_nearest_model()
	#clf6 = gradient_boost_model()

	model = [clf0, clf1, clf2, clf3, clf4, clf5]#, clf6]

	score = make_scorer(my_custom_loss_func, greater_is_better=False)

	for i, clf in enumerate(model): # use f1 scoring method for optimal performance (dummy estimator will have a 0.00 score)
		scores = cross_validation.cross_val_score(clf, X, y, cv = 4, scoring = "f1", n_jobs = -1)
		print "Model %d " % (i+1) + "Accuracy: %0.4f (+/- %0.2f)" % (scores.mean(), scores.std() * 2)
		X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=10)
		clf.fit(X_train, y_train)
		y_pred = clf.predict(X_test)
		print metrics.classification_report(y_test, y_pred)
		cm = confusion_matrix(y_test, y_pred)
		print cm

		#print scores

def search_grid():
	model = random_forest_model()
	score = make_scorer(my_custom_loss_func, greater_is_better=False)
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=10)
	param_grid = dict(select__percentile=np.arange(10,99,10), randf__max_features=["auto", "log2", None], randf__n_estimators=np.arange(1,100,10), randf__min_samples_split=np.arange(1,100,10), randf__max_depth=[None, 1, 10, 20, 30], randf__class_weight=['balanced'])

	#dict(select__percentile=np.arange(10,99,10), extra__n_estimators=np.arange(1,100,10), extra__max_features=[None, 'auto', 'sqrt', 'log2'])
	#dict(select__percentile=np.arange(10,99,10), randf__max_features=["auto", "log2", None], randf__n_estimators=np.arange(1,100,10), randf__min_samples_split=np.arange(1,100,10), randf__max_depth=[None, 1, 10, 20, 30])
	#dict(select__percentile=np.arange(1,99,1), gb__loss=['deviance', 'exponential'], gb__learning_rate = (2.0**np.arange(-10,10,4)), gb__max_depth= np.arange(1,20,2), gb__max_features=[None, "auto", "sqrt", "log2"])
	#dict(select__percentile=np.arange(1,99,1), knear__algorithm=['auto', 'ball_tree', 'kd_tree', 'brute'], knear__n_neighbors=np.arange(1,20,1), knear__weights=['uniform', 'distance'])
	#dict(select__percentile=np.arange(1,99,1), linsvc__C=(2.0**np.arange(-10,20,4)), linsvc__penalty=['l1','l2'], linsvc__tol=[1e-8, 1e-4, 1e-2, 1e-1])
	#dict(select__percentile=np.arange(10,99,10), randf__max_features=["auto", "log2", None], randf__n_estimators=np.arange(1,100,10), randf__min_samples_split=np.arange(1,100,10), randf__max_depth=[None, 1, 10, 20, 30])
	#dict(select__percentile=np.arange(1,99,1), logre__penalty=['l1', 'l2'], logre__C = (2.0**np.arange(-10, 20, 4)), logre__tol=[1e-8, 1e-4, 1e-1])
	grid = GridSearchCV(model, cv=4, param_grid=param_grid, scoring='f1', n_jobs=-1, verbose=4)
	grid.fit(X,y)

	for i in grid.grid_scores_:
		print i
	print "Best params: " + str(grid.best_params_)
	print "Best score: " + str(grid.best_score_)

def feature_rank(X, y):
	scale = MinMaxScaler()
	X = scale.fit_transform(X)
	model = ExtraTreesClassifier(n_estimators=250, random_state=0)

	model.fit(X, y)
	importances = model.feature_importances_
	std = np.std([tree.feature_importances_ for tree in model.estimators_], axis = 0)
	indices = np.argsort(importances)[:: -1]

	for f in range(X.shape[1]):
		print "%d. feature %d (%f)" % (f+1, indices[f], importances[indices[f]])

	plt.figure()
	plt.bar(range(X.shape[1]), importances[indices], color = "r", yerr=std[indices], align="center")
	plt.xticks(range(X.shape[1]), indices)
	plt.xlim([-1, X.shape[1]])
	plt.show()

def plot_confusion_matrix(cm, y, title='Confusion matrix', cmap=plt.cm.Blues):
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = [0,1,2,3]#np.arange(len(y.target_names))
    plt.xticks(tick_marks, y, rotation=45)
    plt.yticks(tick_marks, y)
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')



if __name__ == '__main__':
	GG = GraphGenerator(network_data, subzone_data)
	GG2 = GraphGenerator(wkend_network_data, subzone_data)
	G, OG, BH = GG.get_graphs()
	WG, WOG, WBH = GG2.get_graphs()
	FB = FeatureBuilder(G, OG, BH)
	FB2 = FeatureBuilder(WG,WOG, WBH)
	X, y = FB.build_features()
	X1, y1 = FB2.build_features()
	
	FB.output_gexf("compsnormal.gexf")
	FB2.output_gexf("compsweekend.gexf")

	#apply_model(X,y)
	#apply_model(X1,y1)
	#search_grid()
	#feature_rank(X, y)


	'''
	clf = k_nearest_model()
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=10)
	clf.fit(X_train, y_train)
	pred = clf.predict(X_test)

	target_names = ['passive', 'low', 'medium', 'high']
	#target_names = ['passive', 'active']

	print classification_report(y_test, pred, target_names=target_names)

	
	cm = confusion_matrix(y_test, pred)
	print cm
	np.set_printoptions(precision=2)
	plot_confusion_matrix(cm, y)
	plt.show()

	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=10)
	print metrics.classification_report(y_true, y_pred)
	'''