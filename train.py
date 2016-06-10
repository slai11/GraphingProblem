# _*_ coding: utf-8 _*_

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from sklearn import metrics, cross_validation
from sklearn.pipeline import Pipeline
from sklearn.grid_search import GridSearchCV, RandomizedSearchCV
from sklearn.cross_validation import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, make_scorer, f1_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import ExtraTreesClassifier
from model import *
from features import *
import warnings
from sklearn.svm import LinearSVC

from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import label_binarize
from sklearn.multiclass import OneVsRestClassifier
import xgboost as xgb

from util import *

warnings.filterwarnings("ignore")



def apply_model(X, y, score='f1', ensemble=True, predmove=True):
	use=0
	if predmove:
		use = 1
	clf1 = dummy()[use]
	clf2 = log_reg_model()[use]
	clf3 = linear_svc_model()[use]
	clf4 = rbf_svc_model()[use]
	clf5 = random_forest_model()[use]
	clf6 = extra_trees_model()[use]
	clf7 = k_nearest_model()[use]

	clf8 = voting_ensemble()[use]
	clf9 = gradient_boost_model()[use]
	clf10 = xgboost_model()[use]
	clf11 = bag_model()
	#clf12 = ada_boost_model()[use]

	model_name = ["Dummy", "LogisticRegression", "LinearSVC", "SVC w rbf",\
					"RandomForestClassifier", "ExtraTreesClassifier", "KNearestClassifier", "Voting Classifier",\
					"GradientBoostingClassifier", "XGBoost", "Bagging"]#, "ada"]
	if ensemble:
		model = [clf1, clf2, clf3, clf4 ,clf5, clf6, clf7, clf8, clf9, clf10, clf11]#, clf12]
	else:
		model = [clf1, clf2, clf3, clf4 ,clf5, clf6, clf7]
	
	print "===================================================="
	for i, clf in enumerate(model): # use f1_macro scoring method for optimal performance (dummy estimator will have a 0.00 score)
		scores = cross_validation.cross_val_score(clf, X, y, cv = 5, scoring = score, n_jobs = -1)
		print model_name[i]
		print "CV score ("+score+")"+": %0.4f (+/- %0.2f)" % (scores.mean(), scores.std() * 2)
		print scores
		
		X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=10)
		clf.fit(X_train, y_train)
		y_pred = clf.predict(X_test)
		print "Classification Report:"
		print metrics.classification_report(y_test, y_pred)
		print f1_score(y_test, y_pred, average='macro')
		cm = confusion_matrix(y_test, y_pred)
		
		print "Confusion Matrix:"
		print cm
		print " "
		print "===================================================="
		print " "


def get_model_score(X, y, score='f1', pred='status'):
	use = 1
	if pred is "status":
		use = 0
	clf1 = dummy()[use]
	clf2 = log_reg_model()[use]
	clf3 = linear_svc_model()[use]
	clf4 = rbf_svc_model()[use]
	clf5 = random_forest_model()[use]
	clf6 = extra_trees_model()[use]
	clf7 = k_nearest_model()[use]


	model_name = ["Dummy", "LogisticRegression", "LinearSVC", "SVC w rbf",\
					"RandomForestClassifier", "ExtraTreesClassifier", "KNearestClassifier"]

	model = [clf1, clf2, clf3, clf4 ,clf5, clf6, clf7]
	
	scorelist = []
	for i, clf in enumerate(model): # use f1_macro scoring method for optimal performance (dummy estimator will have a 0.00 score)
		scores = cross_validation.cross_val_score(clf, X, y, cv = 5, scoring = score, n_jobs = -1)
		scorelist.append(scores.mean())

	scoredf = pd.DataFrame(scorelist, index=model_name)
	
	return scoredf.transpose()
	

def search_grid(X, y):
	model = xgboost_model()[0]
	
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=10)
	param_grid = dict(xgb__num_round=np.arange(1,100,5), xgb__max_depth=np.arange(5,20,1))

	#dict(xgb__max_depth=np.arange(1,20,1), xgb__gamma=np.arange(0,10,1), xgb__eta=np.arange(0.01, 1, 0.05), xgb__max_delta_step=np.arange(0,100,1))
	#dict(select__percentile=np.arange(1,99,10), svc__kernel=['rbf'], svc__gamma=np.arange(0,10,1), svc__C = (2.0**np.arange(-10, 10, 4)), svc__tol=[1e-8, 1e-4, 1e-1])
	#dict(select__percentile=np.arange(10,99,10), extra__n_estimators=np.arange(1,100,10), extra__max_features=[None, 'auto', 'sqrt', 'log2'])
	#dict(select__percentile=np.arange(10,99,10), randf__max_features=["auto", "log2", None], randf__n_estimators=np.arange(1,100,10), randf__min_samples_split=np.arange(1,100,10), randf__max_depth=[None, 1, 10, 20, 30])
	#dict(select__percentile=np.arange(1,99,10), gb__loss=['deviance', 'exponential'], gb__learning_rate = (2.0**np.arange(-10,10,4)), gb__max_depth= np.arange(1,20,2), gb__max_features=[None, "auto", "sqrt", "log2"])
	#dict(select__percentile=np.arange(1,99,10), knear__algorithm=['auto', 'ball_tree', 'kd_tree', 'brute'], knear__n_neighbors=np.arange(1,20,1), knear__weights=['uniform', 'distance'])
	#dict(select__percentile=np.arange(1,99,10), linsvc__C=(2.0**np.arange(-10,10,4)), linsvc__penalty=['l1','l2'], linsvc__tol=[1e-8, 1e-4, 1e-2, 1e-1])
	#dict(select__percentile=np.arange(10,99,10), randf__max_features=["auto", "log2", None], randf__n_estimators=np.arange(1,100,10), randf__min_samples_split=np.arange(1,100,10), randf__max_depth=[None, 1, 10, 20, 30])
	#dict(select__percentile=np.arange(1,99,10), logre__penalty=['l1', 'l2'], logre__C = (2.0**np.arange(-10, 20, 4)), logre__tol=[1e-8, 1e-4, 1e-1])
	grid = GridSearchCV(model, cv=5, param_grid=param_grid, scoring='f1', n_jobs=-1, verbose=4)
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
		print "%d. Feature %d (%f)" % (f+1, indices[f], importances[indices[f]])

	
	plt.title("Feature Importance")
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

def plot_pr_auc(X, y, model, modelname):
	# compute precision and recall scores
	title = "PR curve for " + modelname
	random_state=np.random.RandomState(0)
	y = label_binarize(y, classes=[0, 1, 2])
	n_classes = y.shape[1]
	
	n_samples, n_features = X.shape
	
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2,
                                                    random_state=random_state)
	classifier = OneVsRestClassifier(model)
	y_score = classifier.fit(X_train, y_train).decision_function(X_test)
	
	

	precision = dict()
	recall = dict()
	average_precision = dict()
	for i in range(n_classes):
	    precision[i], recall[i], _ = precision_recall_curve(y_test[:, i],
	                                                        y_score[:, i])
	    average_precision[i] = average_precision_score(y_test[:, i], y_score[:, i])

	# Compute micro-average ROC curve and ROC area
	precision["macro"], recall["macro"], _ = precision_recall_curve(y_test.ravel(),
	    y_score.ravel())
	average_precision["macro"] = average_precision_score(y_test, y_score,
	                                                     average="macro")

	# Plot Precision-Recall curve for each class
	plt.clf()
	plt.plot(recall["macro"], precision["macro"],
	         label='macro-average PR curve ({0:0.4f})'
	               ''.format(average_precision["macro"]))
	for i in range(n_classes):
	    plt.plot(recall[i], precision[i],
	             label='PR curve of class {0} ({1:0.4f})'
	                   ''.format(i, average_precision[i]))

	plt.xlim([0.0, 1.0])
	plt.ylim([0.0, 1.05])
	plt.xlabel('Recall')
	plt.ylabel('Precision')
	plt.title(title)
	plt.legend(loc="lower right")
	plt.show()

def check_study_period():
	# range of study period 2-5
	# fix 1 day ahead
	list_=["20160523", "20160524", "20160525", "20160527", "20160531", "20160601", "20160602"]
	i = 2
	stack=[]
	while i < 5+1:
		X, y = load_multiple_data(list_, i, ahead=1, delta=True, pred_movement=False)
		scorelist = get_model_score(X, y, score='f1_macro', pred='status') #contains DF with column names as model + score
		stack.append(scorelist)
		i+=1

	testlist = pd.concat(stack)
	days = [2,3,4,5]

	plt.title("Study period against score")
	for i in testlist.columns:
		plt.plot(days, testlist[i], 'x')

	plt.ylim([0.0, 1.05])
	plt.xlabel('Study period')
	plt.ylabel('f1_macro score')
	plt.show()


def check_ahead_period():
	# range of study period 2-5
	# fix 1 day ahead
	list_=["20160523", "20160524", "20160525", "20160527", "20160531", "20160601", "20160602"]
	i = 1
	stack=[]
	while i < 5:
		X, y = load_multiple_data(list_, 3, i, delta=True, pred_movement=False)
		scorelist = get_model_score(X, y, score='f1_macro', pred='status') #contains DF with column names as model + score
		stack.append(scorelist)
		i+=1

	testlist = pd.concat(stack)
	days = [1,2,3,4]

	plt.title("Days forecasted against score")
	for i in testlist.columns:
		plt.plot(days, testlist[i], 'x')

	plt.ylim([0.0, 1.05])
	plt.xlabel('Forecast range')
	plt.ylabel('f1_macro score')
	plt.show()



def build_X_y(list_, study_period=3, ahead=1, delta=False, pred_movement=True):
	X, y = load_multiple_data(list_, study_period, ahead, delta, pred_movement)
	return X, y


if __name__ == '__main__':
	
	list_ = ["20160523", "20160524", "20160525", "20160527", "20160531", "20160601", "20160602"]
	
	predmove = False
	X, y = build_X_y(list_, study_period=3, ahead=2, delta=True, pred_movement=predmove) # basic + daily

	#X = np.genfromtxt('featuresWdelta.csv', delimiter=',')
	#y = np.genfromtxt('labelStatus.csv', delimiter=',')


	#apply_model(X,y,"f1_macro", ensemble=True,predmove=predmove)
	#plot_pr_auc(X, y, log_reg_model(), "log reg")

	search_grid(X, y)
	#feature_rank(X, y)

