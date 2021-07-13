import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC

from fi_techniques import FeatureImportance

DATASET_URL = 'https://raw.githubusercontent.com/Kuntal-G/Machine-Learning/master/R-machine-learning/data/banknote-authentication.csv'
CLASS_NAMES = ['fake_banknote', 'real_banknote']

def get_dataset(url = DATASET_URL):
	return pd.read_csv(url)

def get_feature_names(dataset):
	return list(dataset.columns[:-1])

def get_dataset_stats(dataset):
	length = len(dataset.columns) - 1
	stats = {}

	for i in range(length):
		feature = dataset.columns[i]
		values = dataset.values[:, i]

		stats[feature] = {
			"min": values.min(),
			"max": values.max(),
			"mean": values.mean()
		}

	return stats

def split_for_target(dataset):
	l = len(dataset.columns) - 1
	values = dataset.iloc[:, :(l)].values
	target = dataset.iloc[:, l].values

	return values, target

def svm_train(dataset):
	values, target = split_for_target(dataset)
	features = get_feature_names(dataset)

	steps  = [
		('scaler', MinMaxScaler(feature_range=(-1, 1))),
		('svm', SVC(probability=True,random_state=77))
	]

	parameters = {
		'svm__C': [100],
		'svm__gamma': [0.1],
		'svm__kernel': ['rbf']
	}

	pipe = Pipeline(steps=steps)
	clf = GridSearchCV(pipe, parameters, scoring='f1', cv=10, n_jobs=-1)
	clf.fit(values, target)

	scaler = clf.best_estimator_.steps[0][1]
	svm = clf.best_estimator_.steps[1][1]

	scaled_values = scaler.transform(values)
	feature_importance = FeatureImportance(scaled_values, target, features, CLASS_NAMES)

	return svm, scaler, scaled_values, feature_importance
