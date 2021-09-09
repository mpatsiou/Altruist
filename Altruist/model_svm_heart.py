import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC

from fi_techniques import FeatureImportance
import urllib
import numpy as np

DATASET_URL = "http://archive.ics.uci.edu/ml/machine-learning-databases/statlog/heart/heart.dat"
CLASS_NAMES = ['absence','presence']

def get_class_names():
	return CLASS_NAMES

def get_credit(url = DATASET_URL):
    raw_data = urllib.request.urlopen(url)
    credit=np.genfromtxt(raw_data)

    return credit

def get_feature_names():
    return ['age', 'sex','chest pain', 'resting blood pressure', 'serum cholestoral',
            'fasting blood sugar', 'resting ecg results', 'maximum heart rate achieved',
            'exercise induced angina', 'oldpeak', 'the slope of the peak exercise',
            'number of major vessels','reversable defect']

def split_for_target(dataset = None):
    credit = get_credit()
    values, target = credit[:,:-1], credit[:,-1].squeeze()
    target = [int(i-1) for i in target]

    return values, target

def get_dataset(values, feature_names):
    return pd.DataFrame(values, columns=feature_names)


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

def svm_train(dataset):
	values, target = split_for_target()
	features = get_feature_names()

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

feature_names = get_feature_names()
values, target = split_for_target()
dataset = get_dataset(values, feature_names)
stats = get_dataset_stats(dataset)

svm, scaler, scaled_values, feature_importance = svm_train(dataset)
