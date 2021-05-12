import re
from nltk.corpus import wordnet
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from Altruist.fi_techniques import FeatureImportance
import matplotlib.cm as cm
from matplotlib.colors import Normalize

import pandas as pd
import numpy as np
import seaborn as sns
import urllib
import networkx as nx
import matplotlib.pyplot as plt
from ipywidgets import interact, interactive, fixed, interact_manual
import ipywidgets as widgets


banknote_datadset = pd.read_csv('https://raw.githubusercontent.com/Kuntal-G/Machine-Learning/master/R-machine-learning/data/banknote-authentication.csv')
feature_names = ['variance','skew','curtosis','entropy']
class_names=['fake banknote','real banknote'] #0: no, 1: yes #or ['not authenticated banknote','authenticated banknote']

# Saves the dataset_statistics for every feature
dataset_statistics = {}
for i in range (len(banknote_datadset.columns) - 1):
    feature = banknote_datadset.columns[i]
    min = banknote_datadset.values[:,i].min()
    max = banknote_datadset.values[:,i].max()
    mean = banknote_datadset.values[:,i].mean()
    dataset_statistics[feature] = [min, max, mean]

dataset_statistics_f = pd.DataFrame(data=dataset_statistics, index = ['min','max', 'mean'])

dataset_values = banknote_datadset.iloc[:, 0:4].values
dataset_class = banknote_datadset.iloc[:, 4].values

# We will use MinMaxScaler scaler to normalize the input
# and the SVM classifier
scaler = MinMaxScaler(feature_range=(-1,1))
classifiers = {}
scalers = {}

pipe = Pipeline(steps=[('scaler', scaler), ('svm', SVC(probability=True,random_state=77))])
parameters = {'svm__C': [100], 'svm__gamma': [0.1], 'svm__kernel': ['rbf']} #best
clf = GridSearchCV(pipe, parameters, scoring='f1', cv=10, n_jobs=-1)
clf.fit(dataset_values, dataset_class)
scaler_svm = clf.best_estimator_.steps[0][1]
svm = clf.best_estimator_.steps[1][1]
classifiers[1] = [svm, str("SVM: "+ str(clf.best_score_))]
scalers[1] = scaler_svm

X_svm = scaler_svm.transform(dataset_values)
fi_svm = FeatureImportance(X_svm, dataset_class, feature_names, class_names)

#
# vars = [0] * len(feature_names)
# for i in range(len(feature_names)):
#     v = input(f"     {feature_names[i]}: ")
#     vars[i] = v
#
# vars = [np.array(vars)]
#
# print(class_names[svm.predict(vars)[0]])
def goBack():
    print("Robin: Do you want to go back?")
    print('\tyes\n\tno')
    user_input = input().lower()
    if 'yes' in user_input:
        return True
    return False

list_words = ['yes', 'bye', 'hello', 'informations', 'interpretation']
list_syn = {}

for word in list_words:
    synonyms = []

    for syn in wordnet.synsets(word):

        for lem in syn.lemmas():
            # Remove any special characters from synonym strings
            lem_name = re.sub('[^a-zA-Z0-9 \n\.]', ' ', lem.name())

            synonyms.append(lem_name)

    list_syn[word] = set(synonyms)


keywords = {}
keywords_dict = {}

for key in list_words:
    keywords[key] = []

for key in list_syn:
    for synonym in list(list_syn[key]):
        keywords[key].append('.*\\b' + synonym + '\\b.*')

for key, values in keywords.items():
    keywords_dict[key]=re.compile('|'.join(values))

responses = {
    'hello': "Robin: Hello! Do you want predict if those banknotes  are valid or not\n",
    'yes': "Robin: Okay! Fill in the following features\n",
    'bye': "Robin: Thank you for visiting.",
    'error': "Robin: I'm a bot programmed to answer only some of the frequent questions. Here are the topics I can help you with.\nRobin: Select the topic or write your answer below\n",
    'phase2': "\t1)informations about LIME\n\t2)informations about Shap\n\t3)informations about Permutation Importance(PI)\n\t4)interpretation of LIME\n\t5)interpretation of Shap\n\t6)interpretation of Permutation Importance(PI)\n",
    'no method': "Robin: Please give also a method",
    'infoLime': "Info of LIME...",
    'infoShap': "Info of Shap...",
    'infoPI': "Info of Permutation Importance"

}

user_name = input(f"\n\nHi​! My name is Robin. Let me know if you have any questions regarding our tool!\nWhat's your name?\n")
print(f"Robin: Hello {user_name}\nAre you ready to predict some banknotes?\n1)yes I am ready!\n2)no bye.")


while(True):
    user_input = input(f"{user_name}: ").lower()

    matched = None
    for key, pattern in keywords_dict.items():
        #Using the re search function
        if re.search(pattern, user_input):
            matched = key

    print('matched=', matched)
    if matched == 'bye':
        print(responses[matched])
        break

    if matched not in keywords:
        print(responses["error"])

    else:
        if matched == 'yes':
            print(responses[matched])

            vars = [0] * len(feature_names)
            for i in range(len(feature_names)):
                v = input(f"     {feature_names[i]}: ")
                vars[i] = v

            vars = [np.array(vars)]
            print("Robin: The prediction is that is a ", class_names[svm.predict(vars)[0]], '\n')
            print("Robin: Perfect! Νow you can see the interpretation of some models, as well as the informations about them")
            print(responses['phase2'])

        elif matched == 'informations':
            if 'lime' in user_input:
                print(responses['infoLime'])
            elif 'shap' in user_input:
                print(responses['infoShap'])
            elif 'pi' in user_input or 'permutation importance' in user_input:
                print(responses['infoPI'])
            else:
                print(responses['no method'])

            if goBack(): print(responses['phase2'])

        elif matched == 'interpretation':
            my_cmap = cm.get_cmap('Greens', 17)
            my_norm = Normalize(vmin = 0, vmax = 4)

            fig, axs = plt.subplots(1, 1, figsize = (10, 4.7), dpi = 150, sharex = True)
            if 'lime' in user_input:
                # axs.bar(feature_names, fi_svm.fi_lime(vars[0],_, svm),color=my_cmap(my_norm([1,2,3,4])))
                # axs.set_title('LIME')
                # axs.set_ylabel('Feature Importance')
                print('interpretation of lime')
            elif 'shap' in user_input:
                print('interpretation of shap')
            elif 'pi' in user_input:
                print('interpretation of pi')
            else:
                print(responses['no method'])

            if goBack(): print(responses['phase2'])
