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
from altruist import Altruist
from fi_techniques import FeatureImportance

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
class_names = ['fake banknote','real banknote'] #0: no, 1: yes #or ['not authenticated banknote','authenticated banknote']
model_names = ['fi_lime', 'fi_shap', 'fi_perm_imp']
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
classifiers[1] = [svm, str("SVM: " + str(clf.best_score_))]
scalers[1] = scaler_svm

X_svm = scalers[1].transform(dataset_values)
fi_svm = FeatureImportance(X_svm, dataset_class, feature_names, class_names)

def metaExplanation(X_t, dataset_class, inst, feature_names, class_names):
    fi = FeatureImportance(X_t, dataset_class, feature_names, class_names)
    fis = [fi.fi_lime, fi.fi_shap, fi.fi_perm_imp]

    altruistino = Altruist(classifiers[1][0], X_t, fis, feature_names, None)
    untruthful_features = altruistino.find_untruthful_features(inst[0])
    return untruthful_features

def counterfactuals(X_t, dataset_class, feature_names, class_names, untruthful_features):
    fi = FeatureImportance(X_t, dataset_class, feature_names, class_names)

    fis = [fi.fi_lime, fi.fi_shap, fi.fi_perm_imp]
    min_un = 100000
    min_pos = 0
    for i in range(len(fis)):
        if min_un > len(untruthful_features[0][i]):
            min_un = len(untruthful_features[0][i])
            min_pos = i

    if not untruthful_features[1][min_pos]:
        print(f'{bot_name}: There are no counterfactual!')
        return

    c = untruthful_features[1][min_pos][0]
    print("The counterfactuals for the feature ", feature_names[c[0] - 1],'is', c[1])

def varsAltruist(X_t, dataset_class, class_names, untruthful_features, feature_names, vars):
    fi = FeatureImportance(X_t, dataset_class, feature_names, class_names)

    fis = [fi.fi_lime, fi.fi_shap, fi.fi_perm_imp]
    min_un = 100000
    min_pos = 0
    min2_pos = 0
    for i in range(len(fis)):
        if min_un >= len(untruthful_features[i]):
            min_un = len(untruthful_features[i])
            min2_pos = min_pos
            min_pos = i

    altruistVars = [0] * len(feature_names)

    for i in range(0, len(feature_names)):
        if feature_names[i] in untruthful_features[min_pos]:
            altruistVars[i] = fis[min2_pos](vars[0], "_", svm)[i]
            continue

        altruistVars[i] = fis[min_pos](vars[0], "_", svm)[i]

    plotMethod(feature_names, altruistVars, "Altruist", 'Feature Importance')


bot_name = "Robin"
list_words = ['yes', 'no', 'bye', 'hello', 'informations', 'interpretation', 'features', 'counterfactual', 'previous']
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

def readBanknote(feature_names):
    print(f"{bot_name}: Okay! Fill in the following features\n")
    vars = [0] * len(feature_names)
    for i in range(len(feature_names)):
        v = input(f"     {feature_names[i]}: ")
        vars[i] = float(v)

    vars = [np.array(vars)]
    vars = scaler_svm.transform(vars)
    return vars

def prediction(vars):
    print(f"{bot_name}: The prediction is that is a ", class_names[svm.predict(vars)[0]], '\n')

def phase2Menu():
    print(f"{bot_name}: Νow you can see the interpretation of some models, as well as the informations about them")
    print("\t1)informations about LIME\n\t2)informations about Shap\n\t3)informations about Permutation Importance(PI)")
    print("\t4)interpretation of LIME\n\t5)interpretation of Shap\n\t6)interpretation of Permutation Importance(PI)\n")

def infoLime():
    print(f"{bot_name}: Info of LIME...")

def infoShap():
    print(f"{bot_name}: Info of Shap...")

def infoPI():
    print(f"{bot_name}: Info of PI...")

def yesOrNo(question):
    print(f"{bot_name}: ",question)
    print('\tyes\n\tno')

    while(True):
        user_input = input(f"{user_name}: ").lower()
        if 'yes' in user_input:
            return True

        elif 'no' in user_input:
            return False

        print(f"{bot_name}: I'm a bot programmed to answer only some of the specific answers. Here are the topics I can help you with.\nRobin: Select the topic or write your answer below\n")
        continue


def phase3Menu():
    print(f"{bot_name}: After that, I suggest you to use the Altruist. A new methodology that aims to tackle a few problems of feature importance-based aproaches.")
    print(f"{bot_name}: You can see:")
    print("\t1)Combinatorial interpretation from Altruist")
    print("\t2)The untruthful features of LIME\n\t3)The untruthful features of Shap\n\t4)The untruthful features of Permutation Importance(PI)")
    print("\t5)The counterfactuals for every feature\n\t6)Go to previous step")

def infoAltruist():
    print(f"{bot_name}: Info of Altruist...")

def plotMethod(feature_names, ar, title, yLabel):
    my_cmap = cm.get_cmap('Greens', 17)
    my_norm = Normalize(vmin = 0, vmax = 4)

    fig, axs = plt.subplots(1, 1, figsize = (10, 4.7), dpi = 150, sharex = True)

    axs.bar(feature_names, ar,color=my_cmap(my_norm([1,2,3,4])))
    axs.set_title(title)
    axs.set_ylabel(yLabel)
    plt.show()

responses = {
    'phase1': {
        'yes': readBanknote,
    },
    'phase2': {
        #'menu': printMenu
        'infoLime': infoLime,
        'infoShap': infoShap,
        'infoPI': infoPI,
        'noMethod': lambda: print(f"{bot_name}: Please give also a method")
    },
    'phase3': {
        'noMethod': lambda: print(f"{bot_name}: Please give also a method"),
        'altruist' : metaExplanation,
        'counterfactual': counterfactuals

    },
    'default': {
        'hello': lambda: print(f"{bot_name}: Hello! Time to predict some banknotes"),
        'quit': lambda: print(f"{bot_name}: Thank you for visiting!"),
        'error': lambda: print(f"{bot_name}: I'm a bot programmed to answer only some of the specific answers. Here are the topics I can help you with.\nRobin: Select the topic or write your answer below\n")
    }
}

user_name = input(f"\n\nHi​! My name is Robin. Let me know if you have any questions regarding our tool!\nWhat's your name?\n")
print(f"Robin: Hello {user_name}\nAre you ready to predict some banknotes?\n1)yes I am ready!\n2)no bye.")

phase = 'phase1'
while(True):
    user_input = input(f"{user_name}: ").lower()

    matched = None
    for key, pattern in keywords_dict.items():
        #Using the re search function
        if re.search(pattern, user_input):
            matched = key
        elif key in user_input:
            matched = key

    if matched == 'bye':
        responses['default']['quit']()
        break

    if matched not in keywords:
        responses['default']["error"]()
        continue

    if matched == 'hello':
        responses['default']['hello']()
        continue

    if phase == 'phase1':
        if matched == 'yes':
            vars = responses[phase][matched](feature_names)
            prediction(vars)
            phase = 'phase2'
            phase2Menu()

        elif matched == 'no':
            responses['default']['quit']()
            break


    elif phase == 'phase2':
        if matched == 'informations':
            if 'lime' in user_input:
                responses[phase]['infoLime']()
            elif 'shap' in user_input:
                responses[phase]['infoShap']()
            elif ('pi' in user_input) or ('permutation importance' in user_input):
                responses[phase]['infoPI']()
            else:
                responses[phase]['noMethod']()
                continue


        elif matched == 'interpretation':
            if 'lime' in user_input:
                plotMethod(feature_names, fi_svm.fi_lime(vars[0], "_", svm), 'LIME', 'Feature Importance')

            elif 'shap' in user_input:
                plotMethod(feature_names, fi_svm.fi_shap(vars[0], "_", svm), 'Shap', 'Feature Importance')

            elif ('pi' in user_input) or ('permutation importance' in user_input):
                plotMethod(feature_names, fi_svm.fi_perm_imp(vars[0], "_", svm), 'PI', 'Feature Importance')

            else:
                responses[phase]['noMethod']()
                continue

        if yesOrNo("Do you want to go back"):
            phase2Menu()
        else:
            phase = "phase3"
            untruthful_features = responses[phase]['altruist'](X_svm, dataset_class, vars, feature_names, class_names)
            print(untruthful_features)
            phase3Menu()

    elif phase == 'phase3':
        if matched == 'interpretation':
            varsAltruist(X_svm, dataset_class, class_names, untruthful_features[0], feature_names, vars)
            print("Altruist plot")

        if matched == 'features':
            if 'lime' in user_input:
                print("Untruthful features LIME: "+str(untruthful_features[0][0])+" ("+str(len(untruthful_features[0][0]))+")")
            elif 'shap' in user_input:
                print("Untruthful features Shap: "+str(untruthful_features[0][1])+" ("+str(len(untruthful_features[0][1]))+")")
            elif ('pi' in user_input) or ('permutation importance' in user_input):
                print("Untruthful features Permutation Importance: "+str(untruthful_features[0][2])+" ("+str(len(untruthful_features[0][2]))+")")
            else:
                responses[phase]['noMethod']()
                continue

        elif matched == 'counterfactual':
            responses[phase][matched](X_svm, dataset_class, feature_names, class_names, untruthful_features)

        elif matched == 'previous':
            phase = 'phase2'
            phase2Menu();
            continue

        if yesOrNo('Do you want to go back'):
            phase3Menu()
        else:
            print('thats it bye')
            break
