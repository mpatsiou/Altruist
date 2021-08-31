import re
import pandas as pd
import numpy as np

from matplotlib.colors import Normalize
import matplotlib.cm as cm
import matplotlib.pyplot as plt

from nltk.corpus import wordnet

from altruist import Altruist
from fi_techniques import FeatureImportance
import model_svm

MODEL_NAMES = ['fi_lime', 'fi_shap', 'fi_perm_imp']
CLASS_NAMES = ['fake_banknote', 'real_banknote']

dataset = model_svm.get_dataset()
dataset_statistics = model_svm.get_dataset_stats(dataset)
feature_names = model_svm.get_feature_names(dataset)
svm, scaler, X_svm, fi_svm = model_svm.svm_train(dataset)
_, target = model_svm.split_for_target(dataset)
fis = [fi_svm.fi_lime, fi_svm.fi_shap, fi_svm.fi_perm_imp]

bot_name = "Robin"

def getSynonyms(list_words):
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

    return keywords_dict, keywords

def readBanknote(feature_names):
    vars = [0] * len(feature_names)
    for i in range(len(feature_names)):
        v = input(f"     {feature_names[i]}: ")
        vars[i] = float(v)

    vars = [np.array(vars)]
    vars = scaler.transform(vars)
    return vars

def prediction(vars):
    return "The prediction is that is a " + CLASS_NAMES[svm.predict(vars)[0]]

def phase2Menu():
    intro = "Now you can see the interpretation of some models, as well as the informations about them."
    choice1 = "1)informations about LIME"
    choice2 = "2)informations about Shap"
    choice3 = "3)informations about Permutation Importance(PI)"
    choice4 = "4)interpretation of LIME"
    choice5 = "5)interpretation of Shap"
    choice6 = "6)interpretation of Permutation Importance(PI)"

    menu = intro + choice1 + choice2 + choice3 + choice4 + choice5 + choice6
    return menu

def infoLime():
    return "Info of LIME"

def infoShap():
    return "Info of Shap"

def infoPI():
    return "Info of PI"

def yesOrNo(question):
    print(f"{bot_name}: ", question)
    print('\tyes\n\tno')

    while(True):
        user_input = input(f"{user_name}: ").lower()
        if 'yes' in user_input:
            return True

        elif 'no' in user_input:
            return False

        print(f"{bot_name}: I'm a bot programmed to answer only some of the specific answers. Here are the topics I can help you with.\nRobin: Select the topic or write your answer below\n")
        return "I'm a bot programmed to answer only some of the specific answers. Here are the topics I can help you with. Select the topic or write your answer below"

        continue

def phase3Menu():
    intro = "After that, I suggest you to use the Altruist. A new methodology that aims to tackle a few problems of feature importance-based aproaches. You can see"
    choice1 = "1)Info of Altruist"
    choice2 = "2)Combinatorial interpretation from Altruist"
    choice3 = "3)The untruthful features of LIME"
    choice4 = "4)The untruthful features of Shap"
    choice5 = "5)The untruthful features of Permutation Importance(PI)"
    choice6 = "6)The counterfactuals for every feature"
    choice7 = "7)Go to previous step"

    menu = intro + choice1 + choice2 + choice3 + choice4 + choice5 + choice6 + choice7
    return menu

def infoAltruist():
    return "Info of Altruist"

def plotMethod(feature_names, ar, title, yLabel):
    my_cmap = cm.get_cmap('Greens', 17)
    my_norm = Normalize(vmin = 0, vmax = 4)

    fig, axs = plt.subplots(1, 1, figsize = (10, 4.7), dpi = 150, sharex = True)

    axs.bar(feature_names, ar,color=my_cmap(my_norm([1,2,3,4])))
    axs.set_title(title)
    axs.set_ylabel(yLabel)
    #change the img name with randomString.png
    imgName ="./images/" + title + '.png'

    plt.savefig(imgName, dpi=300, bbox_inches='tight')
    plt.show()
    return imgName

def metaExplanation(X_t, inst):
    altruistino = Altruist(svm, X_t, fis, feature_names, None)
    untruthful_features = altruistino.find_untruthful_features(inst[0])
    return untruthful_features

def counterfactuals(feature_names, untruthful_features):
    min_un = 100000
    min_pos = 0
    for i in range(len(fis)):
        if min_un > len(untruthful_features[0][i]):
            min_un = len(untruthful_features[0][i])
            min_pos = i

    if not untruthful_features[1][min_pos]:
        print(f'{bot_name}: There are no counterfactual!')
        return "There are no counterfactual!"

    c = untruthful_features[1][min_pos][0]
    
    bot_response = "The counterfactuals for the feature " + feature_names[c[0] - 1] + ' is ' + str(c[1])
    return bot_response

def AltruistPlot(untruthful_features, feature_names, vars):
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

    imgName = plotMethod(feature_names, altruistVars, "Altruist", 'Feature Importance')
    return imgName

def printUntruthful(methodName, untruthful_features, length):
    return "Untruthful features of " + methodName + " : " + untruthful_features + " (" + length + ")"

def findMatched(keywords_dict, user_input):
    matched = None
    for key, pattern in keywords_dict.items():
        #Using the re search function
        if re.search(pattern, user_input):
            matched = key
        elif key in user_input:
            matched = key

    return matched

def getResponse(phase, user_input, aux):
    matched = findMatched(keywords_dict, user_input)
    bot_responses = {}

    if matched == 'bye':
        bot_responses['answer'] = responses['default']['quit']()
        return bot_responses

    if matched not in keywords:
        bot_responses['answer'] = responses['default']["error"]()
        return bot_responses

    if matched == 'hello':
        bot_responses['answer'] = responses['default']["hello"]()
        return bot_responses

    if phase == 'phase1':
        if matched == 'yes':
            #Maybe it will be created in frontend
            #vars = responses[phase][matched](feature_names)
            bot_responses['answer'] = "Okay! Fill in the following features"
            bot_responses['next_question'] = phase2Menu()
            bot_responses['phase'] = "phase2"

        elif matched == 'no':
            bot_responses['answer'] = responses['default']['quit']()

        return bot_responses

    elif phase == 'phase2':
        if matched == 'informations':
            if 'lime' in user_input:
                bot_responses['answer'] = responses[phase]['infoLime']()
            elif 'shap' in user_input:
                bot_responses['answer'] = responses[phase]['infoShap']()
            elif ('pi' in user_input) or ('permutation importance' in user_input):
                bot_responses['answer'] = responses[phase]['infoPI']()
            else:
                bot_responses['answer'] = responses[phase]['noMethod']()

            bot_responses['next_question'] = "Do you want to go back?"

        elif matched == 'interpretation':
            if 'lime' in user_input:
                bot_responses['answer'] = plotMethod(feature_names, fi_svm.fi_lime(aux["vars"][0], "_", svm), 'LIME', 'Feature Importance')

            elif 'shap' in user_input:
                bot_responses['answer'] = plotMethod(feature_names, fi_svm.fi_shap(aux["vars"][0], "_", svm), 'Shap', 'Feature Importance')

            elif ('pi' in user_input) or ('permutation importance' in user_input):
                bot_responses['answer'] = plotMethod(feature_names, fi_svm.fi_perm_imp(aux["vars"][0], "_", svm), 'PI', 'Feature Importance')

            else:
                bot_responses['answer'] = responses[phase]['noMethod']()

            bot_responses['next_question'] = "Do you want to go back?"

        elif matched == 'yes':
            bot_responses['answer'] = phase2Menu()

        elif matched == 'no':
            bot_responses['answer'] = phase3Menu()
            bot_responses['phase'] = "phase3"

        return bot_responses

    elif phase == 'phase3':
        if matched == 'informations':
            bot_responses['answer'] = responses[phase]['infoAltruist']()
            bot_responses['next_question'] = "Do you want to go back?"

        elif matched == 'interpretation':
            bot_responses['answer'] = AltruistPlot(aux["untruthful"][0], feature_names, aux["vars"])
            bot_responses['next_question'] = "Do you want to go back?"

        elif matched == 'features':
            if 'lime' in user_input:
                bot_responses['answer'] = printUntruthful("LIME", str(aux["untruthful"][0][0]), str(len(aux["untruthful"][0][0])))
                bot_responses['next_question'] = "Do you want to go back?"

            elif 'shap' in user_input:
                bot_responses['answer'] = printUntruthful("Shap", str(aux["untruthful"][0][1]), str(len(aux["untruthful"][0][1])))
                bot_responses['next_question'] = "Do you want to go back?"

            elif ('pi' in user_input) or ('permutation importance' in user_input):
                bot_responses['answer'] = printUntruthful("Permutation Importance", str(aux["untruthful"][0][2]), str(len(aux["untruthful"][0][2])))
                bot_responses['next_question'] = "Do you want to go back?"

            else:
                bot_responses['answer'] = responses[phase]['noMethod']()
                bot_responses['next_question'] = phase3Menu()

        elif matched == 'counterfactual':
            bot_responses['answer'] = responses[phase][matched](aux["feature_names"], aux["untruthful"])
            bot_responses['next_question'] = "Do you want to go back?"

        elif matched == 'previous':
            bot_responses['answer'] = phase2Menu();
            bot_responses['phase'] = 'phase2'

        elif matched == 'yes':
            bot_responses['answer'] = phase3Menu()

        elif matched == 'no':
            bot_responses['answer'] = "That's it. Bye!"

        return bot_responses


responses = {
    'phase1': {
        'yes': readBanknote,
    },
    'phase2': {
        'infoLime': infoLime,
        'infoShap': infoShap,
        'infoPI': infoPI,
        'noMethod': lambda: ("Please give also a method")
    },
    'phase3': {
        'infoAltruist': infoAltruist,
        'noMethod': lambda: ("Please give also a method"),
        'altruist' : metaExplanation,
        'counterfactual': counterfactuals

    },
    'default': {
        'hello': lambda: ("Hello! Time to predict some banknotes"),
        'quit': lambda: ("Thank you for visiting!"),
        'error': lambda: ("I'm a bot programmed to answer only some of the specific answers. Here are the topics I can help you with.\nRobin: Select the topic or write your answer below\n")
    }
}

list_words = ['yes', 'no', 'bye', 'hello', 'informations', 'interpretation', 'features', 'counterfactual', 'previous']
keywords_dict, keywords = getSynonyms(list_words)

"""
user_name = input(f"\n\nHi​! My name is Robin. Let me know if you have any questions regarding our tool!\nWhat's your name?\n")
print(f"Robin: Hello {user_name}\nAre you ready to predict some banknotes?\n1)yes I am ready!\n2)no bye.")

phase = 'phase1'
aux = {}

x = getResponse(phase, 'yes', aux)
print("1 ->", x)

aux["vars"] = readBanknote(feature_names)
aux['X_svm'] = X_svm
aux["feature_names"] = feature_names

print(prediction(aux["vars"]))

print('I want the interpretation about lime')
print("2 ->",getResponse('phase2', 'I want the informations about lime', aux))

print('no')
print("3 ->",getResponse('phase2', 'no', aux))

aux['untruthful'] = metaExplanation(aux["X_svm"], aux["vars"])

print("I want to see the informations of Altruist")
print("4 ->",getResponse('phase3', 'I want to see the informations of Altruist', aux))

print('yes')
print("5 ->",getResponse('phase3', 'yes', aux))

print("i want to see the counterfactual")
print("6 ->",getResponse('phase3', 'I want to see the counterfactual', aux))

"""
# while(True):
#     user_input = input(f"{user_name}: ").lower()
#
#     matched = findMatched(keywords_dict)
#
#     if matched == 'bye':
#         responses['default']['quit']()
#         break
#
#     if matched not in keywords:
#         responses['default']["error"]()
#         continue
#
#     if matched == 'hello':
#         responses['default']['hello']()
#         continue
#
#     if phase == 'phase1':
#         if matched == 'yes':
#             vars = responses[phase][matched](feature_names)
#             prediction(vars)
#             phase = 'phase2'
#             phase2Menu()
#
#         elif matched == 'no':
#             responses['default']['quit']()
#             break
#
#
#     elif phase == 'phase2':
#         if matched == 'informations':
#             if 'lime' in user_input:
#                 responses[phase]['infoLime']()
#             elif 'shap' in user_input:
#                 responses[phase]['infoShap']()
#             elif ('pi' in user_input) or ('permutation importance' in user_input):
#                 responses[phase]['infoPI']()
#             else:
#                 responses[phase]['noMethod']()
#                 continue
#
#         elif matched == 'interpretation':
#             if 'lime' in user_input:
#                 pathImg = plotMethod(feature_names, fi_svm.fi_lime(vars[0], "_", svm), 'LIME', 'Feature Importance')
#             elif 'shap' in user_input:
#                 pathImg = plotMethod(feature_names, fi_svm.fi_shap(vars[0], "_", svm), 'Shap', 'Feature Importance')
#
#             elif ('pi' in user_input) or ('permutation importance' in user_input):
#                 pathImg = plotMethod(feature_names, fi_svm.fi_perm_imp(vars[0], "_", svm), 'PI', 'Feature Importance')
#
#             else:
#                 responses[phase]['noMethod']()
#                 continue
#
#         if yesOrNo("Do you want to go back"):
#             phase2Menu()
#         else:
#             phase = "phase3"
#             untruthful_features = responses[phase]['altruist'](X_svm, vars)
#             print(untruthful_features)
#             phase3Menu()
#
#     elif phase == 'phase3':
#         if matched == 'interpretation':
#             varsAltruist(untruthful_features[0], feature_names, vars)
#             print("Altruist plot")
#
#         if matched == 'features':
#             if 'lime' in user_input:
#                 print("Untruthful features LIME: "+str(untruthful_features[0][0])+" ("+str(len(untruthful_features[0][0]))+")")
#             elif 'shap' in user_input:
#                 print("Untruthful features Shap: "+str(untruthful_features[0][1])+" ("+str(len(untruthful_features[0][1]))+")")
#             elif ('pi' in user_input) or ('permutation importance' in user_input):
#                 print("Untruthful features Permutation Importance: "+str(untruthful_features[0][2])+" ("+str(len(untruthful_features[0][2]))+")")
#             else:
#                 responses[phase]['noMethod']()
#                 continue
#
#         elif matched == 'counterfactual':
#             responses[phase][matched](feature_names, untruthful_features)
#
#         elif matched == 'previous':
#             phase = 'phase2'
#             phase2Menu();
#             continue
#
#         if yesOrNo('Do you want to go back'):
#             phase3Menu()
#         else:
#             print('thats it bye')
#             break
