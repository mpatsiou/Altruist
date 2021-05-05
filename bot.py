import re
from nltk.corpus import wordnet

feature_names = ['variance','skew','curtosis','entropy']
class_names=['fake banknote','real banknote'] #0: no, 1: yes #or ['not authenticated banknote','authenticated banknote']

list_words = ['hello', 'prediction', 'interpretation', 'yes']
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

keywords['hello'] = []
keywords['prediction'] = []
keywords['interpretation'] = []
keywords['yes'] = []

for key in list_syn:
    for synonym in list(list_syn[key]):
        keywords[key].append('.*\\b' + synonym + '\\b.*')

for key, values in keywords.items():
    keywords_dict[key]=re.compile('|'.join(values))

responses = {
    'hello': "Robin: Hello! Let's predict if those banknotes  are valid or not",
    'prediction': 'Robin: The prediction is..',
    'interpretation': "Robin: The interpretation of SVM's classification",
}


user_name = input(f"Welcome to Altruist Bot! My name is Robin. \nWhat's your name?  ")
print(f"Robin: Hello {user_name}")

while(True):
    user_input = input(f"{user_name}: ").lower()

    if user_input == 'bye':
        print("Robin: Thank you for visiting.")
        break

    matched = None
    for key, pattern in keywords_dict.items():

        #Using the re search function
        if re.search(pattern, user_input):
            matched = key

    if matched in responses:

        if matched == 'hello':
            print(responses[matched])

            print("Robin: Fill in the banknotes variables")
            vars = {}
            for feature in feature_names:
                vars[feature] = input(f"    {feature}: ")
        else:
            print(responses[matched])

    else:
        print("Robin: Sorry, please specify your answer")
