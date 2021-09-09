from model_svm import get_feature_names, get_dataset, get_dataset_stats, svm_train, split_for_target
from flask import Flask, json, request, jsonify
from flask_cors import CORS
import model_svm
import numpy as np
from altruist import Altruist

dataset = model_svm.get_dataset()
dataset_statistics = model_svm.get_dataset_stats(dataset)
features_names = model_svm.get_feature_names(dataset)
svm, scaler, X_svm, fi_svm = model_svm.svm_train(dataset)
_, target = model_svm.split_for_target(dataset)

fis = {
    'lime': fi_svm.fi_lime,
    'shap': fi_svm.fi_shap,
    'pi': fi_svm.fi_perm_imp
}

CLASS_NAMES = ['fake banknote', 'real banknote']

app = Flask(__name__)
CORS(app)

@app.route("/features_names", methods=['GET'])
def get_feature_names():
    return jsonify(features_names)

@app.route("/predict", methods=['POST'])
def predict():
    features = request.get_json()

    features_values = features.values()
    values = []

    for value in features_values:
        values.append(float(value))

    values = [np.array(values)]
    values = scaler.transform(values)

    prediction = svm.predict(values)[0]

    return CLASS_NAMES[prediction]

@app.route("/feature_importance", methods=['GET'])
def get_feature_importance():
    method = request.args.get('method')
    values = request.args.get('values')

    print('_____________________ON SERVER FEATURE IMPORTANCE____________________')
    print('method: ', method)
    print('values: ', values)
    print("_____________________________________________________________________")

    if not method or not values:
        return "Wrong input", 400

    values = list(map(int, values.split(',')))
    values = [np.array(values)]
    values = scaler.transform(values)
    feature_importance = fis[method](values[0], _, svm)

    #Shap and PI do not return a list
    if not isinstance(feature_importance, list):
        feature_importance = feature_importance.tolist()

    return jsonify(feature_importance)

@app.route("/altruist", methods=['GET'])
def get_metaExplanation():
    values = request.args.get('values')

    if not values:
        return "Wrong input", 400

    values = list(map(int, values.split(',')))
    values = [np.array(values)]
    values = scaler.transform(values)

    fi_method_list = list(fis.values())
    altruistino = Altruist(svm, X_svm, fi_method_list, features_names, None)
    untruthful_features = altruistino.find_untruthful_features(values[0])
    return jsonify(untruthful_features)

@app.route("/", methods=['GET'])
def hello():
    return "It works"

app.run(port=3000)
