from flask import Flask, json, request, jsonify
from bot import getResponse

app = Flask(__name__)

@app.route("/works")
def works():
    return "It works!"

@app.route("/ask", methods=['POST'])
def ask():
    content = request.get_json()

    for input in ['phase', 'user_input', 'aux']:
        if input not in content:
            return input + ' not in request body', 400

    return jsonify(getResponse(content['phase'], content['user_input'], content['aux'])) 

app.run(port=3000)
