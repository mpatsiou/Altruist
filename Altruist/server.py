from flask import Flask, json, request, jsonify
from bot import getResponse
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

@app.route("/works")
def works():
    return "It works!"

@app.route("/ask", methods=['POST'])
def ask():
    content = request.get_json()
    print(content)

    for input in ['phase', 'user_input', 'aux']:
        if input not in content:
            return input + ' not in request body', 400

    response = getResponse(content['phase'], content['user_input'], content['aux'])

    return response

app.run(port=3000)