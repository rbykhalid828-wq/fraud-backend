from flask import Flask, jsonify, request
from flask_cors import CORS
import pickle

app = Flask(__name__)
CORS(app)

model = pickle.load(open("model.pkl", "rb"))

@app.route("/")
def home():
    return "Fraud Detection API Running"

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    features = data.get("features", [0]*30)

    result = model.predict([features])

    return jsonify({
        "prediction": int(result[0]),
        "message": "Fraud Detected" if result[0] == 1 else "Not Fraud"
    })

import os

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
