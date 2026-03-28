from flask import Flask, jsonify, request
from flask_cors import CORS
import pickle

app = Flask(__name__)
CORS(app, resources={r"/predict": {"origins": "*"}})

model = pickle.load(open("model.pkl", "rb"))

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    features = data["features"]

    result = model.predict([features])

    return jsonify({"result": str(result[0])})
