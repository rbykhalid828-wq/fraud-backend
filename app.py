from flask import Flask, jsonify, request
from flask_cors import CORS
import pickle

app = Flask(__name__)
CORS(app)

# Load the model
model = pickle.load(open("model.pkl", "rb"))

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    features = data["features"]

    # Predict result and probability
    result = model.predict([features])
    probability = model.predict_proba([features])[0][1]  # Get the probability of fraud (class 1)

    return jsonify({"result": str(result[0]), "probability": probability})

if __name__ == "__main__":
    app.run(debug=True)
