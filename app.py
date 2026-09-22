from flask import Flask, request, jsonify
import joblib
import os

app = Flask(__name__)

model = joblib.load("headline_classifier.pkl")


@app.route("/")
def home():
    return jsonify({
        "message": "Headline Classifier API is running"
    })


@app.route("/predict", methods=["POST"])
def predict():

    data = request.get_json()

    if not data or "headline" not in data:
        return jsonify({
            "error": "Headline is required"
        }), 400

    headline = data["headline"]

    if not headline.strip():
        return jsonify({
            "error": "Headline cannot be empty"
        }), 400

    prediction = model.predict([headline])[0]

    return jsonify({
        "headline": headline,
        "prediction": int(prediction)
    })


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)