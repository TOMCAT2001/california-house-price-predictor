from pathlib import Path

import joblib
import pandas as pd
from flask import Flask, jsonify, request, send_from_directory

BASE = Path(__file__).parent
COLUMNS = [
    "MedInc", "HouseAge", "AveRooms", "AveBedrms",
    "Population", "AveOccup", "Latitude", "Longitude",
]

model = joblib.load(BASE / "house_price_model.joblib")
app = Flask(__name__)


@app.get("/")
def index():
    return send_from_directory(BASE, "index.html")


@app.post("/predict")
def predict():
    data = request.get_json(silent=True) or {}
    try:
        row = [[float(data[c]) for c in COLUMNS]]
    except (KeyError, TypeError, ValueError):
        return jsonify(error="Missing or invalid input values."), 400

    X = pd.DataFrame(row, columns=COLUMNS)
    price = float(model.predict(X)[0]) * 100_000
    return jsonify(price=price)


if __name__ == "__main__":
    print("Open http://127.0.0.1:5000 in your browser")
    app.run(host="127.0.0.1", port=5000, debug=False)
