# from flask import Flask, render_template, request, jsonify
# import joblib
# import numpy as np
# import pandas as pd

# # Initialize Flask app
# app = Flask(__name__)

# # Load trained model
# model = joblib.load("travel_model.pkl")

# @app.route("/")
# def home():
#     return render_template("index.html")

# @app.route("/predict", methods=["POST"])
# def predict():
#     data = request.get_json()

#     # Extract input data
#     features = [
#         float(data["operator"]),
#         float(data["bus_type"]),
#         float(data["seats_left"]),
#         float(data["window_seats"]),
#         float(data["rating"]),
#         float(data["source"]),
#         float(data["destination"]),
#         float(data["distance"]),
#         float(data["departure_hour"]),
#         float(data["arrival_hour"]),
#         float(data["travel_duration"]),
#     ]

#     # Reshape for model prediction
#     prediction = model.predict([features])[0]

#     return jsonify({"predicted_cost": round(prediction, 2)})

# if __name__ == "__main__":
#     app.run(debug=True)


from flask import Flask, render_template, request, jsonify
import joblib

app = Flask(__name__)

model = joblib.load("travel_multi_model.pkl")

# Distance matrix
distance_matrix = {
    "Agra": {"Delhi": 240, "Jaipur": 250, "Lucknow": 330},
    "Delhi": {"Agra": 240, "Jaipur": 280, "Lucknow": 500},
    "Jaipur": {"Agra": 250, "Delhi": 280, "Lucknow": 600},
    "Lucknow": {"Agra": 330, "Delhi": 500, "Jaipur": 600}
}

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()

    source = data["source"]
    destination = data["destination"]
    distance = distance_matrix[source][destination]

    features = [
        float(data["operator"]),
        float(data["bus_type"]),
        float(data["source_code"]),
        float(data["destination_code"]),
        distance,
        float(data["rating"])
    ]

    prediction = model.predict([features])[0]

    return jsonify({
        "price": round(prediction[0], 2),
        "seats_left": int(prediction[1]),
        "window_seats": int(prediction[2]),
        "departure_hour": int(prediction[3]),
        "arrival_hour": int(prediction[4]),
        "travel_duration": int(prediction[5]),
        "distance": distance
    })

if __name__ == "__main__":
    app.run(debug=True)
