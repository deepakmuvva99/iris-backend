import os
from flask import Flask, request, jsonify
from flask_cors import CORS
from tensorflow.keras.models import load_model
import numpy as np

# Load model
model = load_model("iris_model.h5")

# Label mapping
label_map = ['Setosa', 'Versicolor', 'Virginica']

# Setup Flask
app = Flask(__name__)
CORS(app)  # Allow requests from React frontend

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()
        features = data["features"]  # Expecting 4 input features
        features = np.array(features).reshape(1, -1)
        prediction = model.predict(features)
        predicted_class = label_map[np.argmax(prediction)]

        return jsonify({"prediction": predicted_class})
    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == "__main__":
    # Use dynamic port for Render or default to 5000 locally
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
