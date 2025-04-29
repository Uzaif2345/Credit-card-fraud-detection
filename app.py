from flask import Flask, render_template, request
from fraud_detection import FraudDetectionModel, preprocess_data
import numpy as np
import os
import joblib

app = Flask(__name__)

# Load model and scaler
model_type = "autoencoder"  # Change to "isolation_forest" if needed
model_path = os.path.join("models", "autoencoder.pt")
csv_path = "creditcard.csv"

# Preprocess data to get scaler and dimensions
X, scaler, input_dim = preprocess_data(csv_path)

# Initialize and load model
detector = FraudDetectionModel(model_type=model_type, input_dim=input_dim)
detector.load_model(model_path)

@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    if request.method == "POST":
        values = [float(request.form.get(f"feature{i+1}")) for i in range(input_dim)]
        scaled_input = scaler.transform([values])
        is_fraud, score = detector.predict(scaled_input)
        result = {
            "status": "FRAUD ⚠️" if is_fraud else "NORMAL ✅",
            "confidence": round(score, 4)
        }
    return render_template("index.html", input_dim=input_dim, result=result)

if __name__ == "__main__":
    app.run(debug=True)
