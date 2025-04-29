import joblib
import numpy as np
import pandas as pd
import torch
from fraud_detection import FraudDetectionModel, preprocess_data  # Assuming fraud_detection.py contains the FraudDetectionModel class

def main():
    model_type = "autoencoder"  # or "isolation_forest"
    csv_path = "creditcard.csv"
    model_path = "models/autoencoder.pt"  # Path to the trained model file

    # Load and preprocess dataset
    X, scaler, input_dim = preprocess_data(csv_path)

    # Load model
    detector = FraudDetectionModel(model_type=model_type, input_dim=input_dim)
    detector.load_model(model_path)

    # Example transaction (same shape as features)
    sample_transaction = scaler.transform([X[0]])  # Use the first row as example

    # Predict fraud status
    is_fraud, confidence = detector.predict(sample_transaction)

    print("\n" + "=" * 40)
    print("Transaction Analysis Results:")
    print(f"Model Type: {model_type.upper()}")
    print(f"Status: {'FRAUD' if is_fraud else 'NORMAL'}")
    print(f"Confidence Score: {confidence:.4f}")
    if is_fraud:
        print("🚨 Warning: Potential fraud detected!")
    print("=" * 40 + "\n")

if __name__ == "__main__":
    main()
