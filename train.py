import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
from typing import Union, Tuple
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

class FraudDetectionModel:
    """Unified fraud detection model handler"""

    def __init__(self, model_type: str = "isolation_forest", input_dim: int = 30):
        self.model_type = model_type
        self.model = None
        self.threshold = 1.5  # Default threshold for autoencoder
        self.input_dim = input_dim
        self.scaler = StandardScaler()

        if model_type == "autoencoder":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.autoencoder = self._build_autoencoder(input_dim)

    def _build_autoencoder(self, input_dim: int) -> nn.Module:
        return nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.Linear(16, 32),
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, input_dim),
            nn.Sigmoid()
        ).to(self.device)

    def load_model(self, model_path: Union[str, Path]):
        if self.model_type == "isolation_forest":
            self.model = joblib.load(model_path)
        else:
            self.autoencoder.load_state_dict(torch.load(model_path, map_location=self.device))
            self.autoencoder.eval()

    def save_model(self, model_path: Union[str, Path]):
        if self.model_type == "isolation_forest":
            joblib.dump(self.model, model_path)
        else:
            torch.save(self.autoencoder.state_dict(), model_path)

    def set_threshold(self, threshold: float):
        self.threshold = threshold

    def predict(self, transaction: np.ndarray) -> Tuple[bool, float]:
        if self.model_type == "isolation_forest":
            score = -self.model.score_samples([transaction])[0]
            return score > 0.5, score
        transaction_tensor = torch.FloatTensor(transaction).to(self.device)
        with torch.no_grad():
            reconstruction = self.autoencoder(transaction_tensor).cpu().numpy()
        error = np.mean(np.square(transaction - reconstruction))
        return error > self.threshold, error

    def train(self, X: np.ndarray, model_path: Path, epochs: int = 20, lr: float = 1e-3):
        if self.model_type == "isolation_forest":
            self.model = IsolationForest(contamination=0.01, random_state=42)
            self.model.fit(X)
            self.save_model(model_path)
            print("Isolation Forest training complete.")
        else:
            self.autoencoder.train()
            criterion = nn.MSELoss()
            optimizer = optim.Adam(self.autoencoder.parameters(), lr=lr)
            X_tensor = torch.FloatTensor(X).to(self.device)

            for epoch in range(epochs):
                optimizer.zero_grad()
                output = self.autoencoder(X_tensor)
                loss = criterion(output, X_tensor)
                loss.backward()
                optimizer.step()
                print(f"Epoch {epoch + 1}/{epochs} - Loss: {loss.item():.6f}")

            self.save_model(model_path)
            print("Autoencoder training complete.")

def preprocess_data(filepath: str) -> Tuple[np.ndarray, StandardScaler, int]:
    df = pd.read_csv(filepath)
    df = df.drop(columns=["Time", "Class"])  # Only drop "Class" column, keep "Time"
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df)
    return X_scaled, scaler, X_scaled.shape[1]

def main():
    model_type = "autoencoder"  # or "isolation_forest"
    csv_path = "creditcard.csv"
    model_path = Path("models") / (
        "isolation_forest.pkl" if model_type == "isolation_forest" 
        else "autoencoder.pt"
    )
    model_path.parent.mkdir(parents=True, exist_ok=True)

    # Load and preprocess dataset
    X, scaler, input_dim = preprocess_data(csv_path)

    # Train model
    detector = FraudDetectionModel(model_type=model_type, input_dim=input_dim)
    detector.train(X, model_path=model_path)

    # Example transaction (same shape as features)
    sample_transaction = scaler.transform([X[0]])  # Use the first row as example
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
