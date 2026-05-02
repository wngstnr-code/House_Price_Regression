from pathlib import Path

import joblib
import kagglehub
import pandas as pd
from kagglehub import KaggleDatasetAdapter
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "house_price_linear_regression.pkl"
SCALER_PATH = ROOT / "scaler_linear_regression.pkl"


def main() -> None:
    file_path = "house_price_regression_dataset.csv"
    df = kagglehub.load_dataset(
        KaggleDatasetAdapter.PANDAS,
        "prokshitha/home-value-insights",
        file_path,
    )

    target_col = "House_Price"
    X = df.drop(columns=[target_col]).copy()
    y = df[target_col].copy()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    model = LinearRegression()
    model.fit(X_train_scaled, y_train)

    joblib.dump(scaler, SCALER_PATH)
    joblib.dump(model, MODEL_PATH)

    print("Artifacts saved:")
    print(f"- {SCALER_PATH}")
    print(f"- {MODEL_PATH}")


if __name__ == "__main__":
    main()
