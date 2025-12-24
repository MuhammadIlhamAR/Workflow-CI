import argparse
import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

def train_model(input_file, n_estimators, max_depth):
    mlflow.autolog(log_models=True)

    # Load Data
    df = pd.read_csv(input_file)
    X = df.drop(['RAIN_Category'], axis=1)
    y = df['RAIN_Category'].map({'Tidak Hujan': 0, 'Hujan': 1})

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    with mlflow.start_run():
        model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=42)
        model.fit(X_train, y_train)
        
        print(f"✅ Training Skilled Selesai. Cek MLflow UI/DagsHub untuk hasil autolog.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", type=str)
    parser.add_argument("--n_estimators", type=int, default=100)
    parser.add_argument("--max_depth", type=int, default=20)
    args = parser.parse_args()
    train_model(args.input_file, args.n_estimators, args.max_depth)
