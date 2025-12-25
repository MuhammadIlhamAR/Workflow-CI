import os
import glob
import argparse
import logging

import numpy as np
import pandas as pd

import mlflow
import mlflow.sklearn

# anti Tkinter / GUI error di CI
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


# =========================
# Dataset resolver
# =========================
def resolve_input_path(p: str) -> str:
    if p and os.path.exists(p):
        return p

    if p:
        base = os.path.basename(p)
        if os.path.exists(base):
            logging.info("Input '%s' tidak ada, pakai '%s' di root.", p, base)
            return base

    matches = glob.glob("**/*PRSA*preprocess*.csv", recursive=True)
    if matches:
        logging.info("Auto-found dataset: %s", matches[0])
        return matches[0]

    matches = glob.glob("**/*preprocess*.csv", recursive=True)
    if matches:
        logging.info("Auto-found dataset: %s", matches[0])
        return matches[0]

    raise FileNotFoundError(
        f"Dataset tidak ditemukan. Coba taruh file csv di folder ini atau pakai --input_file PATH_CSV"
    )


# =========================
# Build pipeline
# =========================
def build_pipeline(X: pd.DataFrame, n_estimators: int, max_depth: int | None, seed: int = 42) -> Pipeline:
    num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = X.select_dtypes(exclude=[np.number]).columns.tolist()

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", SimpleImputer(strategy="median"), num_cols),
            ("cat", Pipeline(steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("onehot", OneHotEncoder(handle_unknown="ignore"))
            ]), cat_cols),
        ],
        remainder="drop"
    )

    rf = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=seed,
        n_jobs=-1
    )

    return Pipeline(steps=[
        ("prep", preprocessor),
        ("rf", rf)
    ])


# =========================
# Confusion matrix plot
# =========================
def save_confusion_matrix(cm: np.ndarray, out_path: str, title: str):
    plt.figure(figsize=(6, 5))
    plt.imshow(cm)
    plt.title(title)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.colorbar()

    thresh = cm.max() / 2 if cm.max() > 0 else 0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(
                j, i, f"{cm[i, j]}",
                ha="center", va="center",
                color="white" if cm[i, j] > thresh else "black"
            )

    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()


# =========================
# MAIN
# =========================
def main():
    parser = argparse.ArgumentParser(description="RandomForest Training + MLflow (baseline)")
    parser.add_argument(
        "--input_file",
        type=str,
        default="PRSA_Data_Aotizhongxin_preprocessing.csv",
        help="Path ke dataset preprocessing CSV"
    )
    parser.add_argument("--n_estimators", type=int, default=100)
    parser.add_argument("--max_depth", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--experiment_name", type=str, default="Membangun_Model")
    parser.add_argument("--run_name", type=str, default="rf_baseline")
    args = parser.parse_args()

    # max_depth: kalau user kasih 0 atau negatif, treat jadi None
    max_depth = None if (args.max_depth is None or args.max_depth <= 0) else int(args.max_depth)

    csv_path = resolve_input_path(args.input_file)
    df = pd.read_csv(csv_path)

    target_col = "RAIN_Category"
    if target_col not in df.columns:
        raise ValueError(f"Kolom target '{target_col}' tidak ditemukan di dataset.")

    X = df.drop(columns=[target_col])
    y = df[target_col].map({"Tidak Hujan": 0, "Hujan": 1})
    if y.isna().any():
        bad = df.loc[y.isna(), target_col].unique()
        raise ValueError(f"Label target tidak dikenali: {bad}. Harus 'Tidak Hujan'/'Hujan'.")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=args.seed, stratify=y
    )

    mlflow.set_experiment(args.experiment_name)
    mlflow.sklearn.autolog(log_models=True)

    model = build_pipeline(X_train, n_estimators=args.n_estimators, max_depth=max_depth, seed=args.seed)

    with mlflow.start_run(run_name=args.run_name):
        mlflow.log_param("dataset_path", csv_path)

        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        acc = float(accuracy_score(y_test, y_pred))
        f1 = float(f1_score(y_test, y_pred))

        mlflow.log_metric("test_accuracy_manual", acc)
        mlflow.log_metric("test_f1_manual", f1)

        os.makedirs("artifacts", exist_ok=True)

        report = classification_report(y_test, y_pred, target_names=["Tidak Hujan", "Hujan"])
        report_path = os.path.join("artifacts", "classification_report.txt")
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(report)
        mlflow.log_artifact(report_path)

        cm = confusion_matrix(y_test, y_pred)
        cm_path = os.path.join("artifacts", "confusion_matrix.png")
        save_confusion_matrix(cm, cm_path, title="Confusion Matrix - RF Baseline")
        mlflow.log_artifact(cm_path)

        print(f"✅ TRAINING SELESAI | acc={acc:.4f} f1={f1:.4f}")


if __name__ == "__main__":
    main()
