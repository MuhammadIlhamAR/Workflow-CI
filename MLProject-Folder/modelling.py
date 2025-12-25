import os
import glob
import argparse
import logging
import random

import numpy as np
import pandas as pd

import mlflow
import mlflow.sklearn
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, StratifiedKFold, RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    confusion_matrix,
    classification_report,
    roc_auc_score,
)


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


# =========================
# MLflow local
# =========================
def setup_mlflow(experiment_name: str):
    tracking_dir = os.path.join(os.getcwd(), "mlruns")
    os.makedirs(tracking_dir, exist_ok=True)
    mlflow.set_tracking_uri(f"file:{tracking_dir}")
    mlflow.set_experiment(experiment_name)
    logging.info("MLflow tracking -> %s", tracking_dir)


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

    # cari csv yang mirip PRSA preprocessing
    matches = glob.glob("**/*PRSA*preprocess*.csv", recursive=True)
    if matches:
        logging.info("Auto-found dataset: %s", matches[0])
        return matches[0]

    # fallback cari csv apapun yang ada kata preprocessing
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
def build_pipeline(X: pd.DataFrame, random_state: int = 42) -> Pipeline:
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
        random_state=random_state,
        n_jobs=-1
    )

    pipe = Pipeline(steps=[
        ("prep", preprocessor),
        ("rf", rf)
    ])

    return pipe


# =========================
# Simple confusion matrix plot (matplotlib only)
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
    parser = argparse.ArgumentParser(description="RandomForest Tuning (RandomizedSearchCV) + MLflow (local)")
    parser.add_argument(
        "--input_file",
        type=str,
        default="PRSA_Data_Aotizhongxin_preprocessing.csv",
        help="Path ke dataset preprocessing CSV"
    )
    parser.add_argument("--experiment_name", type=str, default="Membangun_Model")
    parser.add_argument("--run_name", type=str, default="rf_tuning_randomsearch")
    parser.add_argument("--n_iter", type=int, default=25, help="Jumlah kombinasi random search")
    parser.add_argument("--cv", type=int, default=5, help="Jumlah folds CV")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    setup_mlflow(args.experiment_name)

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

    pipeline = build_pipeline(X_train, random_state=args.seed)

    # ---- param space TANPA scipy (biar tidak ribet install) ----
    param_distributions = {
        "rf__n_estimators": [100, 200, 300, 400, 600],
        "rf__max_depth": [None, 5, 10, 15, 20, 30, 40],
        "rf__min_samples_split": [2, 5, 10, 15],
        "rf__min_samples_leaf": [1, 2, 4, 8],
        "rf__max_features": ["sqrt", "log2", None],
        "rf__bootstrap": [True, False],
    }

    cv = StratifiedKFold(n_splits=args.cv, shuffle=True, random_state=args.seed)

    search = RandomizedSearchCV(
        estimator=pipeline,
        param_distributions=param_distributions,
        n_iter=args.n_iter,
        scoring="f1",
        cv=cv,
        n_jobs=-1,
        random_state=args.seed,
        refit=True,
        return_train_score=True,
        verbose=2,  # biar keliatan progres di terminal
    )

    mlflow.sklearn.autolog(log_models=False)

    with mlflow.start_run(run_name=args.run_name):
        mlflow.log_param("tuning_method", "RandomizedSearchCV")
        mlflow.log_param("scoring", "f1")
        mlflow.log_param("n_iter", args.n_iter)
        mlflow.log_param("cv_folds", args.cv)
        mlflow.log_param("seed", args.seed)
        mlflow.log_param("dataset_path", csv_path)

        # Tuning
        logging.info("Mulai tuning RandomizedSearchCV...")
        search.fit(X_train, y_train)
        logging.info("Tuning selesai.")

        best_model = search.best_estimator_
        best_params = search.best_params_
        best_cv_f1 = float(search.best_score_)

        mlflow.log_params(best_params)
        mlflow.log_metric("best_cv_f1", best_cv_f1)

        # Evaluate on test
        y_pred = best_model.predict(X_test)
        acc = float(accuracy_score(y_test, y_pred))
        f1 = float(f1_score(y_test, y_pred))

        mlflow.log_metric("test_accuracy", acc)
        mlflow.log_metric("test_f1", f1)

        # ROC-AUC (jaga-jaga kalau ada masalah proba)
        try:
            y_proba = best_model.predict_proba(X_test)[:, 1]
            auc = float(roc_auc_score(y_test, y_proba))
            mlflow.log_metric("test_roc_auc", auc)
        except Exception as e:
            logging.warning("ROC-AUC tidak bisa dihitung: %s", e)
            auc = None

        # Artifacts
        os.makedirs("artifacts", exist_ok=True)

        # classification report
        report = classification_report(y_test, y_pred, target_names=["Tidak Hujan", "Hujan"])
        report_path = os.path.join("artifacts", "classification_report_tuned.txt")
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(report)
        mlflow.log_artifact(report_path)

        # confusion matrix image
        cm = confusion_matrix(y_test, y_pred)
        cm_path = os.path.join("artifacts", "confusion_matrix_tuned.png")
        save_confusion_matrix(cm, cm_path, title="Confusion Matrix - Tuned RandomForest")
        mlflow.log_artifact(cm_path)

        # cv results
        cv_results = pd.DataFrame(search.cv_results_).sort_values("rank_test_score")
        cv_path = os.path.join("artifacts", "cv_results.csv")
        cv_results.to_csv(cv_path, index=False)
        mlflow.log_artifact(cv_path)

        # log model
        mlflow.sklearn.log_model(
            sk_model=best_model,
            artifact_path="model",
            input_example=X_train.head(5),
        )

        print("\n✅ TUNING SELESAI")
        print(f"Best CV F1 : {best_cv_f1:.4f}")
        print(f"Test Acc   : {acc:.4f}")
        print(f"Test F1    : {f1:.4f}")
        if auc is not None:
            print(f"Test AUC   : {auc:.4f}")


if __name__ == "__main__":
    main()
