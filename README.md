# Rain Prediction Production Workflow - CI/CD & MLflow

Repository ini berisi alur kerja *production* untuk model klasifikasi prediksi hujan. Proyek ini mencakup pelatihan model, *hyperparameter tuning*, pelacakan eksperimen menggunakan MLflow, serta otomatisasi CI/CD menggunakan GitHub Actions.

## Project Overview
Repository ini mengelola script pelatihan dan tuning untuk klasifikasi kejadian hujan (RAIN_Category) menggunakan library **scikit-learn** dan **MLflow**. Sistem ini menyimpan artefak hasil pelatihan dan model dalam format MLflow yang siap digunakan untuk pembuatan Docker image.

## Requirements
- **Python**: 3.8+ recommended
- **Libraries**: `mlflow`, `dagshub`, `scikit-learn`, `pandas`, `numpy`, `matplotlib`, `seaborn` (Library ini akan diinstal otomatis oleh workflow CI)

## Struktur Direktori
- `.github/workflows/CI.yml`: Konfigurasi GitHub Actions untuk otomatisasi retraining.
- `MLProject-Folder/`: Folder utama yang berisi komponen MLflow Project.
  - `MLProject`: File konfigurasi entry point MLflow.
  - `modelling.py`: Script utama pelatihan model.
  - `conda.yaml`: Definisi environment untuk reproduksibilitas.
  - `PRSA_Data_Aotizhongxin_preprocessing.csv`: Dataset yang digunakan untuk training.

## Quick Start
### Menjalankan Training Secara Lokal
Anda dapat menjalankan script pelatihan langsung dari root repository dengan perintah:
```bash
python "MLProject-Folder/modelling.py" --input_file "MLProject-Folder/PRSA_Data_Aotizhongxin_preprocessing.csv" --n_estimators 100 --max_depth 20
