# Gunakan image Python yang sudah ada
FROM python:3.8-slim

# Set working directory di dalam container
WORKDIR /app

# Salin file yang diperlukan (termasuk script, data, dan dependencies)
COPY . /app

# Install dependencies
RUN pip install --upgrade pip
RUN pip install -r MLProject/conda.yaml

# Install MLflow dan dependencies lainnya
RUN pip install mlflow

# Install DagsHub
RUN pip install dagshub

# Install libraries yang dibutuhkan
RUN pip install -U imbalanced-learn
