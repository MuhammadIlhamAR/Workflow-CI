FROM python:3.10-slim

# Install mlflow dan dependensi dasar
RUN pip install mlflow pandas scikit-learn

# Buat folder kerja
WORKDIR /app

# Copy isi model lo ke dalam image
COPY . /app/model

# Set environment variable untuk model
ENV MLFLOW_MODEL_PATH=/app/model

# Ekspos port standar MLflow
EXPOSE 5001

# Jalankan server model saat kontainer nyala
CMD ["mlflow", "models", "serve", "-m", "/app/model", "--host", "0.0.0.0", "--port", "5001", "--no-conda"]