FROM python:3.11-slim

WORKDIR /app

RUN pip install --no-cache-dir mlflow==2.17.2 scikit-learn pandas numpy

COPY model /app/model

EXPOSE 5001
CMD ["mlflow", "models", "serve", "-m", "/app/model", "-p", "5001", "--host", "0.0.0.0", "--no-conda"]
