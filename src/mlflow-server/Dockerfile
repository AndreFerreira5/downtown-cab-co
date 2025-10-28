FROM ghcr.io/mlflow/mlflow:latest

COPY src/mlflow-server/mlflow-entrypoint.sh /mlflow-entrypoint.sh
RUN chmod +x /mlflow-entrypoint.sh

ENTRYPOINT ["/mlflow-entrypoint.sh"]