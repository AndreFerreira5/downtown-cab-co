FROM ghcr.io/mlflow/mlflow:latest

COPY mlflow-entrypoint.sh /mlflow-entrypoint.sh
RUN chmod +x /mlflow-entrypoint.sh

ENTRYPOINT ["/mlflow-entrypoint.sh"]