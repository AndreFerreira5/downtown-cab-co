FROM ghcr.io/mlflow/mlflow:latest

COPY src/mlflow-server/mlflow-entrypoint.sh /mlflow-entrypoint.sh

# Normalize Windows line endings just in case (CRLF -> LF)
RUN sed -i 's/\r$//' /mlflow-entrypoint.sh && chmod +x /mlflow-entrypoint.sh

ENTRYPOINT ["/mlflow-entrypoint.sh"]