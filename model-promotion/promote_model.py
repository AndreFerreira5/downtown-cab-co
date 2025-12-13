import os
import mlflow
from mlflow.tracking import MlflowClient


def main():
    client = MlflowClient(
        tracking_uri=os.getenv("MLFLOW_TRACKING_URI_EXTERNAL"),
    )

    model_name = os.getenv("MODEL_NAME")
    from_alias = os.getenv("FROM_ALIAS", "staging")
    to_alias = os.getenv("TO_ALIAS", "production")

    # Get the version with the staging alias
    model_version = client.get_model_version_by_alias(model_name, from_alias)

    # Promote to production
    client.set_registered_model_alias(model_name, to_alias, model_version.version)
    print(f"Model {model_name} version {model_version.version} promoted to {to_alias}")


if __name__ == "__main__":
    main()