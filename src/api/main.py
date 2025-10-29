import logging
import os
from contextlib import asynccontextmanager
import uvicorn
from fastapi import FastAPI
import mlflow.pyfunc

from logging_config import configure_logging
from routes import (
    health
)

# configure logging globally
configure_logging()
logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    mlflow.set_tracking_uri(
        os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5050")
    )

    models = []
    try:
        models.append(
            mlflow.pyfunc.load_model(
                model_uri=f"models:/taxi/Production" # TODO later offload this to a .env file
            )
        )
        logger.info("Model loaded succesfully!")
    except Exception as e:
        logger.warning(f"Could not load model at startup: {e}.")
        logger.warning(f"The model will be lazy loaded.")



app = FastAPI()
app.include_router(health.router, prefix="/health", tags=["health"])


@app.get("/ping")
async def ping():
    return {"message": "pong"}


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8080, reload=True)
