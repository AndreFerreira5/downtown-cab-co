import logging

import uvicorn
from fastapi import FastAPI

from logging_config import configure_logging
from routes import (
    health
)

# configure logging globally
configure_logging()
logger = logging.getLogger(__name__)

app = FastAPI()
app.include_router(health.router, prefix="/health", tags=["health"])


@app.get("/ping")
async def ping():
    return {"message": "pong"}


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8080, reload=True)
