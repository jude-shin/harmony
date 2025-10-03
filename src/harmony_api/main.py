import logging
from fastapi import FastAPI, Response
import harmony_api.serving.routes as serving
import harmony_api.dataset.routes as dataset
import harmony_api.train.routes as train

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

app: FastAPI = FastAPI(title="Harmony ML API", version="1.0.0")
app.include_router(serving.router)
app.include_router(dataset.router)
app.include_router(train.router)

@app.get("/")
async def read_route():
    return Response("Server is running...\n")
