import logging
from fastapi import FastAPI, Response
from harmony_api.serving.routes import router

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

app = FastAPI(title="Harmony ML API", version="1.0.0")
app.include_router(router)

@app.get("/")
async def read_route():
    return Response("server is running...")
