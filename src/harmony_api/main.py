import logging
from fastapi import FastAPI, Response

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

app = FastAPI(title="Harmony ML API", version="1.0.0")

@app.get("/")
async def read_route():
    return Response("server is running...")
