from __future__ import annotations

import json
import os
from pathlib import Path
from PIL import Image

import numpy as np
import tensorflow as tf

import uvicorn
from pydantic import BaseModel
from fastapi import FastAPI, HTTPException

from harmony_config.structs import GAMES
from utils.data_conversion import label_to_json

# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
GAME = GAMES.LORCANA.value
MODEL_DIR = Path(os.getenv("MODEL_DIR"), GAME)

MODEL_PATH = MODEL_DIR / "model.keras"
DEFAULT_THRESHOLD = float(os.getenv("THRESHOLD", 0.60))
PORT = int(os.getenv("PORT", 8000))

IMG_HEIGHT = 224 
IMG_WIDTH = 22

# --------------------------------------------------------------------------- #
# App & model initialisation
# --------------------------------------------------------------------------- #
app = FastAPI(
        title="Harmony ML API",
        version="1.0.0",
        )




try:
    model = tf.keras.models.load_model(MODEL_PATH)
except (IOError, ValueError) as exc:
    raise SystemExit(f"Could not load model at {MODEL_PATH}: {exc}") from exc


def preprocess_image(img: Image.Image) -> np.ndarray:
    """Resize → scale to [0,1] → add batch dimension."""
    img = img.resize((IMG_WIDTH, IMG_HEIGHT))
    arr = np.asarray(img, dtype="float32") / 255.0
    return np.expand_dims(arr, axis=0)


# --------------------------------------------------------------------------- #
# Routes
# --------------------------------------------------------------------------- #
@app.get("/ping", summary="Health-check")
async def ping() -> dict[str, str]:
    return {"ping": "pong"}

@app.post("/predict/{game}")
async def predict(game: str, body: PredictRequest):
    try:
        # TODO : fill this in
        return {"label": "SomeCard", "confidence": 0.97}
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))

# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    uvicorn.run("src.api.server:app", host="0.0.0.0", port=8000, reload=True)


