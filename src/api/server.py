from __future__ import annotations

import json
import os
from pathlib import Path
from PIL import Image

import numpy as np
import tensorflow as tf

import uvicorn
from pydantic import BaseModel
from fastapi import FastAPI, HTTPException, File, Form, UploadFile

from harmony_config.structs import GAMES
from utils.data_conversion import label_to_json, format_json 
from helper.image_processing import get_tensor_from_image

# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
GAME = GAMES.LORCANA.value
MODEL_DIR = Path(os.getenv("MODEL_DIR"), GAME)

MODEL_PATH = MODEL_DIR / "model.keras"
PORT = int(os.getenv("PORT", 8000))

IMG_HEIGHT = 437 
IMG_WIDTH = 313 
DEFAULT_THRESHOLD = float(os.getenv("THRESHOLD", 0.60))

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
async def predict(
        game: str,
        image: UploadFile = File(..., description="JPEG scan from client"),
        card_index: int = Form(-1, description="Ground-truth index, or -1 if unknown"),
        img_width: int = Form(..., description="Original client image width in pixels"), #TODO : no need for this
        img_height: int = Form(..., description="Original client image height in pixels"), #TODO : no need for this
        threshold: float = Form(
            DEFAULT_THRESHOLD,
            description="Confidence threshold sent by the client",
            ),
        ):

    try:
        pil_image = Image.open(image.file).convert("RGB")
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=400, detail=f"invalid image: {exc}")

    # if (img_width, img_height) != pil_image.size:
    #     # Log or warn; we won’t hard-fail because cropping/resizing may follow
    #     print(
    #         f"[WARN] Client size ({img_width}×{img_height}) "
    #         f"!= actual {pil_image.size} for file {image.filename}"
    #         )


    _, model_img_width, model_img_height, _ = model.input_shape

    img_tensor = get_tensor_from_image(pil_image, model_img_width, model_img_height)

    img_tensor = np.expand_dims(img_tensor, axis=0)
    prediction = model.predict(img_tensor)

    prediction, confidence = np.argmax(
        prediction), prediction[0, np.argmax(prediction)]

    print("\nconfidence: ", confidence)
    print("\nprediction: ", prediction)

    
    raw_json = label_to_json(int(prediction), GAMES.LORCANA)
    formatted_json = format_json(raw_json, GAMES.LORCANA)


    return json.loads(formatted_json)



# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    uvicorn.run("src.api.server:app", host="0.0.0.0", port=8000, reload=True)


