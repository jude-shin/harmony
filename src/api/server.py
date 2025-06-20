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

from harmony_config.productLines import PRODUCTLINES, string_to_productLine
from utils.data_conversion import label_to_json, format_json 
from helper.image_processing import get_tensor_from_image

# Configuration
PRODUCTLINE = PRODUCTLINES.LORCANA.value
MODEL_DIR = Path(os.getenv("MODEL_DIR"), PRODUCTLINE)

MODEL_PATH = MODEL_DIR / "model.keras"
PORT = int(os.getenv("PORT", 5000))

# App & model initialisation
app = FastAPI(
        title="Harmony ML API",
        version="1.0.0",
        )

try:
    model = tf.keras.models.load_model(MODEL_PATH)
except (IOError, ValueError) as exc:
    raise SystemExit(f"Could not load model at {MODEL_PATH}: {exc}") from exc


# Routes
@app.get("/ping", summary="Health-check")
async def ping() -> dict[str, str]:
    return {"ping": "pong"}

@app.post("/predict")
async def predict(
        productLineString: str = Form(..., description="productLine name (e.g., locrana, mtg)"),
        image: UploadFile = File(..., description="image scan from client"),
        ):

    try:
        pil_image = Image.open(image.file).convert("RGB")
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=400, detail=f"invalid image: {exc}")

    _, model_img_width, model_img_height, _ = model.input_shape

    img_tensor = get_tensor_from_image(pil_image, model_img_width, model_img_height)

    img_tensor = np.expand_dims(img_tensor, axis=0)
    prediction = model.predict(img_tensor)

    prediction, confidence = np.argmax(
        prediction), prediction[0, np.argmax(prediction)]

    print("\nconfidence: ", confidence)
    print("\nprediction: ", prediction)

    productLine = string_to_productLine(productLineString)

    raw_json = label_to_json(int(prediction), productLine)
    formatted_json = format_json(raw_json, productLine)

    return json.loads(formatted_json)



# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    uvicorn.run("src.api.server:app", host="0.0.0.0", port=5000, reload=True)


