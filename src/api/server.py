from __future__ import annotations

import json
import os
import logging 
import typeguard
from pathlib import Path
from PIL import Image

import numpy as np
import tensorflow as tf

import uvicorn
from pydantic import BaseModel
from fastapi import FastAPI, HTTPException, File, Form, UploadFile

from harmony_config.product_lines import PRODUCTLINES, string_to_product_line
from utils.data_conversion import label_to_json, format_json
from helper.image_processing import get_tensor_from_image
from evaluation.evaluate import identify 

# Configuration
PRODUCTLINE = PRODUCTLINES.LORCANA.value
MODEL_DIR = Path(os.getenv('MODEL_DIR'), PRODUCTLINE) # change to os.path.join

MODEL_PATH = MODEL_DIR / 'model.keras'

# App & model initialisation
app = FastAPI(
        title='Harmony ML API',
        version='1.0.0',
        )

try:
    model = tf.keras.models.load_model(MODEL_PATH)
except (IOError, ValueError) as exc:
    raise SystemExit(f'Could not load model at {MODEL_PATH}: {exc}') from exc


# Routes
@app.get('/ping', summary='Health-check')
async def ping() -> dict[str, str]:
    return {'ping': 'pong'}

@app.post('/predict')
async def predict(
        product_line_string: str = Form(..., description='productLine name (e.g., locrana, mtg)'),
        image: UploadFile = File(..., description='image scan from client'),
        ):
    # TODO : add some kind of validation to make sure that the file structure is good, and all the imputs and outputs check out

    # TODO : return the top 3 or top 5 predictions in order (instead of just the biggest one)

    # get information prepared
    # image
    # product_line
    try:
        pil_image = Image.open(image.file).convert('RGB')
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=400, detail=f'invalid image: {exc}')

    product_line = string_to_product_line(product_line_string)

    best_prediction = identify(pil_image, 'm0', product_line)



    raw_json = label_to_json(int(best_prediction), product_line)

    

    formatted_json = format_json(raw_json, product_line)

    return json.loads(formatted_json)

# ---------------------------------------------------------------------------
if __name__ == '__main__':
    uvicorn.run('src.api.server:app', host='0.0.0.0', port=5000, reload=True)

