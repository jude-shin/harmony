from __future__ import annotations

import json
import os
import logging 
import typeguard
from PIL import Image
import requests
import numpy as np

import uvicorn
# from pydantic import BaseModel
from fastapi import FastAPI, HTTPException, File, Form, UploadFile

from helper.image_processing import get_tensor_from_image
from utils.product_lines import string_to_product_line
from utils.data_conversion import label_to_json
from utils.tfs_models import identify, get_model_metadata

logging.getLogger().setLevel(0)

app = FastAPI(
        title='Harmony ML API',
        version='1.0.0',
        )

# Routes
@app.get('/ping', summary='Health-check')
async def ping() -> dict[str, str]:
    return {'ping': 'pong'}

@app.post('/predict')
async def predict(
        product_line_string: str = Form(..., description='productLine name (e.g., locrana, mtg)'),
        # TODO : add the ability to request multiple images within the same productline
        image: UploadFile = File(..., description='image scan from client'),
        ):
    # TODO : add some kind of validation to make sure that the file structure is good, and all the imputs and outputs check out

    # TODO : return the top 3 or top 5 predictions in order (instead of just the biggest one)

    try:
        pil_image = Image.open(image.file).convert('RGB')
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=400, detail=f'invalid image: {exc}')

    # _, model_img_width, model_img_height, _ = model.input_shape
    # model_img_width, model_img_height = 450, 650

    pl = string_to_product_line(product_line_string)

    metadata = get_model_metadata('m0', pl)
    model_img_width = int(metadata['metadata']['signature_def']['signature_def']['serve']['inputs']['input_layer']['tensor_shape']['dim'][1]['size'])
    model_img_height = int(metadata['metadata']['signature_def']['signature_def']['serve']['inputs']['input_layer']['tensor_shape']['dim'][2]['size'])

    img_tensor = get_tensor_from_image(pil_image, model_img_width, model_img_height)
    img_tensor = np.expand_dims(img_tensor, axis=0)
    instance = img_tensor.tolist()

    best_prediction = identify(instance, 'm0', pl)

    json_prediction_obj = label_to_json(int(best_prediction), pl)
    return json_prediction_obj

# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
if __name__ == '__main__':
    uvicorn.run('src.api.server:app', host='0.0.0.0', port=os.getenv('API_PORT'), reload=True)

