from __future__ import annotations

import json
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
from utils.data_conversion import label_to_json, format_json
from evaluation.evaluate import identify, CachedModels

logging.getLogger().setLevel(0)


logging.info(' [main] STARTING Caching Tensorflow models... This may take a while... ')
CachedModels()
logging.info(' [main] FINISHED caching Tensorflow models!')


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
    model_img_width, model_img_height = 450, 650
    img_tensor = get_tensor_from_image(pil_image, model_img_width, model_img_height)
    img_tensor = np.expand_dims(img_tensor, axis=0)


    product_line = string_to_product_line(product_line_string)

    # instread, create a post request to the docker containers that
    # best_prediction = identify(pil_image, 'm0', product_line)
    url = 'http://tfs-lorcana:8501/v1/models/m0:predict'
    response = requests.post(url, json={'instances': img_tensor.tolist()})

    return response.json()
    
    # translation
    # json_prediction_obj = label_to_json(int(best_prediction), product_line)
    # return json_prediction_obj


# ---------------------------------------------------------------------------
if __name__ == '__main__':
    uvicorn.run('src.api.server:app', host='0.0.0.0', port=5000, reload=True)

