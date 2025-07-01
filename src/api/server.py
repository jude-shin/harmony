import json, os, logging, requests, numpy as np, uvicorn, typeguard

from fastapi import FastAPI, HTTPException, File, Form, UploadFile
from PIL import Image

from helper.image_processing import get_tensor_from_image
from utils.product_lines import string_to_product_line
from utils.data_conversion import label_to_json, label_to_id
from utils.tfs_models import identify, get_model_metadata

logging.getLogger().setLevel(10)

app = FastAPI(
        title='Harmony ML API',
        version='1.0.0',
        )


# TODO : add some kind of validation to make sure that the file structure is good, and all the imputs and outputs check out


# Routes
@app.get('/ping', summary='Health-check')
async def ping() -> dict[str, str]:
    return {'ping': 'pong'}

@app.post('/predict')
async def predict(
        product_line_string: str = Form(..., description='productLine name (e.g., locrana, mtg)'),
        images: list[UploadFile] = File(..., description='image scans from client that are to be identified'),
        threshold: float = Form(..., description='what percent confidence that is deemed correct')
        ):
    pl = string_to_product_line(product_line_string)
    
    pil_images = []
    for image in images:
        try:
            pil_image = Image.open(image.file).convert('RGB')
        except Exception as exc:  # noqa: BLE001
            logging.error(HTTPException(status_code=400, detail=f'invalid image: {exc}'))
            pil_image = None
            continue
        pil_images.append(pil_image)

    
    instances = []
    # TODO : cache the model metadata somewhere
    metadata = get_model_metadata('m0', pl)
    model_img_width = int(metadata['metadata']['signature_def']['signature_def']['serve']['inputs']['input_layer']['tensor_shape']['dim'][1]['size'])
    model_img_height = int(metadata['metadata']['signature_def']['signature_def']['serve']['inputs']['input_layer']['tensor_shape']['dim'][2]['size'])

    for pil_image in pil_images:
        img_tensor = get_tensor_from_image(pil_image, model_img_width, model_img_height)

        instance = img_tensor.numpy().tolist()
        instances.append(instance)

    predictions, confidences = identify(instances, 'm0', pl)

    json_prediction_obj = {
            'predictions': list(filter(lambda p: ('' if (p is None) else label_to_id(int(p), pl)), predictions)), 
            'confidences': confidences 
            }
    return json_prediction_obj

# ---------------------------------------------------------------------------
if __name__ == '__main__':
    uvicorn.run('src.api.server:app', host='0.0.0.0', port=os.getenv('API_PORT'), reload=True)

