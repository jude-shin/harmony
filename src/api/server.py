import os, logging, uvicorn

from fastapi import FastAPI, HTTPException, File, Form, UploadFile
from PIL import Image

from processing.image_processing import get_tensor_from_image
from utils.product_lines import string_to_product_line
from utils.data_conversion import label_to_id
from utils.tfs_models import identify, CachedConfigs

# from data.collect import generate_keys
# from data.collect import download_images_parallel

logging.getLogger().setLevel(0) # 20

CachedConfigs()

app = FastAPI(
        title='Harmony ML API',
        version='1.0.0',
        )

# Routes
@app.get('/ping', summary='testing')
async def ping() -> dict[str, str]:
    return {'ping': 'pong'}

# @app.get('/validate', summary='validates the structure of our application')
# async def validate():
#     logging.warning('validate endpoint not implemented yet')
#     return {'validate endpoint not implemented yet'}
# 
# @app.post('/process', summary='processes deckdrafterprod for the first time')
# async def process(
#         product_line_string: str = Form(..., description='productLine name (e.g., locrana, mtg)'),
#         ):
#     pl = string_to_product_line(product_line_string)
#     process_deckdrafterprod(pl)
#     return {}
# 
# @app.post('/process_images', summary='pickles the labels from the deckdrafterprod')
# async def process_images(
#         product_line_string: str = Form(..., description='productLine name (e.g., locrana, mtg)'),
#         ):
#     pl = string_to_product_line(product_line_string)
#     download_images_parallel(pl)
#     return {}
# 
# @app.post('/process_keys', summary='pickles the labels from the deckdrafterprod')
# async def process_keys(
#         product_line_string: str = Form(..., description='productLine name (e.g., locrana, mtg)'),
#         ):
#     pl = string_to_product_line(product_line_string)
#     generate_keys(pl)
#     return {}

@app.post('/predict')
async def predict(
        product_line_string: str = Form(..., description='productLine name (e.g., locrana, mtg)'),
        images: list[UploadFile] = File(..., description='image scans from client that are to be identified'),
        threshold: float = Form(..., description='what percent confidence that is deemed correct')
        ):

    pl = string_to_product_line(product_line_string)
    
    # with Process are we able to process all of these in parallel?
    # first find out if these use cpu instructions
    # from multiprocessing import Process
    # p1 = Process(target(func1))
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
    # NOTE: for simplicity we need the models to all comply to the same width and height
    input_width = CachedConfigs().request_config(pl)['m0']['input_width']
    input_height  = CachedConfigs().request_config(pl)['m0']['input_height']

    for pil_image in pil_images:
        img_tensor = get_tensor_from_image(pil_image, input_width, input_height)

        instance = img_tensor.numpy().tolist()
        instances.append(instance)

    predictions, confidences = identify(instances, 'm0', pl)

    json_prediction_obj = {
            'predictions': [label_to_id(int(p), pl) if p is not None else None for p in predictions],
            'confidences': confidences 
            }
    return json_prediction_obj

# ---------------------------------------------------------------------------
if __name__ == '__main__':
    uvicorn.run('src.api.server:app', host='0.0.0.0', port=os.getenv('API_PORT'), reload=True)

