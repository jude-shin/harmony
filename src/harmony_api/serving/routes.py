import logging
import asyncio
import httpx

import tensorflow as tf
import numpy as np

from fastapi import HTTPException, File, Form, UploadFile, APIRouter
from PIL import Image
from logging import Logger
from tensorflow import Tensor
from numpy.typing import NDArray

from utils.product_lines import PRODUCTLINES as PLS, string_to_product_line
from utils.file_handler.toml import load_model_config
from utils.data_conversion import label_to_id
from processing.image_processing import get_tensor_from_image

from harmony_api.serving.services import load_image_from_upload, load_image_from_url, identify

logger: Logger = logging.getLogger(__name__)

router: APIRouter = APIRouter(
    prefix="/serving",
)

@router.post("/predict")
async def predict(
    product_line_string: str = Form(..., description="productLine name (e.g., locrana, mtg)"),
    images: list[UploadFile] | None = File(None, description="image scans to be identified"),
    image_urls: list[str] | None = Form(None, description="http(s) URLs pointing to images"),
    threshold: float = Form(..., description="confidence threshold deemed correct"),
    version: int = Form(..., description="model version to use in the given product line"),
):
    # Require at least one source
    if not images and not image_urls:
        raise HTTPException(status_code=400, detail="provide at least one image file or image URL")

    # TODO: sanitize inputs
    pl = string_to_product_line(product_line_string)

    # Load inputs concurrently
    pil_images: list[Image.Image] = []
    tasks = []

    # file tasks
    for f in images or []:
        tasks.append(load_image_from_upload(f))

    # url tasks
    async with httpx.AsyncClient(
        timeout=httpx.Timeout(connect=5.0, read=20.0, write=5.0, pool=5.0)
    ) as client:
        for u in image_urls or []:
            tasks.append(load_image_from_url(u, client))

        results = await asyncio.gather(*tasks, return_exceptions=True)

    # Collect successes; log failures
    for res in results:
        if isinstance(res, Image.Image):
            pil_images.append(res)
        else:
            logger.warning("skipping input: %s", res)

    if not pil_images:
        raise HTTPException(status_code=400, detail="no valid images provided")

    # Model config
    config = load_model_config(pl)

    # TODO: do not hardcode these
    input_width: int | bool | str | list[str] = config.get('m0', 'img_width')
    input_height: int | bool | str | list[str] = config.get('m0', 'img_height')
    if type(input_width) is not int:
        logger.error('img_width in m0 must be a defined integer')
        raise HTTPException(status_code=500, detail='internal config error')
    if type(input_height) is not int:
        logger.error('img_width in m0 must be a defined integer')
        raise HTTPException(status_code=500, detail='internal config error')

    # Preprocess
    instances: list[Tensor]= []
    for pil_image in pil_images:
        img_tensor: Tensor = get_tensor_from_image(pil_image, input_width, input_height)
        instances.append(img_tensor.numpy().tolist())

    # Inference
    predictions, confidences = identify(instances, "m0", pl, version)

    # Response
    return {
        "predictions": [label_to_id(int(p), pl) if p is not None else None for p in predictions],
        "confidences": confidences,
        # threshold is currently unused in this handler; left as part of API for downstream logic
    }
