import os
import asyncio
import logging
from typing import List, Optional

import httpx
import uvicorn
from fastapi import FastAPI, HTTPException, File, Form, UploadFile
from PIL import Image

from processing.image_processing import get_tensor_from_image
from utils.product_lines import string_to_product_line
from utils.data_conversion import label_to_id
from utils.file_handler.dir import get_config_path
from utils.file_handler.toml import * 

from serving_api.tfs_models import identify
from serving_api.tfs_models import load_image_from_upload
from serving_api.tfs_models import load_image_from_url

# ---------------------------------------------------------------------------
# Logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# ---------------------------------------------------------------------------
app = FastAPI(title="Harmony ML API", version="1.0.0")

@app.get("/ping", summary="testing")
async def ping() -> dict[str, str]:
    return {"ping": "pong"}

@app.post("/predict")
async def predict(
        product_line_string: str = Form(..., description="productLine name (e.g., locrana, mtg)"),
        images: Optional[List[UploadFile]] = File(None, description="image scans to be identified"),
        image_urls: Optional[List[str]] = Form(None, description="http(s) URLs pointing to images"),
        threshold: float = Form(..., description="confidence threshold deemed correct"),
        version: int = Form(..., description="model version to use in the given product line"),
        ):
    # Require at least one source
    if not images and not image_urls:
        raise HTTPException(status_code=400, detail="provide at least one image file or image URL")

    pl = string_to_product_line(product_line_string)

    # Load inputs concurrently
    pil_images: List[Image.Image] = []
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
    input_width = config["m0"]["img_width"]
    input_height = config["m0"]["img_height"]

    # Preprocess
    instances = []
    for pil_image in pil_images:
        img_tensor = get_tensor_from_image(pil_image, input_width, input_height)
        instances.append(img_tensor.numpy().tolist())

    # Inference
    predictions, confidences = identify(instances, "m0", pl, version)

    # Response
    return {
            "predictions": [label_to_id(int(p), pl) if p is not None else None for p in predictions],
            "confidences": confidences,
            # threshold is currently unused in this handler; left as part of API for downstream logic
            }
