import logging
import os
import io
import requests
import numpy as np

import httpx
from fastapi import UploadFile
from PIL import Image
from typing import Optional

from utils.product_lines import PRODUCTLINES as PLS
from utils.singleton import Singleton
from utils.file_handler.pickle import load_ids
from utils.file_handler.dir import get_saved_model_dir, get_config_path
from utils.file_handler.toml import * 


# ================================================

MAX_IMAGE_BYTES = 15 * 1024 * 1024  # 15MB per image
ALLOWED_CONTENT_TYPES = {"image/jpeg", "image/png"}

def validate_content_type(ct: Optional[str]) -> bool:
    if not ct:
        return False
    # strip parameters like `; charset=...`
    return ct.split(";", 1)[0].lower() in ALLOWED_CONTENT_TYPES

async def load_image_from_upload(f: UploadFile) -> Image.Image:
    data = await f.read()
    await f.close()
    if not data:
        raise ValueError("empty file")
    if len(data) > MAX_IMAGE_BYTES:
        raise ValueError("file too large")
    try:
        im = Image.open(io.BytesIO(data)).convert("RGB")
        im.load()
        return im
    except Exception as exc:
        raise ValueError(f"invalid image: {exc}") from exc

async def load_image_from_url(url: str, client: httpx.AsyncClient) -> Image.Image:
    # Basic scheme check
    if not (url.startswith("http://") or url.startswith("https://")):
        raise ValueError("unsupported URL scheme")

    async with client.stream("GET", url) as resp:
        if resp.status_code != 200:
            raise ValueError(f"http status {resp.status_code}")

        ct = resp.headers.get("content-type")
        if not validate_content_type(ct):
            raise ValueError(f"unsupported content-type: {ct}")

        # Optional content-length precheck
        cl = resp.headers.get("content-length")
        if cl and int(cl) > MAX_IMAGE_BYTES:
            raise ValueError("content-length exceeds limit")

        # Read up to MAX_IMAGE_BYTES + 1 to detect overflow
        buf = bytearray()
        async for chunk in resp.aiter_bytes():
            buf.extend(chunk)
            if len(buf) > MAX_IMAGE_BYTES:
                raise ValueError("downloaded image exceeds limit")

    try:
        im = Image.open(io.BytesIO(buf)).convert("RGB")
        im.load()
        return im
    except Exception as exc:
        raise ValueError(f"invalid image: {exc}") from exc


def identify(instances: list, model_name: str, pl: PLS, version: int) -> tuple[list[str], list[float]]:
    '''
    Identifies a card with multiple models, giving the most confident output.

    Args:
        instances (list): The list of preprocessed images that are converted to a tensor, which is converted to a list
        model_name (string): unique identifier for which (sub)model we are using for evaluation
            ex) in the model "m12.keras", the model_name is "m12"
            ex) in the labels toml "m0_labels.toml", the model_name is "m0"
        pl (PRODUCTLINES): The product_line we are working with.

    Returns:
        tuple[list[str | None], list[float]]: a tuple containing:
            - list of most confident labels (or None if there is some sort of error)
            - list of confidences corresponding to each label
    '''

    TFS_PORT = os.getenv('TFS_PORT')
    url = f'http://tfs-{pl.value}:{TFS_PORT}/v1/models/{model_name}/versions/{version}:predict'

    try:
        response = requests.post(url, json={'instances': instances}, timeout=10)
        response.raise_for_status()
        predictions = response.json().get('predictions', [])
    except Exception as e:
        logging.warning('Model [%s] failed to get predictions: %s', model_name, str(e))
        return [None] * len(instances), [0.0] * len(instances)

    final_prediction_labels = [None] * len(instances)
    confidences = [0.0] * len(instances)

    config = load_model_config(pl)

    if config[model_name]['is_final']: 
        for i, p in enumerate(predictions):
            try:
                p_np = np.array(p)
                best_idx = int(np.argmax(p_np))
                confidence = float(p_np[best_idx])
                final_prediction_labels[i] = best_idx
                confidences[i] = confidence
                logging.info('Model [%s] final prediction for image %d: %d (%.4f)', model_name, i, best_idx, confidence)
            except Exception as e:
                logging.warning('Model [%s] failed to process prediction for image %d: %s', model_name, i, str(e))
        return final_prediction_labels, confidences

    # Handle submodel routing
    submodel_inputs = {}
    image_indices_by_submodel = {}

    for i, p in enumerate(predictions):
        try:
            p_np = np.array(p)
            best_idx = int(np.argmax(p_np))
            confidence = float(p_np[best_idx])

            _ids: list[str] = load_ids(pl, model_name, 'rb') # TODO: move this outside of the for enumerate(predictions) loop

            next_model: str = _ids[best_idx]

            logging.info('Model [%s] defers image %d to submodel [%s] (label: %d)', model_name, i, next_model, best_idx)

            if next_model not in submodel_inputs:
                submodel_inputs[next_model] = []
                image_indices_by_submodel[next_model] = []

            submodel_inputs[next_model].append(instances[i])
            image_indices_by_submodel[next_model].append(i)
        except Exception as e:
            logging.warning('Model [%s] failed to defer image %d: %s', model_name, i, str(e))

    # Recurse into submodels
    for next_model, sub_instances in submodel_inputs.items():
        try:
            sub_labels, sub_confidences = identify(sub_instances, next_model, pl, version)
        except Exception as e:
            logging.warning('Submodel [%s] failed to identify batch: %s', next_model, str(e))
            sub_labels = [None] * len(sub_instances)
            sub_confidences = [0.0] * len(sub_instances)

        for idx, sub_label, sub_conf in zip(image_indices_by_submodel[next_model], sub_labels, sub_confidences):
            top_prediction = predictions[idx]
            top_conf = float(np.max(top_prediction))  # confidence of the top-level model
            final_prediction_labels[idx] = sub_label
            confidences[idx] = top_conf * sub_conf

    return final_prediction_labels, confidences
