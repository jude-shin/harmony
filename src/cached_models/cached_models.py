from __future__ import annotations

import json
import logging 
import typeguard
import os
from PIL import Image

import uvicorn
# from pydantic import BaseModel
from fastapi import FastAPI, HTTPException, File, Form, UploadFile

from data_defs.product_lines import string_to_product_line
from data_defs.product_lines import PRODUCTLINES as PLS 
from utils.data_conversion import label_to_json, format_json
from evaluation.evaluate import identify, get_model_config, get_model


class CachedModels(object):
    # hashmap of hashmaps_
    # productline -> models

    # instead of .getIntance(), just instantiate a new  
    def __new__(cls):
        if not hasatrr(cls, 'instance'):
            logging.info(' CachedModels: no singleton instance. Intantiating a new one.')
            cls.instance = super(CachedModels, cls).__new__(cls)

            cls.instance._models = {}

            for pl in PLS:
                models = {}
                config_toml = get_model_config(pl)

                for model_name in config_toml:
                    models[model_name] = get_model(model_name, pl)

                cls.instance._models[pl.value] = models

        return cls.instance

    # I may have just done all of this for no reason...
    # TODO: either move this into the evaluate.py file, or move the evaluate functions into the utils file

