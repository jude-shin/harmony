import os
import tomllib 

from utils.product_lines import PRODUCTLINES as PLS
from utils.file_handler.dir import * 

def load_config(pl: PLS) -> dict:
    config_path = get_config_path(pl)

    # # TODO: if there is no config, load a set of defaults
    # config_defaults = {
    #         'images': {'height': 12, 'width': 12},
    #         }

    with open(config_path, 'rb') as f:
        config = tomllib.load(f)

    return config 
