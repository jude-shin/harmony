import os
import tomllib 

from utils.product_lines import PRODUCTLINES as PLS
from utils.file_handler.dir import get_data_dir

def load_config() -> dict:
    data_dir = get_data_dir()
    config_path = os.path.join(data_dir, 'config.toml')

    # TODO: if there is no config, load a set of defaults
    config_defaults = {
            'images': {'height': 12, 'width': 12},
            }

    with open(config_path, 'rb+') as f:
        config = tomllib.load(f)

    return config 
