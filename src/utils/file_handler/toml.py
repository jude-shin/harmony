import toml
import tomllib 

from utils.product_lines import PRODUCTLINES as PLS
from utils.file_handler.dir import * 

def load_config(config_path: str) -> dict:
    with open(config_path, 'rb') as f:
        config = tomllib.load(f)

    return config 

def save_config(new_config_path: str, model: str, config: dict) -> dict:
    new_config_path = os.path.join(new_config_path, model+'.toml')
    with open(new_config_path, 'w+') as f:
        toml.dump(config, f) 

    return config 
