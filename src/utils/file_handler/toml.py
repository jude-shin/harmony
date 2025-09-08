import toml
import tomllib 

from utils.product_lines import PRODUCTLINES as PLS
from utils.file_handler.dir import * 

# TODO: rename this package to "config" or something
def load_model_config(pl: PLS) -> dict:
    path = get_config_path(pl)

    with open(path, 'rb') as f:
        config = tomllib.load(f)

    return config 

def save_model_config(new_config_path: str, model: str, config: dict) -> dict:
    new_config_path = os.path.join(new_config_path, model+'.toml')
    with open(new_config_path, 'w+') as f:
        toml.dump(config, f) 

    return config 
