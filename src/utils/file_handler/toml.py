import os
import toml
import tomllib

from utils.product_lines import PRODUCTLINES as PLS
from utils.file_handler.dir import get_config_path

# TODO: rename this package to "config" or something
def load_model_config(pl: PLS) -> dict[str, str | int | bool | list[str]]:
    # string, int, float, dict of strings, boolean
    path: str = get_config_path(pl)

    with open(path, 'rb') as f:
        config = tomllib.load(f)

    return config

def save_model_config(new_config_path: str, model: str, config: dict[str, str | int | bool | list[str]]) -> None:
    new_config_path = os.path.join(new_config_path, model+'.toml')
    with open(new_config_path, 'w+') as f:
        _ = toml.dump(config, f)
