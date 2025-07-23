import os
import json

from utils.product_lines import PRODUCTLINES as PLS
from utils.file_handler.dir import get_data_dir

def load_deckdrafterprod(pl: PLS, mode: str) -> dict:
    data_dir = get_data_dir()
    deckdrafterprod_path = os.path.join(data_dir, pl.value, 'deckdrafterprod.json')

    with open(deckdrafterprod_path, mode) as f:
        deckdrafterprod = json.load(f)

    return deckdrafterprod

