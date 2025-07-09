# import os
# import logging 
# import json
# 
# from utils.product_lines import PRODUCTLINES as PLS
# 
# 
# 
# def load_dict(pl: PLS) -> dict:
#     # depickle a master dict or something
#     data_dir = os.getenv('DATA_DIR')
#     if data_dir is None:
#         msg = 'DATA_DIR env var is not set...'
#         logging.error(msg)
#         raise KeyError(msg)
# 
#     deckdrafterprod_path = os.path.join(data_dir, pl.value, 'deckdrafterprod.json')
# 
#     with open(deckdrafterprod_path, mode) as f:
#         deckdrafterprod = json.load(f)
# 
#     return deckdrafterprod

