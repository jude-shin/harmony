import os
import typeguard

from enum import Enum

class PRODUCTLINES(Enum):
    # MTG = 'mtg'
    LORCANA = 'lorcana' 
    POKEMON = 'pokemon'

# make this all lowercase before looking it up
product_line_names = {
        # 'magic: the gathering': PRODUCTLINES.MTG,
        # 'mtg': PRODUCTLINES.MTG,
        'lorcana': PRODUCTLINES.LORCANA,
        'pokemon': PRODUCTLINES.POKEMON,
        }

def string_to_product_line(s : str) -> PRODUCTLINES:
    s = s.lower()
    return product_line_names[s]

def get_port(pl: PRODUCTLINES) -> str:
    return os.getenv(pl.value.upper() + '_PORT')



