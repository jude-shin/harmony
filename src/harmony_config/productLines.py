from enum import Enum

class PRODUCTLINES(Enum):
    MTG = 'mtg'
    LORCANA = 'lorcana' 
    POKEMON = 'pokemon'

# make this all lowercase before looking it up
productLine_names = {
        'magic: the gathering': PRODUCTLINES.MTG,
        'mtg': PRODUCTLINES.MTG,
        'lorcana': PRODUCTLINES.LORCANA,
        'pokemon': PRODUCTLINES.POKEMON,
        }

def string_to_productLine(s : str) -> PRODUCTLINES:
    s = s.lower()
    return productLine_names[s]


