from enum import Enum

class PRODUCTLINES(Enum):
    '''
        mapping from an enum to a string
        everything in harmony's system will be using this string's naming convention (folders, models, etc)
        ex) harmony/data/lorcana/images]/ will hold LORCANA's images
    '''

    # MTG = 'mtg'
    LORCANA = 'lorcana' 
    POKEMON = 'pokemon'

def string_to_product_line(s : str) -> PRODUCTLINES:
    '''
        looks up the various ways a product line could be referenced outside of harmony's system
        ex) PRODUCTLINES.MTG can be referenced from storepass as mtg, MTG, 'Magic: the Gathering', 'magic the gathering'....
        there is probably a better way to do this
    '''

    s = s.lower()

    product_line_names = {
            # 'magic: the gathering': PRODUCTLINES.MTG,
            # 'mtg': PRODUCTLINES.MTG,
            'lorcana': PRODUCTLINES.LORCANA,
            'pokemon': PRODUCTLINES.POKEMON,
        }

    return product_line_names[s]

