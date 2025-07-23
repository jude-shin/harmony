import os
import pickle

from utils.product_lines import PRODUCTLINES as PLS
from utils.file_handler.dir import get_data_dir

def load_ids(pl: PLS, prefix: str, mode: str) -> list[str]:
    '''
    Depickle the list of labels. The index is the label, and the string inside is the _id

    Args:
        pl (PLS): TODO
        prefix (str): the name of the sub_model that is going to be used
            ex) the prefix of master_ids.pkl is 'master'
            ex) the prefix of m0_ids.pkl is 'm0'
    Returns:
        list[str]: the list of the _ids that were pickled
        
    '''
    data_dir = get_data_dir()
    _ids_path = os.path.join(data_dir, pl.value, f'{prefix}_ids.pkl')

    with open(_ids_path, mode) as f:
        _ids = pickle.load(f)

    return _ids 
